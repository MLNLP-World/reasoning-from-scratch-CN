# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》配套源码：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

from .utils import download_file

from pathlib import Path
import re

import torch
import torch.nn as nn


# 6 亿参数
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,     # 词表大小
    "context_length": 40_960,  # 原始训练所用的序列长度
    "emb_dim": 1024,           # 嵌入维度
    "n_heads": 16,             # 注意力头数量
    "n_layers": 28,            # 层数
    "hidden_dim": 3072,        # 前馈网络中间维度的大小
    "head_dim": 128,           # GQA 中每个头的维度
    "qk_norm": True,           # 是否对 GQA 中的查询和键进行归一化
    "n_kv_groups": 8,          # GQA 的键值组数量
    "rope_base": 1_000_000.0,  # RoPE 中 theta 的基数
    "dtype": torch.bfloat16,   # 使用较低精度的数据类型以减少内存占用
}


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 主要模型参数
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # 使用 ModuleList，因为 Sequential 只能接收一个输入，而这里需要 `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # 可复用的辅助组件
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0  # 记录 KV 缓存中的当前位置

    def forward(self, in_idx, cache=None):
        # 前向传播
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0  # 并非必需，但有助于 torch.compile
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1
            )
        # 预填充（无缓存）时，掩码初始形状为 (num_tokens, num_tokens)
        # 缓存解码时，掩码初始形状为 (num_tokens, prev_k_number_tokens + num_tokens)
        #
        # 添加两个前导维度，使掩码变为
        # 预填充时的 (1, 1, num_tokens, num_tokens)，以及
        # 缓存解码时的 (1, 1, num_tokens, total_key_tokens)。
        # 这些额外维度使 PyTorch 能够广播同一个掩码
        # 并在将其应用于以下形状的 attn_scores 时覆盖所有批次和注意力头：
        # 形状：(batch, num_heads, num_tokens, total_key_tokens)。
        mask = mask[None, None, :, :]  # 广播掩码

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     start_pos=pos_start,
                                     cache=blk_cache)
            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        # 注意力块的残差连接
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)  # 形状为 [batch_size, num_tokens, emb_size]
        x = x + shortcut  # 加回原始输入

        # 前馈网络块的残差连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # 加回原始输入

        return x, next_cache


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads 必须能被 num_kv_groups 整除"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "未设置 `head_dim` 时，`d_in` 必须能被 `num_heads` 整除"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape

        # 应用投影
        queries = self.W_query(x)  # 形状：(b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # 形状：(b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # 形状：(b, num_tokens, num_kv_groups * head_dim)

        # 重塑为注意力头/键值组
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # 可选的归一化
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # 应用 RoPE
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            start_pos = 0  # 重置 RoPE
            keys, values = keys_new, values_new
        next_cache = (keys, values)

        # 扩展 K 和 V 以匹配注意力头数量
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # 注意力计算
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context), next_cache


# ==============================================================================
# RoPE 实现概述
#
#
# RoPE 有两种常见实现方式，它们在数学上
# 是等价的；
# 主要区别在于旋转矩阵如何对维度进行配对。
#
# 1）对半拆分方式（本仓库及 Hugging Face Transformers）：
#
# 以隐藏维度 d = 4 为例：
#
#       [ x0   x1 | x2   x3 ]
#         │    │    │    │
#         ▼    ▼    ▼    ▼
#        cos  cos  sin  sin
#
# 旋转矩阵：
#
#       [ cosθ0   0    -sinθ0   0   ]
#       [  0    cosθ1    0    -sinθ1]
#       [ sinθ0   0     cosθ0   0   ]
#       [  0    sinθ1    0     cosθ1]
#
# 这里先将嵌入维度拆分成两半，然后
# 分别按块进行旋转。
#
#
# 2）交错（偶数/奇数）方式（原论文及 Llama 仓库）：
#
# 以隐藏维度 d = 4 为例：
#
#       [ x0   x1   x2   x3 ]
#         │    │    │    │
#         ▼    ▼    ▼    ▼
#        cos  sin  cos  sin
#
# 旋转矩阵：
#
#       [ cosθ0  -sinθ0   0       0    ]
#       [ sinθ0   cosθ0   0       0    ]
#       [  0        0    cosθ1  -sinθ1 ]
#       [  0        0    sinθ1   cosθ1 ]
#
#
# 这里将嵌入维度按偶数/奇数的余弦/正弦对交错排列。
#
# 两种布局编码的相对位置相同；唯一的区别是
# 维度的配对方式。
# ==============================================================================


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "嵌入维度必须为偶数"

    # 计算逆频率
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # 生成位置索引
    positions = torch.arange(context_length, dtype=dtype)

    # 计算角度
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # 形状：(context_length, head_dim // 2)

    # 扩展角度以匹配 head_dim
    angles = torch.cat([angles, angles], dim=1)  # 形状：(context_length, head_dim)

    # 预先计算正弦和余弦
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin, offset=0):
    # x 的形状：(batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "注意力头维度必须为偶数"

    # 将 x 拆分为前半部分和后半部分
    x1 = x[..., : head_dim // 2]  # 前半部分
    x2 = x[..., head_dim // 2:]  # 后半部分

    # 调整 sin 和 cos 的形状
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)  # 形状：(1, 1, seq_len, head_dim)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    # 应用旋转变换
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # 应用 cos 和 sin 旋转后可以使用较低精度
    return x_rotated.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(self, tokenizer_file_path="tokenizer-base.json",
                 apply_chat_template=False,
                 add_generation_prompt=False,
                 add_thinking=False):
        from tokenizers import Tokenizer

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_path = Path(tokenizer_file_path)
        if not tok_path.is_file():
            raise FileNotFoundError(
                f"未找到分词器文件 '{tok_path}'。请确认该文件可用。"
            )

        self._tok = Tokenizer.from_file(str(tok_path))
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}

        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._special_to_id.get(self.pad_token)

        # 与 HF 行为保持一致：对话模型使用 <|im_end|>，基础模型使用 <|endoftext|>
        fname = tok_path.name.lower()
        if "base" in fname and "reasoning" not in fname:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = "<|im_end|>"
        self.eos_token_id = self._special_to_id.get(self.eos_token)

    def encode(self, prompt, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = prompt.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            prompt = self._wrap_chat(prompt)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, token_ids):
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"  # 不插入 <think> 标签，只添加一个换行符
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s


class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None


def download_qwen3_small(kind="base", tokenizer_only=False, out_dir="."):
    files = {
        "base": {"model": "qwen3-0.6B-base.pth", "tokenizer": "tokenizer-base.json"},
        "reasoning": {"model": "qwen3-0.6B-reasoning.pth", "tokenizer": "tokenizer-reasoning.json"},
    }
    if kind not in files:
        raise ValueError("kind 必须为 'base' 或 'reasoning'")

    repo = "rasbt/qwen3-from-scratch"
    hf_fmt = "https://huggingface.co/{repo}/resolve/main/{file}"
    backup_root = "https://f001.backblazeb2.com/file/reasoning-from-scratch/qwen3-0.6B"
    targets = ["tokenizer"] if tokenizer_only else ["model", "tokenizer"]

    for key in targets:
        fname = files[kind][key]
        primary = hf_fmt.format(repo=repo, file=fname)
        backup = f"{backup_root}/{fname}"
        download_file(primary, out_dir=out_dir, backup_url=backup)


def download_qwen3_grpo_checkpoints(
    grpo_type="no_kl",
    step="00050",
    out_dir=".",
):
    mapper = {
        "no_kl": "grpo_original_no_kl",
        "tracking": "7_3_plus_tracking/checkpoints",
        "clip_ratio": "7_4_plus_clip_ratio/checkpoints",
        "kl": "7_5_plus_kl/checkpoints",
        "format_reward": "7_6_plus_format_reward/checkpoints",
    }
    if grpo_type not in mapper:
        raise ValueError(f"目前仅支持以下 grpo_type：{mapper.keys()}")

    repo = "rasbt/qwen3-from-scratch-grpo-checkpoints"
    step = str(step)
    if step.isdigit():
        step = step.zfill(5)
    fname = f"qwen3-0.6B-rlvr-grpo-step{step}.pth"
    primary = f"https://huggingface.co/{repo}/resolve/main/{mapper[grpo_type]}/{fname}"

    backup = None
    if grpo_type == "no_kl" and step == "00050":
        backup_root = (
            "https://f001.backblazeb2.com/file/"
            "reasoning-from-scratch/qwen3-0.6B-checkpoints"
        )
        fname = (
            "grpo_original_no_kl/qwen3-0.6B-rlvr-grpo-step00050.pth"
        )
        backup = f"{backup_root}/{fname}"

    return download_file(primary, out_dir=out_dir, backup_url=backup)


def download_qwen3_distill_checkpoints(
    distill_type="deepseek_r1",
    step="06682",
    out_dir=".",
):
    mapper = {
        "deepseek_r1": {
            "06682": "qwen3-0.6B-distill-step06682-epoch1.pth",
            "13364": "qwen3-0.6B-distill-step13364-epoch2.pth",
            "20046": "qwen3-0.6B-distill-step20046-epoch3.pth",
        },
        "qwen3_235b_a22b": {
            "05746": "qwen3-0.6B-distill-step05746-epoch1.pth",
            "11492": "qwen3-0.6B-distill-step11492-epoch2.pth",
            "17238": "qwen3-0.6B-distill-step17238-epoch3.pth",
        },
    }
    folder_map = {
        "deepseek_r1": "ch08_distill_deepseek_r1/checkpoints",
        "qwen3_235b_a22b": "ch08_distill_qwen3_235b_a22b/checkpoints",
    }
    if distill_type not in mapper:
        raise ValueError(f"目前仅支持以下 distill_type：{mapper.keys()}")

    step = str(step)
    if step.isdigit():
        step = step.zfill(5)
    if step not in mapper[distill_type]:
        raise ValueError(
            f"仅支持以下 step：{mapper[distill_type].keys()}（distill_type={distill_type}）"
        )

    repo = "rasbt/qwen3-from-scratch-distill-checkpoints"
    fname = mapper[distill_type][step]
    primary = f"https://huggingface.co/{repo}/resolve/main/{folder_map[distill_type]}/{fname}"
    return download_file(primary, out_dir=out_dir)


def load_hf_weights_into_qwen(model, param_config, params):
    """
    仅在附录 D 中用于加载其他 Qwen3 变体。
    """
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"张量 '{tensor_name}' 的形状不匹配。左侧：{left.shape}，右侧：{right.shape}")

        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

        return left

    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):  # noqa: E741
        block = model.trf_blocks[l]
        att = block.att

        # Q、K、V 投影
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )

        # 输出投影
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )

        # QK 归一化层
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )

        # 注意力层归一化
        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # 前馈网络权重
        if "num_experts" in param_config:
            # 加载路由器（门控）权重
            block.ff.gate.weight = assign(
                block.ff.gate.weight,
                params[f"model.layers.{l}.mlp.gate.weight"],
                f"model.layers.{l}.mlp.gate.weight"
            )
            # 加载专家权重
            for e in range(param_config["num_experts"]):
                prefix = f"model.layers.{l}.mlp.experts.{e}"
                block.ff.fc1[e].weight = assign(
                    block.ff.fc1[e].weight,
                    params[f"{prefix}.gate_proj.weight"],
                    f"{prefix}.gate_proj.weight"
                )
                block.ff.fc2[e].weight = assign(
                    block.ff.fc2[e].weight,
                    params[f"{prefix}.up_proj.weight"],
                    f"{prefix}.up_proj.weight"
                )
                block.ff.fc3[e].weight = assign(
                    block.ff.fc3[e].weight,
                    params[f"{prefix}.down_proj.weight"],
                    f"{prefix}.down_proj.weight"
                )
                # 分配权重后，将专家层从 meta 设备移至 CPU
                block.ff.fc1[e] = block.ff.fc1[e].to("cpu")
                block.ff.fc2[e] = block.ff.fc2[e].to("cpu")
                block.ff.fc3[e] = block.ff.fc3[e].to("cpu")

        else:
            block.ff.fc1.weight = assign(
                block.ff.fc1.weight,
                params[f"model.layers.{l}.mlp.gate_proj.weight"],
                f"model.layers.{l}.mlp.gate_proj.weight"
            )
            block.ff.fc2.weight = assign(
                block.ff.fc2.weight,
                params[f"model.layers.{l}.mlp.up_proj.weight"],
                f"model.layers.{l}.mlp.up_proj.weight"
            )
            block.ff.fc3.weight = assign(
                block.ff.fc3.weight,
                params[f"model.layers.{l}.mlp.down_proj.weight"],
                f"model.layers.{l}.mlp.down_proj.weight"
            )

        block.norm2.scale = assign(
            block.norm2.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # 最终归一化层和输出头
    model.final_norm.scale = assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = model.tok_emb.weight
        print("模型使用权重绑定。")
