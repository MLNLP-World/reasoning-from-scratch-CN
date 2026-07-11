# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》配套源码：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

from .qwen3 import KVCache, download_qwen3_small, Qwen3Tokenizer

from pathlib import Path

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
    """支持填充掩码的 Qwen3 批处理实现。

    参数：
        cfg：模型配置字典。
        float32_upcast：默认让注意力分数和 softmax 计算路径保持
            float32。批处理实现同时使用因果掩码和左填充掩码；对于
            很长或包含大量填充的批次，其数值稳定性比单序列路径更
            脆弱。因此，在批量生成和评估中使用 float32 是更安全的
            默认选择。简而言之，float32_upcast 可使结果与无填充的
            单样本版本等价，但速度更慢且内存占用更多。如果更看重
            较低的内存占用和更快的训练速度，可以通过
            `Qwen3Model(..., float32_upcast=False)` 将其关闭。
    """

    def __init__(self, cfg, float32_upcast=True):
        super().__init__()

        # 主要模型参数
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # 使用 ModuleList，因为 Sequential 只能接收一个输入，而这里需要 `x, mask, cos, sin`
            [
                TransformerBlock(cfg, float32_upcast=float32_upcast)
                for _ in range(cfg["n_layers"])
            ]
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

    def forward(self, in_idx, cache=None, attn_mask=None):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        B, num_tokens = x.shape[0], x.shape[1]

        # 如果存在缓存，则根据其内容（第 0 层 K 的长度）推导 pos_start
        if cache is not None and cache.get(0) is not None:
            prev_k0, _ = cache.get(0)                 # 形状：(B, G_kv, L_prev, D)
            pos_start = prev_k0.size(2)               # L_prev
        else:
            pos_start = 0

        pos_end = pos_start + num_tokens

        # 为 [Q=num_tokens, K=pos_end] 构建因果掩码
        base = torch.triu(
            torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
        )
        causal4d = base[pos_start:pos_end, :pos_end][None, None, :, :]

        has_pad = attn_mask is not None and (~attn_mask[:, :pos_end]).any().item()
        if has_pad:
            # 屏蔽填充键，使其不出现在 softmax 的分母中
            kpm = (attn_mask[:, :pos_end] == 0).view(B, 1, 1, pos_end)
            mask = causal4d | kpm
        else:
            mask = causal4d

        pos_ids_current = torch.arange(pos_start, pos_end, device=x.device).unsqueeze(0).expand(B, -1)

        # 将填充查询行置零，使其 Q/K/V 均为零且不影响缓存
        if attn_mask is not None:
            qmask = attn_mask[:, pos_start:pos_end].unsqueeze(-1)
            x = x * qmask.to(x.dtype)

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     cache=blk_cache,
                                     pos_ids=pos_ids_current)
            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    # 保留此项以兼容常规的非批处理 generate_text_basic_cache 函数
    def reset_kv_cache(self):
        pass


class TransformerBlock(nn.Module):
    def __init__(self, cfg, float32_upcast=True):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"],
            float32_upcast=float32_upcast,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, cache=None, pos_ids=None):
        # 注意力块的残差连接
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, cache=cache, pos_ids=pos_ids)  # 形状为 [batch_size, num_tokens, emb_size]
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
        self,
        d_in,
        num_heads,
        num_kv_groups,
        head_dim=None,
        qk_norm=False,
        dtype=None,
        float32_upcast=True,
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
        self.float32_upcast = float32_upcast

    def forward(self, x, mask, cos, sin, cache=None, pos_ids=None):
        b, num_tokens, _ = x.shape

        # 应用投影
        queries = self.W_query(x)  # 形状：(b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # 形状：(b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # 形状：(b, num_tokens, num_kv_groups * head_dim)

        # 重塑形状
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # 可选的归一化
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # 应用 RoPE（使用逐词元位置 ID）
        queries = apply_rope_with_pos_ids(queries, cos, sin, pos_ids)
        keys_new = apply_rope_with_pos_ids(keys_new, cos, sin, pos_ids)
        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            keys, values = keys_new, values_new
        next_cache = (keys, values)

        # 扩展 K 和 V 以匹配注意力头数量
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        score_dtype = torch.float32 if self.float32_upcast else queries.dtype
        attn_scores = torch.matmul(
            queries.to(score_dtype),
            keys.transpose(2, 3).to(score_dtype),
        )
        attn_scores = attn_scores / self.head_dim**0.5

        # 用 -inf 应用掩码，使被屏蔽项在 softmax 后严格为零
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        # 在未屏蔽集合上稳定地计算 log-sum-exp
        row_max = attn_scores.amax(dim=-1, keepdim=True)
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        exp_scores = torch.exp(attn_scores - row_max)
        exp_scores = exp_scores.masked_fill(mask, 0.0)

        denom = exp_scores.sum(dim=-1, keepdim=True)
        attn_weights = exp_scores / denom.clamp(min=torch.finfo(exp_scores.dtype).tiny)

        # 当分数路径提升精度后，使上下文矩阵乘法的数据类型与 value 一致
        attn_weights = attn_weights.to(values.dtype)

        # 与之前相同
        context = torch.matmul(attn_weights, values)
        context = context.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context), next_cache


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "嵌入维度必须为偶数"

    # 计算逆频率
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # 生成位置索引
    positions = torch.arange(context_length, dtype=dtype)

    # 计算角度
    angles = positions[:, None] * inv_freq[None, :]  # 形状：(context_length, head_dim // 2)

    # 扩展角度以匹配 head_dim
    angles = torch.cat([angles, angles], dim=1)  # 形状：(context_length, head_dim)

    # 预先计算正弦和余弦
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope_with_pos_ids(x, cos, sin, pos_ids):
    B, H, L, D = x.shape
    cos_sel = cos[pos_ids]  # (B, L, D)
    sin_sel = sin[pos_ids]  # (B, L, D)
    cos_sel = cos_sel.unsqueeze(1)  # (B, 1, L, D)
    sin_sel = sin_sel.unsqueeze(1)  # (B, 1, L, D)
    x1 = x[..., : D // 2]
    x2 = x[..., D // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos_sel) + (rotated * sin_sel)
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


@torch.inference_mode()
def generate_text_basic_batched_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    attn_mask=None,
    pad_id=None,
):
    device = token_ids.device
    model.eval()

    batch_size, input_length = token_ids.shape

    if attn_mask is None and pad_id is not None:
        attn_mask = (token_ids != pad_id).to(torch.bool)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)

    # 初始化缓存和模型位置
    cache = KVCache(n_layers=model.cfg["n_layers"])

    # 预填充
    out = model(token_ids, cache=cache, attn_mask=attn_mask)[:, -1]

    # 记录哪些序列已经生成 EOS
    if eos_token_id is not None:
        # 如果提示词已经以 EOS 结尾，则视为已完成
        finished = (token_ids[:, -1] == eos_token_id)
    else:
        finished = None

    # 解码
    cur_attn = attn_mask
    generated_tokens = []
    for _ in range(max_new_tokens):
        # 如果所有序列都已完成，则停止
        if eos_token_id is not None and finished is not None and torch.all(finished):
            break

        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if eos_token_id is not None:
            # 强制已完成的行继续生成 EOS，以维持张量形状
            eos_tok = next_token.new_full((batch_size, 1), eos_token_id)
            next_token = torch.where(finished.view(batch_size, 1), eos_tok, next_token)

        # 扩展掩码以包含新生成的词元
        if cur_attn is not None:
            ones = torch.ones((batch_size, 1), dtype=cur_attn.dtype, device=device)
            cur_attn = torch.cat([cur_attn, ones], dim=1)

        # 使用 KV 缓存向前生成一个词元
        out = model(next_token, cache=cache, attn_mask=cur_attn)[:, -1]
        generated_tokens.append(next_token)

        # 追加本步骤词元后更新完成状态掩码
        if eos_token_id is not None:
            finished = finished | (next_token.squeeze(1) == eos_token_id)

    if generated_tokens:
        return torch.cat(generated_tokens, dim=1)
    return token_ids[:, input_length:]


@torch.inference_mode()
def generate_text_basic_batched_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    attn_mask=None,
    pad_id=None,
):
    device = token_ids.device
    model.eval()

    B, T = token_ids.shape

    if attn_mask is None and pad_id is not None:
        attn_mask = (token_ids != pad_id).to(torch.bool)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)

    # 初始化缓存和模型位置
    cache = KVCache(n_layers=model.cfg["n_layers"])

    # 预填充
    out = model(token_ids, cache=cache, attn_mask=attn_mask)[:, -1]

    # 解码
    cur_attn = attn_mask
    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if eos_token_id is not None and torch.all(next_token.squeeze(-1) == eos_token_id):
            break

        yield next_token

        # 扩展掩码以包含新生成的词元
        if cur_attn is not None:
            ones = torch.ones((B, 1), dtype=cur_attn.dtype, device=device)
            cur_attn = torch.cat([cur_attn, ones], dim=1)

        # 使用 KV 缓存向前生成一个词元
        out = model(next_token, cache=cache, attn_mask=cur_attn)[:, -1]
        token_ids = torch.cat([token_ids, next_token], dim=1)


def shrink_kv_cache_inplace(cache, keep_mask, n_layers):
    if keep_mask.dtype != torch.bool:
        keep_mask = keep_mask.to(torch.bool)
    for i in range(n_layers):
        kv = cache.get(i)
        if kv is None:
            continue
        K, V = kv
        K = K[keep_mask]  # 沿批次维度缩减
        V = V[keep_mask]
        cache.update(i, (K, V))


@torch.inference_mode()
def generate_text_basic_batched_cache_stop(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None,
    attn_mask=None,
    pad_id=None,
):
    """与 generate_text_basic_batched_cache 相同，但支持各序列独立提前停止。

    也就是说，已经写入 EOS 的已完成行不再参与前向传播。
    """
    device = token_ids.device
    model.eval()

    B, T0 = token_ids.shape

    # 构建注意力掩码
    if attn_mask is None and pad_id is not None:
        attn_mask = (token_ids != pad_id)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)

    # 初始化缓存，并对完整批次执行一次预填充
    cache = KVCache(n_layers=model.cfg["n_layers"])
    out = model(token_ids, cache=cache, attn_mask=attn_mask)[:, -1]  # (B, V)

    finished_full = torch.zeros(B, dtype=torch.bool, device=device)
    active_idx = torch.arange(B, device=device)  # 活动行 -> 原始行
    cur_attn_active = attn_mask                  # 与活动缓存对应
    generated_full_steps = []                    # 由 (B, 1) 单步张量组成的列表

    for _ in range(max_new_tokens):
        # 活动子批次的下一个词元
        next_token_active = torch.argmax(out, dim=-1, keepdim=True)  # 形状：(B_active, 1)

        # 散布到完整大小的 (B, 1) 单步张量中（已完成行填 EOS）
        fill_val = int(eos_token_id) if eos_token_id is not None else 0
        step_full = torch.full((B, 1), fill_value=fill_val,
                               dtype=token_ids.dtype, device=device)
        step_full.index_copy_(0, active_idx, next_token_active)
        generated_full_steps.append(step_full)

        # 在完整批次坐标中更新完成状态记录
        if eos_token_id is not None:
            newly_finished_active = (next_token_active.squeeze(1) == eos_token_id)
            finished_full.index_put_(
                (active_idx,),
                newly_finished_active | finished_full.index_select(0, active_idx)
            )
        else:
            newly_finished_active = torch.zeros_like(
                next_token_active.squeeze(1), dtype=torch.bool, device=device
            )

        if eos_token_id is not None and torch.all(finished_full):
            break

        # 计算批次中仅保留尚未完成的序列
        keep_mask_active = ~newly_finished_active
        if keep_mask_active.ndim == 0:
            keep_any = bool(keep_mask_active.item())
        else:
            keep_any = bool(keep_mask_active.any().item())
        if not keep_any:
            break

        next_token_survivors = next_token_active[keep_mask_active]  # 形状：(B_surv, 1)
        active_idx = active_idx[keep_mask_active]

        # 缩减注意力掩码，并为生成的词元追加一个 "1"
        if cur_attn_active is not None:
            cur_attn_active = cur_attn_active[keep_mask_active]
            ones = torch.ones((cur_attn_active.size(0), 1),
                              dtype=cur_attn_active.dtype, device=device)
            cur_attn_active = torch.cat([cur_attn_active, ones], dim=1)

        # 沿批次维度将 KV 缓存缩减到尚未完成的序列
        shrink_kv_cache_inplace(cache, keep_mask_active, model.cfg["n_layers"])

        # 仅为尚未完成的序列向前生成一个词元
        out = model(next_token_survivors, cache=cache, attn_mask=cur_attn_active)[:, -1]

    # 拼接逐步张量；只返回生成部分
    if generated_full_steps:
        return torch.cat(generated_full_steps, dim=1)  # 形状：(B, L_generated)
    else:
        return torch.empty((B, 0), dtype=token_ids.dtype, device=device)


@torch.inference_mode()
def generate_text_basic_batched_stream_cache_stop(
    model,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    attn_mask: torch.Tensor | None = None,
    pad_id: int | None = None,
):
    """与 generate_text_basic_batched_stream_cache 相同，但支持各序列独立提前停止。
    """
    device = token_ids.device
    model.eval()

    B, T0 = token_ids.shape

    if attn_mask is None and pad_id is not None:
        attn_mask = (token_ids != pad_id)
    if attn_mask is not None:
        attn_mask = attn_mask.to(torch.bool).to(device)

    cache = KVCache(n_layers=model.cfg["n_layers"])
    out = model(token_ids, cache=cache, attn_mask=attn_mask)[:, -1]  # (B, V)

    finished_full = torch.zeros(B, dtype=torch.bool, device=device)
    active_idx = torch.arange(B, device=device)
    cur_attn_active = attn_mask

    for _ in range(max_new_tokens):
        next_token_active = torch.argmax(out, dim=-1, keepdim=True)  # 形状：(B_active, 1)

        # 构建完整大小的单步张量并产出
        fill_val = int(eos_token_id) if eos_token_id is not None else 0
        step_full = torch.full((B, 1), fill_value=fill_val,
                               dtype=token_ids.dtype, device=device)
        step_full.index_copy_(0, active_idx, next_token_active)

        if eos_token_id is not None:
            newly_finished_active = (next_token_active.squeeze(1) == eos_token_id)
            finished_full.index_put_(
                (active_idx,),
                newly_finished_active | finished_full.index_select(0, active_idx)
            )
        else:
            newly_finished_active = torch.zeros_like(
                next_token_active.squeeze(1), dtype=torch.bool, device=device
            )

        # 在缩减前产出，使调用方每一步仍恰好看到一个 (B, 1) 张量
        yield step_full

        if eos_token_id is not None and torch.all(finished_full):
            break

        keep_mask_active = ~newly_finished_active
        if keep_mask_active.ndim == 0:
            keep_any = bool(keep_mask_active.item())
        else:
            keep_any = bool(keep_mask_active.any().item())
        if not keep_any:
            break

        next_token_survivors = next_token_active[keep_mask_active]
        active_idx = active_idx[keep_mask_active]

        if cur_attn_active is not None:
            cur_attn_active = cur_attn_active[keep_mask_active]
            ones = torch.ones((cur_attn_active.size(0), 1),
                              dtype=cur_attn_active.dtype, device=device)
            cur_attn_active = torch.cat([cur_attn_active, ones], dim=1)

        shrink_kv_cache_inplace(cache, keep_mask_active, model.cfg["n_layers"])

        out = model(next_token_survivors, cache=cache, attn_mask=cur_attn_active)[:, -1]


def load_model_and_tokenizer(
    which_model,
    device,
    use_compile,
    local_dir="qwen3",
    float32_upcast=True,
):
    if which_model == "base":

        download_qwen3_small(
            kind="base", tokenizer_only=False, out_dir=local_dir
        )

        tokenizer_path = Path(local_dir) / "tokenizer-base.json"
        model_path = Path(local_dir) / "qwen3-0.6B-base.pth"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    elif which_model == "reasoning":

        download_qwen3_small(
            kind="reasoning", tokenizer_only=False, out_dir=local_dir
        )

        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        model_path = Path(local_dir) / "qwen3-0.6B-reasoning.pth"
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_path,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True,
        )

    else:
        raise ValueError(f"无效选项：which_model={which_model}")

    model = Qwen3Model(QWEN_CONFIG_06_B, float32_upcast=float32_upcast)
    model.load_state_dict(torch.load(model_path))

    model.to(device)

    if use_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    return model, tokenizer
