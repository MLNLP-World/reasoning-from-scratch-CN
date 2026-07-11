
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》配套源码：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

import os
import json
import torch


QWEN3_CONFIG_1_7B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 2048,                 # 比 0.6B 模型大 2 倍
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 6144,              # 比 0.6B 模型大 2 倍
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 40 亿参数
QWEN3_CONFIG_4B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 2560,                 # 比上面的模型大 25%
    "n_heads": 32,                   # 比上面的模型大 2 倍
    "n_layers": 36,                  # 比上面的模型大 29%
    "hidden_dim": 9728,              # 约为上面模型的 3 倍
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 80 亿参数
QWEN3_CONFIG_8B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 4096,                 # 比上面的模型大 60%
    "n_heads": 32,
    "n_layers": 36,                  # 比上面的模型大 26%
    "hidden_dim": 12288,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 140 亿参数
QWEN3_CONFIG_14B = {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 5120,                 # 比上面的模型大 25%
        "n_heads": 40,                   # 比上面的模型大 25%
        "n_layers": 40,                  # 比上面的模型大 11%
        "hidden_dim": 17408,             # 比上面的模型大 42%
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
}

QWEN3_CONFIG_32B = {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 5120,
        "n_heads": 64,                   # 比上面的模型大 60%
        "n_layers": 64,                  # 比上面的模型大 60%
        "hidden_dim": 25600,             # 比上面的模型大 47%
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
}


def load_weights_into_qwen(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"张量 '{tensor_name}' 的形状不匹配。左侧：{left.shape}，右侧：{right.shape}")
        return torch.nn.Parameter(right.clone().detach() if isinstance(right, torch.Tensor) else torch.tensor(right))

    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for ln in range(param_config["n_layers"]):
        block = model.trf_blocks[ln]
        att = block.att

        # Q、K、V 投影
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{ln}.self_attn.q_proj.weight"],
            f"model.layers.{ln}.self_attn.q_proj.weight"
        )
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{ln}.self_attn.k_proj.weight"],
            f"model.layers.{ln}.self_attn.k_proj.weight"
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{ln}.self_attn.v_proj.weight"],
            f"model.layers.{ln}.self_attn.v_proj.weight"
        )

        # 输出投影
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{ln}.self_attn.o_proj.weight"],
            f"model.layers.{ln}.self_attn.o_proj.weight"
        )

        # QK 归一化层
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{ln}.self_attn.q_norm.weight"],
                f"model.layers.{ln}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{ln}.self_attn.k_norm.weight"],
                f"model.layers.{ln}.self_attn.k_norm.weight"
            )

        # 注意力层归一化
        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{ln}.input_layernorm.weight"],
            f"model.layers.{ln}.input_layernorm.weight"
        )

        # 前馈网络权重
        block.ff.fc1.weight = assign(
            block.ff.fc1.weight,
            params[f"model.layers.{ln}.mlp.gate_proj.weight"],
            f"model.layers.{ln}.mlp.gate_proj.weight"
        )
        block.ff.fc2.weight = assign(
            block.ff.fc2.weight,
            params[f"model.layers.{ln}.mlp.up_proj.weight"],
            f"model.layers.{ln}.mlp.up_proj.weight"
        )
        block.ff.fc3.weight = assign(
            block.ff.fc3.weight,
            params[f"model.layers.{ln}.mlp.down_proj.weight"],
            f"model.layers.{ln}.mlp.down_proj.weight"
        )
        block.norm2.scale = assign(
            block.norm2.scale,
            params[f"model.layers.{ln}.post_attention_layernorm.weight"],
            f"model.layers.{ln}.post_attention_layernorm.weight"
        )

    # 最终归一化层和输出头
    model.final_norm.scale = assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")

    # 模型采用权重绑定，因此这里复用嵌入层的权重
    model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")


def download_from_huggingface_from_snapshots(repo_id, local_dir):
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file  # 也可以使用你偏好的加载器

    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)

    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    single_file_path = os.path.join(repo_dir, "model.safetensors")

    if os.path.exists(index_path):
        # 多分片模型
        with open(index_path, "r") as f:
            index = json.load(f)

        weights_dict = {}
        for filename in set(index["weight_map"].values()):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights_dict.update(shard)
    elif os.path.exists(single_file_path):
        # 单分片模型
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        weights_dict = load_file(weights_file)
    else:
        raise FileNotFoundError("未找到 model.safetensors 或 model.safetensors.index.json。")

    return weights_dict
