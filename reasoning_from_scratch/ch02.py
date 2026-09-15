# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》配套源码：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

from .qwen3 import KVCache
import warnings
import torch


def get_device(enable_tensor_cores=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("正在使用 NVIDIA CUDA GPU")

        if enable_tensor_cores:
            major, minor = map(int, torch.__version__.split(".")[:2])
            if (major, minor) >= (2, 9):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("正在使用 Apple Silicon GPU（MPS）")

    elif torch.xpu.is_available():
        device = torch.device("xpu")
        print("正在使用 Intel GPU")

    else:
        device = torch.device("cpu")
        print("正在使用 CPU")

    return device


@torch.inference_mode()
def generate_text_basic(model, token_ids, max_new_tokens, eos_token_id=None):
    input_length = token_ids.shape[1]
    model.eval()

    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        # 如果批次中的所有序列都已生成 EOS，则停止
        if (eos_token_id is not None
                and next_token.item() == eos_token_id):
            break

        token_ids = torch.cat([token_ids, next_token], dim=1)
    return token_ids[:, input_length:]


# 第 2 章此前使用非流式函数，而
# *_stream 函数是在练习中引入的
# 虽然较简单的 generate_text_basic_cache 已不再使用
# 但出于向后兼容的考虑仍保留在这里
@torch.inference_mode()
def generate_text_basic_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None
):
    input_length = token_ids.shape[1]
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    out = model(token_ids, cache=cache)[:, -1]
    generated_tokens = []

    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if (eos_token_id is not None
                and next_token.item() == eos_token_id):
            break

        generated_tokens.append(next_token)
        out = model(next_token, cache=cache)[:, -1]

    if generated_tokens:
        return torch.cat(generated_tokens, dim=1)
    return token_ids[:, input_length:]


@torch.inference_mode()
def generate_text_basic_stream(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None
):
    model.eval()

    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break

        yield next_token

        token_ids = torch.cat([token_ids, next_token], dim=1)


@torch.inference_mode()
def generate_text_basic_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None
):
    # input_length = token_ids.shape[1]
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()

    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break

        yield next_token  # 新增：每生成一个词元就立即产出
        # token_ids = torch.cat([token_ids, next_token], dim=1)
        out = model(next_token, cache=cache)[:, -1]

    # return token_ids[:, input_length:]


def _print_peak_memory_stats(device):
    device = torch.device(device)
    backend = getattr(torch, device.type, None)
    if backend is None or not hasattr(backend, "is_available"):
        return
    if not backend.is_available():
        return

    sync_fn = getattr(backend, "synchronize", None)
    if callable(sync_fn):
        try:
            sync_fn(device=device)
        except TypeError:
            sync_fn()

    try:
        max_mem_bytes = backend.max_memory_allocated(device=device)
    except TypeError:
        max_mem_bytes = backend.max_memory_allocated()

    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"{device.type.upper()} 最大已分配内存：{max_mem_gb:.2f} GB")

    try:
        backend.reset_peak_memory_stats(device=device)
    except TypeError:
        backend.reset_peak_memory_stats()


def generate_stats(output_token_ids, tokenizer, start_time,
                   end_time):
    total_time = end_time - start_time
    print(f"\n\n耗时：{total_time:.2f} 秒")
    print(f"{int(output_token_ids.numel() / total_time)} 词元/秒")

    for name, backend in (("CUDA", getattr(torch, "cuda", None)),
                          ("XPU", getattr(torch, "xpu", None))):
        if backend is not None and backend.is_available():

            # 检查是否确实在使用此后端
            device_type = output_token_ids.device.type
            if device_type != name.lower():
                warnings.warn(
                    f"{name} 可用，但张量位于 {device_type}。"
                    "内存统计值可能为 0。"
                )

            # 如果支持则进行同步（对异步后端很重要）
            if hasattr(backend, "synchronize"):
                backend.synchronize()

            max_mem_bytes = backend.max_memory_allocated()
            max_mem_gb = max_mem_bytes / (1024 ** 3)
            print(f"{name} 最大已分配内存：{max_mem_gb:.2f} GB")
            backend.reset_peak_memory_stats()
