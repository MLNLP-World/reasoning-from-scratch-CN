# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

import torch

from reasoning_from_scratch.ch02 import (
    get_device,
    generate_text_basic,
    generate_text_basic_cache,
    generate_stats
)


# 用于 generate_text_basic 测试的虚拟模型。
class DummyModel:
    def __init__(self, fixed_token, vocab_size=5):
        self.fixed_token = fixed_token
        self.vocab_size = vocab_size
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, token_ids, cache=None):
        batch_size, seq_len = token_ids.size()
        out = torch.zeros(batch_size, seq_len, self.vocab_size)
        # 将 fixed_token 列设为最大值，使 argmax 返回 fixed_token
        out[..., self.fixed_token] = 1.0
        return out


class DummyModelCache(DummyModel):
    def __init__(self, fixed_token, vocab_size=5, n_layers=2):
        super().__init__(fixed_token, vocab_size)
        self.cfg = {"n_layers": n_layers}
        self.reset_called = False

    def reset_kv_cache(self):
        self.reset_called = True


class DummyTokenizer:
    def decode(self, token_list):
        return " ".join(str(t) for t in token_list)


def test_get_device_returns_torch_device(capsys):
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda", "mps")


def test_generate_text_basic_stops_on_eos():
    # 批量大小 = 1
    # 序列长度 = 3
    max_new_tokens = 10
    fixed_token = 2

    dummy_model = DummyModel(fixed_token=fixed_token)
    token_ids = torch.tensor([[1, 3, 4]])  # 形状（批量, 序列长度）

    # 将 eos_token_id 设为 fixed_token，使生成立即停止
    output = generate_text_basic(dummy_model, token_ids, max_new_tokens, eos_token_id=fixed_token)
    assert output.size(1) == 0
    assert dummy_model.eval_called is True


def test_generate_text_basic_generates_tokens_without_eos():
    # 批量大小 = 1
    # 序列长度 = 2
    max_new_tokens = 3
    fixed_token = 1

    dummy_model = DummyModel(fixed_token=fixed_token)
    token_ids = torch.tensor([[0, 4]])
    output = generate_text_basic(dummy_model, token_ids, max_new_tokens, eos_token_id=None)
    assert output.size(1) == max_new_tokens
    assert torch.all(output == fixed_token)


def test_generate_text_basic_cache_stops_on_eos():
    # 批量大小 = 1
    # 序列长度 = 2
    max_new_tokens = 10
    fixed_token = 3

    dummy_model = DummyModelCache(fixed_token=fixed_token, n_layers=4)
    token_ids = torch.tensor([[2, 2]])
    output = generate_text_basic_cache(dummy_model, token_ids, max_new_tokens, eos_token_id=fixed_token)
    assert output.size(1) == 0
    assert dummy_model.reset_called is True


def test_generate_text_basic_cache_generates_tokens_without_eos():
    # 批量大小 = 1
    # 序列长度 = 1
    max_new_tokens = 4
    fixed_token = 0

    dummy_model = DummyModelCache(fixed_token=fixed_token, n_layers=3)
    token_ids = torch.tensor([[5]])

    output = generate_text_basic_cache(dummy_model, token_ids, max_new_tokens, eos_token_id=None)
    assert output.size(1) == max_new_tokens
    assert torch.all(output == fixed_token)
    assert dummy_model.reset_called is True


def test_generate_stats_prints_output(monkeypatch, capsys):
    output_token_ids = torch.tensor([[10, 20, 30]])
    tokenizer = DummyTokenizer()
    start_time = 100.0
    end_time = 102.0

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    generate_stats(output_token_ids, tokenizer, start_time, end_time)

    captured = capsys.readouterr().out
    assert "Time:" in captured
    assert "tokens/sec" in captured
