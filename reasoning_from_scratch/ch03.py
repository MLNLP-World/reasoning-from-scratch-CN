# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》配套源码：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

from pathlib import Path
import json
import re
import time

import requests
from sympy import simplify
from sympy.parsing import sympy_parser as spp
from sympy.core.sympify import SympifyError
from sympy.polys.polyerrors import PolynomialError
from tokenize import TokenError
import torch

from .qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
    Qwen3Model,
    QWEN_CONFIG_06_B
)
from .ch02 import (
    generate_text_basic_stream_cache
)

RE_NUMBER = re.compile(
    r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

LATEX_FIXES = [  # 需要替换的 LaTeX 格式
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"\u00B7|\u00D7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"°", ""),
]

RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")  # 移除 <|assistant|> 等对话特殊词元
SUPERSCRIPT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")",
}


def load_model_and_tokenizer(which_model, device, use_compile, local_dir="qwen3"):
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

    model = Qwen3Model(QWEN_CONFIG_06_B)
    model.load_state_dict(torch.load(model_path))

    model.to(device)

    if use_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        model = torch.compile(model)

    return model, tokenizer


def load_tokenizer_only(which_model, local_dir="qwen3"):
    if which_model == "base":
        download_qwen3_small(
            kind="base", tokenizer_only=True, out_dir=local_dir
        )

        tokenizer_path = Path(local_dir) / "tokenizer-base.json"
        tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    elif which_model == "reasoning":
        download_qwen3_small(
            kind="reasoning", tokenizer_only=True, out_dir=local_dir
        )

        tokenizer_path = Path(local_dir) / "tokenizer-reasoning.json"
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_path,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=True,
        )

    else:
        raise ValueError(f"无效选项：which_model={which_model}")

    return tokenizer


def generate_text_stream_concat(
    model, tokenizer, prompt, device, max_new_tokens,
    verbose=False,
):
    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
        ).unsqueeze(0)

    generated_ids = []
    for token in generate_text_basic_stream_cache(
        model=model,
        token_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        next_token_id = token.squeeze(0)
        generated_ids.append(next_token_id.item())

        if verbose:
            print(
                tokenizer.decode(next_token_id.tolist()),
                end="",
                flush=True
            )
    return tokenizer.decode(generated_ids)


def get_last_boxed(text):
    # 查找最后一次出现的 "\boxed"
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None

    # 获取 "\boxed" 后面的位置
    current_idx = boxed_start_idx + len(r"\boxed")

    # 跳过 "\boxed" 后的所有空白字符
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    # 此处应为左花括号 "{"
    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    # 解析可能嵌套的花括号
    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    # 处理花括号不匹配的情况
    if brace_depth != 0:
        return None

    # 提取最外层花括号中的内容
    return text[content_start_idx:current_idx-1]


def extract_final_candidate(text, fallback="number_then_full"):
    # 没有匹配项时的默认返回值
    result = ""

    if text:
        # 如果存在 boxed 表达式，优先取最后一个
        boxed = get_last_boxed(text.strip())
        if boxed:
            result = boxed.strip().strip("$ ")

        # 如果没有 boxed 表达式，则尝试回退策略
        elif fallback in ("number_then_full", "number_only"):
            m = RE_NUMBER.findall(text)
            if m:
                # 使用最后一个数字
                result = m[-1]
            elif fallback == "number_then_full":
                # 否则在找不到数字时返回完整文本
                result = text
    return result


def normalize_text(text):
    if not text:
        return ""
    text = RE_SPECIAL.sub("", text).strip()

    # 移除开头的选择题选项标签
    # 例如将 "c. 3" 转为 3，将 "b: 2" 转为 2
    match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if match:
        text = match.group(1)

    # 移除角度符号
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)   # ^{\circ}
    text = re.sub(r"\^\s*\\circ", "", text)             # ^\circ
    text = text.replace("°", "")                        # Unicode 度数符号

    # 如果整个字符串由 \text{...} 包裹，则去掉外层包装
    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if match:
        text = match.group("x")

    # 移除行内或行间数学公式包装符 \( \) \[ \]
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)

    # 对 LaTeX 做轻量规范化
    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)

    # 将 Unicode 上标转换为幂形式（例如 2² -> 2**2）
    def convert_superscripts(s, base=None):
        converted = "".join(
            SUPERSCRIPT_MAP[ch] if ch in SUPERSCRIPT_MAP else ch
            for ch in s
        )
        if base is None:
            return converted
        return f"{base}**{converted}"

    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: convert_superscripts(m.group(2), base=m.group(1)),
        text,
    )
    text = convert_superscripts(text)

    # 数字和根式
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )

    # 分数
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )

    # 指数和带分数
    text = text.replace("^", "**")
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda match: "+" + match.group(1),
        text,
    )

    # 1,234 -> 1234
    text = re.sub(
        r"(?<=\d),(?=\d\d\d(\D|$))",
        "",
        text,
    )

    return text.replace("{", "").replace("}", "").strip().lower()


def sympy_parser(expr):
    # 避免因过长的无效回复而崩溃
    # 某些训练不佳的模型（第 6 章）可能会生成此类回复
    if expr is None or len(expr) > 2000:
        return None
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                # 标准转换，例如处理括号
                *spp.standard_transformations,

                # 允许省略乘号（例如 "2x" -> 2*x）
                spp.implicit_multiplication_application,
            ),

            # 解析时求值，使简单的常量表达式得以化简（例如 2+3 -> 5）
            evaluate=True,
        )
    except (SympifyError, SyntaxError, TypeError, AttributeError,
            IndexError, TokenError, ValueError, PolynomialError):
        return None


def equality_check(expr_gtruth, expr_pred):
    # 首先检查两个表达式的字符串是否完全相同
    if expr_gtruth == expr_pred:
        return True

    # 将两个表达式解析为 SymPy 对象（解析失败时返回 None）
    gtruth, pred = sympy_parser(expr_gtruth), sympy_parser(expr_pred)

    # 如果两个表达式都解析成功，则尝试符号比较
    if gtruth is not None and pred is not None:
        try:
            # 如果二者之差为 0，则它们等价
            return simplify(gtruth - pred) == 0
        except (SympifyError, TypeError):
            pass

    return False


def split_into_parts(text):
    result = [text]

    if text:
        # 检查文本是否类似元组或列表，例如 "(a, b)" 或 "[a, b]"
        if (
            len(text) >= 2
            and text[0] in "([" and text[-1] in ")]"
            and "," in text[1:-1]
        ):
            # 按括号内的逗号拆分，并移除空白字符
            items = [p.strip() for p in text[1:-1].split(",")]
            if all(items):
                result = items
    else:
        # 如果文本为空，则返回空列表
        result = []

    return result


def grade_answer(pred_text, gt_text):
    result = False  # 检查失败时的默认结果

    # 仅当两个输入都是非空字符串时才继续
    if pred_text is not None and gt_text is not None:
        gt_parts = split_into_parts(
            normalize_text(gt_text)
        )  # 将标准答案拆分为可比较的部分

        pred_parts = split_into_parts(
            normalize_text(pred_text)
        )  # 将预测答案拆分为可比较的部分

        # 确保两边有效部分的数量相同
        if (gt_parts and pred_parts
           and len(gt_parts) == len(pred_parts)):
            result = all(
                equality_check(gt, pred)
                for gt, pred in zip(gt_parts, pred_parts)
            )  # 逐一检查各部分在数学上是否等价

    return result  # 仅当所有检查均通过时才为 True


def run_demos_table(tests):
    header = ("Test", "Expect", "Got", "Status")
    rows = []
    for name, pred, gtruth, expect in tests:
        got = grade_answer(pred, gtruth)  # 运行相等性检查
        status = "PASS" if got == expect else "FAIL"
        rows.append((name, str(expect), str(got), status))

    data = [header] + rows

    # 计算每列的最大宽度，使表格整齐对齐
    col_widths = [
        max(len(row[i]) for row in data)
        for i in range(len(header))
    ]

    # 逐行打印表格
    for row in data:
        line = " | ".join(
            row[i].ljust(col_widths[i])
            for i in range(len(header))
        )
        print(line)

    # 打印测试通过情况摘要
    passed = sum(r[3] == "PASS" for r in rows)
    print(f"\n通过 {passed}/{len(rows)}")


def render_prompt(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{prompt}\n\nAnswer:"
    )
    return template


def load_math500_test(local_path="math500_test.json", save_copy=True):
    local_path = Path(local_path)
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()

        if save_copy:  # 保存本地副本
            with local_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    return data


def mini_eval_demo(model, tokenizer, device):
    ex = {  # 使用包含 "problem" 和 "answer" 字段的示例进行测试
        "problem": "Compute 1/2 + 1/6.",
        "answer": "2/3"
    }
    prompt = render_prompt(ex["problem"])     # 1. 应用提示词模板
    gen_text = generate_text_stream_concat(   # 2. 生成回复
        model, tokenizer, prompt, device,
        max_new_tokens=64,
    )
    pred_answer = extract_final_candidate(gen_text)  # 3. 提取并规范化答案
    is_correct = grade_answer(                       # 4. 对答案评分
        pred_answer, ex["answer"]
    )
    print(f"设备：{device}")
    print(f"预测答案：{pred_answer}")
    print(f"标准答案：{ex['answer']}")
    print(f"是否正确：{is_correct}")


def eta_progress_message(
    processed,
    total,
    start_time,
    show_eta=False,
    label="进度",
):
    progress = f"{label}：{processed}/{total}"
    pad_width = len(f"{label}：{total}/{total} | 预计剩余时间：00时 00分 00秒")
    if not show_eta or processed <= 0:
        return progress.ljust(pad_width)

    elapsed = time.time() - start_time
    if elapsed <= 0:
        return progress.ljust(pad_width)

    remaining = max(total - processed, 0)

    if processed:
        avg_time = elapsed / processed
        eta_seconds = avg_time * remaining
    else:
        eta_seconds = 0

    eta_seconds = max(int(round(eta_seconds)), 0)
    minutes, rem_seconds = divmod(eta_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        eta = f"{hours}时 {minutes:02d}分 {rem_seconds:02d}秒"
    elif minutes:
        eta = f"{minutes:02d}分 {rem_seconds:02d}秒"
    else:
        eta = f"{rem_seconds:02d}秒"

    message = f"{progress} | 预计剩余时间：{eta}"
    return message.ljust(pad_width)


def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=512,
    verbose=False,
):

    if out_path is None:
        dev_name = str(device).replace(":", "-")  # 使文件名兼容 Windows
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    total_len = 0  # 计算平均回复长度（参见练习 3.2）
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:  # 保存结果以供检查
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])    # 1. 应用提示词模板
            gen_text = generate_text_stream_concat(   # 2. 生成回复
                model, tokenizer, prompt, device,
                max_new_tokens=max_new_tokens,
                verbose=verbose,
            )
            total_len += len(tokenizer.encode(gen_text))

            extracted = extract_final_candidate(  # 3. 提取并规范化答案
                gen_text
            )
            is_correct = grade_answer(            # 4. 对答案评分
                extracted, row["answer"]
            )
            num_correct += int(is_correct)

            record = {  # 保存记录以供检查
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)
            if verbose:  # 在生成过程中打印回复
                print(
                    f"\n\n{'='*50}\n{progress_msg}\n"
                    f"{'='*50}\n提取结果：{extracted}\n"
                    f"预期答案：{row['answer']}\n"
                    f"当前正确数：{num_correct}\n{'-'*50}"
                )

    # 打印摘要信息
    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\n准确率：{acc*100:.1f}%（{num_correct}/{num_examples}）")
    print(f"总耗时：{seconds_elapsed/60:.1f} 分钟")
    avg_len = total_len / num_examples
    print(f"平均回复长度：{avg_len:.2f} 个词元")
    print(f"日志已写入：{out_path}")
    return num_correct, num_examples, acc
