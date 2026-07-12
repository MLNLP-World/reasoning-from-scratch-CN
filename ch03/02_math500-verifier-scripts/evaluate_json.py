# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# 《从零构建推理模型》来源：https://mng.bz/lZ5B
# 代码仓库：https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
from pathlib import Path

from reasoning_from_scratch.ch03 import (
    extract_final_candidate,
    grade_answer,
)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="记录文件（.json 或 .jsonl）的路径。",
    )
    parser.add_argument(
        "--gtruth_answer",
        type=str,
        default="gtruth_answer",
        help="真实答案的键名",
    )
    parser.add_argument(
        "--generated_text",
        type=str,
        default="generated_text",
        help="模型生成输出的键名",
    )
    return parser.parse_args()


def load_records(json_path):
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件： {path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            parsed = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            records = []
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"以下行包含无效 JSON： {line_num} in {path}: {exc}"
                    ) from exc
            return records

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if "records" in parsed and isinstance(parsed["records"], list):
            return parsed["records"]
        return [parsed]

    raise ValueError(
        f"不支持的 JSON 根类型，文件： {path}: {type(parsed).__name__}"
    )


def evaluate_records(records, gtruth_key, generated_text_key):
    num_examples = len(records)
    num_correct = 0

    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(
                f"Record {idx} 不是 JSON 对象： {type(record).__name__}"
            )

        if gtruth_key not in record:
            raise KeyError(f"Record {idx} 缺少键： {gtruth_key}")
        if generated_text_key not in record:
            raise KeyError(f"Record {idx} 缺少键： {generated_text_key}")

        extracted = extract_final_candidate(record[generated_text_key])
        is_correct = grade_answer(extracted, record[gtruth_key])
        num_correct += int(is_correct)

    acc = num_correct / num_examples if num_examples else 0.0
    return num_correct, num_examples, acc


if __name__ == "__main__":
    args = parse_args()
    records = load_records(args.json_path)
    num_correct, num_examples, acc = evaluate_records(
        records=records,
        gtruth_key=args.gtruth_answer,
        generated_text_key=args.generated_text,
    )
    print(f"Accuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
