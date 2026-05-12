"""Qwen 文本分类提示策略对比实验。

运行方式：
    python3 code/chapter3/Qwen.py

可选环境变量：
    QWEN_MODEL_ID=Qwen/Qwen1.5-0.5B-Chat

实验会比较 Zero-shot、Few-shot、Chain-of-Thought、Few-shot + CoT
四种提示策略在中文情感分类任务上的效果差异。
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# 增加 HF_ENDPOINT，避免国内网络环境下出现 Connection aborted。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen1.5-0.5B-Chat")
LABELS = ("正面", "负面", "中性")
RESULT_DIR = Path(__file__).resolve().parent / "qwen_classification_results"


@dataclass(frozen=True)
class Sample:
    id: str
    text: str
    label: str


@dataclass(frozen=True)
class PromptStrategy:
    name: str
    build_prompt: Callable[[str], str]
    max_new_tokens: int


FEW_SHOT_EXAMPLES = [
    Sample("ex_pos_1", "这款耳机音质很好，降噪效果也很明显。", "正面"),
    Sample("ex_pos_2", "客服回复很快，问题当天就解决了。", "正面"),
    Sample("ex_pos_3", "这次旅行体验非常舒服，酒店也很干净。", "正面"),
    Sample("ex_neg_1", "外卖送到的时候已经凉了，味道也很一般。", "负面"),
    Sample("ex_neg_2", "软件一直闪退，根本没法正常使用。", "负面"),
    Sample("ex_neg_3", "包装破损严重，里面的东西也少了一件。", "负面"),
    Sample("ex_neu_1", "会议时间改到明天下午三点。", "中性"),
    Sample("ex_neu_2", "订单已提交，预计三个工作日内发货。", "中性"),
    Sample("ex_neu_3", "今天北京气温为二十六摄氏度。", "中性"),
]


TEST_SAMPLES = [
    Sample("pos_1", "这家店的牛肉面汤很鲜，服务员也很热情。", "正面"),
    Sample("pos_2", "新版本界面清爽了很多，操作也更顺手。", "正面"),
    Sample("pos_3", "医生解释得很耐心，让我安心了不少。", "正面"),
    Sample("pos_4", "快递比预计早到一天，包装也很完整。", "正面"),
    Sample("pos_5", "这本书内容扎实，案例也很有启发。", "正面"),
    Sample("pos_6", "售后主动联系我补发配件，处理结果满意。", "正面"),
    Sample("neg_1", "排队等了四十分钟，最后菜还是上错了。", "负面"),
    Sample("neg_2", "页面加载特别慢，点几次都没有反应。", "负面"),
    Sample("neg_3", "衣服色差很大，线头也特别多。", "负面"),
    Sample("neg_4", "说明书写得不清楚，安装过程非常折腾。", "负面"),
    Sample("neg_5", "客服一直推脱，没有给出明确解决方案。", "负面"),
    Sample("neg_6", "电影节奏拖沓，看到一半就想离场。", "负面"),
    Sample("neu_1", "本周五公司将进行网络维护。", "中性"),
    Sample("neu_2", "这款手机提供黑色、白色和蓝色三个版本。", "中性"),
    Sample("neu_3", "请在表格中填写姓名、电话和收货地址。", "中性"),
    Sample("neu_4", "航班计划于晚上八点二十分起飞。", "中性"),
    Sample("neu_5", "系统将在登录后展示最近三十天的数据。", "中性"),
    Sample("neu_6", "展览开放时间为上午九点到下午五点。", "中性"),
]


def select_device() -> str:
    """选择当前机器可用的推理设备。"""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_zero_shot_prompt(text: str) -> str:
    return f"""请判断下面文本的情感类别。

只允许从以下三个标签中选择一个：正面、负面、中性。
请只输出标签，不要解释。

文本：{text}
"""


def build_few_shot_prompt(text: str) -> str:
    examples = "\n\n".join(
        f"文本：{sample.text}\n标签：{sample.label}" for sample in FEW_SHOT_EXAMPLES
    )
    return f"""请判断文本的情感类别。标签只能是：正面、负面、中性。
请只输出标签，不要解释。

示例：
{examples}

现在判断：
文本：{text}
标签："""


def build_cot_prompt(text: str) -> str:
    return f"""请判断下面文本的情感类别。

标签只能是：正面、负面、中性。
请先简要分析理由，最后用固定格式输出：
最终标签：正面/负面/中性

文本：{text}
"""


def build_few_shot_cot_prompt(text: str) -> str:
    return f"""请判断文本的情感类别。标签只能是：正面、负面、中性。
请先简要分析理由，最后用固定格式输出“最终标签：标签”。

示例：
文本：这款耳机音质很好，降噪效果也很明显。
分析：文本表达了对产品音质和降噪效果的满意。
最终标签：正面

文本：软件一直闪退，根本没法正常使用。
分析：文本表达了使用失败和不满。
最终标签：负面

文本：会议时间改到明天下午三点。
分析：文本只是陈述时间变更，没有明显情绪。
最终标签：中性

现在判断：
文本：{text}
"""


PROMPT_STRATEGIES = [
    PromptStrategy("zero_shot", build_zero_shot_prompt, max_new_tokens=24),
    PromptStrategy("few_shot", build_few_shot_prompt, max_new_tokens=24),
    PromptStrategy("cot", build_cot_prompt, max_new_tokens=128),
    PromptStrategy("few_shot_cot", build_few_shot_cot_prompt, max_new_tokens=128),
]


def load_qwen_model():
    device = select_device()
    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_ID}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    model.eval()

    print("模型和分词器加载完成。")
    return tokenizer, model, device


def generate_response(tokenizer, model, device: str, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": "你是一个严格的中文文本分类助手。"},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = model_inputs.input_ids.shape[1]
    generated_ids = generated_ids[:, input_length:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def parse_label(response: str) -> str | None:
    final_label = re.search(r"最终标签\s*[:：]\s*(正面|负面|中性)", response)
    if final_label:
        return final_label.group(1)

    stripped = response.strip()
    if stripped in LABELS:
        return stripped

    matches = list(re.finditer(r"正面|负面|中性", response))
    if matches:
        return matches[-1].group(0)
    return None


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_metrics(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(row["gold_label"] == row["pred_label"] for row in rows)
    parse_failed = sum(row["pred_label"] is None for row in rows)

    per_label = {}
    f1_scores = []
    for label in LABELS:
        tp = sum(row["gold_label"] == label and row["pred_label"] == label for row in rows)
        fp = sum(row["gold_label"] != label and row["pred_label"] == label for row in rows)
        fn = sum(row["gold_label"] == label and row["pred_label"] != label for row in rows)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        f1_scores.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(row["gold_label"] == label for row in rows),
        }

    confusion_matrix = {
        gold: {pred: 0 for pred in [*LABELS, "解析失败"]} for gold in LABELS
    }
    for row in rows:
        pred = row["pred_label"] if row["pred_label"] in LABELS else "解析失败"
        confusion_matrix[row["gold_label"]][pred] += 1

    return {
        "total": total,
        "accuracy": safe_divide(correct, total),
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "parse_failed_rate": safe_divide(parse_failed, total),
        "avg_latency_seconds": sum(row["latency_seconds"] for row in rows) / total,
        "avg_output_chars": sum(len(row["raw_response"]) for row in rows) / total,
        "per_label": per_label,
        "confusion_matrix": confusion_matrix,
    }


def run_strategy(tokenizer, model, device: str, strategy: PromptStrategy) -> tuple[list[dict], dict]:
    print(f"\n=== Running strategy: {strategy.name} ===")
    rows = []

    for index, sample in enumerate(TEST_SAMPLES, start=1):
        prompt = strategy.build_prompt(sample.text)
        start = time.perf_counter()
        response = generate_response(
            tokenizer=tokenizer,
            model=model,
            device=device,
            prompt=prompt,
            max_new_tokens=strategy.max_new_tokens,
        )
        latency = time.perf_counter() - start
        pred_label = parse_label(response)
        is_correct = pred_label == sample.label

        row = {
            "strategy": strategy.name,
            "sample_id": sample.id,
            "text": sample.text,
            "gold_label": sample.label,
            "pred_label": pred_label,
            "is_correct": is_correct,
            "latency_seconds": latency,
            "raw_response": response,
        }
        rows.append(row)

        status = "正确" if is_correct else "错误"
        print(
            f"[{index:02d}/{len(TEST_SAMPLES)}] {sample.id} "
            f"gold={sample.label} pred={pred_label} {status}"
        )

    return rows, calculate_metrics(rows)


def save_results(all_rows: list[dict], summary: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULT_DIR / "predictions.csv"
    json_path = RESULT_DIR / "summary.json"

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "strategy",
                "sample_id",
                "text",
                "gold_label",
                "pred_label",
                "is_correct",
                "latency_seconds",
                "raw_response",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"\n明细结果已保存到：{csv_path}")
    print(f"汇总结果已保存到：{json_path}")


def print_summary(summary: dict) -> None:
    print("\n=== 实验汇总 ===")
    print(
        f"{'策略':<16} {'Accuracy':>10} {'Macro-F1':>10} "
        f"{'解析失败率':>10} {'平均耗时(s)':>12} {'平均输出字符':>12}"
    )
    for strategy_name, metrics in summary.items():
        print(
            f"{strategy_name:<16} "
            f"{metrics['accuracy']:>10.3f} "
            f"{metrics['macro_f1']:>10.3f} "
            f"{metrics['parse_failed_rate']:>10.3f} "
            f"{metrics['avg_latency_seconds']:>12.2f} "
            f"{metrics['avg_output_chars']:>12.1f}"
        )


def main() -> None:
    tokenizer, model, device = load_qwen_model()

    all_rows = []
    summary = {}
    for strategy in PROMPT_STRATEGIES:
        rows, metrics = run_strategy(tokenizer, model, device, strategy)
        all_rows.extend(rows)
        summary[strategy.name] = metrics

    print_summary(summary)
    save_results(all_rows, summary)


if __name__ == "__main__":
    main()
