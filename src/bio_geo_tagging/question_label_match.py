"""Export original definitions and score existing question-label pairs with DeepSeek."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


FIELDS = ("definition", "keywords", "exam_methods", "distinction")
SYSTEM_PROMPT = """你是高中地理题目与原有知识点释义的匹配度评估员。
只评估当前题目是否实际考查给定知识点，不重新打标，也不判断其他知识点。
依据知识点完整路径和原有释义判断解题所需知识；共同题干仅供理解小题，不把背景词当作小题考点。
常见考查方式是示例，不是穷举；不要仅因未出现相同字词或例题而扣分。解析可以辅助判断，但不能只凭解析中的提及判定考点。
0.90-1.00：直接核心依据；0.80-0.89：明确考查；0.70-0.79：可归入但偏边界/辅助；
0.40-0.69：有关联但不足以作为该题知识点；0.10-0.39：背景或弱关联；
0.01-0.09：基本无关；0.00：完全无关。
如果题目明确依赖缺失的图片或图表，现有文字不足以可靠判断，score设为null；不要把缺图当作低匹配。
只输出JSON对象：{"score":0到1的数字或null,"reason":"一句简短中文理由"}。"""


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield number, json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path} 第 {number} 行不是有效JSON") from error


def export_definitions(comparison: str, output: str) -> int:
    """Use only completed original definitions; never substitute generated text."""
    definitions: dict[str, dict[str, Any]] = {}
    for number, record in read_jsonl(comparison):
        if record.get("status") != "completed":
            continue
        path = record.get("full_path")
        existing = record.get("existing_interpretation")
        if not isinstance(path, str) or not isinstance(existing, dict):
            raise ValueError(f"对比文件第 {number} 行缺少原释义或路径")
        if any(not isinstance(existing.get(field), str) for field in FIELDS):
            raise ValueError(f"对比文件第 {number} 行原释义字段不完整")
        label = path.replace("->", "@")
        definitions[label] = {
            "label_id": str(record["label_id"]),
            "knw_label": label,
            "existing_interpretation": {field: existing[field] for field in FIELDS},
        }

    if not definitions:
        raise ValueError("没有可用的已完成原释义记录")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for record in definitions.values():
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(definitions)


def load_definitions(path: str) -> dict[str, dict[str, Any]]:
    definitions = {}
    for number, record in read_jsonl(path):
        label = record.get("knw_label")
        if not isinstance(label, str) or not isinstance(record.get("existing_interpretation"), dict):
            raise ValueError(f"释义文件第 {number} 行字段不完整")
        if label in definitions:
            raise ValueError(f"释义文件中重复的标签: {label}")
        definitions[label] = record
    return definitions


def request_score(client: OpenAI, model: str, unit: dict, definition: dict) -> dict:
    payload = {
        "knowledge_path": definition["knw_label"],
        "existing_interpretation": definition["existing_interpretation"],
        "input_role": unit.get("input_role"),
        "context_stem": unit.get("context_stem", ""),
        "stem": unit.get("stem", ""),
        "options": unit.get("options", ""),
        "analysis": unit.get("analysis", ""),
    }
    chunks = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0,
        max_tokens=256,
        stream=True,
    )
    content = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices).strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    answer = json.loads(content)
    score = answer.get("score")
    reason = answer.get("reason")
    if (score is not None and (type(score) not in (float, int) or not 0 <= score <= 1)) or not isinstance(reason, str) or not reason.strip():
        raise ValueError("模型返回的score或reason无效")
    return {"score": score, "match": score >= 0.70 if score is not None else None, "reason": reason.strip()}


def load_completed(output: str) -> set[tuple[str, str]]:
    if not Path(output).exists():
        return set()
    return {
        (str(record["question_id"]), record["knw_label"])
        for _, record in read_jsonl(output)
        if record.get("status") == "completed"
    }


def score_units(
    units_file: str,
    definitions_file: str,
    output: str,
    base_url: str,
    model: str,
    workers: int = 1,
    limit: int | None = None,
    timeout: float = 180.0,
) -> dict[str, int]:
    if workers < 1 or limit is not None and limit < 1:
        raise ValueError("workers和limit必须为正整数")
    definitions = load_definitions(definitions_file)
    completed = load_completed(output)
    client = OpenAI(api_key="not-required", base_url=base_url, timeout=timeout, max_retries=0)
    summary = {"attempted": 0, "completed": 0, "errors": 0, "already_completed": 0}

    def pending_pairs():
        for _, unit in read_jsonl(units_file):
            for label in dict.fromkeys(unit.get("knw_labels", [])):
                key = (str(unit["question_id"]), label)
                if key in completed:
                    summary["already_completed"] += 1
                    continue
                yield unit, label

    def evaluate(pair):
        unit, label = pair
        result = {
            "question_id": str(unit["question_id"]),
            "root_question_id": str(unit.get("root_question_id", unit.get("parent_id", ""))),
            "input_role": unit.get("input_role"),
            "knw_label": label,
            "label_id": definitions.get(label, {}).get("label_id"),
            "model": model,
        }
        try:
            if label not in definitions:
                raise ValueError("该标签在原释义文件中不存在")
            result.update(request_score(client, model, unit, definitions[label]))
            result["status"] = "completed"
        except Exception as error:
            result.update(status="error", error_type=type(error).__name__, error=str(error))
        return result

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        from itertools import islice
        pairs = pending_pairs()
        if limit is not None:
            pairs = islice(pairs, limit)
        with tqdm(desc="Scoring question-label pairs") as progress:
            while batch := list(islice(pairs, workers * 2)):
                for result in pool.map(evaluate, batch):
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    summary["attempted"] += 1
                    summary["completed" if result["status"] == "completed" else "errors"] += 1
                    progress.update(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export original definitions or score question-label pairs")
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("export-definitions")
    extraction.add_argument("--comparison-jsonl", required=True)
    extraction.add_argument("--output-jsonl", required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("--input-jsonl", required=True)
    scoring.add_argument("--definitions-jsonl", required=True)
    scoring.add_argument("--output-jsonl", required=True)
    scoring.add_argument("--base-url", required=True)
    scoring.add_argument("--model", default="DeepSeek-V4-Flash")
    scoring.add_argument("--workers", type=int, default=1)
    scoring.add_argument("--limit", type=int)
    scoring.add_argument("--timeout", type=float, default=180.0)
    arguments = parser.parse_args()
    if arguments.command == "export-definitions":
        print(json.dumps({"definitions": export_definitions(arguments.comparison_jsonl, arguments.output_jsonl)}, ensure_ascii=False))
    else:
        print(json.dumps(score_units(arguments.input_jsonl, arguments.definitions_jsonl, arguments.output_jsonl, arguments.base_url, arguments.model, arguments.workers, arguments.limit, arguments.timeout), ensure_ascii=False))


if __name__ == "__main__":
    main()
