"""Export original definitions and score existing question-label pairs with DeepSeek."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import time
from typing import Any

from openai import OpenAI
from tqdm import tqdm


FIELDS = ("definition", "keywords", "exam_methods", "distinction")
SYSTEM_PROMPT = """你是高中地理题目与原有知识点释义的匹配度评估员。
当前输入是一道完整大题，包含公共题干和全部小题。只评估整道大题是否实际考查给定的原标签，不重新打标，也不判断其他知识点。
依据知识点完整路径和原有释义判断解题所需知识。
只要公共题干或至少一个小题明确考查该知识点，整道大题就可以匹配，不要求每个小题都考查它；不要把仅作为材料背景出现的词语当作考点。
常见考查方式是示例，不是穷举；不要仅因未出现相同字词或例题而扣分。解析可以辅助判断，但不能只凭解析中的提及判定考点。
0.90-1.00：直接核心依据；0.80-0.89：明确考查；0.70-0.79：可归入但偏边界/辅助；
0.40-0.69：有关联但不足以作为该题知识点；0.10-0.39：背景或弱关联；
0.01-0.09：基本无关；0.00：完全无关。
如果现有题干、选项和解析足以判断，即使图片缺失也正常评分。只有缺失的图片或图表是判断匹配度不可替代的信息时，才判为unjudgeable；不要把材料缺失当作低匹配。
只输出JSON对象，且必须遵守以下二选一格式：
可评分：{"judgement":"scored","score":0到1的数字,"reason":"一句简短中文理由"}
无法判断：{"judgement":"unjudgeable","score":null,"reason":"一句简短中文理由"}。"""


def configure_run_logger(log_file: str | None) -> logging.Logger:
    """Create an overwritten log for progress, results, and request errors."""
    logger = logging.getLogger("question_label_match.run")
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, mode="w", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


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


def read_complete_questions(path: str):
    """Read each merged record as one complete question scoring unit."""
    for number, question in read_jsonl(path):
        if not str(question.get("question_id", "")):
            raise ValueError(f"聚合题目文件第 {number} 行缺少question_id")
        sub_question_records = question.get("sub_questions", [])
        if not isinstance(sub_question_records, list):
            raise ValueError(f"聚合题目文件第 {number} 行的sub_questions不是列表")
        sub_questions = [
            {
                "question_id": str(sub_question.get("question_id", "")),
                "stem": sub_question.get("stem", ""),
                "options": sub_question.get("options", ""),
                "analysis": sub_question.get("analysis", ""),
            }
            for sub_question in sub_question_records
            if isinstance(sub_question, dict)
        ]
        if len(sub_questions) != len(sub_question_records):
            raise ValueError(f"聚合题目文件第 {number} 行包含无效小题记录")
        unit = dict(question)
        unit["scoring_sub_questions"] = sub_questions
        yield unit


def request_score(client: OpenAI, model: str, unit: dict, definition: dict) -> dict:
    payload = {
        "knowledge_path": definition["knw_label"],
        "existing_interpretation": definition["existing_interpretation"],
        "complete_question": {
            "stem": unit.get("stem", ""),
            "options": unit.get("options", ""),
            "analysis": unit.get("analysis", ""),
            "sub_questions": unit.get("scoring_sub_questions", []),
        },
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
    judgement = answer.get("judgement")
    score = answer.get("score")
    reason = answer.get("reason")
    if judgement not in {"scored", "unjudgeable"}:
        raise ValueError("模型返回的judgement无效")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("模型返回的reason无效")
    if judgement == "unjudgeable":
        if score is not None:
            raise ValueError("unjudgeable必须对应score=null")
        match = None
    else:
        if type(score) not in (float, int) or not 0 <= score <= 1:
            raise ValueError("scored必须对应0到1的score")
        match = score >= 0.70
    return {
        "judgement": judgement,
        "score": score,
        "match": match,
        "reason": reason.strip(),
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
    log_file: str | None = None,
) -> dict[str, int]:
    if workers < 1 or limit is not None and limit < 1:
        raise ValueError("workers和limit必须为正整数")
    definitions = load_definitions(definitions_file)
    client = OpenAI(api_key="not-required", base_url=base_url, timeout=timeout, max_retries=0)
    run_logger = configure_run_logger(log_file)
    summary = {"attempted": 0, "completed": 0, "errors": 0}

    def pending_pairs():
        for unit in read_complete_questions(units_file):
            for label in dict.fromkeys(unit.get("knw_labels", [])):
                yield unit, label

    total_pairs = sum(
        len(dict.fromkeys(unit.get("knw_labels", [])))
        for unit in read_complete_questions(units_file)
    )
    if limit is not None:
        total_pairs = min(total_pairs, limit)
    started_at = time.monotonic()
    run_logger.info(
        "run_started input=%s output=%s total_pairs=%s workers=%s "
        "timeout=%s model=%s base_url=%s",
        units_file,
        output,
        total_pairs,
        workers,
        timeout,
        model,
        base_url,
    )

    def evaluate(pair):
        unit, label = pair
        result = {
            "question_id": str(unit["question_id"]),
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
            run_logger.error(
                "question_id=%s knw_label=%s error_type=%s error=%s",
                result["question_id"],
                label,
                type(error).__name__,
                error,
                exc_info=True,
            )
        return result

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        from itertools import islice
        pairs = pending_pairs()
        if limit is not None:
            pairs = islice(pairs, limit)
        with tqdm(total=total_pairs, desc="Scoring question-label pairs") as progress:
            while batch := list(islice(pairs, workers * 2)):
                for result in pool.map(evaluate, batch):
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    summary["attempted"] += 1
                    summary["completed" if result["status"] == "completed" else "errors"] += 1
                    run_logger.info(
                        "progress=%s/%s status=%s question_id=%s label_id=%s "
                        "score=%s match=%s elapsed=%.1fs",
                        summary["attempted"],
                        total_pairs,
                        result["status"],
                        result["question_id"],
                        result.get("label_id"),
                        result.get("score"),
                        result.get("match"),
                        time.monotonic() - started_at,
                    )
                    progress.update(1)
    run_logger.info(
        "run_finished attempted=%s completed=%s errors=%s elapsed=%.1fs",
        summary["attempted"],
        summary["completed"],
        summary["errors"],
        time.monotonic() - started_at,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export original definitions or score question-label pairs")
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("export-definitions")
    extraction.add_argument("--comparison-jsonl", required=True)
    extraction.add_argument("--output-jsonl", required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("--input-jsonl", required=True, help="Merged question JSONL file")
    scoring.add_argument("--definitions-jsonl", required=True)
    scoring.add_argument("--output-jsonl", required=True)
    scoring.add_argument("--base-url", required=True)
    scoring.add_argument("--model", default="DeepSeek-V4-Flash")
    scoring.add_argument("--workers", type=int, default=1)
    scoring.add_argument("--limit", type=int)
    scoring.add_argument("--timeout", type=float, default=180.0)
    scoring.add_argument(
        "--log-file",
        required=True,
        help="Overwrite this file with run progress and request errors",
    )
    arguments = parser.parse_args()
    if arguments.command == "export-definitions":
        print(json.dumps({"definitions": export_definitions(arguments.comparison_jsonl, arguments.output_jsonl)}, ensure_ascii=False))
    else:
        print(json.dumps(score_units(arguments.input_jsonl, arguments.definitions_jsonl, arguments.output_jsonl, arguments.base_url, arguments.model, arguments.workers, arguments.limit, arguments.timeout, arguments.log_file), ensure_ascii=False))


if __name__ == "__main__":
    main()
