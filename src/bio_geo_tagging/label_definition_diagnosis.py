"""Ask DeepSeek whether observed label-match issues can be fixed by definitions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import json
import logging
from pathlib import Path
import time
from typing import Any

from openai import OpenAI
from tqdm import tqdm


DEFINITION_STATUSES = {
    "adequate",
    "too_broad",
    "too_narrow",
    "ambiguous",
    "boundary_missing",
    "multiple_issues",
    "insufficient_evidence",
}

SYSTEM_PROMPT = """你是高中地理知识点释义诊断员。每次只诊断一个知识点。
输入包含原释义、相邻知识点、A/B级高匹配题和C/D级低匹配题。第一阶段分数只用于抽样，不代表最终正确。
这些题目只是从该标签现有题目中抽取的阶段性样本，不代表全部题目。不得把样本结论表述为“所有高匹配题”或“所有低匹配题”，也不得把样本直接外推为整个标签；证据不足时必须输出insufficient_evidence。
每个样本都是一道完整大题。大题标签是公共题干以及所有小题所考查知识点的并集：只要公共题干或任意一道小题直接考查当前知识点，该大题就与当前标签有效匹配。不得因为当前知识点只出现在部分小题、不是整道大题的主要主题，就将其判为假阳性、无关误标或释义问题。

你的目标是判断：当前问题是否能通过修改释义解决。
必须同时检查两个方向：
1. 高匹配题是否真正需要该知识点；如果题目只是背景相关或应属于相邻标签，却因释义过宽而被接受，属于too_broad证据。
2. 低匹配题是否本应属于该知识点；如果因释义遗漏、表述歧义或边界缺失而被排除，属于too_narrow、ambiguous或boundary_missing证据。

如果题目与标签完全无关，应归为unrelated_mislabel。这是历史误标或不可控噪声，不得通过扩大释义把它纳入。
如果第一阶段理由明显违背题目或释义，应归为model_misjudgement。
如果大题的任意一道小题直接考查当前知识点，但第一阶段因为它只出现在部分小题而给出低分，应归为model_misjudgement，不得归为low_score_definition_issue。
高匹配率本身不能证明释义太宽，低匹配率本身也不能证明释义太窄；必须依据题目证据。
只有存在可由修改释义解决的问题时，definition_fixable和teacher_review_required才为true。
每道A/B级高匹配样本必须且只能归入high_score_valid_ids、high_score_false_positive_ids或model_misjudgement_ids之一。
每道C/D级低匹配样本必须且只能归入low_score_definition_issue_ids、unrelated_mislabel_ids或model_misjudgement_ids之一。
凡是用于证明释义存在问题的题目，必须写入high_score_false_positive_ids或low_score_definition_issue_ids。analysis中不要写题目ID，题目ID只写入对应数组。

只输出JSON对象，字段必须为：
{
  "definition_status":"adequate|too_broad|too_narrow|ambiguous|boundary_missing|multiple_issues|insufficient_evidence",
  "definition_fixable":true或false,
  "analysis":"一段简洁中文结论",
  "high_score_valid_ids":["题目ID"],
  "high_score_false_positive_ids":["题目ID"],
  "low_score_definition_issue_ids":["题目ID"],
  "unrelated_mislabel_ids":["题目ID"],
  "model_misjudgement_ids":["题目ID"],
  "teacher_review_required":true或false,
  "revision_direction":"需要修改时说明修改方向；无需修改时为空字符串"
}。"""


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield number, json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {number} 行不是有效JSON") from error


def configure_run_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("label_definition_diagnosis.run")
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def strip_code_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```")
        content = content.removesuffix("```").strip()
    return content


def validate_id_list(value: Any, field: str, allowed_ids: set[str]) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"模型返回的{field}必须是字符串数组")
    if any(item not in allowed_ids for item in value):
        raise ValueError(f"模型返回的{field}包含输入样本之外的题目ID")
    return list(dict.fromkeys(value))


def validate_diagnosis(answer: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    status = answer.get("definition_status")
    if status not in DEFINITION_STATUSES:
        raise ValueError("模型返回的definition_status无效")
    fixable = answer.get("definition_fixable")
    teacher_review = answer.get("teacher_review_required")
    if type(fixable) is not bool or type(teacher_review) is not bool:
        raise ValueError("模型返回的布尔字段无效")
    if fixable != teacher_review:
        raise ValueError("definition_fixable必须与teacher_review_required一致")
    if status in {"adequate", "insufficient_evidence"} and fixable:
        raise ValueError("释义正常或证据不足时definition_fixable必须为false")
    if status not in {"adequate", "insufficient_evidence"} and not fixable:
        raise ValueError("存在释义问题时definition_fixable必须为true")

    analysis = answer.get("analysis")
    revision = answer.get("revision_direction")
    if not isinstance(analysis, str) or not analysis.strip():
        raise ValueError("模型返回的analysis无效")
    if not isinstance(revision, str):
        raise ValueError("模型返回的revision_direction无效")
    if fixable and not revision.strip():
        raise ValueError("需要修改释义时revision_direction不能为空")
    if not fixable and revision.strip():
        raise ValueError("无需修改释义时revision_direction必须为空")

    high_ids = {
        str(example.get("question_id", ""))
        for example in review.get("high_score_examples", [])
    }
    low_ids = {
        str(example.get("question_id", ""))
        for example in review.get("low_score_examples", [])
    }
    all_ids = high_ids | low_ids
    high_valid_ids = validate_id_list(
        answer.get("high_score_valid_ids"),
        "high_score_valid_ids",
        high_ids,
    )
    high_false_positive_ids = validate_id_list(
        answer.get("high_score_false_positive_ids"),
        "high_score_false_positive_ids",
        high_ids,
    )
    low_definition_issue_ids = validate_id_list(
        answer.get("low_score_definition_issue_ids"),
        "low_score_definition_issue_ids",
        low_ids,
    )
    unrelated_mislabel_ids = validate_id_list(
        answer.get("unrelated_mislabel_ids"),
        "unrelated_mislabel_ids",
        low_ids,
    )
    model_misjudgement_ids = validate_id_list(
        answer.get("model_misjudgement_ids"),
        "model_misjudgement_ids",
        all_ids,
    )
    category_sets = [
        set(high_valid_ids),
        set(high_false_positive_ids),
        set(low_definition_issue_ids),
        set(unrelated_mislabel_ids),
        set(model_misjudgement_ids),
    ]
    if sum(len(values) for values in category_sets) != len(set().union(*category_sets)):
        raise ValueError("同一道题不能同时归入多个诊断类别")
    classified_high_ids = (
        set(high_valid_ids)
        | set(high_false_positive_ids)
        | (set(model_misjudgement_ids) & high_ids)
    )
    if classified_high_ids != high_ids:
        raise ValueError("每道高匹配样本必须归入一个诊断类别")
    classified_low_ids = (
        set(low_definition_issue_ids)
        | set(unrelated_mislabel_ids)
        | (set(model_misjudgement_ids) & low_ids)
    )
    if classified_low_ids != low_ids:
        raise ValueError("每道低匹配样本必须归入一个诊断类别")

    definition_evidence_ids = high_false_positive_ids + low_definition_issue_ids
    if fixable and not definition_evidence_ids:
        raise ValueError("释义存在问题时必须提供证据题目ID")
    if not fixable and definition_evidence_ids:
        raise ValueError("释义无需修改时不能提供释义问题证据题目ID")
    if status == "too_broad" and not high_false_positive_ids:
        raise ValueError("too_broad必须提供高匹配假阳性题目ID")
    if status == "too_narrow" and not low_definition_issue_ids:
        raise ValueError("too_narrow必须提供低匹配释义问题题目ID")

    example_by_id = {
        str(example.get("question_id", "")): example
        for example in review.get("high_score_examples", [])
        + review.get("low_score_examples", [])
    }
    evidence_questions = []
    for evidence_type, question_ids in (
        ("high_score_false_positive", high_false_positive_ids),
        ("low_score_definition_issue", low_definition_issue_ids),
    ):
        for question_id in question_ids:
            evidence = dict(example_by_id[question_id])
            evidence["evidence_type"] = evidence_type
            evidence_questions.append(evidence)

    return {
        "definition_status": status,
        "definition_fixable": fixable,
        "analysis": analysis.strip(),
        "high_score_valid_ids": high_valid_ids,
        "high_score_false_positive_ids": high_false_positive_ids,
        "low_score_definition_issue_ids": low_definition_issue_ids,
        "unrelated_mislabel_ids": unrelated_mislabel_ids,
        "model_misjudgement_ids": model_misjudgement_ids,
        "definition_issue_evidence_questions": evidence_questions,
        "teacher_review_required": teacher_review,
        "revision_direction": revision.strip(),
    }


def request_diagnosis(
    client: OpenAI,
    model: str,
    review: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "label_id": review.get("label_id"),
        "knowledge_path": review.get("knw_label"),
        "existing_interpretation": review.get("existing_interpretation", {}),
        "adjacent_labels": review.get("adjacent_labels", []),
        "distribution": {
            "evaluated_count": review.get("evaluated_count"),
            "grade_counts": review.get("grade_counts", {}),
            "match_rate": review.get("match_rate"),
        },
        "sampling_note": "每组最多10题：一半为接近分级阈值的边界题，一半为固定抽样题；这些题仅代表样本。",
        "root_label_rule": "大题标签是公共题干和所有小题知识点的并集；任意小题直接考查该知识点即为有效匹配。",
        "high_score_examples": review.get("high_score_examples", []),
        "low_score_examples": review.get("low_score_examples", []),
    }
    chunks = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
    )
    content = "".join(
        chunk.choices[0].delta.content or ""
        for chunk in chunks
        if chunk.choices
    )
    answer = json.loads(strip_code_fence(content))
    if not isinstance(answer, dict):
        raise ValueError("模型返回内容不是JSON对象")
    return validate_diagnosis(answer, review)


def diagnose(
    review_samples_jsonl: str,
    output_jsonl: str,
    log_file: str,
    base_url: str,
    model: str,
    workers: int = 1,
    limit: int | None = None,
    timeout: float = 300.0,
    max_tokens: int = 768,
    retry_errors: bool = False,
) -> dict[str, int]:
    if workers < 1 or limit is not None and limit < 1:
        raise ValueError("workers和limit必须为正整数")
    if max_tokens < 1:
        raise ValueError("max_tokens必须为正整数")
    all_reviews = [record for _, record in read_jsonl(review_samples_jsonl)]
    target = Path(output_jsonl)
    target.parent.mkdir(parents=True, exist_ok=True)
    previous_results: dict[str, dict[str, Any]] = {}
    if retry_errors:
        if not target.exists():
            raise ValueError("使用--retry-errors时输出文件必须已经存在")
        previous_results = {
            str(record.get("label_id", "")): record
            for _, record in read_jsonl(output_jsonl)
        }
        selected_reviews = [
            review
            for review in all_reviews
            if previous_results.get(str(review.get("label_id", "")), {}).get("status")
            != "completed"
        ]
    else:
        selected_reviews = all_reviews
    if limit is not None:
        selected_reviews = selected_reviews[:limit]
    total = len(selected_reviews)
    logger = configure_run_logger(log_file)
    client = OpenAI(
        api_key="not-required",
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )
    summary = {"attempted": 0, "completed": 0, "errors": 0}
    started_at = time.monotonic()
    logger.info(
        "run_started input=%s output=%s total_labels=%s workers=%s "
        "timeout=%s model=%s base_url=%s retry_errors=%s",
        review_samples_jsonl,
        output_jsonl,
        total,
        workers,
        timeout,
        model,
        base_url,
        retry_errors,
    )

    def evaluate(review):
        result = {
            "label_id": str(review.get("label_id", "")),
            "knw_label": review.get("knw_label"),
            "evaluated_count": review.get("evaluated_count"),
            "grade_counts": review.get("grade_counts", {}),
            "model": model,
        }
        try:
            result.update(request_diagnosis(client, model, review, max_tokens))
            result["status"] = "completed"
        except Exception as error:
            result.update(
                status="error",
                error_type=type(error).__name__,
                error=str(error),
            )
            logger.error(
                "label_id=%s knw_label=%s error_type=%s error=%s",
                result["label_id"],
                result["knw_label"],
                type(error).__name__,
                error,
                exc_info=True,
            )
        return result

    def process_reviews(reviews, on_result):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            with tqdm(total=total, desc="Diagnosing label definitions") as progress:
                while batch := list(islice(reviews, workers * 2)):
                    for result in pool.map(evaluate, batch):
                        on_result(result)
                        summary["attempted"] += 1
                        summary["completed" if result["status"] == "completed" else "errors"] += 1
                        logger.info(
                            "progress=%s/%s status=%s label_id=%s definition_status=%s "
                            "fixable=%s elapsed=%.1fs",
                            summary["attempted"],
                            total,
                            result["status"],
                            result["label_id"],
                            result.get("definition_status"),
                            result.get("definition_fixable"),
                            time.monotonic() - started_at,
                        )
                        progress.update(1)

    if retry_errors:
        retried_results = {}

        def keep_result(result):
            retried_results[result["label_id"]] = result

        process_reviews(iter(selected_reviews), keep_result)
        temporary_target = target.with_name(target.name + ".tmp")
        with temporary_target.open("w", encoding="utf-8") as stream:
            for review in all_reviews:
                label_id = str(review.get("label_id", ""))
                result = retried_results.get(label_id) or previous_results.get(label_id)
                if result is not None:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        temporary_target.replace(target)
    else:
        with target.open("w", encoding="utf-8") as stream:

            def write_result(result):
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()

            process_reviews(iter(selected_reviews), write_result)
    logger.info(
        "run_finished attempted=%s completed=%s errors=%s elapsed=%.1fs",
        summary["attempted"],
        summary["completed"],
        summary["errors"],
        time.monotonic() - started_at,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether label definitions can be improved")
    parser.add_argument("--review-samples-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="DeepSeek-V4-Flash")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Preserve completed rows and retry only error or missing labels",
    )
    args = parser.parse_args()
    result = diagnose(
        args.review_samples_jsonl,
        args.output_jsonl,
        args.log_file,
        args.base_url,
        args.model,
        args.workers,
        args.limit,
        args.timeout,
        args.max_tokens,
        args.retry_errors,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
