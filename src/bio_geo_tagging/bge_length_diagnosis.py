"""Diagnose whether BGE query truncation is associated with retrieval misses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

from .hybrid_candidate_retrieval import (
    as_text,
    is_region_label_path,
    is_strict_comprehensive_label_path,
    question_gold_labels,
    question_query_text,
    read_jsonl,
)


def candidate_paths(record: dict[str, Any], field: str) -> list[str]:
    values = record.get(field) or []
    if not isinstance(values, list):
        raise ValueError(f"候选结果字段{field}必须是数组")
    paths: list[str] = []
    for item in values:
        if not isinstance(item, dict):
            raise ValueError(f"候选结果字段{field}中的元素必须是对象")
        label_path = as_text(item.get("label_path"))
        if not label_path:
            raise ValueError(f"候选结果字段{field}中的元素缺少label_path")
        paths.append(label_path)
    return paths


def nonregional_gold_labels(question: dict[str, Any]) -> list[str]:
    return [
        label
        for label in question_gold_labels(question)
        if not is_region_label_path(label)
        and not is_strict_comprehensive_label_path(label)
    ]


def empty_metrics() -> dict[str, int]:
    return {
        "questions": 0,
        "questions_with_nonregional_gold": 0,
        "nonregional_gold_labels": 0,
        "bge_recalled_gold_labels": 0,
        "bge_fully_covered_questions": 0,
        "final_recalled_gold_labels": 0,
        "final_fully_covered_questions": 0,
    }


def update_metrics(
    metrics: dict[str, int],
    gold_labels: list[str],
    bge_labels: set[str],
    final_labels: set[str],
) -> None:
    metrics["questions"] += 1
    if not gold_labels:
        return
    gold = set(gold_labels)
    metrics["questions_with_nonregional_gold"] += 1
    metrics["nonregional_gold_labels"] += len(gold)
    metrics["bge_recalled_gold_labels"] += len(gold & bge_labels)
    metrics["bge_fully_covered_questions"] += int(gold <= bge_labels)
    metrics["final_recalled_gold_labels"] += len(gold & final_labels)
    metrics["final_fully_covered_questions"] += int(gold <= final_labels)


def finalize_metrics(metrics: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = dict(metrics)
    question_count = metrics["questions_with_nonregional_gold"]
    label_count = metrics["nonregional_gold_labels"]
    result["bge_full_coverage_rate"] = (
        round(metrics["bge_fully_covered_questions"] / question_count, 6)
        if question_count
        else None
    )
    result["bge_label_recall"] = (
        round(metrics["bge_recalled_gold_labels"] / label_count, 6)
        if label_count
        else None
    )
    result["final_full_coverage_rate"] = (
        round(metrics["final_fully_covered_questions"] / question_count, 6)
        if question_count
        else None
    )
    result["final_label_recall"] = (
        round(metrics["final_recalled_gold_labels"] / label_count, 6)
        if label_count
        else None
    )
    return result


def percentile(sorted_values: list[int], fraction: float) -> int | None:
    if not sorted_values:
        return None
    index = max(0, math.ceil(fraction * len(sorted_values)) - 1)
    return sorted_values[index]


def load_candidate_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for line_number, record in read_jsonl(path):
        question_id = as_text(record.get("question_id"))
        if not question_id:
            raise ValueError(f"候选结果第{line_number}行缺少question_id")
        if question_id in results:
            raise ValueError(f"候选结果中question_id重复：{question_id}")
        results[question_id] = record
    return results


def diagnose(
    questions: Iterable[dict[str, Any]],
    candidate_results: dict[str, dict[str, Any]],
    token_counter: Callable[[str], int],
    max_seq_length: int,
    instruction: str,
    bge_top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_seq_length <= 0:
        raise ValueError("max_seq_length必须大于0")
    if bge_top_k <= 0:
        raise ValueError("bge_top_k必须大于0")

    question_records = list(questions)
    details: list[dict[str, Any]] = []
    token_counts: list[int] = []
    metrics = {
        "within_limit": empty_metrics(),
        "truncated": empty_metrics(),
        "all": empty_metrics(),
    }

    for question in question_records:
        question_id = as_text(question.get("question_id"))
        if not question_id:
            raise ValueError("题目缺少question_id")
        candidate = candidate_results.get(question_id)
        if candidate is None:
            raise ValueError(f"题目{question_id}没有候选结果")

        query = question_query_text(question)
        if not query:
            raise ValueError(f"题目{question_id}没有可用于检索的文本")
        token_count = token_counter(f"{instruction}{query}")
        token_counts.append(token_count)
        is_truncated = token_count > max_seq_length
        bucket = "truncated" if is_truncated else "within_limit"

        gold = nonregional_gold_labels(question)
        bge = candidate_paths(candidate, "bge_candidates")[:bge_top_k]
        final = candidate_paths(candidate, "nonregional_final_candidates")
        for metric in (metrics[bucket], metrics["all"]):
            update_metrics(metric, gold, set(bge), set(final))

        if is_truncated:
            details.append(
                {
                    "question_id": question_id,
                    "parent_id": as_text(question.get("parent_id")),
                    "subquestion_count": len(question.get("sub_questions") or []),
                    "query_character_count": len(query),
                    "token_count": token_count,
                    "max_seq_length": max_seq_length,
                    "excess_tokens": token_count - max_seq_length,
                    "nonregional_gold_labels": gold,
                    "bge_top_k": bge_top_k,
                    "bge_missing_labels": [label for label in gold if label not in bge],
                    "final_missing_labels": [
                        label for label in gold if label not in final
                    ],
                }
            )

    extra_candidate_ids = sorted(set(candidate_results) - {
        as_text(question.get("question_id")) for question in question_records
    })
    sorted_counts = sorted(token_counts)
    truncated_count = len(details)
    summary = {
        "input_questions": len(token_counts),
        "candidate_results": len(candidate_results),
        "candidate_results_without_input_question": len(extra_candidate_ids),
        "max_seq_length": max_seq_length,
        "bge_top_k": bge_top_k,
        "length": {
            "minimum_tokens": min(sorted_counts) if sorted_counts else None,
            "average_tokens": round(sum(sorted_counts) / len(sorted_counts), 2)
            if sorted_counts
            else None,
            "median_tokens": percentile(sorted_counts, 0.5),
            "p90_tokens": percentile(sorted_counts, 0.9),
            "p95_tokens": percentile(sorted_counts, 0.95),
            "maximum_tokens": max(sorted_counts) if sorted_counts else None,
            "truncated_questions": truncated_count,
            "truncated_rate": round(truncated_count / len(token_counts), 6)
            if token_counts
            else None,
        },
        "metrics": {
            name: finalize_metrics(value) for name, value in metrics.items()
        },
    }
    return details, summary


def load_token_counter(
    model_name_or_path: str, device: str | None
) -> tuple[Callable[[str], int], int]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "缺少sentence-transformers，请安装项目的retrieval可选依赖"
        ) from error

    arguments: dict[str, Any] = {}
    if device:
        arguments["device"] = device
    model = SentenceTransformer(model_name_or_path, **arguments)
    tokenizer = model.tokenizer

    def count_tokens(text: str) -> int:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded["input_ids"]
        return len(input_ids)

    return count_tokens, int(model.max_seq_length)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose BGE query length and retrieval recall by truncation status"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--embedding-model")
    parser.add_argument("--device")
    parser.add_argument("--bge-top-k", type=int, default=23)
    args = parser.parse_args()

    manifest = json.loads(
        (args.index_dir / "manifest.json").read_text(encoding="utf-8")
    )
    model_name = args.embedding_model or manifest["embedding"]["model"]
    instruction = manifest["embedding"].get("query_instruction", "")
    token_counter, max_seq_length = load_token_counter(model_name, args.device)
    questions = [record for _, record in read_jsonl(args.input)]
    candidates = load_candidate_results(args.candidates)
    details, summary = diagnose(
        questions=questions,
        candidate_results=candidates,
        token_counter=token_counter,
        max_seq_length=max_seq_length,
        instruction=instruction,
        bge_top_k=args.bge_top_k,
    )
    summary["embedding_model"] = model_name
    summary["query_instruction"] = instruction

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in details:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
