"""Compare BM25 label-field weights for ordinary geography labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .build_retrieval_index import build_bm25_index
from .hybrid_candidate_retrieval import (
    NONREGION_BM25_PRIMARY_QUOTA,
    as_text,
    bm25_scores,
    combine_nonregion_candidates,
    encode_queries,
    finalize_metrics,
    is_region_label_path,
    is_strict_comprehensive_label_path,
    load_index,
    metric_record,
    question_gold_labels,
    question_query_text,
    ranked_candidates,
    read_jsonl,
    top_indices_from_pool,
    update_metrics,
)


WEIGHT_SCHEMES: dict[str, dict[str, int]] = {
    "b0_baseline": {
        "label_name": 0,
        "label_path": 1,
        "definition": 1,
        "keywords": 1,
        "assessment_scope": 1,
    },
    "w1_name": {
        "label_name": 4,
        "label_path": 1,
        "definition": 1,
        "keywords": 1,
        "assessment_scope": 1,
    },
    "w2_keywords": {
        "label_name": 0,
        "label_path": 1,
        "definition": 1,
        "keywords": 3,
        "assessment_scope": 1,
    },
    "w3_definition": {
        "label_name": 0,
        "label_path": 1,
        "definition": 3,
        "keywords": 1,
        "assessment_scope": 1,
    },
    "w4_assessment": {
        "label_name": 0,
        "label_path": 1,
        "definition": 1,
        "keywords": 1,
        "assessment_scope": 3,
    },
    "w5_positive_balanced": {
        "label_name": 4,
        "label_path": 1,
        "definition": 2,
        "keywords": 2,
        "assessment_scope": 1,
    },
    "w6_without_path": {
        "label_name": 4,
        "label_path": 0,
        "definition": 2,
        "keywords": 2,
        "assessment_scope": 1,
    },
}

BM25_EVALUATION_LIMITS = (5, 10, 15, NONREGION_BM25_PRIMARY_QUOTA)
NONREGION_CANDIDATE_LIMIT = 30


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(
            text for item in value if (text := normalize_text(item))
        )
    return str(value).strip()


def load_definition_fields(path: Path) -> dict[str, dict[str, str]]:
    definitions: dict[str, dict[str, str]] = {}
    for line_number, record in read_jsonl(path):
        label_path = normalize_text(record.get("label_path"))
        if not label_path:
            raise ValueError(f"检索释义第{line_number}行缺少label_path")
        if label_path in definitions:
            raise ValueError(f"检索释义标签路径重复：{label_path}")
        definitions[label_path] = {
            "label_name": label_path.rsplit("@", 1)[-1],
            "label_path": label_path,
            "definition": normalize_text(
                record.get("positive_definition") or record.get("definition")
            ),
            "keywords": normalize_text(record.get("keywords")),
            "assessment_scope": normalize_text(
                record.get("assessment_scope") or record.get("exam_methods")
            ),
        }
    if not definitions:
        raise ValueError("检索释义文件中没有标签")
    return definitions


def build_weighted_bm25_text(
    fields: dict[str, str], weights: dict[str, int]
) -> str:
    parts: list[str] = []
    for field in (
        "label_name",
        "label_path",
        "definition",
        "keywords",
        "assessment_scope",
    ):
        weight = weights[field]
        if weight < 0:
            raise ValueError(f"字段权重不能为负数：{field}={weight}")
        value = fields[field]
        if value and weight:
            parts.extend([value] * weight)
    text = " ".join(parts)
    if not text:
        raise ValueError(f"权重方案生成了空BM25文本：{fields['label_path']}")
    return text


def weighted_records(
    labels: list[dict[str, Any]],
    definitions: dict[str, dict[str, str]],
    ordinary_indices: list[int],
    weights: dict[str, int],
) -> list[dict[str, Any]]:
    ordinary_index_set = set(ordinary_indices)
    records: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        label_path = label["label_path"]
        bm25_text = label["bm25_text"]
        if index in ordinary_index_set:
            fields = definitions.get(label_path)
            if fields is None:
                raise ValueError(f"检索释义缺少普通标签：{label_path}")
            bm25_text = build_weighted_bm25_text(fields, weights)
        records.append({**label, "bm25_text": bm25_text})
    return records


def split_name(question_id: str) -> str:
    bucket = int(hashlib.sha256(question_id.encode("utf-8")).hexdigest(), 16) % 10
    return "development" if bucket < 7 else "validation"


def mean_or_none(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def finalized_metric_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    result = {
        "bm25": {
            f"top_{limit}": finalize_metrics(bundle["bm25"][limit])
            for limit in BM25_EVALUATION_LIMITS
        },
        "final_30": finalize_metrics(bundle["final_30"]),
        "gold_rank": {
            "mean_reciprocal_rank": mean_or_none(bundle["reciprocal_ranks"]),
            "mean_rank": mean_or_none(bundle["gold_ranks"]),
            "unranked_gold_labels": bundle["unranked_gold_labels"],
        },
    }
    return result


def empty_metric_bundle() -> dict[str, Any]:
    return {
        "bm25": {limit: metric_record() for limit in BM25_EVALUATION_LIMITS},
        "final_30": metric_record(),
        "reciprocal_ranks": [],
        "gold_ranks": [],
        "unranked_gold_labels": 0,
    }


def evaluate_scheme(
    questions: list[dict[str, Any]],
    queries: list[str],
    labels: list[dict[str, Any]],
    ordinary_indices: list[int],
    bm25: dict[str, Any],
    dense_candidates: list[list[dict[str, Any]]],
    allowed_labels: set[str],
    candidate_limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, set[str]]]:
    bundles = {
        name: empty_metric_bundle()
        for name in ("all", "development", "validation")
    }
    diagnostics: list[dict[str, Any]] = []
    selected_by_question: dict[str, set[str]] = {}
    for question, query, dense in zip(questions, queries, dense_candidates):
        question_id = as_text(question.get("question_id"))
        sparse_scores = bm25_scores(query, bm25)
        sparse = ranked_candidates(
            top_indices_from_pool(
                sparse_scores,
                ordinary_indices,
                len(ordinary_indices),
            ),
            sparse_scores,
            labels,
        )
        final, _ = combine_nonregion_candidates(sparse, dense, candidate_limit)
        selected = {item["label_path"] for item in final}
        selected_by_question[question_id] = selected
        gold = [
            label
            for label in question_gold_labels(question)
            if label in allowed_labels
            and not is_region_label_path(label)
            and not is_strict_comprehensive_label_path(label)
        ]
        split = split_name(question_id)
        sparse_paths = [item["label_path"] for item in sparse]
        ranks = {label: rank for rank, label in enumerate(sparse_paths, 1)}
        for bundle_name in ("all", split):
            bundle = bundles[bundle_name]
            for limit in BM25_EVALUATION_LIMITS:
                update_metrics(
                    bundle["bm25"][limit],
                    gold,
                    set(sparse_paths[:limit]),
                )
            update_metrics(bundle["final_30"], gold, selected)
            for label in gold:
                rank = ranks.get(label)
                if rank is None:
                    bundle["unranked_gold_labels"] += 1
                else:
                    bundle["gold_ranks"].append(float(rank))
                    bundle["reciprocal_ranks"].append(1.0 / rank)
        diagnostics.append(
            {
                "question_id": question_id,
                "split": split,
                "gold_labels": gold,
                "bm25_gold_ranks": {
                    label: ranks.get(label) for label in gold
                },
                "final_recalled_labels": sorted(set(gold) & selected),
                "final_missing_labels": sorted(set(gold) - selected),
            }
        )
    return (
        {name: finalized_metric_bundle(bundle) for name, bundle in bundles.items()},
        diagnostics,
        selected_by_question,
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_experiment(
    input_path: Path,
    index_dir: Path,
    definitions_path: Path,
    output_dir: Path,
    batch_size: int,
    device: str | None,
) -> dict[str, Any]:
    manifest, labels, baseline_bm25, label_embeddings = load_index(index_dir)
    if baseline_bm25.get("tokenizer") != "unicode_char_ngram":
        raise ValueError("权重实验只支持字符n-gram BM25索引")
    definitions = load_definition_fields(definitions_path)
    allowed_labels = {label["label_path"] for label in labels}
    ordinary_indices = [
        index
        for index, label in enumerate(labels)
        if not is_region_label_path(label["label_path"])
        and not is_strict_comprehensive_label_path(label["label_path"])
    ]
    missing_definitions = sorted(
        labels[index]["label_path"]
        for index in ordinary_indices
        if labels[index]["label_path"] not in definitions
    )
    if missing_definitions:
        raise ValueError("检索释义缺少普通标签：" + "、".join(missing_definitions))

    questions: list[dict[str, Any]] = []
    queries: list[str] = []
    for line_number, question in read_jsonl(input_path):
        query = question_query_text(question)
        if not query:
            raise ValueError(f"题目文件第{line_number}行没有可用于检索的文本")
        questions.append(question)
        queries.append(query)

    model_name = manifest["embedding"]["model"]
    instruction = manifest["embedding"].get("query_instruction", "")
    query_embeddings = encode_queries(
        queries,
        model_name,
        instruction,
        batch_size,
        device,
    )
    dense_candidates: list[list[dict[str, Any]]] = []
    for query_embedding in query_embeddings:
        dense_scores = label_embeddings @ query_embedding
        dense_candidates.append(
            ranked_candidates(
                top_indices_from_pool(
                    dense_scores,
                    ordinary_indices,
                    len(ordinary_indices),
                ),
                dense_scores,
                labels,
            )
        )

    scheme_results: dict[str, Any] = {}
    scheme_diagnostics: dict[str, list[dict[str, Any]]] = {}
    selections: dict[str, dict[str, set[str]]] = {}
    for scheme_name, weights in WEIGHT_SCHEMES.items():
        if scheme_name == "b0_baseline":
            bm25 = baseline_bm25
        else:
            records = weighted_records(
                labels,
                definitions,
                ordinary_indices,
                weights,
            )
            bm25 = build_bm25_index(
                records,
                tuple(baseline_bm25["ngram_sizes"]),
                float(baseline_bm25["k1"]),
                float(baseline_bm25["b"]),
            )
        metrics, diagnostics, selected = evaluate_scheme(
            questions,
            queries,
            labels,
            ordinary_indices,
            bm25,
            dense_candidates,
            allowed_labels,
            NONREGION_CANDIDATE_LIMIT,
        )
        scheme_results[scheme_name] = {
            "weights": weights,
            "metrics": metrics,
        }
        scheme_diagnostics[scheme_name] = diagnostics
        selections[scheme_name] = selected

    baseline_selection = selections["b0_baseline"]
    baseline_diagnostics = {
        item["question_id"]: item
        for item in scheme_diagnostics["b0_baseline"]
    }
    change_counts = {
        scheme_name: {
            split: {"gained_labels": 0, "lost_labels": 0}
            for split in ("all", "development", "validation")
        }
        for scheme_name in WEIGHT_SCHEMES
        if scheme_name != "b0_baseline"
    }
    changes: list[dict[str, Any]] = []
    for scheme_name, diagnostics in scheme_diagnostics.items():
        if scheme_name == "b0_baseline":
            continue
        diagnostic_by_question = {
            item["question_id"]: item for item in diagnostics
        }
        for question in questions:
            question_id = as_text(question.get("question_id"))
            gold = set(diagnostic_by_question[question_id]["gold_labels"])
            baseline_recalled = gold & baseline_selection[question_id]
            experiment_recalled = gold & selections[scheme_name][question_id]
            gained = sorted(experiment_recalled - baseline_recalled)
            lost = sorted(baseline_recalled - experiment_recalled)
            if gained or lost:
                question_split = split_name(question_id)
                for split in ("all", question_split):
                    change_counts[scheme_name][split]["gained_labels"] += len(
                        gained
                    )
                    change_counts[scheme_name][split]["lost_labels"] += len(lost)
                changes.append(
                    {
                        "scheme": scheme_name,
                        "question_id": question_id,
                        "split": question_split,
                        "gained_labels": gained,
                        "lost_labels": lost,
                        "baseline_bm25_gold_ranks": baseline_diagnostics[
                            question_id
                        ]["bm25_gold_ranks"],
                        "experiment_bm25_gold_ranks": diagnostic_by_question[
                            question_id
                        ]["bm25_gold_ranks"],
                    }
                )

    baseline_metrics = scheme_results["b0_baseline"]["metrics"]
    for scheme_name, result in scheme_results.items():
        for split in ("all", "development", "validation"):
            current = result["metrics"][split]["final_30"]
            baseline = baseline_metrics[split]["final_30"]
            current["recalled_label_change"] = (
                current["recalled_gold_labels"] - baseline["recalled_gold_labels"]
            )
            current["fully_covered_question_change"] = (
                current["fully_covered_questions"]
                - baseline["fully_covered_questions"]
            )
            if scheme_name != "b0_baseline":
                current.update(change_counts[scheme_name][split])

    summary = {
        "input": str(input_path),
        "index": str(index_dir),
        "definitions": str(definitions_path),
        "input_questions": len(questions),
        "ordinary_label_count": len(ordinary_indices),
        "candidate_limit": NONREGION_CANDIDATE_LIMIT,
        "bm25_primary_quota": NONREGION_BM25_PRIMARY_QUOTA,
        "bge_supplement_limit": (
            NONREGION_CANDIDATE_LIMIT - NONREGION_BM25_PRIMARY_QUOTA
        ),
        "split_rule": "sha256(question_id) mod 10; 0-6 development, 7-9 validation",
        "embedding_model": model_name,
        "schemes": scheme_results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_dir / "changes.jsonl", changes)
    comparison_lines = [
        "\t".join(
            (
                "scheme",
                "label_name",
                "label_path",
                "definition",
                "keywords",
                "assessment_scope",
                "all_recalled_labels",
                "all_label_recall",
                "all_fully_covered_questions",
                "all_full_coverage_rate",
                "all_gained_labels",
                "all_lost_labels",
                "validation_recalled_labels",
                "validation_label_recall",
                "validation_fully_covered_questions",
                "validation_full_coverage_rate",
            )
        )
    ]
    for scheme_name, result in scheme_results.items():
        weights = result["weights"]
        all_metrics = result["metrics"]["all"]["final_30"]
        validation_metrics = result["metrics"]["validation"]["final_30"]
        comparison_lines.append(
            "\t".join(
                str(value)
                for value in (
                    scheme_name,
                    weights["label_name"],
                    weights["label_path"],
                    weights["definition"],
                    weights["keywords"],
                    weights["assessment_scope"],
                    all_metrics["recalled_gold_labels"],
                    all_metrics["label_recall"],
                    all_metrics["fully_covered_questions"],
                    all_metrics["full_coverage_rate"],
                    all_metrics.get("gained_labels", 0),
                    all_metrics.get("lost_labels", 0),
                    validation_metrics["recalled_gold_labels"],
                    validation_metrics["label_recall"],
                    validation_metrics["fully_covered_questions"],
                    validation_metrics["full_coverage_rate"],
                )
            )
        )
    (output_dir / "comparison.tsv").write_text(
        "\n".join(comparison_lines) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BM25 field weights for nonregional non-comprehensive labels"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--definitions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    args = parser.parse_args()
    summary = run_experiment(
        args.input,
        args.index_dir,
        args.definitions,
        args.output_dir,
        args.batch_size,
        args.device,
    )
    compact = {
        name: {
            "weights": result["weights"],
            "all": result["metrics"]["all"]["final_30"],
            "development": result["metrics"]["development"]["final_30"],
            "validation": result["metrics"]["validation"]["final_30"],
        }
        for name, result in summary["schemes"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
