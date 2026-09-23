"""Build automatic diagnostics for adjudication additions over legacy labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} line {line_number} must be a JSON object")
            rows.append(value)
    return rows


def as_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def label_paths(rows: Iterable[dict[str, Any]]) -> set[str]:
    paths = {
        as_text(row.get("label_path") or row.get("knw_label")) for row in rows
    }
    paths.discard("")
    if not paths:
        raise ValueError("labels file contains no label_path or knw_label")
    return paths


def gold_by_root(rows: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in rows:
        root_id = as_text(row.get("question_id") or row.get("parent_id"))
        if not root_id:
            raise ValueError("gold candidate row lacks question_id and parent_id")
        raw_labels = row.get("knw_labels") or []
        if not isinstance(raw_labels, list) or not all(
            isinstance(label, str) for label in raw_labels
        ):
            raise ValueError(f"knw_labels for {root_id} must be a string array")
        labels = [label.strip() for label in raw_labels if label.strip()]
        if root_id in result:
            labels = result[root_id] + labels
        result[root_id] = list(dict.fromkeys(labels))
    return result


def selected_paths(row: dict[str, Any]) -> set[str]:
    selected = row.get("selected_labels") or []
    if not isinstance(selected, list):
        raise ValueError("selected_labels must be an array")
    paths: set[str] = set()
    for label in selected:
        if not isinstance(label, dict):
            raise ValueError("selected label must be an object")
        path = as_text(label.get("label_path") or label.get("label_id"))
        if not path:
            raise ValueError("selected label lacks label_path and label_id")
        paths.add(path)
    return paths


def safe_rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def metric_summary(details: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(details)
    true_positive = sum(len(row["true_positive_labels"]) for row in rows)
    false_positive = sum(len(row["false_positive_labels"]) for row in rows)
    false_negative = sum(len(row["false_negative_labels"]) for row in rows)
    precision = safe_rate(true_positive, true_positive + false_positive)
    recall = safe_rate(true_positive, true_positive + false_negative)
    f1 = (
        round(2 * precision * recall / (precision + recall), 6)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    exact_matches = sum(int(row["exact_match"]) for row in rows)
    return {
        "questions": len(rows),
        "exact_match_questions": exact_matches,
        "exact_match_rate": safe_rate(exact_matches, len(rows)),
        "true_positive_labels": true_positive,
        "false_positive_labels": false_positive,
        "false_negative_labels": false_negative,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
    }


def label_category(path: str) -> str:
    if "综合" in path.rsplit("@", 1)[-1]:
        return "comprehensive"
    if "@中国地理@" in path or "@世界地理@" in path:
        return "regional"
    return "ordinary"


def hierarchy_redundancy_pairs(
    additions: set[str], final_labels: set[str]
) -> list[list[str]]:
    pairs: set[tuple[str, str]] = set()
    for addition in additions:
        for other in final_labels - {addition}:
            if addition.startswith(other + "@") or other.startswith(addition + "@"):
                pairs.add(tuple(sorted((addition, other))))
    return [list(pair) for pair in sorted(pairs)]


def evidence_diagnostics(prediction: dict[str, Any]) -> dict[str, int]:
    result = {
        "selected_labels": 0,
        "labels_without_evidence": 0,
        "empty_evidence_items": 0,
        "evidence_items_over_300_chars": 0,
    }
    selected = prediction.get("selected_labels") or []
    for label in selected:
        result["selected_labels"] += 1
        evidence_items = label.get("evidence_by_unit") or []
        if not evidence_items:
            result["labels_without_evidence"] += 1
            continue
        for item in evidence_items:
            evidence = as_text(item.get("evidence")) if isinstance(item, dict) else ""
            if not evidence:
                result["empty_evidence_items"] += 1
            elif len(evidence) > 300:
                result["evidence_items_over_300_chars"] += 1
    return result


def addition_summary(details: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(details)
    addition_counts = [len(row["ds_added_labels"]) for row in rows]
    category_counts = {"ordinary": 0, "regional": 0, "comprehensive": 0}
    for row in rows:
        for path in row["ds_added_labels"]:
            category_counts[label_category(path)] += 1
    questions_with_additions = sum(count > 0 for count in addition_counts)
    questions_over_five = sum(count > 5 for count in addition_counts)
    return {
        "questions": len(rows),
        "total_ds_added_labels": sum(addition_counts),
        "questions_with_additions": questions_with_additions,
        "questions_with_additions_rate": safe_rate(questions_with_additions, len(rows)),
        "average_ds_added_labels_per_question": (
            round(sum(addition_counts) / len(rows), 6) if rows else None
        ),
        "maximum_ds_added_labels_on_one_question": max(addition_counts, default=0),
        "questions_with_more_than_five_additions": questions_over_five,
        "addition_category_counts": category_counts,
    }


def agreement_summary(details: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(details)
    selected = sum(len(row["ds_predicted_labels"]) for row in rows)
    legacy = sum(len(row["legacy_labels"]) for row in rows)
    matched = sum(len(row["ds_matched_legacy_labels"]) for row in rows)
    ds_only = sum(len(row["ds_added_labels"]) for row in rows)
    legacy_not_selected = sum(
        len(row["legacy_labels_not_selected_by_ds"]) for row in rows
    )
    exact = sum(
        row["ds_predicted_labels"] == row["legacy_labels"] for row in rows
    )
    return {
        "questions": len(rows),
        "exact_agreement_questions": exact,
        "exact_agreement_rate": safe_rate(exact, len(rows)),
        "ds_selected_labels": selected,
        "legacy_labels": legacy,
        "matched_legacy_labels": matched,
        "ds_only_labels": ds_only,
        "legacy_labels_not_selected_by_ds": legacy_not_selected,
        "ds_selected_legacy_share": safe_rate(matched, selected),
        "legacy_label_coverage_rate": safe_rate(matched, legacy),
    }


def stability_summary(
    current: dict[str, dict[str, Any]], comparison: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    common = sorted(set(current) & set(comparison))
    exact = 0
    jaccards: list[float] = []
    for root_id in common:
        left = selected_paths(current[root_id])
        right = selected_paths(comparison[root_id])
        exact += int(left == right)
        union = left | right
        jaccards.append(len(left & right) / len(union) if union else 1.0)
    return {
        "available": True,
        "common_complete_questions": len(common),
        "exact_selected_set_matches": exact,
        "exact_selected_set_match_rate": safe_rate(exact, len(common)),
        "mean_jaccard": round(sum(jaccards) / len(jaccards), 6) if jaccards else None,
    }


def evaluate_adjudication(
    run_dir: Path,
    gold_candidates_path: Path,
    labels_path: Path,
    output_path: Path | None = None,
    details_path: Path | None = None,
    stability_run_dir: Path | None = None,
    final_labels_path: Path | None = None,
) -> dict[str, Any]:
    predictions_path = run_dir / "question_predictions.jsonl"
    evidence_path = run_dir / "evidence.jsonl"
    predictions = read_jsonl(predictions_path)
    attempted_roots = {
        as_text(row.get("root_question_id") or row.get("question_id"))
        for row in read_jsonl(evidence_path)
    }
    attempted_roots.discard("")
    allowed_labels = label_paths(read_jsonl(labels_path))
    gold = gold_by_root(read_jsonl(gold_candidates_path))

    prediction_by_root: dict[str, dict[str, Any]] = {}
    for row in predictions:
        root_id = as_text(row.get("root_question_id"))
        if not root_id:
            raise ValueError("question prediction lacks root_question_id")
        if root_id in prediction_by_root:
            raise ValueError(f"duplicate question prediction: {root_id}")
        prediction_by_root[root_id] = row

    complete_predictions = {
        root_id: row
        for root_id, row in prediction_by_root.items()
        if row.get("components_complete", True) is True
    }

    details: list[dict[str, Any]] = []
    no_legacy_source = 0
    empty_legacy_labels = 0
    unmapped_counts: dict[str, int] = {}
    for root_id, prediction in complete_predictions.items():
        if root_id not in gold:
            no_legacy_source += 1
            continue
        raw_gold = gold[root_id]
        empty_legacy_labels += int(not raw_gold)
        legacy_labels = set(raw_gold)
        known_gold = {label for label in raw_gold if label in allowed_labels}
        unmapped = sorted(set(raw_gold) - known_gold)
        for label in unmapped:
            unmapped_counts[label] = unmapped_counts.get(label, 0) + 1
        predicted = selected_paths(prediction)
        true_positive = sorted(known_gold & predicted)
        false_positive = sorted(predicted - known_gold)
        false_negative = sorted(known_gold - predicted)
        ds_added = predicted - legacy_labels
        final_labels = legacy_labels | ds_added
        redundancy_pairs = hierarchy_redundancy_pairs(ds_added, final_labels)
        evidence_checks = evidence_diagnostics(prediction)
        component_units = prediction.get("component_units") or []
        context_insufficient = any(
            bool(component.get("context_insufficient"))
            for component in component_units
            if isinstance(component, dict)
        )
        need_expand = any(
            bool(component.get("need_expand_recall"))
            for component in component_units
            if isinstance(component, dict)
        )
        details.append(
            {
                "root_question_id": root_id,
                "legacy_labels": sorted(legacy_labels),
                "ds_predicted_labels": sorted(predicted),
                "ds_matched_legacy_labels": sorted(legacy_labels & predicted),
                "ds_added_labels": sorted(ds_added),
                "legacy_labels_not_selected_by_ds": sorted(legacy_labels - predicted),
                "final_labels": sorted(final_labels),
                "ds_addition_category_counts": {
                    category: sum(
                        label_category(path) == category for path in ds_added
                    )
                    for category in ("ordinary", "regional", "comprehensive")
                },
                "hierarchy_redundancy_pairs": redundancy_pairs,
                "evidence_diagnostics": evidence_checks,
                "automatic_risk_flags": {
                    "context_insufficient_with_selected_labels": bool(
                        context_insufficient and predicted
                    ),
                    "need_expand_recall_with_selected_labels": bool(
                        need_expand and predicted
                    ),
                    "more_than_five_additions": len(ds_added) > 5,
                    "hierarchy_redundancy": bool(redundancy_pairs),
                    "evidence_validation_failure": any(
                        evidence_checks[key]
                        for key in (
                            "labels_without_evidence",
                            "empty_evidence_items",
                            "evidence_items_over_300_chars",
                        )
                    ),
                },
                # Deprecated compatibility fields. These measure agreement with
                # mapped legacy labels, not semantic correctness.
                "gold_labels": sorted(known_gold),
                "predicted_labels": sorted(predicted),
                "true_positive_labels": true_positive,
                "false_positive_labels": false_positive,
                "false_negative_labels": false_negative,
                "unmapped_gold_labels": unmapped,
                "exact_match": not false_positive and not false_negative,
                "usable_for_training": bool(prediction.get("usable_for_training")),
                "needs_review": bool(prediction.get("needs_review")),
            }
        )

    usable_details = [row for row in details if row["usable_for_training"]]
    evidence_totals = {
        key: sum(row["evidence_diagnostics"][key] for row in details)
        for key in (
            "selected_labels",
            "labels_without_evidence",
            "empty_evidence_items",
            "evidence_items_over_300_chars",
        )
    }
    risk_flag_counts = {
        key: sum(row["automatic_risk_flags"][key] for row in details)
        for key in (
            "context_insufficient_with_selected_labels",
            "need_expand_recall_with_selected_labels",
            "more_than_five_additions",
            "hierarchy_redundancy",
            "evidence_validation_failure",
        )
    }
    if stability_run_dir is None:
        stability = {
            "available": False,
            "reason": "provide --stability-run-dir with an independent repeated run",
        }
    else:
        comparison_rows = read_jsonl(stability_run_dir / "question_predictions.jsonl")
        comparison = {
            as_text(row.get("root_question_id")): row
            for row in comparison_rows
            if row.get("components_complete", True) is True
        }
        stability = stability_summary(complete_predictions, comparison)
    summary = {
        "report_kind": "automatic_adjudication_diagnostics",
        "semantic_addition_accuracy_available": False,
        "semantic_addition_accuracy_note": (
            "DS additions are not manually adjudicated; automatic diagnostics "
            "must not be reported as semantic precision or accuracy."
        ),
        "attempted_questions": len(attempted_roots),
        "question_prediction_rows": len(prediction_by_root),
        "completed_question_predictions": len(complete_predictions),
        "incomplete_question_predictions": len(prediction_by_root)
        - len(complete_predictions),
        "attempted_questions_without_prediction": len(
            attempted_roots - set(prediction_by_root)
        ),
        "attempted_questions_without_complete_prediction": len(
            attempted_roots - set(complete_predictions)
        ),
        "completed_predictions_without_legacy_source": no_legacy_source,
        "completed_predictions_without_gold": empty_legacy_labels,
        "completed_predictions_with_only_unmapped_gold": sum(
            1
            for root_id in complete_predictions
            if gold.get(root_id)
            and not any(label in allowed_labels for label in gold[root_id])
        ),
        "unmapped_gold_label_counts": dict(sorted(unmapped_counts.items())),
        "addition_volume": addition_summary(details),
        "usable_addition_volume": addition_summary(usable_details),
        "redundancy": {
            "questions_with_hierarchy_redundancy": risk_flag_counts[
                "hierarchy_redundancy"
            ],
            "hierarchy_redundancy_pairs": sum(
                len(row["hierarchy_redundancy_pairs"]) for row in details
            ),
        },
        "evidence_validation": evidence_totals,
        "automatic_risk_flag_counts": risk_flag_counts,
        "stability": stability,
        "legacy_label_agreement": agreement_summary(details),
        "usable_legacy_label_agreement": agreement_summary(usable_details),
        # Deprecated aliases retained for existing report readers.
        "all_completed_with_known_gold": metric_summary(details),
        "usable_for_training_with_known_gold": metric_summary(usable_details),
    }

    output_path = output_path or run_dir / "evaluation.json"
    details_path = details_path or run_dir / "evaluation_details.jsonl"
    final_labels_path = final_labels_path or run_dir / "final_labels.jsonl"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with details_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with final_labels_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in details:
            final_row = {
                "root_question_id": row["root_question_id"],
                "legacy_labels": row["legacy_labels"],
                "ds_added_labels": row["ds_added_labels"],
                "final_labels": row["final_labels"],
                "components_complete": True,
                "usable_for_training": row["usable_for_training"],
                "needs_review": row["needs_review"],
            }
            handle.write(
                json.dumps(final_row, ensure_ascii=False, sort_keys=True) + "\n"
            )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--legacy-candidates",
        "--gold-candidates",
        dest="gold_candidates",
        type=Path,
        required=True,
        help="candidate file containing the legacy knw_labels",
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--details-output", type=Path)
    parser.add_argument("--final-labels-output", type=Path)
    parser.add_argument(
        "--stability-run-dir",
        type=Path,
        help="independent repeated run used only for selected-label stability",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = evaluate_adjudication(
        args.run_dir,
        args.gold_candidates,
        args.labels,
        args.output,
        args.details_output,
        args.stability_run_dir,
        args.final_labels_output,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
