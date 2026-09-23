"""Evaluate whole-question adjudication predictions against existing labels."""

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


def evaluate_adjudication(
    run_dir: Path,
    gold_candidates_path: Path,
    labels_path: Path,
    output_path: Path | None = None,
    details_path: Path | None = None,
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
    no_gold = 0
    unmapped_counts: dict[str, int] = {}
    for root_id, prediction in complete_predictions.items():
        raw_gold = gold.get(root_id, [])
        if not raw_gold:
            no_gold += 1
            continue
        known_gold = {label for label in raw_gold if label in allowed_labels}
        unmapped = sorted(set(raw_gold) - known_gold)
        for label in unmapped:
            unmapped_counts[label] = unmapped_counts.get(label, 0) + 1
        if not known_gold:
            continue
        predicted = selected_paths(prediction)
        true_positive = sorted(known_gold & predicted)
        false_positive = sorted(predicted - known_gold)
        false_negative = sorted(known_gold - predicted)
        details.append(
            {
                "root_question_id": root_id,
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
    summary = {
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
        "completed_predictions_without_gold": no_gold,
        "completed_predictions_with_only_unmapped_gold": sum(
            1
            for root_id in complete_predictions
            if gold.get(root_id)
            and not any(label in allowed_labels for label in gold[root_id])
        ),
        "unmapped_gold_label_counts": dict(sorted(unmapped_counts.items())),
        "all_completed_with_known_gold": metric_summary(details),
        "usable_for_training_with_known_gold": metric_summary(usable_details),
    }

    output_path = output_path or run_dir / "evaluation.json"
    details_path = details_path or run_dir / "evaluation_details.jsonl"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with details_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gold-candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--details-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = evaluate_adjudication(
        args.run_dir,
        args.gold_candidates,
        args.labels,
        args.output,
        args.details_output,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
