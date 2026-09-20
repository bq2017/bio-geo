"""Evaluate call-one candidate recall against existing knw_labels."""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .call1_candidate_retrieval import (
    as_text,
    get_input_role,
    has_question_content,
    load_catalog,
    make_unit_key,
    validate_result,
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}第{line_number}行JSON无效：{error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}第{line_number}行不是JSON对象")
            yield line_number, record


def load_units_to_db(
    connection: sqlite3.Connection, input_path: Path
) -> tuple[int, int]:
    eligible_units = 0
    skipped_empty_stem = 0
    rows: list[tuple[str, str, str, str, str]] = []
    for line_number, unit in read_jsonl(input_path):
        unit_key = make_unit_key(unit)
        if not has_question_content(unit):
            skipped_empty_stem += 1
            continue
        labels = unit.get("knw_labels")
        if labels is None:
            labels = []
        if not isinstance(labels, list) or not all(
            isinstance(label, str) for label in labels
        ):
            raise ValueError(f"题目文件第{line_number}行knw_labels必须是字符串数组")
        labels = list(dict.fromkeys(label.strip() for label in labels if label.strip()))
        rows.append(
            (
                unit_key,
                as_text(unit.get("question_id")),
                as_text(
                    unit.get("root_question_id")
                    or unit.get("parent_id")
                    or unit.get("question_id")
                ),
                get_input_role(unit),
                json.dumps(labels, ensure_ascii=False),
            )
        )
        eligible_units += 1
        if len(rows) == 10_000:
            _insert_units(connection, rows)
            rows.clear()
    if rows:
        _insert_units(connection, rows)
    connection.commit()
    return eligible_units, skipped_empty_stem


def _insert_units(
    connection: sqlite3.Connection, rows: list[tuple[str, str, str, str, str]]
) -> None:
    try:
        connection.executemany(
            "INSERT INTO units VALUES (?, ?, ?, ?, ?)", rows
        )
    except sqlite3.IntegrityError as error:
        raise ValueError("题目文件存在重复unit_key") from error


def load_candidates_to_db(
    connection: sqlite3.Connection,
    candidates_path: Path,
    allowed_paths: set[str],
) -> int:
    count = 0
    rows: list[tuple[str, str]] = []
    for line_number, record in read_jsonl(candidates_path):
        if record.get("status") != "completed":
            continue
        unit_key = as_text(record.get("unit_key"))
        if not unit_key:
            raise ValueError(f"候选文件第{line_number}行缺少unit_key")
        labels, _ = validate_result(record, allowed_paths)
        rows.append((unit_key, json.dumps(labels, ensure_ascii=False)))
        count += 1
        if len(rows) == 10_000:
            _insert_candidates(connection, rows)
            rows.clear()
    if rows:
        _insert_candidates(connection, rows)
    connection.commit()
    return count


def _insert_candidates(
    connection: sqlite3.Connection, rows: list[tuple[str, str]]
) -> None:
    try:
        connection.executemany("INSERT INTO candidates VALUES (?, ?)", rows)
    except sqlite3.IntegrityError as error:
        raise ValueError("候选文件存在重复unit_key") from error


def load_pre_candidates_to_db(
    connection: sqlite3.Connection,
    trace_path: Path,
    allowed_paths: set[str],
) -> int:
    count = 0
    rows: list[tuple[str, str]] = []
    for line_number, record in read_jsonl(trace_path):
        unit_key = as_text(record.get("unit_key"))
        labels = record.get("global_pre_candidate_labels")
        if not unit_key:
            raise ValueError(f"trace文件第{line_number}行缺少unit_key")
        if not isinstance(labels, list) or not all(
            isinstance(label, str) for label in labels
        ):
            raise ValueError(
                f"trace文件第{line_number}行缺少global_pre_candidate_labels"
            )
        labels = list(dict.fromkeys(label.strip() for label in labels))
        unknown = [label for label in labels if label not in allowed_paths]
        if unknown:
            raise ValueError(
                f"trace文件第{line_number}行含目录外预候选：{unknown}"
            )
        rows.append((unit_key, json.dumps(labels, ensure_ascii=False)))
        count += 1
        if len(rows) == 10_000:
            _insert_pre_candidates(connection, rows)
            rows.clear()
    if rows:
        _insert_pre_candidates(connection, rows)
    connection.commit()
    return count


def _insert_pre_candidates(
    connection: sqlite3.Connection, rows: list[tuple[str, str]]
) -> None:
    try:
        connection.executemany("INSERT INTO pre_candidates VALUES (?, ?)", rows)
    except sqlite3.IntegrityError as error:
        raise ValueError("trace文件存在重复unit_key") from error


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def evaluate_global_pre_recall(
    connection: sqlite3.Connection,
    allowed_paths: set[str],
) -> dict[str, Any]:
    evaluated_units = 0
    units_with_known_gold = 0
    fully_covered_units = 0
    known_gold_labels = 0
    recalled_gold_labels = 0
    candidate_sizes: list[int] = []
    missing_counts: Counter[str] = Counter()

    query = """
        SELECT u.knw_labels, p.candidate_labels
        FROM units AS u
        JOIN candidates AS c ON c.unit_key = u.unit_key
        JOIN pre_candidates AS p ON p.unit_key = u.unit_key
        ORDER BY u.rowid
    """
    for gold_json, candidate_json in connection.execute(query):
        gold = [
            label for label in json.loads(gold_json) if label in allowed_paths
        ]
        candidates = json.loads(candidate_json)
        candidate_set = set(candidates)
        missing = [label for label in gold if label not in candidate_set]
        evaluated_units += 1
        candidate_sizes.append(len(candidates))
        if gold:
            units_with_known_gold += 1
            known_gold_labels += len(gold)
            recalled_gold_labels += len(gold) - len(missing)
            if not missing:
                fully_covered_units += 1
            else:
                missing_counts.update(missing)

    return {
        "evaluated_units": evaluated_units,
        "units_with_known_gold": units_with_known_gold,
        "fully_covered_units": fully_covered_units,
        "full_coverage_rate": safe_rate(
            fully_covered_units, units_with_known_gold
        ),
        "known_gold_labels": known_gold_labels,
        "recalled_gold_labels": recalled_gold_labels,
        "label_recall": safe_rate(recalled_gold_labels, known_gold_labels),
        "candidate_count": {
            "average": safe_rate(sum(candidate_sizes), len(candidate_sizes)),
            "minimum": min(candidate_sizes) if candidate_sizes else None,
            "maximum": max(candidate_sizes) if candidate_sizes else None,
        },
        "missing_label_counts": dict(missing_counts.most_common()),
    }


def new_role_stats() -> dict[str, int]:
    return {
        "evaluated_units": 0,
        "units_with_known_gold": 0,
        "fully_covered_units": 0,
        "known_gold_labels": 0,
        "recalled_gold_labels": 0,
    }


def evaluate_joined_records(
    connection: sqlite3.Connection,
    allowed_paths: set[str],
    missing_output: Path,
) -> dict[str, Any]:
    evaluated_units = 0
    units_with_no_knw_labels = 0
    units_with_unmapped_labels = 0
    units_with_known_gold = 0
    fully_covered_units = 0
    known_gold_count = 0
    recalled_gold_count = 0
    candidate_count_sum = 0
    candidate_count_min: int | None = None
    candidate_count_max = 0
    missing_counts: Counter[str] = Counter()
    unmapped_counts: Counter[str] = Counter()
    role_stats: defaultdict[str, dict[str, int]] = defaultdict(new_role_stats)

    missing_output.parent.mkdir(parents=True, exist_ok=True)
    query = """
        SELECT u.unit_key, u.question_id, u.root_question_id, u.input_role,
               u.knw_labels, c.candidate_labels
        FROM units AS u
        JOIN candidates AS c ON c.unit_key = u.unit_key
        ORDER BY u.rowid
    """
    with missing_output.open("w", encoding="utf-8", newline="\n") as output:
        for row in connection.execute(query):
            unit_key, question_id, root_question_id, input_role, gold_json, candidate_json = row
            gold_labels = json.loads(gold_json)
            candidate_labels = json.loads(candidate_json)
            candidate_set = set(candidate_labels)
            known_gold = [label for label in gold_labels if label in allowed_paths]
            unmapped = [label for label in gold_labels if label not in allowed_paths]
            missing = [label for label in known_gold if label not in candidate_set]

            evaluated_units += 1
            role = role_stats[input_role]
            role["evaluated_units"] += 1
            candidate_count = len(candidate_labels)
            candidate_count_sum += candidate_count
            candidate_count_min = (
                candidate_count
                if candidate_count_min is None
                else min(candidate_count_min, candidate_count)
            )
            candidate_count_max = max(candidate_count_max, candidate_count)

            if not gold_labels:
                units_with_no_knw_labels += 1
            if unmapped:
                units_with_unmapped_labels += 1
                unmapped_counts.update(unmapped)
            if known_gold:
                units_with_known_gold += 1
                role["units_with_known_gold"] += 1
                known_gold_count += len(known_gold)
                recalled = len(known_gold) - len(missing)
                recalled_gold_count += recalled
                role["known_gold_labels"] += len(known_gold)
                role["recalled_gold_labels"] += recalled
                if not missing:
                    fully_covered_units += 1
                    role["fully_covered_units"] += 1
                else:
                    missing_counts.update(missing)

            if missing or unmapped:
                detail = {
                    "unit_key": unit_key,
                    "question_id": question_id,
                    "root_question_id": root_question_id,
                    "input_role": input_role,
                    "knw_labels": gold_labels,
                    "candidate_labels": candidate_labels,
                    "missing_labels": missing,
                    "unmapped_knw_labels": unmapped,
                }
                output.write(json.dumps(detail, ensure_ascii=False) + "\n")

    role_summary: dict[str, dict[str, int | float | None]] = {}
    for role, stats in sorted(role_stats.items()):
        role_summary[role] = {
            **stats,
            "full_coverage_rate": safe_rate(
                stats["fully_covered_units"], stats["units_with_known_gold"]
            ),
            "label_recall": safe_rate(
                stats["recalled_gold_labels"], stats["known_gold_labels"]
            ),
        }

    return {
        "evaluated_units": evaluated_units,
        "units_with_no_knw_labels": units_with_no_knw_labels,
        "units_with_unmapped_knw_labels": units_with_unmapped_labels,
        "units_with_known_gold": units_with_known_gold,
        "fully_covered_units": fully_covered_units,
        "full_coverage_rate": safe_rate(fully_covered_units, units_with_known_gold),
        "known_gold_labels": known_gold_count,
        "recalled_gold_labels": recalled_gold_count,
        "label_recall": safe_rate(recalled_gold_count, known_gold_count),
        "candidate_count": {
            "average": safe_rate(candidate_count_sum, evaluated_units),
            "minimum": candidate_count_min,
            "maximum": candidate_count_max if evaluated_units else None,
        },
        "missing_label_counts": dict(missing_counts.most_common()),
        "unmapped_knw_label_counts": dict(unmapped_counts.most_common()),
        "by_input_role": role_summary,
    }


def evaluate_grouped_records(
    connection: sqlite3.Connection,
    allowed_paths: set[str],
    missing_output: Path,
) -> dict[str, Any]:
    totals = Counter(
        {
            "input_groups": 0,
            "incomplete_groups": 0,
            "evaluated_groups": 0,
            "groups_with_no_knw_labels": 0,
            "groups_with_unmapped_knw_labels": 0,
            "groups_with_known_gold": 0,
            "fully_covered_groups": 0,
            "known_gold_labels": 0,
            "recalled_gold_labels": 0,
        }
    )
    missing_counts: Counter[str] = Counter()
    unmapped_counts: Counter[str] = Counter()
    type_stats: defaultdict[str, Counter[str]] = defaultdict(Counter)
    candidate_sizes: list[int] = []

    query = """
        SELECT u.root_question_id, u.unit_key, u.knw_labels, c.candidate_labels
        FROM units AS u
        LEFT JOIN candidates AS c ON c.unit_key = u.unit_key
        ORDER BY u.root_question_id, u.rowid
    """

    def evaluate_group(
        root_id: str | None,
        rows: list[tuple[str, str, str | None]],
        output,
    ) -> None:
        if not rows:
            return
        totals["input_groups"] += 1
        if any(candidate_json is None for _, _, candidate_json in rows):
            totals["incomplete_groups"] += 1
            return

        group_type = "ordinary_question" if len(rows) == 1 else "big_question"
        gold_labels: list[str] = []
        candidate_labels: list[str] = []
        for _, gold_json, candidate_json in rows:
            gold_labels.extend(json.loads(gold_json))
            candidate_labels.extend(json.loads(candidate_json))
        gold_labels = list(dict.fromkeys(gold_labels))
        candidate_labels = list(dict.fromkeys(candidate_labels))
        candidate_set = set(candidate_labels)
        known_gold = [label for label in gold_labels if label in allowed_paths]
        unmapped = [label for label in gold_labels if label not in allowed_paths]
        missing = [label for label in known_gold if label not in candidate_set]

        totals["evaluated_groups"] += 1
        type_stats[group_type]["evaluated_groups"] += 1
        candidate_sizes.append(len(candidate_labels))
        if not gold_labels:
            totals["groups_with_no_knw_labels"] += 1
        if unmapped:
            totals["groups_with_unmapped_knw_labels"] += 1
            unmapped_counts.update(unmapped)
        if known_gold:
            totals["groups_with_known_gold"] += 1
            totals["known_gold_labels"] += len(known_gold)
            recalled = len(known_gold) - len(missing)
            totals["recalled_gold_labels"] += recalled
            stats = type_stats[group_type]
            stats["groups_with_known_gold"] += 1
            stats["known_gold_labels"] += len(known_gold)
            stats["recalled_gold_labels"] += recalled
            if not missing:
                totals["fully_covered_groups"] += 1
                stats["fully_covered_groups"] += 1
            else:
                missing_counts.update(missing)
        if missing or unmapped:
            output.write(
                json.dumps(
                    {
                        "root_question_id": root_id,
                        "unit_keys": [row[0] for row in rows],
                        "group_type": group_type,
                        "knw_labels": gold_labels,
                        "candidate_labels": candidate_labels,
                        "missing_labels": missing,
                        "unmapped_knw_labels": unmapped,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    missing_output.parent.mkdir(parents=True, exist_ok=True)
    with missing_output.open("w", encoding="utf-8", newline="\n") as output:
        current_root: str | None = None
        group_rows: list[tuple[str, str, str | None]] = []
        for root_id, unit_key, gold_json, candidate_json in connection.execute(query):
            if current_root is not None and root_id != current_root:
                evaluate_group(current_root, group_rows, output)
                group_rows = []
            current_root = root_id
            group_rows.append((unit_key, gold_json, candidate_json))
        evaluate_group(current_root, group_rows, output)

    by_group_type: dict[str, dict[str, int | float | None]] = {}
    for group_type, stats in sorted(type_stats.items()):
        by_group_type[group_type] = {
            "evaluated_groups": stats["evaluated_groups"],
            "groups_with_known_gold": stats["groups_with_known_gold"],
            "fully_covered_groups": stats["fully_covered_groups"],
            "known_gold_labels": stats["known_gold_labels"],
            "recalled_gold_labels": stats["recalled_gold_labels"],
            "full_coverage_rate": safe_rate(
                stats["fully_covered_groups"], stats["groups_with_known_gold"]
            ),
            "label_recall": safe_rate(
                stats["recalled_gold_labels"], stats["known_gold_labels"]
            ),
        }
    return {
        **totals,
        "full_coverage_rate": safe_rate(
            totals["fully_covered_groups"], totals["groups_with_known_gold"]
        ),
        "label_recall": safe_rate(
            totals["recalled_gold_labels"], totals["known_gold_labels"]
        ),
        "candidate_count": {
            "average": safe_rate(sum(candidate_sizes), len(candidate_sizes)),
            "minimum": min(candidate_sizes) if candidate_sizes else None,
            "maximum": max(candidate_sizes) if candidate_sizes else None,
        },
        "missing_label_counts": dict(missing_counts.most_common()),
        "unmapped_knw_label_counts": dict(unmapped_counts.most_common()),
        "by_group_type": by_group_type,
    }


def evaluate_files(
    input_path: Path,
    candidates_path: Path,
    catalog_path: Path,
    summary_output: Path,
    missing_output: Path,
    group_by_root: bool = False,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    _, allowed_paths = load_catalog(catalog_path)
    with tempfile.TemporaryDirectory(prefix="call1-evaluation-") as temp_dir:
        database_path = Path(temp_dir) / "evaluation.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                """CREATE TABLE units (
                    unit_key TEXT PRIMARY KEY,
                    question_id TEXT,
                    root_question_id TEXT,
                    input_role TEXT,
                    knw_labels TEXT
                )"""
            )
            connection.execute(
                """CREATE TABLE candidates (
                    unit_key TEXT PRIMARY KEY,
                    candidate_labels TEXT
                )"""
            )
            connection.execute(
                """CREATE TABLE pre_candidates (
                    unit_key TEXT PRIMARY KEY,
                    candidate_labels TEXT
                )"""
            )
            eligible_units, skipped_empty_stem = load_units_to_db(
                connection, input_path
            )
            completed_candidate_units = load_candidates_to_db(
                connection, candidates_path, allowed_paths
            )
            pre_candidate_units = None
            if trace_path is not None:
                pre_candidate_units = load_pre_candidates_to_db(
                    connection, trace_path, allowed_paths
                )
            stale_candidates = connection.execute(
                """SELECT COUNT(*) FROM candidates AS c
                   LEFT JOIN units AS u ON u.unit_key = c.unit_key
                   WHERE u.unit_key IS NULL"""
            ).fetchone()[0]
            if stale_candidates:
                raise ValueError(
                    f"候选文件中有{stale_candidates}条记录不属于当前题目文件"
                )
            global_pre_recall = None
            if trace_path is not None:
                global_pre_recall = evaluate_global_pre_recall(
                    connection, allowed_paths
                )
            if group_by_root:
                summary = evaluate_grouped_records(
                    connection, allowed_paths, missing_output
                )
            else:
                summary = evaluate_joined_records(
                    connection, allowed_paths, missing_output
                )
        finally:
            connection.close()

    summary = {
        "eligible_input_units": eligible_units,
        "skipped_empty_stem": skipped_empty_stem,
        "completed_candidate_units": completed_candidate_units,
        "units_without_candidate_result": eligible_units
        - completed_candidate_units,
        "evaluation_grain": "question_group" if group_by_root else "tagging_unit",
        **summary,
    }
    if trace_path is not None:
        summary["global_pre_candidate_units"] = pre_candidate_units
        summary["global_pre_recall"] = global_pre_recall
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate call-one candidate recall with knw_labels."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--missing-output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--group-by-root", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate_files(
        args.input,
        args.candidates,
        args.catalog,
        args.summary_output,
        args.missing_output,
        args.group_by_root,
        args.trace,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
