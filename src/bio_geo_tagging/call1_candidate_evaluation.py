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
        if not as_text(unit.get("stem")):
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
                as_text(unit.get("input_role")) or "root",
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


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


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


def evaluate_files(
    input_path: Path,
    candidates_path: Path,
    catalog_path: Path,
    summary_output: Path,
    missing_output: Path,
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
            eligible_units, skipped_empty_stem = load_units_to_db(
                connection, input_path
            )
            completed_candidate_units = load_candidates_to_db(
                connection, candidates_path, allowed_paths
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
        - summary["evaluated_units"],
        **summary,
    }
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate_files(
        args.input,
        args.candidates,
        args.catalog,
        args.summary_output,
        args.missing_output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
