"""Sample complete question groups for candidate-recall evaluation."""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from .call1_candidate_evaluation import read_jsonl
from .call1_candidate_retrieval import as_text, load_catalog


def root_question_id(unit: dict[str, Any]) -> str:
    question_id = as_text(unit.get("question_id"))
    if not question_id:
        raise ValueError("题目缺少question_id")
    return as_text(
        unit.get("root_question_id") or unit.get("parent_id") or question_id
    )


def update_group(
    connection: sqlite3.Connection,
    root_id: str,
    order_index: int,
    is_root: bool,
    stem_is_empty: bool,
    has_gold: bool,
    has_unmapped: bool,
) -> None:
    connection.execute(
        """
        INSERT INTO groups (
            root_question_id, first_order, unit_count, root_count,
            has_empty_stem, has_gold, has_unmapped
        ) VALUES (?, ?, 1, ?, ?, ?, ?)
        ON CONFLICT(root_question_id) DO UPDATE SET
            unit_count = unit_count + 1,
            root_count = root_count + excluded.root_count,
            has_empty_stem = MAX(has_empty_stem, excluded.has_empty_stem),
            has_gold = MAX(has_gold, excluded.has_gold),
            has_unmapped = MAX(has_unmapped, excluded.has_unmapped)
        """,
        (
            root_id,
            order_index,
            int(is_root),
            int(stem_is_empty),
            int(has_gold),
            int(has_unmapped),
        ),
    )


def profile_groups(
    connection: sqlite3.Connection,
    input_path: Path,
    allowed_paths: set[str],
) -> int:
    total_units = 0
    for line_number, unit in read_jsonl(input_path):
        root_id = root_question_id(unit)
        labels = unit.get("knw_labels")
        if labels is None:
            labels = []
        if not isinstance(labels, list) or not all(
            isinstance(label, str) for label in labels
        ):
            raise ValueError(f"题目文件第{line_number}行knw_labels必须是字符串数组")
        labels = {label.strip() for label in labels if label.strip()}
        input_role = as_text(unit.get("input_role")) or "root"
        update_group(
            connection,
            root_id,
            total_units,
            input_role == "root",
            not as_text(unit.get("stem")),
            bool(labels),
            any(label not in allowed_paths for label in labels),
        )
        total_units += 1
        if total_units % 10_000 == 0:
            connection.commit()
    connection.commit()
    return total_units


def reservoir_sample_group_ids(
    connection: sqlite3.Connection, group_count: int, seed: int
) -> tuple[set[str], int]:
    rng = random.Random(seed)
    sample: list[str] = []
    eligible_count = 0
    query = """
        SELECT root_question_id
        FROM groups
        WHERE root_count = 1
          AND has_empty_stem = 0
          AND has_gold = 1
          AND has_unmapped = 0
        ORDER BY first_order
    """
    for (root_id,) in connection.execute(query):
        eligible_count += 1
        if len(sample) < group_count:
            sample.append(root_id)
            continue
        replacement = rng.randrange(eligible_count)
        if replacement < group_count:
            sample[replacement] = root_id
    if eligible_count < group_count:
        raise ValueError(
            f"符合条件的完整题组只有{eligible_count}个，不足{group_count}个"
        )
    return set(sample), eligible_count


def write_sampled_units(
    input_path: Path, output_path: Path, sampled_ids: set[str]
) -> tuple[int, int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_units = 0
    root_units = 0
    subquestion_units = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for _, unit in read_jsonl(input_path):
            if root_question_id(unit) not in sampled_ids:
                continue
            output.write(json.dumps(unit, ensure_ascii=False) + "\n")
            sampled_units += 1
            if (as_text(unit.get("input_role")) or "root") == "root":
                root_units += 1
            else:
                subquestion_units += 1
    return sampled_units, root_units, subquestion_units


def scalar(connection: sqlite3.Connection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def sample_groups(
    input_path: Path,
    catalog_path: Path,
    output_path: Path,
    summary_output: Path,
    group_count: int,
    seed: int,
) -> dict[str, Any]:
    _, allowed_paths = load_catalog(catalog_path)
    with tempfile.TemporaryDirectory(prefix="tagging-sample-") as temp_dir:
        database_path = Path(temp_dir) / "groups.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                """
                CREATE TABLE groups (
                    root_question_id TEXT PRIMARY KEY,
                    first_order INTEGER,
                    unit_count INTEGER,
                    root_count INTEGER,
                    has_empty_stem INTEGER,
                    has_gold INTEGER,
                    has_unmapped INTEGER
                )
                """
            )
            total_units = profile_groups(connection, input_path, allowed_paths)
            sampled_ids, eligible_groups = reservoir_sample_group_ids(
                connection, group_count, seed
            )
            total_groups = scalar(connection, "SELECT COUNT(*) FROM groups")
            excluded_empty_stem_groups = scalar(
                connection, "SELECT COUNT(*) FROM groups WHERE has_empty_stem = 1"
            )
            excluded_no_gold_groups = scalar(
                connection, "SELECT COUNT(*) FROM groups WHERE has_gold = 0"
            )
            excluded_unmapped_groups = scalar(
                connection, "SELECT COUNT(*) FROM groups WHERE has_unmapped = 1"
            )
            excluded_invalid_structure_groups = scalar(
                connection, "SELECT COUNT(*) FROM groups WHERE root_count != 1"
            )
            sampled_big_questions = sum(
                1
                for root_id, unit_count in connection.execute(
                    "SELECT root_question_id, unit_count FROM groups WHERE unit_count > 1"
                )
                if root_id in sampled_ids
            )
        finally:
            connection.close()

    sampled_units, root_units, subquestion_units = write_sampled_units(
        input_path, output_path, sampled_ids
    )
    summary = {
        "seed": seed,
        "requested_groups": group_count,
        "sampled_groups": len(sampled_ids),
        "sampled_ordinary_questions": len(sampled_ids) - sampled_big_questions,
        "sampled_big_questions": sampled_big_questions,
        "sampled_units": sampled_units,
        "sampled_root_units": root_units,
        "sampled_subquestion_units": subquestion_units,
        "source_units": total_units,
        "source_groups": total_groups,
        "eligible_groups": eligible_groups,
        "excluded_empty_stem_groups": excluded_empty_stem_groups,
        "excluded_no_gold_groups": excluded_no_gold_groups,
        "excluded_unmapped_groups": excluded_unmapped_groups,
        "excluded_invalid_structure_groups": excluded_invalid_structure_groups,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample complete question groups for tagging evaluation."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.groups <= 0:
        raise SystemExit("groups必须大于0")
    summary = sample_groups(
        args.input,
        args.catalog,
        args.output,
        args.summary_output,
        args.groups,
        args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
