import json

import pytest

from bio_geo_tagging.call1_candidate_evaluation import evaluate_files


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_evaluate_files_uses_known_knw_labels_and_reports_unmapped(tmp_path):
    catalog = tmp_path / "catalog.txt"
    catalog.write_text(
        "\n".join(
            f"知识点@标签{index}｜释义{index}" for index in range(414)
        )
        + "\n",
        encoding="utf-8",
    )
    units = [
        {
            "question_id": "q1",
            "root_question_id": "q1",
            "input_role": "root",
            "stem": "题目一",
            "knw_labels": ["知识点@标签0", "知识点@标签1"],
        },
        {
            "question_id": "q2",
            "root_question_id": "q1",
            "input_role": "subquestion",
            "stem": "题目二",
            "knw_labels": ["知识点@标签2"],
        },
        {
            "question_id": "q3",
            "root_question_id": "q3",
            "input_role": "root",
            "stem": "题目三",
            "knw_labels": ["知识点@旧标签"],
        },
        {
            "question_id": "q4",
            "root_question_id": "q4",
            "input_role": "root",
            "stem": "题目四",
            "knw_labels": [],
        },
        {
            "question_id": "q5",
            "root_question_id": "q5",
            "input_role": "root",
            "stem": "",
            "knw_labels": ["知识点@标签0"],
        },
    ]
    candidates = [
        {
            "unit_key": "q1|q1|root",
            "candidate_labels": ["知识点@标签0", "知识点@标签2"],
            "uncovered_topic": None,
            "status": "completed",
        },
        {
            "unit_key": "q1|q2|subquestion",
            "candidate_labels": ["知识点@标签2"],
            "uncovered_topic": None,
            "status": "completed",
        },
        {
            "unit_key": "q3|q3|root",
            "candidate_labels": [],
            "uncovered_topic": None,
            "status": "completed",
        },
        {
            "unit_key": "q4|q4|root",
            "candidate_labels": ["知识点@标签0"],
            "uncovered_topic": None,
            "status": "completed",
        },
    ]
    input_path = tmp_path / "units.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    summary_path = tmp_path / "summary.json"
    missing_path = tmp_path / "missing.jsonl"
    write_jsonl(input_path, units)
    write_jsonl(candidates_path, candidates)

    summary = evaluate_files(
        input_path, candidates_path, catalog, summary_path, missing_path
    )

    assert summary["eligible_input_units"] == 4
    assert summary["skipped_empty_stem"] == 1
    assert summary["completed_candidate_units"] == 4
    assert summary["units_without_candidate_result"] == 0
    assert summary["units_with_known_gold"] == 2
    assert summary["fully_covered_units"] == 1
    assert summary["full_coverage_rate"] == 0.5
    assert summary["known_gold_labels"] == 3
    assert summary["recalled_gold_labels"] == 2
    assert summary["label_recall"] == pytest.approx(2 / 3, abs=1e-6)
    assert summary["units_with_no_knw_labels"] == 1
    assert summary["units_with_unmapped_knw_labels"] == 1
    assert summary["missing_label_counts"] == {"知识点@标签1": 1}
    assert summary["unmapped_knw_label_counts"] == {"知识点@旧标签": 1}

    details = [json.loads(line) for line in missing_path.read_text(encoding="utf-8").splitlines()]
    assert [detail["unit_key"] for detail in details] == [
        "q1|q1|root",
        "q3|q3|root",
    ]
    assert details[0]["missing_labels"] == ["知识点@标签1"]
    assert details[1]["unmapped_knw_labels"] == ["知识点@旧标签"]
