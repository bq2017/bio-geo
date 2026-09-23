import json

from bio_geo_tagging import label_definition_impact as impact


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def score_record(question_id, label_id, label, score):
    return {
        "question_id": question_id,
        "label_id": label_id,
        "knw_label": label,
        "status": "completed",
        "judgement": "scored",
        "score": score,
        "match": score >= 0.70,
        "reason": f"分数{score}",
    }


def test_compare_builds_pairs_statistics_and_review_evidence(tmp_path):
    definition_file = tmp_path / "definition.jsonl"
    name_file = tmp_path / "name.jsonl"
    questions_file = tmp_path / "questions.jsonl"
    definitions_file = tmp_path / "definitions.jsonl"
    pairs_output = tmp_path / "pairs.jsonl"
    statistics_output = tmp_path / "statistics.json"
    review_output = tmp_path / "review.jsonl"
    report_output = tmp_path / "report.md"
    label = "知识点@自然地理@测试标签"
    definition_scores = [0.10, 0.10, 0.20, 0.20, 0.30, 0.95]
    name_scores = [0.95] * 6
    write_jsonl(
        definition_file,
        [score_record(f"q{i}", "l1", label, score) for i, score in enumerate(definition_scores, 1)],
    )
    write_jsonl(
        name_file,
        [score_record(f"q{i}", "l1", label, score) for i, score in enumerate(name_scores, 1)],
    )
    write_jsonl(
        questions_file,
        [
            {
                "question_id": f"q{i}",
                "stem": f"题干{i}",
                "options": "A. 选项",
                "analysis": "解析",
                "sub_questions": [],
            }
            for i in range(1, 7)
        ],
    )
    write_jsonl(
        definitions_file,
        [
            {
                "label_id": "l1",
                "knw_label": label,
                "existing_interpretation": {"definition": "原释义"},
            }
        ],
    )

    result = impact.compare(
        str(definition_file),
        str(name_file),
        str(questions_file),
        str(definitions_file),
        str(pairs_output),
        str(statistics_output),
        str(review_output),
        str(report_output),
    )

    assert result == {
        "definition_records": 6,
        "name_records": 6,
        "comparable_records": 6,
        "covered_labels": 1,
        "review_candidates": 1,
        "missing_evidence_questions": 0,
    }
    pairs = [json.loads(line) for line in pairs_output.read_text(encoding="utf-8").splitlines()]
    assert [row["comparison_type"] for row in pairs].count("definition_suppressed") == 5
    assert pairs[0]["score_delta"] == -0.85
    statistics = json.loads(statistics_output.read_text(encoding="utf-8"))
    label_statistics = statistics["labels"][0]
    assert label_statistics["definition_suppressed_count"] == 5
    assert label_statistics["review_candidate"] is True
    review = json.loads(review_output.read_text(encoding="utf-8"))
    assert review["existing_interpretation"]["definition"] == "原释义"
    assert len(review["definition_suppressed_examples"]) == 5
    assert review["definition_suppressed_examples"][0]["question"]["stem"] == "题干1"
    assert "需要进一步诊断的标签" in report_output.read_text(encoding="utf-8")


def test_comparison_type_covers_four_directions():
    assert impact.comparison_type(0.90, 0.90) == "both_match"
    assert impact.comparison_type(0.20, 0.90) == "definition_suppressed"
    assert impact.comparison_type(0.90, 0.20) == "definition_expanded"
    assert impact.comparison_type(0.20, 0.20) == "both_mismatch"
