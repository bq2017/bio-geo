import json
from collections import Counter

from bio_geo_tagging import label_match_analysis as analysis


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_score_grade_uses_history_thresholds():
    assert analysis.score_grade(0.95) == "A"
    assert analysis.score_grade(0.80) == "A"
    assert analysis.score_grade(0.79) == "B"
    assert analysis.score_grade(0.70) == "B"
    assert analysis.score_grade(0.69) == "C"
    assert analysis.score_grade(0.40) == "C"
    assert analysis.score_grade(0.39) == "D"


def test_analyze_builds_anomaly_examples_and_report(tmp_path):
    matches = tmp_path / "matches.jsonl"
    questions = tmp_path / "questions.jsonl"
    definitions = tmp_path / "definitions.jsonl"
    statistics = tmp_path / "statistics.json"
    anomalies = tmp_path / "anomalies.jsonl"
    report = tmp_path / "report.md"
    label = "知识点@自然地理@行星地球"
    scores = [0.0, 0.1, 0.3, 0.85, 0.95]
    write_jsonl(
        matches,
        [
            {
                "question_id": f"q{index}",
                "knw_label": label,
                "label_id": "label-1",
                "judgement": "scored",
                "score": score,
                "match": score >= 0.70,
                "reason": f"理由{index}",
                "status": "completed",
            }
            for index, score in enumerate(scores, 1)
        ],
    )
    write_jsonl(
        questions,
        [
            {
                "question_id": f"q{index}",
                "stem": f"题干{index}",
                "options": "A. 选项",
                "analysis": "解析",
                "sub_questions": [
                    {"question_id": f"q{index}-1", "stem": "小题", "options": "", "analysis": ""}
                ],
            }
            for index in range(1, 6)
        ],
    )
    write_jsonl(
        definitions,
        [
            {
                "label_id": "label-1",
                "knw_label": label,
                "existing_interpretation": {
                    "definition": "原定义",
                    "keywords": "关键词",
                    "exam_methods": "考查方式",
                    "distinction": "区分",
                },
            }
        ],
    )

    result = analysis.analyze(
        str(matches),
        str(questions),
        str(definitions),
        str(statistics),
        str(anomalies),
        str(report),
    )

    assert result == {
        "total_records": 5,
        "covered_labels": 1,
        "anomaly_labels": 1,
        "missing_question_examples": 0,
    }
    stats = json.loads(statistics.read_text(encoding="utf-8"))
    assert stats["grade_counts"] == {"A": 2, "B": 0, "C": 0, "D": 3}
    anomaly = json.loads(anomalies.read_text(encoding="utf-8"))
    assert [example["question_id"] for example in anomaly["low_score_examples"]] == ["q1", "q2", "q3"]
    assert len(anomaly["high_score_examples"]) == 2
    report_text = report.read_text(encoding="utf-8")
    assert "40%阶段性快照" in report_text
    assert "原定义" in report_text
    assert "题干1" in report_text
    assert "A/B级对照题" in report_text


def test_all_d_three_samples_is_anomaly():
    grades = Counter({"D": 3})
    assert analysis.anomaly_reason(3, grades, 5, 3, 0.30, 3)
