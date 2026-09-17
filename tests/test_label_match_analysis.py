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
    reviews = tmp_path / "reviews.jsonl"
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
        ]
        + [
            {
                "question_id": "q6",
                "knw_label": "知识点@自然地理@地球仪",
                "label_id": "label-2",
                "judgement": "scored",
                "score": 0.95,
                "match": True,
                "reason": "直接考查",
                "status": "completed",
            }
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
            for index in range(1, 7)
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
            },
            {
                "label_id": "label-2",
                "knw_label": "知识点@自然地理@地球仪",
                "existing_interpretation": {
                    "definition": "地球仪定义",
                    "keywords": "地轴",
                    "exam_methods": "识图",
                    "distinction": "区别于地图",
                },
            },
        ],
    )

    result = analysis.analyze(
        str(matches),
        str(questions),
        str(definitions),
        str(statistics),
        str(anomalies),
        str(report),
        str(reviews),
    )

    assert result == {
        "total_records": 6,
        "covered_labels": 2,
        "review_labels": 2,
        "anomaly_labels": 1,
        "missing_question_examples": 0,
    }
    stats = json.loads(statistics.read_text(encoding="utf-8"))
    assert stats["grade_counts"] == {"A": 3, "B": 0, "C": 0, "D": 3}
    anomaly = json.loads(anomalies.read_text(encoding="utf-8"))
    assert [example["question_id"] for example in anomaly["low_score_examples"]] == ["q3", "q2", "q1"]
    assert len(anomaly["high_score_examples"]) == 2
    review_rows = [json.loads(line) for line in reviews.read_text(encoding="utf-8").splitlines()]
    assert {review["label_id"] for review in review_rows} == {"label-1", "label-2"}
    review = next(review for review in review_rows if review["label_id"] == "label-1")
    assert len(review["low_score_examples"]) == 3
    assert len(review["high_score_examples"]) == 2
    report_text = report.read_text(encoding="utf-8")
    assert "40%阶段性快照" in report_text
    assert "原定义" in report_text
    assert "题干1" in report_text
    assert "A/B级对照题" in report_text


def test_all_d_three_samples_is_anomaly():
    grades = Counter({"D": 3})
    assert analysis.anomaly_reason(3, grades, 5, 3, 0.30, 3)


def test_review_sampling_keeps_boundaries_and_is_deterministic():
    high_records = [
        {"question_id": f"h{index}", "score": score}
        for index, score in enumerate(
            [0.70, 0.71, 0.72, 0.73, 0.74, 0.80, 0.85, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95],
            1,
        )
    ]
    low_records = [
        {"question_id": f"l{index}", "score": score}
        for index, score in enumerate(
            [0.00, 0.05, 0.10, 0.20, 0.30, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.50, 0.60, 0.69],
            1,
        )
    ]

    selected_high = analysis.select_review_records(high_records, 10, "label-1", "high")
    selected_low = analysis.select_review_records(low_records, 10, "label-1", "low")

    assert [record["score"] for record in selected_high[:5]] == [0.70, 0.71, 0.72, 0.73, 0.74]
    assert [record["score"] for record in selected_low[:5]] == [0.69, 0.60, 0.50, 0.40, 0.39]
    assert len({record["question_id"] for record in selected_high}) == 10
    assert len({record["question_id"] for record in selected_low}) == 10
    assert selected_high == analysis.select_review_records(high_records, 10, "label-1", "high")
    assert selected_low == analysis.select_review_records(low_records, 10, "label-1", "low")
