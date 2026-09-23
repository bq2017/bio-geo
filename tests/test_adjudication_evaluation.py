import json

import pytest

from bio_geo_tagging.adjudication_evaluation import evaluate_adjudication


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_evaluate_adjudication_reports_multilabel_metrics_and_usable_subset(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_jsonl(
        run_dir / "evidence.jsonl",
        [
            {"root_question_id": "q1"},
            {"root_question_id": "q1"},
            {"root_question_id": "q2"},
            {"root_question_id": "q3"},
        ],
    )
    write_jsonl(
        run_dir / "question_predictions.jsonl",
        [
            {
                "root_question_id": "q1",
                "selected_labels": [
                    {
                        "label_path": "知识点@甲",
                        "evidence_by_unit": [{"evidence": "甲证据"}],
                    },
                    {
                        "label_path": "知识点@多选",
                        "evidence_by_unit": [{"evidence": "多选证据"}],
                    },
                ],
                "usable_for_training": True,
                "needs_review": False,
            },
            {
                "root_question_id": "q2",
                "selected_labels": [
                    {
                        "label_path": "知识点@乙",
                        "evidence_by_unit": [{"evidence": "乙证据"}],
                    }
                ],
                "usable_for_training": False,
                "needs_review": True,
            },
        ],
    )
    candidates = tmp_path / "candidates.jsonl"
    write_jsonl(
        candidates,
        [
            {"question_id": "q1", "knw_labels": ["知识点@甲", "知识点@乙"]},
            {"question_id": "q2", "knw_labels": ["知识点@乙", "知识点@旧"]},
            {"question_id": "q3", "knw_labels": ["知识点@甲"]},
        ],
    )
    labels = tmp_path / "labels.jsonl"
    write_jsonl(
        labels,
        [
            {"label_path": "知识点@甲"},
            {"label_path": "知识点@乙"},
            {"label_path": "知识点@多选"},
        ],
    )

    result = evaluate_adjudication(run_dir, candidates, labels)

    assert result["attempted_questions"] == 3
    assert result["question_prediction_rows"] == 2
    assert result["completed_question_predictions"] == 2
    assert result["incomplete_question_predictions"] == 0
    assert result["attempted_questions_without_prediction"] == 1
    assert result["attempted_questions_without_complete_prediction"] == 1
    assert result["unmapped_gold_label_counts"] == {"知识点@旧": 1}
    assert result["report_kind"] == "automatic_adjudication_diagnostics"
    assert result["semantic_addition_accuracy_available"] is False
    assert result["addition_volume"]["total_ds_added_labels"] == 1
    assert result["addition_volume"]["questions_with_additions"] == 1
    assert result["addition_volume"]["addition_category_counts"] == {
        "ordinary": 1,
        "regional": 0,
        "comprehensive": 0,
    }
    assert result["evidence_validation"]["labels_without_evidence"] == 0
    assert result["stability"]["available"] is False
    assert result["legacy_label_agreement"]["matched_legacy_labels"] == 2
    assert result["legacy_label_agreement"]["ds_only_labels"] == 1
    assert result["legacy_label_agreement"]["legacy_labels_not_selected_by_ds"] == 2
    overall = result["all_completed_with_known_gold"]
    assert overall["questions"] == 2
    assert overall["true_positive_labels"] == 2
    assert overall["false_positive_labels"] == 1
    assert overall["false_negative_labels"] == 1
    assert overall["micro_precision"] == pytest.approx(2 / 3, abs=1e-6)
    assert overall["micro_recall"] == pytest.approx(2 / 3, abs=1e-6)
    assert overall["micro_f1"] == pytest.approx(2 / 3, abs=1e-6)
    assert overall["exact_match_rate"] == 0.5
    usable = result["usable_for_training_with_known_gold"]
    assert usable["questions"] == 1
    assert usable["micro_precision"] == 0.5
    assert usable["micro_recall"] == 0.5

    details = [
        json.loads(line)
        for line in (run_dir / "evaluation_details.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert details[0]["false_positive_labels"] == ["知识点@多选"]
    assert details[0]["false_negative_labels"] == ["知识点@乙"]
    assert details[0]["legacy_labels"] == ["知识点@乙", "知识点@甲"]
    assert details[0]["ds_added_labels"] == ["知识点@多选"]
    assert details[0]["final_labels"] == ["知识点@乙", "知识点@多选", "知识点@甲"]
    final_rows = [
        json.loads(line)
        for line in (run_dir / "final_labels.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert final_rows[0]["legacy_labels"] == ["知识点@乙", "知识点@甲"]
    assert final_rows[0]["ds_added_labels"] == ["知识点@多选"]
    assert final_rows[0]["final_labels"] == ["知识点@乙", "知识点@多选", "知识点@甲"]


def test_evaluate_adjudication_skips_empty_and_only_unmapped_gold(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_jsonl(run_dir / "evidence.jsonl", [{"root_question_id": "q1"}])
    write_jsonl(
        run_dir / "question_predictions.jsonl",
        [
            {
                "root_question_id": "q1",
                "selected_labels": [],
                "usable_for_training": False,
                "needs_review": True,
            },
            {
                "root_question_id": "q2",
                "selected_labels": [],
                "usable_for_training": False,
                "needs_review": True,
            },
        ],
    )
    candidates = tmp_path / "candidates.jsonl"
    write_jsonl(
        candidates,
        [
            {"question_id": "q1", "knw_labels": []},
            {"question_id": "q2", "knw_labels": ["知识点@旧"]},
        ],
    )
    labels = tmp_path / "labels.jsonl"
    write_jsonl(labels, [{"knw_label": "知识点@甲"}])

    result = evaluate_adjudication(run_dir, candidates, labels)

    assert result["completed_predictions_without_gold"] == 1
    assert result["completed_predictions_with_only_unmapped_gold"] == 1
    assert result["all_completed_with_known_gold"]["questions"] == 2
    assert result["all_completed_with_known_gold"]["micro_f1"] is None
    assert result["addition_volume"]["total_ds_added_labels"] == 0


def test_evaluate_adjudication_excludes_incomplete_questions_from_metrics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_jsonl(run_dir / "evidence.jsonl", [{"root_question_id": "q1"}])
    write_jsonl(
        run_dir / "question_predictions.jsonl",
        [
            {
                "root_question_id": "q1",
                "selected_labels": [{"label_path": "知识点@甲"}],
                "components_complete": False,
                "usable_for_training": False,
                "needs_review": True,
            }
        ],
    )
    candidates = tmp_path / "candidates.jsonl"
    write_jsonl(candidates, [{"question_id": "q1", "knw_labels": ["知识点@甲"]}])
    labels = tmp_path / "labels.jsonl"
    write_jsonl(labels, [{"label_path": "知识点@甲"}])

    result = evaluate_adjudication(run_dir, candidates, labels)

    assert result["question_prediction_rows"] == 1
    assert result["completed_question_predictions"] == 0
    assert result["incomplete_question_predictions"] == 1
    assert result["attempted_questions_without_complete_prediction"] == 1
    assert result["all_completed_with_known_gold"]["questions"] == 0
