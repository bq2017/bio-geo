import json
from types import SimpleNamespace

import pytest

from bio_geo_tagging import label_definition_diagnosis as diagnosis


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def review_record():
    return {
        "label_id": "label-1",
        "knw_label": "知识点@行星地球",
        "evaluated_count": 2,
        "grade_counts": {"A": 1, "B": 0, "C": 0, "D": 1},
        "match_rate": 0.5,
        "existing_interpretation": {"definition": "地球生命条件"},
        "adjacent_labels": [],
        "high_score_examples": [
            {
                "question_id": "high-1",
                "score": 0.85,
                "grade": "A",
                "model_reason": "高分理由",
                "question": {"stem": "高分题", "options": "A. 选项", "analysis": "解析"},
            }
        ],
        "low_score_examples": [
            {
                "question_id": "low-1",
                "score": 0.20,
                "grade": "D",
                "model_reason": "低分理由",
                "question": {"stem": "低分题", "options": "A. 选项", "analysis": "解析"},
            }
        ],
    }


class FakeClient:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        answer = {
            "definition_status": "too_narrow",
            "definition_fixable": True,
            "analysis": "低分题本应属于该标签，但原释义遗漏比较其他行星。",
            "high_score_valid_ids": ["high-1"],
            "high_score_false_positive_ids": [],
            "low_score_definition_issue_ids": ["low-1"],
            "unrelated_mislabel_ids": [],
            "model_misjudgement_ids": [],
            "teacher_review_required": True,
            "revision_direction": "补充通过其他行星比较地球普通性与特殊性的范围。",
        }
        return [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=json.dumps(answer, ensure_ascii=False)
                        )
                    )
                ]
            )
        ]


def test_diagnose_writes_valid_definition_result(tmp_path, monkeypatch):
    source = tmp_path / "reviews.jsonl"
    output = tmp_path / "diagnosis.jsonl"
    log = tmp_path / "diagnosis.log"
    write_jsonl(source, [review_record()])
    monkeypatch.setattr(diagnosis, "OpenAI", FakeClient)

    summary = diagnosis.diagnose(
        str(source),
        str(output),
        str(log),
        "http://example/v1",
        "DeepSeek-V4-Flash",
    )

    assert summary == {"attempted": 1, "completed": 1, "errors": 0}
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["definition_status"] == "too_narrow"
    assert result["definition_fixable"] is True
    assert result["low_score_definition_issue_ids"] == ["low-1"]
    assert result["definition_issue_evidence_questions"] == [
        {
            "question_id": "low-1",
            "score": 0.20,
            "grade": "D",
            "model_reason": "低分理由",
            "question": {"stem": "低分题", "options": "A. 选项", "analysis": "解析"},
            "evidence_type": "low_score_definition_issue",
        }
    ]
    log_text = log.read_text(encoding="utf-8")
    assert "run_started" in log_text
    assert "definition_status=too_narrow" in log_text
    assert "run_finished attempted=1 completed=1 errors=0" in log_text


def test_validate_rejects_question_id_outside_samples():
    answer = {
        "definition_status": "too_broad",
        "definition_fixable": True,
        "analysis": "释义太宽。",
        "high_score_valid_ids": ["high-1"],
        "high_score_false_positive_ids": ["not-in-input"],
        "low_score_definition_issue_ids": [],
        "unrelated_mislabel_ids": [],
        "model_misjudgement_ids": [],
        "teacher_review_required": True,
        "revision_direction": "收窄范围。",
    }
    with pytest.raises(ValueError, match="输入样本之外"):
        diagnosis.validate_diagnosis(answer, review_record())


def test_validate_requires_every_low_sample_to_be_classified():
    answer = {
        "definition_status": "adequate",
        "definition_fixable": False,
        "analysis": "释义充分。",
        "high_score_valid_ids": ["high-1"],
        "high_score_false_positive_ids": [],
        "low_score_definition_issue_ids": [],
        "unrelated_mislabel_ids": [],
        "model_misjudgement_ids": [],
        "teacher_review_required": False,
        "revision_direction": "",
    }
    with pytest.raises(ValueError, match="每道低匹配样本"):
        diagnosis.validate_diagnosis(answer, review_record())


def test_validate_requires_every_high_sample_to_be_classified():
    answer = {
        "definition_status": "adequate",
        "definition_fixable": False,
        "analysis": "释义充分。",
        "high_score_valid_ids": [],
        "high_score_false_positive_ids": [],
        "low_score_definition_issue_ids": [],
        "unrelated_mislabel_ids": ["low-1"],
        "model_misjudgement_ids": [],
        "teacher_review_required": False,
        "revision_direction": "",
    }
    with pytest.raises(ValueError, match="每道高匹配样本"):
        diagnosis.validate_diagnosis(answer, review_record())


def test_prompt_defines_root_question_label_union_rule():
    assert "大题标签是公共题干以及所有小题所考查知识点的并集" in diagnosis.SYSTEM_PROMPT
    assert "任意一道小题直接考查当前知识点" in diagnosis.SYSTEM_PROMPT
    assert "应归为model_misjudgement" in diagnosis.SYSTEM_PROMPT


def test_insufficient_evidence_still_requires_complete_classification():
    answer = {
        "definition_status": "insufficient_evidence",
        "definition_fixable": False,
        "analysis": "现有样本不足以判断释义是否存在问题。",
        "high_score_valid_ids": [],
        "high_score_false_positive_ids": [],
        "low_score_definition_issue_ids": [],
        "unrelated_mislabel_ids": ["low-1"],
        "model_misjudgement_ids": [],
        "teacher_review_required": False,
        "revision_direction": "",
    }
    with pytest.raises(ValueError, match="每道高匹配样本"):
        diagnosis.validate_diagnosis(answer, review_record())


def test_retry_errors_replaces_error_without_duplicating_completed_rows(tmp_path, monkeypatch):
    source = tmp_path / "reviews.jsonl"
    output = tmp_path / "diagnosis.jsonl"
    log = tmp_path / "diagnosis.log"
    completed_review = review_record()
    completed_review["label_id"] = "label-completed"
    error_review = review_record()
    error_review["label_id"] = "label-error"
    write_jsonl(source, [completed_review, error_review])
    write_jsonl(
        output,
        [
            {
                "label_id": "label-completed",
                "knw_label": "知识点@行星地球",
                "status": "completed",
                "preserved": True,
            },
            {
                "label_id": "label-error",
                "knw_label": "知识点@行星地球",
                "status": "error",
                "error_type": "ValueError",
                "error": "分类不完整",
            }
        ],
    )
    monkeypatch.setattr(diagnosis, "OpenAI", FakeClient)

    summary = diagnosis.diagnose(
        str(source),
        str(output),
        str(log),
        "http://example/v1",
        "DeepSeek-V4-Flash",
        retry_errors=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary == {"attempted": 1, "completed": 1, "errors": 0}
    assert len(rows) == 2
    assert rows[0] == {
        "label_id": "label-completed",
        "knw_label": "知识点@行星地球",
        "status": "completed",
        "preserved": True,
    }
    assert rows[1]["label_id"] == "label-error"
    assert rows[1]["status"] == "completed"
