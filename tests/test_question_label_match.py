import json
from types import SimpleNamespace

import pytest

from bio_geo_tagging import question_label_match as matching


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_export_uses_only_completed_original_definitions(tmp_path):
    source = tmp_path / "comparison.jsonl"
    output = tmp_path / "definitions.jsonl"
    original = dict(definition="土壤形成因素", keywords="成土母质", exam_methods="选择题", distinction="区别于肥力")
    write_jsonl(source, [
        dict(status="error", label_id="1", full_path="知识点->无效"),
        dict(status="completed", label_id="2", full_path="知识点->土壤",
             existing_interpretation=original, generated_interpretation={"definition": "错误的生成释义"}),
    ])

    assert matching.export_definitions(str(source), str(output)) == 1
    assert [row for _, row in matching.read_jsonl(str(output))] == [
        dict(label_id="2", knw_label="知识点@土壤", existing_interpretation=original)
    ]


def test_score_pairs_and_overwrite_without_network(tmp_path, monkeypatch):
    definitions = tmp_path / "definitions.jsonl"
    units = tmp_path / "units.jsonl"
    output = tmp_path / "scores.jsonl"
    write_jsonl(definitions, [
        dict(label_id="2", knw_label="知识点@土壤", existing_interpretation=dict.fromkeys(matching.FIELDS, "释义")),
        dict(label_id="3", knw_label="知识点@降水", existing_interpretation=dict.fromkeys(matching.FIELDS, "释义")),
    ])
    write_jsonl(units, [
        dict(
            parent_id="p",
            question_id="p",
            stem="公共题干",
            knw_labels=["知识点@土壤"],
            sub_questions=[
                dict(
                    parent_id="p",
                    question_id="c",
                    stem="小题",
                    options="",
                    analysis="",
                    knw_labels=["知识点@降水", "知识点@土壤"],
                )
            ],
        ),
    ])

    payloads = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            data = json.loads(kwargs["messages"][1]["content"])
            payloads.append(data)
            answer = {
                "judgement": "scored",
                "score": 0.84,
                "reason": "直接考查",
            }
            return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=json.dumps(answer, ensure_ascii=False)))])]

    monkeypatch.setattr(matching, "OpenAI", FakeClient)
    arguments = (str(units), str(definitions), str(output), "http://example/v1", "DeepSeek-V4-Flash")
    assert matching.score_units(*arguments, workers=2)["completed"] == 1
    rows = [row for _, row in matching.read_jsonl(str(output))]
    assert [(r["question_id"], r["knw_label"], r["score"], r["match"]) for r in rows] == [
        ("p", "知识点@土壤", 0.84, True),
    ]
    assert "input_role" not in rows[0]
    assert "group_model_score" not in rows[0]
    assert len(payloads) == 1
    assert payloads[0]["knowledge_path"] == "知识点@土壤"
    assert payloads[0]["complete_question"]["stem"] == "公共题干"
    assert payloads[0]["complete_question"]["sub_questions"] == [
        {
            "question_id": "c",
            "stem": "小题",
            "options": "",
            "analysis": "",
        }
    ]
    output.write_text("stale output\n", encoding="utf-8")
    assert matching.score_units(*arguments)["completed"] == 1
    assert len(list(matching.read_jsonl(str(output)))) == 1


def test_score_writes_error_details_to_log(tmp_path, monkeypatch):
    definitions = tmp_path / "definitions.jsonl"
    units = tmp_path / "units.jsonl"
    output = tmp_path / "scores.jsonl"
    log_file = tmp_path / "errors.log"
    write_jsonl(definitions, [
        dict(label_id="2", knw_label="知识点@土壤", existing_interpretation=dict.fromkeys(matching.FIELDS, "释义")),
    ])
    write_jsonl(units, [
        dict(
            parent_id="q1",
            question_id="q1",
            stem="题目",
            knw_labels=["知识点@土壤"],
            sub_questions=[],
        ),
    ])

    class FailingClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            raise ConnectionError("connection reset")

    monkeypatch.setattr(matching, "OpenAI", FailingClient)
    log_file.write_text("stale error\n", encoding="utf-8")
    summary = matching.score_units(
        str(units),
        str(definitions),
        str(output),
        "http://example/v1",
        "DeepSeek-V4-Flash",
        log_file=str(log_file),
    )

    assert summary["errors"] == 1
    log_text = log_file.read_text(encoding="utf-8")
    assert "stale error" not in log_text
    assert "question_id=q1" in log_text
    assert "knw_label=知识点@土壤" in log_text
    assert "error_type=ConnectionError" in log_text
    assert "connection reset" in log_text


def test_request_score_rejects_unjudgeable_with_numeric_score():
    class InvalidClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            answer = {
                "judgement": "unjudgeable",
                "score": 0.0,
                "reason": "缺少图示",
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

    unit = {
        "stem": "读图回答",
        "scoring_sub_questions": [],
    }
    definition = {
        "knw_label": "知识点@土壤",
        "existing_interpretation": dict.fromkeys(matching.FIELDS, "释义"),
    }
    with pytest.raises(ValueError, match="unjudgeable必须对应score=null"):
        matching.request_score(
            InvalidClient(), "DeepSeek-V4-Flash", unit, definition
        )
