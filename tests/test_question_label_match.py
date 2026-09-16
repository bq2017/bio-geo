import json
from types import SimpleNamespace

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


def test_score_pairs_and_resume_without_network(tmp_path, monkeypatch):
    definitions = tmp_path / "definitions.jsonl"
    units = tmp_path / "units.jsonl"
    output = tmp_path / "scores.jsonl"
    write_jsonl(definitions, [
        dict(label_id="2", knw_label="知识点@土壤", existing_interpretation=dict.fromkeys(matching.FIELDS, "释义")),
        dict(label_id="3", knw_label="知识点@降水", existing_interpretation=dict.fromkeys(matching.FIELDS, "释义")),
    ])
    write_jsonl(units, [
        dict(question_id="p", root_question_id="p", input_role="root", stem="公共题干", knw_labels=["知识点@土壤"]),
        dict(question_id="c", root_question_id="p", input_role="subquestion", context_stem="公共题干", stem="小题", knw_labels=["知识点@降水", "知识点@土壤"]),
    ])

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            data = json.loads(kwargs["messages"][1]["content"])
            if data["stem"] == "小题" and data["knowledge_path"] == "知识点@降水":
                answer = {"score": None, "reason": "缺少图示"}
            else:
                answer = {"score": 0.84, "reason": "直接考查"}
            return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=json.dumps(answer, ensure_ascii=False)))])]

    monkeypatch.setattr(matching, "OpenAI", FakeClient)
    arguments = (str(units), str(definitions), str(output), "http://example/v1", "DeepSeek-V4-Flash")
    assert matching.score_units(*arguments, workers=2)["completed"] == 3
    rows = [row for _, row in matching.read_jsonl(str(output))]
    assert [(r["question_id"], r["knw_label"], r["score"], r["match"]) for r in rows] == [
        ("p", "知识点@土壤", 0.84, True),
        ("c", "知识点@降水", None, None),
        ("c", "知识点@土壤", 0.84, True),
    ]
    assert matching.score_units(*arguments)["already_completed"] == 3
    assert len(list(matching.read_jsonl(str(output)))) == 3
