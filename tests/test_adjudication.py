import json
from pathlib import Path

import pytest

from bio_geo_tagging.adjudication import (
    CandidateIndex,
    build_adjudication_inputs,
    build_adjudication_prompt,
    candidates_for_unit,
    load_labels,
    normalize_candidates,
    run_adjudication,
    validate_adjudication_result,
    write_question_predictions,
)
from bio_geo_tagging.ds import DSResponse


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


class FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def chat(self, messages, *, max_tokens=1024):
        self.calls += 1
        assert messages[0]["role"] == "system"
        assert "高中地理" in messages[0]["content"]
        return DSResponse(
            content=self.content,
            endpoint="http://test/v1/chat/completions",
            attempts=1,
            latency_seconds=0.01,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


def test_normalize_candidates_supports_all_current_formats():
    assert normalize_candidates(
        {"candidate_labels": ["知识点@甲", "知识点@乙"]}
    ) == [
        {
            "label_id": "知识点@甲",
            "candidate_rank": 1,
            "sources": [],
        },
        {
            "label_id": "知识点@乙",
            "candidate_rank": 2,
            "sources": [],
        },
    ]
    hybrid = normalize_candidates(
        {
            "combined_candidates": [
                {"label_path": "知识点@甲", "bm25_rank": 2},
                {
                    "label_path": "知识点@区域@乙",
                    "region_evidence_type": "direct_name",
                },
            ]
        }
    )
    assert hybrid[0]["sources"] == ["bm25"]
    assert hybrid[1]["sources"] == ["region"]
    assert normalize_candidates(
        {"candidates": [{"label_id": "知识点@甲", "candidate_rank": 7}]}
    )[0]["candidate_rank"] == 7


def test_candidate_index_reuses_root_candidates_for_subquestion():
    index = CandidateIndex(
        [
            {
                "question_id": "root-1",
                "combined_candidates": [{"label_path": "知识点@甲"}],
            }
        ]
    )
    candidates = index.resolve(
        {
            "question_id": "child-1",
            "root_question_id": "root-1",
            "input_role": "subquestion",
        }
    )
    assert candidates[0]["label_id"] == "知识点@甲"


def test_load_labels_accepts_existing_geography_definition_fields(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    write_jsonl(
        labels_path,
        [
            {
                "label_path": "知识点@地理工具@经纬网",
                "positive_definition": "利用经纬网判断方向",
                "assessment_scope": "方向、距离和位置判断",
            },
            {
                "label_path": "知识点@自然地理@气候",
                "knowledge_scope": "分析气候特征",
                "common_exam_content": "气温和降水",
                "distinction_from_similar_labels": "不含天气现象",
            },
        ],
    )
    labels = load_labels(labels_path)
    assert labels["知识点@地理工具@经纬网"]["label_name"] == "经纬网"
    assert labels["知识点@自然地理@气候"]["distinctions"] == "不含天气现象"


def test_load_labels_accepts_nested_existing_interpretation_schema(tmp_path):
    labels_path = tmp_path / "labels.jsonl"
    write_jsonl(
        labels_path,
        [
            {
                "label_id": "2276111643787415552",
                "knw_label": "知识点@地理工具@地球仪",
                "existing_interpretation": {
                    "definition": "认识地球仪的构造及空间意义",
                    "keywords": "地轴、两极、赤道、经线、纬线",
                    "exam_methods": "根据经纬网进行基础定位",
                    "distinction": "不含经纬网推理性应用",
                },
            }
        ],
    )
    labels = load_labels(labels_path)
    label = labels["知识点@地理工具@地球仪"]
    assert label["label_id"] == "知识点@地理工具@地球仪"
    assert label["taxonomy_label_id"] == "2276111643787415552"
    assert label["definition"] == "认识地球仪的构造及空间意义"
    assert label["core_concepts"] == "地轴、两极、赤道、经线、纬线"
    assert label["assessment_scope"] == "根据经纬网进行基础定位"
    assert label["distinctions"] == "不含经纬网推理性应用"


def test_comprehensive_candidates_are_reserved_for_whole_question_pass():
    labels = {
        "知识点@交通区位": {
            "label_name": "交通区位",
        },
        "知识点@交通综合": {
            "label_name": "交通综合",
        },
    }
    candidates = [
        {"label_id": "知识点@交通区位"},
        {"label_id": "知识点@交通综合"},
    ]
    subquestion = candidates_for_unit(
        {"input_role": "subquestion"}, candidates, labels
    )
    whole = candidates_for_unit(
        {"input_role": "whole_question_comprehensive"}, candidates, labels
    )
    assert [item["label_id"] for item in subquestion] == ["知识点@交通区位"]
    assert [item["label_id"] for item in whole] == ["知识点@交通综合"]


def test_build_prompt_maps_geography_fields_and_detects_missing_image():
    unit = {
        "question_id": "child-1",
        "root_question_id": "root-1",
        "input_role": "subquestion",
        "context_stem": "读图完成下题",
        "stem": "判断甲乙两地气温差异",
        "answer": "甲地气温低",
        "explanation": "甲地海拔更高",
    }
    labels = {
        "知识点@气温": {
            "label_id": "知识点@气温",
            "label_name": "气温",
            "label_path": "知识点@气温",
            "definition": "分析气温差异",
            "core_concepts": "海拔影响气温",
            "distinctions": "不含降水",
        }
    }
    question, _, _ = build_adjudication_inputs(
        unit,
        [{"label_id": "知识点@气温", "candidate_rank": 1}],
        labels,
    )
    assert question["parent_stem"] == "读图完成下题"
    assert question["answer_text"] == "甲地气温低"
    assert question["analysis"] == "甲地海拔更高"
    assert question["image_context_missing"] is True
    prompt, _, _ = build_adjudication_prompt(
        unit,
        [{"label_id": "知识点@气温", "candidate_rank": 1}],
        labels,
    )
    assert "空间或时间尺度不一致" in prompt
    assert "区域Label" in prompt


def test_validate_result_requires_short_verbatim_evidence():
    question = {
        "parent_stem": "",
        "stem": "判断甲乙两地气温差异",
        "options": "",
        "answer_text": "甲地气温低",
        "analysis": "甲地海拔更高",
        "image_description": "",
        "parent_image_description": "",
    }
    parsed = validate_adjudication_result(
        {
            "selected": ["C01"],
            "evidence": {"C01": "甲地海拔更高"},
            "context_insufficient": False,
            "need_expand_recall": False,
            "reason": "直接考查海拔对气温的影响",
        },
        {"C01"},
        question,
    )
    assert parsed["selected"] == ["C01"]
    with pytest.raises(ValueError, match="not copied"):
        validate_adjudication_result(
            {
                "selected": ["C01"],
                "evidence": {"C01": "这是模型自行改写的证据"},
                "context_insufficient": False,
                "need_expand_recall": False,
                "reason": "直接考查",
            },
            {"C01"},
            question,
        )


def test_run_adjudication_materializes_and_resumes(tmp_path):
    units_path = tmp_path / "units.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    run_dir = tmp_path / "run"
    write_jsonl(
        units_path,
        [
            {
                "question_id": "child-1",
                "root_question_id": "root-1",
                "parent_id": "root-1",
                "input_role": "subquestion",
                "context_stem": "某山地剖面图",
                "stem": "判断甲乙两地气温差异",
                "answer": "甲地气温低",
                "analysis": "甲地海拔更高",
                "image_description": "甲地位于较高海拔",
            }
        ],
    )
    write_jsonl(
        candidates_path,
        [
            {
                "question_id": "root-1",
                "retrieval_version": "hybrid-test-v1",
                "combined_candidates": [
                    {"label_path": "知识点@自然地理@气温", "bm25_rank": 1}
                ],
            }
        ],
    )
    write_jsonl(
        labels_path,
        [
            {
                "label_path": "知识点@自然地理@气温",
                "definition": "分析气温的空间差异及原因",
                "core_concepts": "纬度、海拔和下垫面影响气温",
                "distinctions": "不含降水差异",
            }
        ],
    )
    client = FakeClient(
        json.dumps(
            {
                "selected": ["C01"],
                "evidence": {"C01": "甲地海拔更高"},
                "context_insufficient": False,
                "need_expand_recall": False,
                "reason": "直接考查海拔造成的气温差异",
            },
            ensure_ascii=False,
        )
    )
    report = run_adjudication(
        units_path,
        candidates_path,
        labels_path,
        run_dir,
        client,
        model="test-model",
        workers=2,
        max_tokens=512,
        enable_thinking=False,
    )
    assert report["success"] == 1
    assert report["usable_for_training"] == 1
    assert client.calls == 1
    prediction = json.loads(
        (run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert prediction["selected_labels"][0]["label_path"] == "知识点@自然地理@气温"
    assert prediction["selected_labels"][0]["candidate_rank"] == 1
    assert prediction["root_question_id"] == "root-1"
    question_prediction = json.loads(
        (run_dir / "question_predictions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert question_prediction["root_question_id"] == "root-1"
    assert question_prediction["selected_labels"][0]["evidence_by_unit"][0][
        "question_id"
    ] == "child-1"

    second = run_adjudication(
        units_path,
        candidates_path,
        labels_path,
        run_dir,
        client,
        model="test-model",
        workers=2,
        max_tokens=512,
        enable_thinking=False,
    )
    assert second["success"] == 1
    assert second["requests_succeeded"] == 0
    assert client.calls == 1


def test_audited_exclusion_removes_selected_label(tmp_path):
    units_path = tmp_path / "units.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    exclusions_path = tmp_path / "exclusions.json"
    write_jsonl(
        units_path,
        [{"question_id": "q1", "stem": "判断气温差异", "input_role": "root"}],
    )
    write_jsonl(
        candidates_path,
        [{"question_id": "q1", "candidate_labels": ["知识点@气温"]}],
    )
    write_jsonl(
        labels_path,
        [{"label_path": "知识点@气温", "definition": "判断气温差异"}],
    )
    exclusions_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "question_id": "q1",
                        "label_path": "知识点@气温",
                        "action": "remove_label_and_exclude_training",
                        "reason": "人工确认误标",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    client = FakeClient(
        json.dumps(
            {
                "selected": ["C01"],
                "evidence": {"C01": "判断气温差异"},
                "context_insufficient": False,
                "need_expand_recall": False,
                "reason": "直接考查气温差异",
            },
            ensure_ascii=False,
        )
    )
    report = run_adjudication(
        units_path,
        candidates_path,
        labels_path,
        tmp_path / "run",
        client,
        model="test-model",
        audited_exclusions_path=exclusions_path,
    )
    assert report["audited_excluded_labels"] == 1
    prediction = json.loads(
        (tmp_path / "run" / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert prediction["selected_labels"] == []
    assert prediction["usable_for_training"] is False


def test_question_predictions_union_subquestions_and_comprehensive_label(tmp_path):
    base = {
        "root_question_id": "root-1",
        "none_of_candidates": False,
        "need_expand_recall": False,
        "context_insufficient": False,
        "needs_review": False,
        "model": "test-model",
        "prompt_version": "test-prompt",
    }
    predictions = [
        {
            **base,
            "unit_key": "root-1|child-1|subquestion",
            "question_id": "child-1",
            "input_role": "subquestion",
            "selected_labels": [
                {
                    "label_id": "知识点@交通区位",
                    "label_path": "知识点@交通区位",
                    "label_name": "交通区位",
                    "candidate_rank": 2,
                    "evidence": "分析港口建设的区位条件",
                }
            ],
        },
        {
            **base,
            "unit_key": "root-1|root-1|whole_question_comprehensive",
            "question_id": "root-1",
            "input_role": "whole_question_comprehensive",
            "selected_labels": [
                {
                    "label_id": "知识点@交通综合",
                    "label_path": "知识点@交通综合",
                    "label_name": "交通综合",
                    "candidate_rank": 35,
                    "evidence": "综合分析交通布局与区域发展的关系",
                }
            ],
        },
    ]
    summary = write_question_predictions(tmp_path / "questions.jsonl", predictions)
    record = json.loads(
        (tmp_path / "questions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert summary["whole_questions"] == 1
    assert [label["label_path"] for label in record["selected_labels"]] == [
        "知识点@交通区位",
        "知识点@交通综合",
    ]
