import json
from types import SimpleNamespace

import pytest
from openpyxl import Workbook

from bio_geo_tagging.label_definition_validation import (
    COLUMNS,
    DeepSeekValidator,
    build_generation_messages,
    load_generated_records,
    load_labels,
    run_comparison,
    run_generation,
)


class FakeValidator:
    def generate(self, full_path):
        assert full_path == "知识点->地理工具->地球仪"
        return {
            "definition": "生成定义",
            "keywords": ["地轴", "经线"],
            "exam_methods": "生成考法",
            "distinction": "生成区分",
        }

    def compare(self, full_path, existing, generated):
        assert full_path == "知识点->地理工具->地球仪"
        assert existing["definition"] == "现有定义"
        assert generated["definition"] == "生成定义"
        return {
            "definition_analysis": {
                "difference_types": ["consistent"],
                "detail": "定义一致",
            },
            "keywords_analysis": {
                "difference_types": ["consistent"],
                "detail": "关键词一致",
            },
            "exam_methods_analysis": {
                "difference_types": ["consistent"],
                "detail": "考查方式一致",
            },
            "distinction_analysis": {
                "difference_types": ["consistent"],
                "detail": "边界一致",
            },
            "overall_difference_types": ["consistent"],
            "summary": "语义一致",
            "recommendation": "keep_existing",
            "review_required": False,
        }


def create_workbook(path):
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "知识点及试题量详情"
    worksheet.append(list(COLUMNS.values()))
    worksheet.append(
        [
            "知识点->地理工具->地球仪",
            "2276111643787415552",
            "现有定义",
            "现有关键词",
            "现有考法",
            "现有区分",
        ]
    )
    workbook.save(path)


def test_generation_prompt_contains_path_but_not_existing_definition():
    messages = build_generation_messages("知识点->地理工具->地球仪")
    prompt = json.dumps(messages, ensure_ascii=False)
    assert "知识点->地理工具->地球仪" in prompt
    assert "现有定义" not in prompt


def test_local_service_does_not_require_api_key():
    validator = DeepSeekValidator(
        model="DeepSeek-V4-Flash",
        base_url="http://172.22.0.35:9092/v1",
    )
    assert validator.model == "DeepSeek-V4-Flash"


def test_request_json_reads_streamed_content():
    validator = DeepSeekValidator(
        model="DeepSeek-V4-Flash",
        base_url="http://172.22.0.35:9092/v1",
    )
    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content='{"ok":'))]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="true}"))]
        ),
    ]
    validator.client.chat.completions.create = lambda **request: chunks

    assert validator.request_json([], 64) == {"ok": True}


def test_compare_accepts_scope_difference_without_teacher_review():
    validator = DeepSeekValidator(
        model="DeepSeek-V4-Flash",
        base_url="http://172.22.0.35:9092/v1",
    )
    comparison = {
        "definition_analysis": {
            "difference_types": ["scope_difference"],
            "detail": "生成释义范围更宽",
        },
        "keywords_analysis": {
            "difference_types": ["scope_difference"],
            "detail": "生成释义包含上位概念",
        },
        "exam_methods_analysis": {
            "difference_types": ["consistent"],
            "detail": "核心考查方式一致",
        },
        "distinction_analysis": {
            "difference_types": ["boundary_difference"],
            "detail": "生成释义未遵守末级知识点边界",
        },
        "overall_difference_types": ["scope_difference", "boundary_difference"],
        "summary": "生成理解超出末级知识点范围",
        "recommendation": "improve_generation",
        "review_required": False,
    }
    validator.request_json = lambda messages, max_tokens: comparison

    result = validator.compare("知识点->地理工具->地球仪", {}, {})

    assert result["recommendation"] == "improve_generation"
    assert result["review_required"] is False


def test_generation_and_comparison_run_separately(tmp_path):
    input_file = tmp_path / "knowledge-graph.xlsx"
    generation_file = tmp_path / "generation.jsonl"
    comparison_file = tmp_path / "comparison.jsonl"
    create_workbook(input_file)

    labels = load_labels(str(input_file), "知识点及试题量详情")
    generation_summary = run_generation(
        labels=labels,
        output_jsonl=str(generation_file),
        validator=FakeValidator(),
        model="test-model",
        limit=None,
    )
    generation_result = json.loads(generation_file.read_text(encoding="utf-8"))

    assert generation_summary["completed"] == 1
    assert "existing_interpretation" not in generation_result
    assert generation_result["generated_interpretation"]["definition"] == "生成定义"

    generated_records = load_generated_records(str(generation_file))
    comparison_summary = run_comparison(
        labels=labels,
        generated_records=generated_records,
        output_jsonl=str(comparison_file),
        validator=FakeValidator(),
        model="test-model",
        limit=None,
    )

    result = json.loads(comparison_file.read_text(encoding="utf-8"))
    assert comparison_summary["completed"] == 1
    assert comparison_summary["missing_generation"] == 0
    assert result["label_id"] == "2276111643787415552"
    assert result["status"] == "completed"
    assert result["comparison_analysis"]["overall_difference_types"] == [
        "consistent"
    ]
    assert result["comparison_analysis"]["recommendation"] == "keep_existing"


def test_comparison_stops_when_generation_is_incomplete(tmp_path):
    input_file = tmp_path / "knowledge-graph.xlsx"
    create_workbook(input_file)
    labels = load_labels(str(input_file), "知识点及试题量详情")

    with pytest.raises(ValueError, match="Generation stage is incomplete: 1"):
        run_comparison(
            labels=labels,
            generated_records={},
            output_jsonl=str(tmp_path / "comparison.jsonl"),
            validator=FakeValidator(),
            model="test-model",
            limit=None,
        )
