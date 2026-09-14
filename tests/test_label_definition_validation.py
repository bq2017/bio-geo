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
    base_url = "test://primary"

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
            "same_understanding": True,
            "difference_summary": "",
            "needs_teacher_review": False,
        }


class AnyFakeValidator:
    def __init__(self, base_url):
        self.base_url = base_url

    def generate(self, full_path):
        return {
            "definition": full_path,
            "keywords": [],
            "exam_methods": "",
            "distinction": "",
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


def test_compare_marks_difference_for_teacher_review():
    validator = DeepSeekValidator(
        model="DeepSeek-V4-Flash",
        base_url="http://172.22.0.35:9092/v1",
    )
    comparison = {
        "same_understanding": False,
        "difference_summary": "原释义和生成释义的知识范围不同",
        "needs_teacher_review": True,
    }
    validator.request_json = lambda messages, max_tokens: comparison

    result = validator.compare("知识点->地理工具->地球仪", {}, {})

    assert set(result) == {
        "same_understanding",
        "difference_summary",
        "needs_teacher_review",
    }
    assert result["same_understanding"] is False
    assert result["needs_teacher_review"] is True


def test_generation_and_comparison_run_separately(tmp_path):
    input_file = tmp_path / "knowledge-graph.xlsx"
    generation_file = tmp_path / "generation.jsonl"
    comparison_file = tmp_path / "comparison.jsonl"
    create_workbook(input_file)

    labels = load_labels(str(input_file), "知识点及试题量详情")
    generation_summary = run_generation(
        labels=labels,
        output_jsonl=str(generation_file),
        validators=[FakeValidator()],
        model="test-model",
        limit=None,
    )
    generation_result = json.loads(generation_file.read_text(encoding="utf-8"))

    assert generation_summary["completed"] == 1
    assert generation_summary["workers"] == 1
    assert "existing_interpretation" not in generation_result
    assert generation_result["endpoint"] == "test://primary"
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
    assert result["comparison_analysis"]["same_understanding"] is True
    assert result["comparison_analysis"]["needs_teacher_review"] is False


def test_generation_distributes_workers_across_two_endpoints(tmp_path):
    labels = [
        {
            "source_row": index + 2,
            "label_id": str(index),
            "full_path": f"知识点->{index}",
            "existing_interpretation": {},
        }
        for index in range(4)
    ]
    validators = [
        AnyFakeValidator("test://9093"),
        AnyFakeValidator("test://9104"),
        AnyFakeValidator("test://9093"),
        AnyFakeValidator("test://9104"),
    ]
    output_file = tmp_path / "generation.jsonl"

    summary = run_generation(
        labels, str(output_file), validators, "test-model", limit=None
    )
    records = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]

    assert summary["completed"] == 4
    assert summary["workers"] == 4
    assert [record["endpoint"] for record in records].count("test://9093") == 2
    assert [record["endpoint"] for record in records].count("test://9104") == 2


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
