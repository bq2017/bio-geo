import json
from types import SimpleNamespace

import pytest
from openpyxl import Workbook

from bio_geo_tagging.label_definition_validation import (
    COLUMNS,
    COMPARISON_SYSTEM_PROMPT,
    DEFINITION_REVIEW_SYSTEM_PROMPT,
    DeepSeekValidator,
    build_definition_review_messages,
    build_generation_messages,
    load_comparison_records,
    load_generated_records,
    load_labels,
    run_comparison,
    run_definition_review,
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

    def compare(self, full_path, existing, generated):
        return {
            "same_understanding": True,
            "difference_summary": "",
            "needs_teacher_review": False,
        }


class ReviewFakeValidator:
    base_url = "test://review"

    def __init__(self):
        self.reviewed = []

    def review_definition(self, record):
        self.reviewed.append(record["label_id"])
        return {
            "needs_definition_review": False,
            "review_fields": [],
            "review_reason": "",
            "needs_teacher_review": False,
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


def test_comparison_prompt_prioritizes_overall_scope():
    assert "明确排除某项内容" in COMPARISON_SYSTEM_PROMPT
    assert "核心教学目标与主要知识范围" in COMPARISON_SYSTEM_PROMPT
    assert "不要抓住单个词句" in COMPARISON_SYSTEM_PROMPT
    assert "只是解释、示例、公式展开或应用场景" in COMPARISON_SYSTEM_PROMPT
    assert "构成可独立考查的教学目标" in COMPARISON_SYSTEM_PROMPT
    assert "是否指向同一具体内容" in COMPARISON_SYSTEM_PROMPT
    assert "必须将 same_understanding 设为 false" in COMPARISON_SYSTEM_PROMPT
    assert "局部措辞不严谨但不影响整体边界" in COMPARISON_SYSTEM_PROMPT


def test_definition_review_prompt_focuses_on_existing_interpretation():
    assert "现有释义是否需要复核" in DEFINITION_REVIEW_SYSTEM_PROMPT
    assert "不要按文字相似度判断" in DEFINITION_REVIEW_SYSTEM_PROMPT
    assert "只输出 json，不要补充解释或修改后的释义" in DEFINITION_REVIEW_SYSTEM_PROMPT


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
        validators=[FakeValidator()],
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


def test_comparison_distributes_workers_across_three_endpoints(tmp_path):
    labels = [
        {
            "source_row": index + 2,
            "label_id": str(index),
            "full_path": f"知识点->{index}",
            "existing_interpretation": {},
        }
        for index in range(6)
    ]
    generated_records = {
        label["label_id"]: {
            "full_path": label["full_path"],
            "generated_interpretation": {},
            "model": "test-model",
        }
        for label in labels
    }
    validators = [
        AnyFakeValidator("test://9102"),
        AnyFakeValidator("test://9103"),
        AnyFakeValidator("test://9104"),
    ]
    output_file = tmp_path / "comparison.jsonl"

    summary = run_comparison(
        labels,
        generated_records,
        str(output_file),
        validators,
        "test-model",
        limit=None,
    )
    records = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]

    assert summary["completed"] == 6
    assert summary["workers"] == 3
    assert [record["endpoint"] for record in records].count("test://9102") == 2
    assert [record["endpoint"] for record in records].count("test://9103") == 2
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
            validators=[FakeValidator()],
            model="test-model",
            limit=None,
        )


def test_definition_review_runs_sequentially_and_resumes(tmp_path):
    input_file = tmp_path / "different.jsonl"
    output_file = tmp_path / "review.jsonl"
    records = [
        {
            "source_row": index + 2,
            "label_id": str(index),
            "full_path": f"知识点->{index}",
            "existing_interpretation": {},
            "generated_interpretation": {},
            "comparison_analysis": {"same_understanding": False},
            "status": "completed",
        }
        for index in range(2)
    ]
    input_file.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = load_comparison_records(str(input_file))
    messages = build_definition_review_messages(loaded[0])
    assert "知识点->0" in messages[1]["content"]

    validator = ReviewFakeValidator()
    summary = run_definition_review(
        loaded, str(output_file), validator, "test-model", limit=None
    )
    assert validator.reviewed == ["0", "1"]
    assert summary["completed"] == 2

    second_validator = ReviewFakeValidator()
    resumed = run_definition_review(
        loaded, str(output_file), second_validator, "test-model", limit=None
    )
    assert second_validator.reviewed == []
    assert resumed["already_completed"] == 2
