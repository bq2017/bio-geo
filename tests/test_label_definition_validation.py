import json

from openpyxl import Workbook

from bio_geo_tagging.label_definition_validation import (
    COLUMNS,
    DeepSeekValidator,
    build_generation_messages,
    load_labels,
    run_validation,
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
            "definition_status": "consistent",
            "keywords_status": "consistent",
            "exam_methods_status": "consistent",
            "distinction_status": "consistent",
            "overall_status": "consistent",
            "reason": "语义一致",
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


def test_load_and_validate_one_label(tmp_path):
    input_file = tmp_path / "knowledge-graph.xlsx"
    output_file = tmp_path / "validation.jsonl"
    create_workbook(input_file)

    labels = load_labels(str(input_file), "知识点及试题量详情")
    summary = run_validation(
        labels=labels,
        output_jsonl=str(output_file),
        validator=FakeValidator(),
        model="test-model",
        limit=None,
    )

    result = json.loads(output_file.read_text(encoding="utf-8"))
    assert summary["completed"] == 1
    assert result["label_id"] == "2276111643787415552"
    assert result["status"] == "completed"
    assert result["consistency"]["overall_status"] == "consistent"
