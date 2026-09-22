import json

from bio_geo_tagging.question_tagging_units import expand_question, process_file


def test_expand_question_propagates_only_parent_image_description():
    units = list(
        expand_question(
            {
                "question_id": "parent-1",
                "stem": "公共题干",
                "image_description": "等高线图，甲地海拔高于乙地",
                "stem_image_url": "https://example.test/parent.png",
                "sub_questions": [
                    {"question_id": "child-1", "stem": "判断两地气温差异"}
                ],
            }
        )
    )
    assert units[0]["input_role"] == "whole_question_region"
    assert units[0]["sub_questions"][0]["question_id"] == "child-1"
    assert units[1]["input_role"] == "whole_question_comprehensive"
    assert units[2]["context_image_description"] == "等高线图，甲地海拔高于乙地"
    assert "context_stem_image_url" not in units[2]


def test_process_file_expands_root_and_sub_questions(tmp_path):
    input_file = tmp_path / "merged.jsonl"
    output_file = tmp_path / "units.jsonl"
    log_file = tmp_path / "units.log"

    merged_question = {
        "parent_id": "parent-1",
        "question_id": "parent-1",
        "stem": "公共题干",
        "options": "",
        "analysis": "",
        "knw_labels": ["知识点@大题标签"],
        "sub_questions": [
            {
                "parent_id": "parent-1",
                "question_id": "child-1",
                "stem": "小题一",
                "options": "A. 选项",
                "analysis": "小题解析",
                "knw_labels": ["知识点@小题标签"],
            }
        ],
    }
    input_file.write_text(
        json.dumps(merged_question, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    process_file(str(input_file), str(output_file), str(log_file))

    units = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]
    assert units == [
        {
            "parent_id": "parent-1",
            "question_id": "parent-1",
            "stem": "公共题干",
            "options": "",
            "analysis": "",
            "root_question_id": "parent-1",
            "context_stem": "",
            "sub_questions": [
                {
                    "parent_id": "parent-1",
                    "question_id": "child-1",
                    "stem": "小题一",
                    "options": "A. 选项",
                    "analysis": "小题解析",
                }
            ],
            "input_role": "whole_question_region",
        },
        {
            "parent_id": "parent-1",
            "question_id": "parent-1",
            "stem": "公共题干",
            "options": "",
            "analysis": "",
            "input_role": "whole_question_comprehensive",
            "root_question_id": "parent-1",
            "context_stem": "",
            "sub_questions": [
                {
                    "parent_id": "parent-1",
                    "question_id": "child-1",
                    "stem": "小题一",
                    "options": "A. 选项",
                    "analysis": "小题解析",
                }
            ],
        },
        {
            "parent_id": "parent-1",
            "question_id": "child-1",
            "stem": "小题一",
            "options": "A. 选项",
            "analysis": "小题解析",
            "input_role": "subquestion",
            "root_question_id": "parent-1",
            "context_stem": "公共题干",
        },
    ]
    assert log_file.exists()
