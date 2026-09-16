import json

from bio_geo_tagging.question_info_merge import process_file


def test_process_file_cleans_and_merges_questions(tmp_path):
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "output.jsonl"
    log_file = tmp_path / "merge.log"
    taxonomy_file = tmp_path / "taxonomy.json"

    taxonomy_file.write_text(
        json.dumps(
            {
                "knowledge-parent": "知识点@大题标签",
                "knowledge-child": "知识点@小题标签",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = [
        {
            "question_id": "parent-1",
            "parent_id": "parent-1",
            "structure_type": "fuhe",
            "answered_count": "10",
            "percent_correct": "80.00",
            "difficulty": "2",
            "knw_ids": ["knowledge-parent"],
            "question_info": {
                "stem": "大题<br><img src=\"image.png\">",
                "options": [],
                "analysis": "",
            },
        },
        {
            "question_id": "child-1",
            "parent_id": "parent-1",
            "structure_type": "danxuan",
            "answered_count": "8",
            "percent_correct": "75.00",
            "difficulty": "2",
            "knw_ids": ["knowledge-child"],
            "question_info": {
                "stem": "小题$(\\qquad)$<br>",
                "options": [
                    {"title": "A", "htmlCode": "选项一<br>"},
                    {"title": "B", "htmlCode": "选项二"},
                ],
                "analysis": "解析<br>",
            },
        },
    ]
    input_file.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    process_file(
        str(input_file),
        str(output_file),
        str(log_file),
        str(taxonomy_file),
    )

    output_records = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
    ]
    assert output_records == [
        {
            "parent_id": "parent-1",
            "question_id": "parent-1",
            "stem": "大题",
            "options": "",
            "analysis": "",
            "structure_type": "fuhe",
            "answered_count": "10",
            "percent_correct": "80.00",
            "difficulty": "2",
            "knw_labels": ["知识点@大题标签"],
            "sub_questions": [
                {
                    "parent_id": "parent-1",
                    "question_id": "child-1",
                    "stem": "小题( )",
                    "options": "A. 选项一\nB. 选项二",
                    "analysis": "解析",
                    "knw_labels": ["知识点@小题标签"],
                }
            ],
        }
    ]
    assert log_file.exists()
