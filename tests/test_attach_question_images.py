import json

from bio_geo_tagging.attach_question_images import attach_question_images


def test_attach_images_to_root_and_subquestions(tmp_path):
    input_path = tmp_path / "questions.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "question_id": "100",
                "stem": "公共题干",
                "knw_labels": ["标签"],
                "sub_questions": [
                    {"question_id": "101", "stem": "小题一"},
                    {"question_id": "102", "stem": "小题二"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    image_map = tmp_path / "four_subject_image_urls.json"
    image_map.write_text(
        "{\n"
        '"other":{"stemImageUrl":"https://example.com/other.png"},\n'
        '"100":{"stemImageUrl":"https://example.com/root.png",'
        '"analysisImageUrl":"https://example.com/root-analysis.png"},\n'
        '"101":{"stemImageUrl":"https://example.com/sub.png"}\n'
        "}\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "with-images.jsonl"
    summary_path = tmp_path / "summary.json"

    summary = attach_question_images(
        input_path, image_map, output_path, summary_path
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["stem_image_url"] == "https://example.com/root.png"
    assert (
        result["analysis_image_url"]
        == "https://example.com/root-analysis.png"
    )
    assert (
        result["sub_questions"][0]["stem_image_url"]
        == "https://example.com/sub.png"
    )
    assert "stem_image_url" not in result["sub_questions"][1]
    assert summary == {
        "input_questions": 1,
        "input_subquestions": 2,
        "target_question_ids": 3,
        "image_records_found": 2,
        "matched_root_questions": 1,
        "matched_subquestions": 1,
        "matched_question_ids": 2,
        "attached_url_fields": 3,
        "image_records_without_usable_url": 0,
        "question_ids_without_image_record": 1,
    }


def test_image_record_with_empty_urls_is_reported(tmp_path):
    input_path = tmp_path / "questions.jsonl"
    input_path.write_text(
        '{"question_id":"200","stem":"题目","sub_questions":[]}\n',
        encoding="utf-8",
    )
    image_map = tmp_path / "images.json"
    image_map.write_text(
        '{\n"200":{"stemImageUrl":"","analysisImageUrl":null}\n}\n',
        encoding="utf-8",
    )
    output_path = tmp_path / "output.jsonl"
    summary_path = tmp_path / "summary.json"

    summary = attach_question_images(
        input_path, image_map, output_path, summary_path
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert "stem_image_url" not in result
    assert summary["image_records_found"] == 1
    assert summary["image_records_without_usable_url"] == 1
