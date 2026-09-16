import json

from bio_geo_tagging.sample_tagging_groups import sample_groups


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_sample_groups_keeps_complete_eligible_groups(tmp_path):
    catalog = tmp_path / "catalog.txt"
    catalog.write_text(
        "\n".join(
            f"知识点@标签{index}｜释义{index}" for index in range(414)
        )
        + "\n",
        encoding="utf-8",
    )
    units = [
        {
            "question_id": "ordinary",
            "root_question_id": "ordinary",
            "input_role": "root",
            "stem": "普通题",
            "knw_labels": ["知识点@标签0"],
        },
        {
            "question_id": "big",
            "root_question_id": "big",
            "input_role": "root",
            "stem": "公共题干",
            "knw_labels": ["知识点@标签1"],
        },
        {
            "question_id": "child-1",
            "root_question_id": "big",
            "input_role": "subquestion",
            "stem": "小题一",
            "knw_labels": ["知识点@标签1"],
        },
        {
            "question_id": "child-2",
            "root_question_id": "big",
            "input_role": "subquestion",
            "stem": "小题二",
            "knw_labels": ["知识点@标签1"],
        },
        {
            "question_id": "empty-root",
            "root_question_id": "empty-root",
            "input_role": "root",
            "stem": "",
            "knw_labels": ["知识点@标签2"],
        },
        {
            "question_id": "no-gold",
            "root_question_id": "no-gold",
            "input_role": "root",
            "stem": "没有标签",
            "knw_labels": [],
        },
        {
            "question_id": "unmapped",
            "root_question_id": "unmapped",
            "input_role": "root",
            "stem": "旧标签",
            "knw_labels": ["知识点@不存在"],
        },
        {
            "question_id": "orphan-child",
            "root_question_id": "orphan",
            "input_role": "subquestion",
            "stem": "缺少根题目",
            "knw_labels": ["知识点@标签3"],
        },
    ]
    input_path = tmp_path / "units.jsonl"
    output_path = tmp_path / "sample.jsonl"
    summary_path = tmp_path / "summary.json"
    write_jsonl(input_path, units)

    summary = sample_groups(
        input_path,
        catalog,
        output_path,
        summary_path,
        group_count=2,
        seed=7,
    )

    sampled = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [unit["question_id"] for unit in sampled] == [
        "ordinary",
        "big",
        "child-1",
        "child-2",
    ]
    assert summary["sampled_groups"] == 2
    assert summary["sampled_ordinary_questions"] == 1
    assert summary["sampled_big_questions"] == 1
    assert summary["sampled_units"] == 4
    assert summary["sampled_root_units"] == 2
    assert summary["sampled_subquestion_units"] == 2
    assert summary["eligible_groups"] == 2
    assert summary["excluded_empty_stem_groups"] == 1
    assert summary["excluded_no_gold_groups"] == 1
    assert summary["excluded_unmapped_groups"] == 1
    assert summary["excluded_invalid_structure_groups"] == 1
