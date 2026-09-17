import json

from bio_geo_tagging.sample_tagging_groups import (
    load_high_score_valid_pairs,
    sample_groups,
)


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


def test_sample_groups_accepts_nested_complete_questions(tmp_path):
    catalog = tmp_path / "catalog.txt"
    catalog.write_text(
        "\n".join(f"知识点@标签{index}｜释义{index}" for index in range(414))
        + "\n",
        encoding="utf-8",
    )
    questions = [
        {
            "parent_id": "big",
            "question_id": "big",
            "stem": "公共材料",
            "knw_labels": ["知识点@标签0"],
            "sub_questions": [
                {"question_id": "child-1", "stem": "小题一"},
                {"question_id": "child-2", "stem": "小题二"},
            ],
        },
        {
            "parent_id": "ordinary",
            "question_id": "ordinary",
            "stem": "普通题",
            "knw_labels": ["知识点@标签1"],
            "sub_questions": [],
        },
    ]
    input_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "sample.jsonl"
    summary_path = tmp_path / "summary.json"
    write_jsonl(input_path, questions)

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
    assert sampled == questions
    assert summary["sampled_groups"] == 2
    assert summary["sampled_units"] == 2
    assert summary["sampled_big_questions"] == 1
    assert summary["sampled_ordinary_questions"] == 1


def test_sample_groups_selects_requested_big_question_count(tmp_path):
    catalog = tmp_path / "catalog.txt"
    catalog.write_text(
        "\n".join(f"知识点@标签{index}｜释义{index}" for index in range(414)) + "\n",
        encoding="utf-8",
    )
    questions = [
        {
            "parent_id": str(index),
            "question_id": str(index),
            "stem": f"题目{index}",
            "knw_labels": ["知识点@标签0"],
            "sub_questions": [{"question_id": f"{index}-1", "stem": "小题"}]
            if index < 3 else [],
        }
        for index in range(10)
    ]
    input_path = tmp_path / "questions.jsonl"
    output_path = tmp_path / "sample.jsonl"
    summary_path = tmp_path / "summary.json"
    write_jsonl(input_path, questions)

    summary = sample_groups(
        input_path, catalog, output_path, summary_path,
        group_count=6, seed=7, big_questions=2,
    )

    sampled = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(sampled) == 6
    assert sum(bool(question["sub_questions"]) for question in sampled) == 2
    assert summary["sampled_big_questions"] == 2
    assert summary["sampled_ordinary_questions"] == 4


def test_sample_groups_requires_every_label_to_be_high_score_valid(tmp_path):
    catalog = tmp_path / "catalog.txt"
    catalog.write_text(
        "\n".join(f"知识点@标签{index}｜释义{index}" for index in range(414)) + "\n",
        encoding="utf-8",
    )
    questions = [
        {
            "parent_id": "invalid-multi",
            "question_id": "invalid-multi",
            "stem": "两个标签但只有一个通过",
            "knw_labels": ["知识点@标签0", "知识点@标签1"],
            "sub_questions": [],
        },
        {
            "parent_id": "ordinary",
            "question_id": "ordinary",
            "stem": "普通题",
            "knw_labels": ["知识点@标签0"],
            "sub_questions": [],
        },
        {
            "parent_id": "big",
            "question_id": "big",
            "stem": "大题",
            "knw_labels": ["知识点@标签1"],
            "sub_questions": [{"question_id": "child", "stem": "小题"}],
        },
    ]
    diagnoses = [
        {
            "status": "completed",
            "knw_label": "知识点@标签0",
            "high_score_valid_ids": ["invalid-multi", "ordinary"],
        },
        {
            "status": "completed",
            "knw_label": "知识点@标签1",
            "high_score_valid_ids": ["big"],
        },
    ]
    input_path = tmp_path / "questions.jsonl"
    diagnosis_path = tmp_path / "diagnosis.jsonl"
    output_path = tmp_path / "sample.jsonl"
    summary_path = tmp_path / "summary.json"
    write_jsonl(input_path, questions)
    write_jsonl(diagnosis_path, diagnoses)

    assert load_high_score_valid_pairs(diagnosis_path) == {
        ("invalid-multi", "知识点@标签0"),
        ("ordinary", "知识点@标签0"),
        ("big", "知识点@标签1"),
    }
    summary = sample_groups(
        input_path,
        catalog,
        output_path,
        summary_path,
        group_count=2,
        seed=7,
        big_questions=1,
        diagnosis_path=diagnosis_path,
    )

    sampled = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert {question["question_id"] for question in sampled} == {"ordinary", "big"}
    assert summary["eligible_groups"] == 2
    assert summary["excluded_unvalidated_groups"] == 1
