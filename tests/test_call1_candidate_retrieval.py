import json

import pytest

from bio_geo_tagging.call1_candidate_retrieval import (
    build_question_text,
    load_catalog,
    load_units,
    normalize_shard_result,
    parse_or_recover_result,
    split_catalog,
    validate_result,
)


def test_load_catalog_returns_exact_paths(tmp_path):
    catalog = tmp_path / "catalog.txt"
    catalog.write_text(
        "知识点@自然地理@标签一｜释义一\n知识点@人文地理@标签二｜释义二\n",
        encoding="utf-8",
    )

    text, paths = load_catalog(catalog, expected_count=2)

    assert text.splitlines() == [
        "知识点@自然地理@标签一｜释义一",
        "知识点@人文地理@标签二｜释义二",
    ]
    assert paths == {"知识点@自然地理@标签一", "知识点@人文地理@标签二"}


def test_split_catalog_covers_each_label_once():
    catalog = "\n".join(
        f"知识点@标签{index}｜释义{index}" for index in range(10)
    )

    parts = split_catalog(catalog, parts=3)

    assert [len(paths) for _, paths in parts] == [4, 3, 3]
    all_paths = [path for _, paths in parts for path in paths]
    assert len(all_paths) == len(set(all_paths)) == 10
    assert set(all_paths) == {f"知识点@标签{index}" for index in range(10)}


def test_build_question_text_separates_context_and_excludes_existing_labels():
    unit = {
        "question_id": "child-1",
        "input_role": "subquestion",
        "context_stem": "公共材料",
        "stem": "当前小题",
        "options": [{"key": "A", "text": "选项A"}],
        "answer": "A",
        "analysis": "解析内容",
        "image_description": "等高线图",
        "knw_labels": ["知识点@不应泄漏"],
    }

    rendered = build_question_text(unit)

    assert "【公共题干，仅作为上下文】\n公共材料" in rendered
    assert "【当前打标对象】\n当前小题" in rendered
    assert json.dumps(unit["options"], ensure_ascii=False) in rendered
    assert "【答案】\nA" in rendered
    assert "【解析】\n解析内容" in rendered
    assert "【图片描述】\n等高线图" in rendered
    assert "不应泄漏" not in rendered


def test_load_units_skips_empty_root_but_keeps_its_subquestion(tmp_path):
    input_path = tmp_path / "units.jsonl"
    units = [
        {
            "question_id": "root-1",
            "root_question_id": "root-1",
            "input_role": "root",
            "stem": "",
        },
        {
            "question_id": "child-1",
            "root_question_id": "root-1",
            "input_role": "subquestion",
            "context_stem": "",
            "stem": "当前小题",
        },
    ]
    input_path.write_text(
        "".join(json.dumps(unit, ensure_ascii=False) + "\n" for unit in units),
        encoding="utf-8",
    )

    valid_units, skipped_empty_stem = load_units(input_path)

    assert valid_units == [units[1]]
    assert skipped_empty_stem == 1


def test_validate_result_accepts_known_unique_labels():
    allowed = {"知识点@标签一", "知识点@标签二"}

    labels, uncovered = validate_result(
        {
            "candidate_labels": ["知识点@标签一", "知识点@标签二"],
            "uncovered_topic": None,
        },
        allowed,
    )

    assert labels == ["知识点@标签一", "知识点@标签二"]
    assert uncovered is None


def test_normalize_shard_result_repairs_format_and_deduplicates():
    allowed = {"知识点@自然地理@地球仪", "知识点@自然地理@经纬网"}

    labels = normalize_shard_result(
        {
            "candidate_labels": [
                "自然地理@地球仪",
                "知识点@自然地理@地球仪｜标签释义",
                "知识点@自然地理@经纬网",
                "知识点@其他批次@标签",
            ]
        },
        allowed,
    )

    assert labels == ["知识点@自然地理@地球仪", "知识点@自然地理@经纬网"]


def test_parse_or_recover_result_recovers_labels_from_broken_json():
    allowed = {
        "知识点@自然地理@地球仪",
        "知识点@自然地理@地球仪@经纬网",
        "知识点@人文地理@人口",
    }
    broken = (
        '{"candidate_labels":["自然地理@地球仪@经纬网",'
        '"知识点@人文地理@人口"'
    )

    result = parse_or_recover_result(broken, allowed)

    assert result == {
        "candidate_labels": [
            "知识点@自然地理@地球仪@经纬网",
            "知识点@人文地理@人口",
        ]
    }


@pytest.mark.parametrize(
    "candidate_labels",
    [
        ["知识点@不存在"],
        ["知识点@标签一", "知识点@标签一"],
        [f"知识点@标签{i}" for i in range(21)],
    ],
)
def test_validate_result_rejects_invalid_candidates(candidate_labels):
    allowed = {"知识点@标签一"} | {f"知识点@标签{i}" for i in range(21)}

    with pytest.raises(ValueError):
        validate_result(
            {"candidate_labels": candidate_labels, "uncovered_topic": None},
            allowed,
        )
