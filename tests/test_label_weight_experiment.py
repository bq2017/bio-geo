import json

from bio_geo_tagging.label_weight_experiment import (
    build_weighted_bm25_text,
    load_definition_fields,
    split_name,
    weighted_records,
)


def test_load_definition_fields_and_build_weighted_text(tmp_path):
    path = tmp_path / "definitions.jsonl"
    path.write_text(
        json.dumps(
            {
                "label_path": "知识点@自然地理@水循环",
                "positive_definition": "水在地理环境中循环运动",
                "keywords": ["蒸发", "降水"],
                "assessment_scope": "判断水循环环节",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    fields = load_definition_fields(path)["知识点@自然地理@水循环"]
    text = build_weighted_bm25_text(
        fields,
        {
            "label_name": 2,
            "label_path": 0,
            "definition": 1,
            "keywords": 3,
            "assessment_scope": 1,
        },
    )

    assert text.split().count("水循环") == 2
    assert text.split().count("蒸发") == 3
    assert "知识点@自然地理@水循环" not in text
    assert "水在地理环境中循环运动" in text


def test_weighted_records_change_only_ordinary_labels():
    ordinary = {
        "document_index": 0,
        "label_path": "知识点@自然地理@水循环",
        "bm25_text": "原普通文本",
        "embedding_text": "普通向量文本",
    }
    comprehensive = {
        "document_index": 1,
        "label_path": "知识点@自然地理@自然地理综合",
        "bm25_text": "专用综合文本",
        "embedding_text": "综合向量文本",
    }
    weights = {
        "label_name": 1,
        "label_path": 0,
        "definition": 1,
        "keywords": 0,
        "assessment_scope": 0,
    }
    definitions = {
        ordinary["label_path"]: {
            "label_name": "水循环",
            "label_path": ordinary["label_path"],
            "definition": "水循环定义",
            "keywords": "",
            "assessment_scope": "",
        }
    }

    result = weighted_records(
        [ordinary, comprehensive], definitions, [0], weights
    )

    assert result[0]["bm25_text"] == "水循环 水循环定义"
    assert result[0]["embedding_text"] == "普通向量文本"
    assert result[1]["bm25_text"] == "专用综合文本"


def test_split_name_is_deterministic():
    assert split_name("question-1") == split_name("question-1")
    assert split_name("question-1") in {"development", "validation"}
