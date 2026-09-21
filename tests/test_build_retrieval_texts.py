import json

from bio_geo_tagging.build_retrieval_texts import (
    build_record,
    build_review_record,
    load_source,
)


def test_load_source_normalizes_path_and_keywords(tmp_path):
    source = tmp_path / "definitions.jsonl"
    source.write_text(
        json.dumps(
            {
                "full_path": "知识点->地图->坡度",
                "existing_interpretation": {
                    "definition": "坡面倾斜程度。",
                    "keywords": "坡度、等高线疏密；坡度",
                    "exam_methods": "比较坡度大小。",
                    "distinction": "与相关计算区分。",
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_source(source)

    assert records[0]["label_path"] == "知识点@地图@坡度"
    assert records[0]["keywords"] == ["坡度", "等高线疏密"]


def test_build_record_uses_only_positive_source_fields():
    source = {
        "label_path": "知识点@地图@坡度",
        "definition": "坡面倾斜程度。",
        "keywords": ["坡度", "等高线疏密"],
        "exam_methods": "比较坡度大小。",
        "distinction": "与相关计算区分。",
    }

    result = build_record(source)

    assert result["positive_definition"] == "坡面倾斜程度。"
    assert result["assessment_scope"] == "比较坡度大小。"
    assert "相关计算" not in result["bm25_text"]
    assert "相关计算" not in result["embedding_text"]
    assert result["bm25_text"] == (
        "知识点@地图@坡度 坡度 等高线疏密 坡面倾斜程度。 比较坡度大小。"
    )


def test_missing_source_field_creates_review_record():
    source = {
        "label_path": "知识点@地图@坡度",
        "definition": "坡面倾斜程度。",
        "keywords": [],
        "exam_methods": "比较坡度大小。",
        "distinction": "",
    }

    assert build_review_record(source) == {
        "label_path": "知识点@地图@坡度",
        "review_reason": "原释义缺少字段：keywords",
    }
