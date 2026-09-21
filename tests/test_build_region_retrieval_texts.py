import json
from pathlib import Path

import pytest

from bio_geo_tagging.build_region_retrieval_texts import (
    build_record,
    build_region_retrieval_texts,
)


RESOURCE_PATH = (
    Path(__file__).parents[1] / "resources" / "geography-region-name-info.jsonl"
)


def test_checked_region_name_resource_has_all_78_labels():
    records = [
        json.loads(line)
        for line in RESOURCE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    built = [build_record(record, index) for index, record in enumerate(records, 1)]
    assert len(built) == 78
    assert len({record["label_path"] for record in built}) == 78
    assert all(len(name) >= 2 for record in built for name in record["exact_names"])
    assert all("气候" not in record["bm25_text"] for record in built)
    assert all("农业" not in record["bm25_text"] for record in built)


def test_build_record_uses_place_names_only():
    record = build_record(
        {
            "label_path": "知识点@世界地理@世界重要的地区@欧洲西部",
            "region_name": "欧洲西部",
            "region_type": "世界重要地区",
            "aliases": ["西欧"],
            "contained_places": ["英国", "法国", "德国", "荷兰"],
            "representative_places": ["鲁尔区", "莱茵河沿岸"],
            "parent_regions": ["欧洲"],
        },
        1,
    )
    assert record["exact_names"] == ["欧洲西部", "西欧"]
    assert "荷兰" in record["bm25_text"]
    assert "鲁尔区" in record["embedding_text"]
    assert "鲁尔区" not in record["bm25_text"]
    assert "经济发达" not in record["bm25_text"]
    assert "温带海洋性气候" not in record["embedding_text"]


def test_region_name_can_differ_from_label_name():
    record = build_record(
        {
            "label_path": "知识点@世界地理@世界主要的大洲@非洲概况",
            "region_name": "非洲",
            "region_type": "世界主要大洲",
            "aliases": [],
            "contained_places": ["埃及", "南非"],
            "representative_places": ["东非大裂谷"],
            "parent_regions": ["世界"],
        },
        1,
    )
    assert record["label_name"] == "非洲概况"
    assert record["region_name"] == "非洲"
    assert record["exact_names"] == ["非洲"]


def test_builder_requires_exact_taxonomy_coverage(tmp_path):
    label_path = "知识点@世界地理@世界重要的国家@英国"
    taxonomy = tmp_path / "taxonomy.jsonl"
    taxonomy.write_text(
        json.dumps({"knw_label": label_path}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    source = tmp_path / "regions.jsonl"
    source.write_text(
        json.dumps(
            {
                "label_path": label_path,
                "region_name": "英国",
                "region_type": "世界重要国家",
                "aliases": ["大不列颠", "英伦"],
                "contained_places": ["英格兰", "苏格兰"],
                "representative_places": ["伦敦"],
                "parent_regions": ["欧洲西部", "欧洲"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output.jsonl"
    summary = build_region_retrieval_texts(
        source, taxonomy, output, expected_count=1
    )
    built = json.loads(output.read_text(encoding="utf-8"))
    assert summary["taxonomy_region_labels"] == 1
    assert built["exact_names"] == ["英国", "大不列颠", "英伦"]

    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="区域名称信息与标签体系不一致"):
        build_region_retrieval_texts(source, taxonomy, output, expected_count=1)
