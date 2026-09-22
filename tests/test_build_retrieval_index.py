import json

import pytest

from bio_geo_tagging.build_retrieval_index import (
    build_bm25_index,
    build_region_phrase_bm25_index,
    load_retrieval_records,
    parse_ngram_sizes,
    tokenize_char_ngrams,
)


def test_load_retrieval_records_preserves_exact_names(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        json.dumps(
            {
                "label_path": "知识点@世界地理@世界重要的国家@英国",
                "bm25_text": "英国 大不列颠",
                "embedding_text": "英国，也称大不列颠。",
                "exact_names": ["英国", "大不列颠"],
                "contained_places": ["英格兰"],
                "representative_places": ["伦敦"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_retrieval_records(path)[0]["exact_names"] == [
        "英国",
        "大不列颠",
    ]
    assert load_retrieval_records(path)[0]["contained_places"] == ["英格兰"]
    assert load_retrieval_records(path)[0]["representative_places"] == ["伦敦"]


def test_parse_ngram_sizes_sorts_and_deduplicates():
    assert parse_ngram_sizes("2,1,2") == (1, 2)
    with pytest.raises(Exception):
        parse_ngram_sizes("0,2")


def test_tokenize_char_ngrams_keeps_bigrams_trigrams_short_segment_and_latin():
    tokens = tokenize_char_ngrams("海水温度 111 km", (2, 3))

    assert tokens == [
        "海水",
        "水温",
        "温度",
        "海水温",
        "水温度",
        "海水温度",
        "111",
        "km",
    ]


def test_tokenize_char_ngrams_does_not_keep_long_chinese_segment():
    segment = "一二三四五六七八九十甲乙丙"

    tokens = tokenize_char_ngrams(segment, (2, 3))

    assert segment not in tokens


def test_build_bm25_index_creates_postings_and_positive_idf():
    records = [
        {
            "label_path": "标签一",
            "bm25_text": "海水温度",
            "embedding_text": "海水温度。",
        },
        {
            "label_path": "标签二",
            "bm25_text": "海水盐度",
            "embedding_text": "海水盐度。",
        },
    ]

    index = build_bm25_index(records, (2,), 1.5, 0.75)

    assert index["document_count"] == 2
    assert index["document_lengths"] == [4, 4]
    assert index["postings"]["温度"] == [[0, 1]]
    assert index["postings"]["盐度"] == [[1, 1]]
    assert index["idf"]["海水"] > 0


def test_region_phrase_bm25_uses_complete_names_only():
    records = [
        {
            "label_path": "北京",
            "exact_names": ["北京", "北京市"],
            "contained_places": ["东城区"],
        },
        {
            "label_path": "黄淮海平原",
            "exact_names": ["黄淮海平原", "华北平原"],
            "contained_places": ["河北平原"],
        },
    ]
    index = build_region_phrase_bm25_index(records, 1.5, 0.75)

    assert index["tokenizer"] == "region_phrase"
    assert "北京" in index["postings"]
    assert "华北平原" in index["postings"]
    assert "北平" not in index["postings"]


def test_load_retrieval_records_rejects_duplicate_path(tmp_path):
    path = tmp_path / "labels.jsonl"
    record = {
        "label_path": "知识点@海水温度",
        "bm25_text": "海水温度",
        "embedding_text": "海水温度。",
    }
    path.write_text(
        json.dumps(record, ensure_ascii=False)
        + "\n"
        + json.dumps(record, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="标签路径重复"):
        load_retrieval_records(path)
