import json

import pytest

from bio_geo_tagging.build_retrieval_index import (
    build_bm25_index,
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


def test_parse_ngram_sizes_sorts_and_deduplicates():
    assert parse_ngram_sizes("2,1,2") == (1, 2)
    with pytest.raises(Exception):
        parse_ngram_sizes("0,2")


def test_tokenize_char_ngrams_keeps_chinese_unigrams_bigrams_and_latin():
    tokens = tokenize_char_ngrams("海水温度 111 km", (1, 2))

    assert tokens == [
        "海",
        "水",
        "温",
        "度",
        "海水",
        "水温",
        "温度",
        "111",
        "km",
    ]


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
    assert index["document_lengths"] == [3, 3]
    assert index["postings"]["温度"] == [[0, 1]]
    assert index["postings"]["盐度"] == [[1, 1]]
    assert index["idf"]["海水"] > 0


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
