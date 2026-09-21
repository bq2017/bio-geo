import json
from pathlib import Path

import pytest

from bio_geo_tagging.build_retrieval_index import (
    build_bm25_index,
    build_region_phrase_bm25_index,
    write_json,
)
from bio_geo_tagging.hybrid_candidate_retrieval import (
    admitted_region_label_paths,
    bm25_scores,
    exact_region_candidates,
    fuse_candidates,
    is_region_label_path,
    question_gold_labels,
    question_query_text,
    question_region_exact_text,
    region_phrase_evidence,
    rank_fused_candidates,
    rank_region_candidates,
    run_retrieval,
)


def test_region_label_path_classification():
    assert is_region_label_path(
        "知识点@中国地理@中国地理微区域@云贵高原"
    )
    assert is_region_label_path(
        "知识点@世界地理@世界重要的国家@美国"
    )
    assert not is_region_label_path(
        "知识点@中国地理@中国地理全貌@中国农业"
    )
    assert not is_region_label_path("知识点@自然地理@地貌@地貌观察")


def test_exact_region_candidates_uses_leaf_name():
    labels = [
        {"label_path": "知识点@世界地理@世界重要的国家@美国"},
        {"label_path": "知识点@人文地理@农业@农业区位"},
    ]
    assert exact_region_candidates("分析美国产业结构", labels, [0]) == [
        {
            "label_path": "知识点@世界地理@世界重要的国家@美国",
            "matched_name": "美国",
        }
    ]


def test_exact_region_candidates_uses_aliases():
    labels = [
        {
            "label_path": "知识点@世界地理@世界重要的国家@英国",
            "exact_names": ["英国", "大不列颠", "英伦"],
        }
    ]
    assert exact_region_candidates("读大不列颠岛地形图", labels, [0]) == [
        {
            "label_path": "知识点@世界地理@世界重要的国家@英国",
            "matched_name": "大不列颠",
        }
    ]


def test_question_text_and_gold_cover_root_and_subquestions():
    question = {
        "stem": "公共材料",
        "options": "",
        "analysis": "根解析",
        "knw_labels": ["标签甲"],
        "sub_questions": [
            {
                "stem": "小题题干",
                "options": "A.选项",
                "analysis": "小题解析",
                "knw_labels": ["标签乙", "标签甲"],
            }
        ],
    }
    text = question_query_text(question)
    assert "公共材料" in text
    assert "小题题干" in text
    assert "标签甲" not in text
    assert question_gold_labels(question) == ["标签甲", "标签乙"]


def test_region_exact_text_includes_analysis_but_excludes_options():
    question = {
        "stem": "读北欧区域图",
        "options": "A.非洲 B.北美洲",
        "analysis": "非洲和北美洲均不符合题意",
        "sub_questions": [],
    }
    text = question_region_exact_text(question)
    assert "北欧" in text
    assert "非洲和北美洲均不符合题意" in text
    assert "A.非洲 B.北美洲" not in text


def test_region_admission_requires_direct_or_multiple_weak_clues():
    labels = [
        {
            "label_path": "知识点@中国地理@中国地理微区域@黄淮海平原",
            "exact_names": ["黄淮海平原", "华北平原"],
            "contained_places": ["河北平原"],
            "representative_places": ["黄河", "渤海"],
        },
        {
            "label_path": "知识点@中国地理@中国地理分区@北方地区@北京",
            "exact_names": ["北京", "北京市"],
            "contained_places": ["东城区"],
            "representative_places": ["太行山", "永定河"],
        },
    ]
    evidence = region_phrase_evidence(
        "太行山是黄土高原和华北平原的分界线",
        labels,
        [0, 1],
    )
    admitted = admitted_region_label_paths(
        [
            {"rank": 6, "label_path": labels[0]["label_path"], "score": 1.0},
            {"rank": 3, "label_path": labels[1]["label_path"], "score": 2.0},
        ],
        [
            {"rank": 5, "label_path": labels[0]["label_path"], "score": 0.6},
            {"rank": 12, "label_path": labels[1]["label_path"], "score": 0.5},
        ],
        evidence,
    )
    assert labels[0]["label_path"] in admitted
    assert labels[1]["label_path"] not in admitted


def test_region_phrase_bm25_does_not_match_character_fragments():
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
    scores = bm25_scores("太行山是黄土高原和华北平原的分界线", index)

    assert scores[0] == 0
    assert scores[1] > 0


def test_fusion_keeps_both_routes_and_deduplicates():
    bm25 = [
        {"rank": 1, "label_path": "甲", "score": 2.0},
        {"rank": 2, "label_path": "乙", "score": 1.0},
    ]
    bge = [
        {"rank": 1, "label_path": "乙", "score": 0.9},
        {"rank": 2, "label_path": "丙", "score": 0.8},
    ]
    assert fuse_candidates(bm25, bge) == [
        {"label_path": "甲", "bm25_rank": 1, "bge_rank": None},
        {"label_path": "乙", "bm25_rank": 2, "bge_rank": 1},
        {"label_path": "丙", "bm25_rank": None, "bge_rank": 2},
    ]


def test_region_fusion_applies_strict_limit():
    bm25 = [
        {"rank": 1, "label_path": "甲", "score": 2.0},
        {"rank": 2, "label_path": "乙", "score": 1.0},
    ]
    bge = [
        {"rank": 1, "label_path": "乙", "score": 0.9},
        {"rank": 2, "label_path": "丙", "score": 0.8},
    ]
    result = rank_region_candidates(
        bm25,
        bge,
        [{"label_path": "丙", "matched_name": "丙地"}],
        limit=2,
    )
    assert {item["label_path"] for item in result} == {"乙", "丙"}
    assert next(item for item in result if item["label_path"] == "丙")[
        "matched_name"
    ] == "丙地"
    assert len(result) == 2


def test_fusion_keeps_strong_single_route_above_two_weak_routes():
    result = rank_fused_candidates(
        {
            "bm25": [
                {"rank": 1, "label_path": "单路强", "score": 10.0},
                {"rank": 10, "label_path": "双路弱", "score": 1.0},
            ],
            "bge": [
                {"rank": 10, "label_path": "双路弱", "score": 0.4},
            ],
        },
        agreement_weight=0.25,
    )
    assert [item["label_path"] for item in result] == ["单路强", "双路弱"]
    assert result[0]["best_route_score"] == 1.0
    assert result[1]["support_count"] == 2


def make_index(index_dir: Path) -> None:
    np = pytest.importorskip("numpy")
    records = [
        {"label_path": "知识点@中国地理@中国地理微区域@甲地", "bm25_text": "甲地 河流 水文", "embedding_text": "甲地河流水文"},
        {"label_path": "知识点@乙", "bm25_text": "农业 区位", "embedding_text": "农业区位"},
        {"label_path": "知识点@世界地理@世界重要的国家@丙国", "bm25_text": "丙国 城市 空间", "embedding_text": "丙国城市空间"},
    ]
    index_dir.mkdir()
    with (index_dir / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for document_index, record in enumerate(records):
            handle.write(json.dumps({"document_index": document_index, **record}, ensure_ascii=False) + "\n")
    write_json(index_dir / "bm25-index.json", build_bm25_index(records, (1, 2), 1.5, 0.75))
    np.save(
        index_dir / "bge-embeddings.npy",
        np.asarray([[1.0, 0.0], [0.0, 1.0], [0.70710677, 0.70710677]], dtype=np.float32),
        allow_pickle=False,
    )
    write_json(
        index_dir / "manifest.json",
        {
            "label_count": 3,
            "labels_file": "labels.jsonl",
            "bm25": {"file": "bm25-index.json"},
            "embedding": {
                "file": "bge-embeddings.npy",
                "model": "fake-model",
                "query_instruction": "检索：",
            },
        },
    )


def make_region_index(index_dir: Path) -> None:
    np = pytest.importorskip("numpy")
    records = [
        {
            "label_path": "知识点@中国地理@中国地理微区域@甲地",
            "bm25_text": "甲地",
            "embedding_text": "区域名称：甲地。",
            "exact_names": ["甲地"],
        },
        {
            "label_path": "知识点@世界地理@世界重要的国家@丙国",
            "bm25_text": "丙国 丙岛",
            "embedding_text": "区域名称：丙国。别称：丙岛。",
            "exact_names": ["丙国", "丙岛"],
        },
    ]
    index_dir.mkdir()
    with (index_dir / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for document_index, record in enumerate(records):
            handle.write(
                json.dumps(
                    {"document_index": document_index, **record},
                    ensure_ascii=False,
                )
                + "\n"
            )
    write_json(
        index_dir / "bm25-index.json",
        build_region_phrase_bm25_index(records, 1.5, 0.75),
    )
    np.save(
        index_dir / "bge-embeddings.npy",
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        allow_pickle=False,
    )
    write_json(
        index_dir / "manifest.json",
        {
            "label_count": 2,
            "labels_file": "labels.jsonl",
            "bm25": {"file": "bm25-index.json"},
            "embedding": {
                "file": "bge-embeddings.npy",
                "model": "fake-model",
                "query_instruction": "检索：",
            },
        },
    )


def test_run_retrieval_writes_results_and_route_metrics(tmp_path):
    np = pytest.importorskip("numpy")
    index_dir = tmp_path / "index"
    make_index(index_dir)
    input_path = tmp_path / "questions.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "parent_id": "1",
                "question_id": "1",
                "stem": "河流水文特征",
                "options": "",
                "analysis": "",
                "knw_labels": ["知识点@中国地理@中国地理微区域@甲地"],
                "sub_questions": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_encoder(texts, model, instruction, batch_size, device):
        assert texts == ["河流水文特征"]
        assert instruction == "检索："
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    output_path = tmp_path / "results.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = run_retrieval(
        input_path,
        index_dir,
        output_path,
        summary_path,
        nonregion_candidate_limit=1,
        region_candidate_limit=1,
        batch_size=8,
        device="cpu",
        embedding_model=None,
        query_encoder=fake_encoder,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["bm25_candidates"][0]["label_path"] == "知识点@乙"
    assert result["bge_candidates"][0]["label_path"] == "知识点@乙"
    assert result["fusion_missing_labels"] == []
    assert summary["metrics"]["nonregional_final"]["label_recall"] is None
    assert summary["regional_label_count"] == 2
    assert summary["metrics"]["combined"]["label_recall"] == 1.0
    assert summary["metrics"]["regional_final"]["label_recall"] == 1.0
    assert result["candidate_count"] <= summary["maximum_combined_candidates"]


def test_run_retrieval_uses_separate_region_index(tmp_path):
    np = pytest.importorskip("numpy")
    index_dir = tmp_path / "index"
    region_index_dir = tmp_path / "region-index"
    make_index(index_dir)
    make_region_index(region_index_dir)
    input_path = tmp_path / "questions.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "parent_id": "1",
                "question_id": "1",
                "stem": "读丙岛地形图",
                "options": "",
                "analysis": "",
                "knw_labels": ["知识点@世界地理@世界重要的国家@丙国"],
                "sub_questions": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_encoder(texts, model, instruction, batch_size, device):
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    output_path = tmp_path / "results.jsonl"
    summary = run_retrieval(
        input_path=input_path,
        index_dir=index_dir,
        region_index_dir=region_index_dir,
        output_path=output_path,
        summary_path=tmp_path / "summary.json",
        nonregion_candidate_limit=1,
        region_candidate_limit=1,
        batch_size=8,
        device="cpu",
        embedding_model=None,
        query_encoder=fake_encoder,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    expected = "知识点@世界地理@世界重要的国家@丙国"
    assert result["regional_bm25_candidates"][0]["label_path"] == expected
    assert result["regional_exact_matches"] == [
        {"label_path": expected, "matched_name": "丙岛"}
    ]
    assert result["regional_final_candidates"][0]["label_path"] == expected
    assert summary["regional_index"] == str(region_index_dir)


def test_irrelevant_region_candidates_are_not_forced_into_final_results(tmp_path):
    np = pytest.importorskip("numpy")
    index_dir = tmp_path / "index"
    region_index_dir = tmp_path / "region-index"
    make_index(index_dir)
    make_region_index(region_index_dir)
    input_path = tmp_path / "questions.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "parent_id": "1",
                "question_id": "1",
                "stem": "无关内容",
                "options": "",
                "analysis": "",
                "knw_labels": ["知识点@乙"],
                "sub_questions": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_encoder(texts, model, instruction, batch_size, device):
        return np.asarray([[0.0, 0.0]], dtype=np.float32)

    output_path = tmp_path / "results.jsonl"
    run_retrieval(
        input_path=input_path,
        index_dir=index_dir,
        region_index_dir=region_index_dir,
        output_path=output_path,
        summary_path=tmp_path / "summary.json",
        nonregion_candidate_limit=1,
        region_candidate_limit=1,
        batch_size=8,
        device="cpu",
        embedding_model=None,
        query_encoder=fake_encoder,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["regional_final_candidates"] == []


def test_retrieval_rejects_invalid_agreement_weight(tmp_path):
    with pytest.raises(ValueError, match="agreement_weight"):
        run_retrieval(
            input_path=tmp_path / "unused.jsonl",
            index_dir=tmp_path / "unused-index",
            output_path=tmp_path / "unused-output.jsonl",
            summary_path=tmp_path / "unused-summary.json",
            nonregion_candidate_limit=25,
            region_candidate_limit=5,
            agreement_weight=1.0,
            batch_size=8,
            device="cpu",
            embedding_model=None,
        )
