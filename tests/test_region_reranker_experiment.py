from bio_geo_tagging.region_reranker_experiment import (
    build_candidate_pool,
    rerank_candidates,
)


def test_candidate_pool_merges_routes_exact_matches_and_original_results():
    result = {
        "regional_bm25_candidates": [
            {"rank": 1, "label_path": "甲"},
            {"rank": 2, "label_path": "乙"},
        ],
        "regional_bge_candidates": [
            {"rank": 1, "label_path": "乙"},
            {"rank": 2, "label_path": "丙"},
        ],
        "regional_exact_matches": [
            {"label_path": "丁", "matched_name": "丁地"}
        ],
        "regional_final_candidates": [{"label_path": "戊"}],
    }

    pool = build_candidate_pool(result, bm25_top_k=1, bge_top_k=1)
    by_label = {item["label_path"]: item for item in pool}

    assert set(by_label) == {"甲", "乙", "丁", "戊"}
    assert by_label["甲"]["bm25_rank"] == 1
    assert by_label["乙"]["bge_rank"] == 1
    assert by_label["丁"]["direct_name_match"] is True
    assert by_label["戊"]["original_selected"] is True


def test_reranking_preserves_direct_name_match_before_higher_model_score():
    candidates = [
        {"label_path": "直接命中", "direct_name_match": True},
        {"label_path": "语义候选", "direct_name_match": False},
        {"label_path": "低分候选", "direct_name_match": False},
    ]

    ranked = rerank_candidates(candidates, [-3.0, 8.0, 1.0], limit=2)

    assert [item["label_path"] for item in ranked] == [
        "直接命中",
        "语义候选",
    ]


def test_reranking_uses_model_score_within_same_evidence_level():
    candidates = [
        {"label_path": "甲", "direct_name_match": False},
        {"label_path": "乙", "direct_name_match": False},
    ]

    ranked = rerank_candidates(candidates, [0.2, 0.8], limit=2)

    assert [item["label_path"] for item in ranked] == ["乙", "甲"]
