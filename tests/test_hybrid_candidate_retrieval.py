import json
from pathlib import Path

import pytest

from bio_geo_tagging.build_retrieval_index import build_bm25_index, write_json
from bio_geo_tagging.hybrid_candidate_retrieval import (
    exact_region_candidates,
    fuse_candidates,
    is_region_label_path,
    question_gold_labels,
    question_query_text,
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
        bm25_top_k=1,
        bge_top_k=1,
        batch_size=8,
        device="cpu",
        embedding_model=None,
        region_top_k=1,
        query_encoder=fake_encoder,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    expected = "知识点@中国地理@中国地理微区域@甲地"
    assert result["bm25_candidates"][0]["label_path"] == expected
    assert result["bge_candidates"][0]["label_path"] == expected
    assert result["bm25_missing_labels"] == []
    assert result["bge_missing_labels"] == []
    assert result["fusion_missing_labels"] == []
    assert summary["metrics"]["fusion"]["label_recall"] == 1.0
    assert summary["metrics"]["fusion"]["full_coverage_rate"] == 1.0
    assert summary["regional_label_count"] == 2
    assert summary["metrics"]["combined"]["label_recall"] == 1.0
