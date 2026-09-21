"""Retrieve geography label candidates with cached BM25 and BGE indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .build_retrieval_index import tokenize_char_ngrams


QUESTION_TEXT_FIELDS = (
    "stem",
    "options",
    "answer",
    "analysis",
    "explanation",
    "image_description",
)

REGION_EXACT_TEXT_FIELDS = (
    "stem",
    "answer",
    "image_description",
)

CHINA_REGION_BRANCHES = {"中国地理分区", "中国地理微区域"}
WORLD_REGION_BRANCHES = {
    "世界主要的大洲",
    "世界重要的地区",
    "世界重要的国家",
    "世界地理微区域",
}
DEFAULT_REGION_RECALL_KS = (1, 3, 5, 10, 20)


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}第{line_number}行不是JSON对象")
            yield line_number, value


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(as_text(item) for item in value if as_text(item))
    if isinstance(value, dict):
        return "\n".join(
            f"{as_text(key)} {as_text(item)}".strip()
            for key, item in value.items()
            if as_text(item)
        )
    return str(value).strip()


def question_query_text(question: dict[str, Any]) -> str:
    parts: list[str] = []

    def append_fields(record: dict[str, Any]) -> None:
        for field in QUESTION_TEXT_FIELDS:
            text = as_text(record.get(field))
            if text:
                parts.append(text)

    append_fields(question)
    sub_questions = question.get("sub_questions") or []
    if not isinstance(sub_questions, list) or not all(
        isinstance(item, dict) for item in sub_questions
    ):
        raise ValueError("sub_questions必须是对象数组")
    for sub_question in sub_questions:
        append_fields(sub_question)
    return "\n".join(parts)


def question_region_exact_text(question: dict[str, Any]) -> str:
    """Return fields where a place-name occurrence is direct question evidence."""
    parts: list[str] = []

    def append_fields(record: dict[str, Any]) -> None:
        for field in REGION_EXACT_TEXT_FIELDS:
            text = as_text(record.get(field))
            if text:
                parts.append(text)

    append_fields(question)
    for sub_question in question.get("sub_questions") or []:
        append_fields(sub_question)
    return "\n".join(parts)


def question_gold_labels(question: dict[str, Any]) -> list[str]:
    labels: list[str] = []

    def append_labels(record: dict[str, Any]) -> None:
        value = record.get("knw_labels") or []
        if not isinstance(value, list) or not all(
            isinstance(label, str) for label in value
        ):
            raise ValueError("knw_labels必须是字符串数组")
        labels.extend(label.strip() for label in value if label.strip())

    append_labels(question)
    for sub_question in question.get("sub_questions") or []:
        append_labels(sub_question)
    return list(dict.fromkeys(labels))


def load_labels(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, record in read_jsonl(path):
        document_index = record.get("document_index")
        label_path = as_text(record.get("label_path"))
        if document_index != len(records):
            raise ValueError(
                f"标签映射第{line_number}行document_index不连续：{document_index}"
            )
        if not label_path:
            raise ValueError(f"标签映射第{line_number}行缺少label_path")
        records.append(record)
    return records


def load_index(
    index_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], Any]:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("缺少numpy，请安装项目的retrieval可选依赖") from error

    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    labels = load_labels(index_dir / manifest["labels_file"])
    bm25 = json.loads(
        (index_dir / manifest["bm25"]["file"]).read_text(encoding="utf-8")
    )
    embeddings = np.load(
        index_dir / manifest["embedding"]["file"], allow_pickle=False
    )
    label_count = int(manifest["label_count"])
    if len(labels) != label_count or bm25["document_count"] != label_count:
        raise ValueError("标签映射、BM25索引与manifest的标签数量不一致")
    if embeddings.shape[0] != label_count:
        raise ValueError("BGE向量与manifest的标签数量不一致")
    return manifest, labels, bm25, embeddings


def bm25_scores(query: str, index: dict[str, Any]) -> list[float]:
    scores = [0.0] * int(index["document_count"])
    terms = set(tokenize_char_ngrams(query, tuple(index["ngram_sizes"])))
    k1 = float(index["k1"])
    b = float(index["b"])
    average_length = float(index["average_document_length"])
    document_lengths = index["document_lengths"]
    for term in terms:
        term_idf = index["idf"].get(term)
        if term_idf is None:
            continue
        for document_index, frequency in index["postings"][term]:
            denominator = frequency + k1 * (
                1 - b + b * document_lengths[document_index] / average_length
            )
            scores[document_index] += (
                float(term_idf) * frequency * (k1 + 1) / denominator
            )
    return scores


def is_region_label_path(label_path: str) -> bool:
    parts = label_path.split("@")
    if len(parts) < 4 or parts[0] != "知识点":
        return False
    if parts[1] == "中国地理":
        return parts[2] in CHINA_REGION_BRANCHES
    if parts[1] == "世界地理":
        return parts[2] in WORLD_REGION_BRANCHES
    return False


def top_indices_from_pool(
    scores: Any, document_indices: list[int], limit: int
) -> list[int]:
    return sorted(
        document_indices,
        key=lambda index: (-float(scores[index]), index),
    )[:limit]


def exact_region_candidates(
    query: str,
    labels: list[dict[str, Any]],
    region_indices: list[int],
) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []
    for index in region_indices:
        label_path = labels[index]["label_path"]
        label_name = label_path.rsplit("@", 1)[-1]
        exact_names = labels[index].get("exact_names") or [label_name]
        matched_name = next(
            (
                name
                for name in sorted(exact_names, key=len, reverse=True)
                if name and name in query
            ),
            None,
        )
        if matched_name:
            matches.append(
                {"label_path": label_path, "matched_name": matched_name}
            )
    return matches


def encode_queries(
    texts: list[str],
    model_name_or_path: str,
    instruction: str,
    batch_size: int,
    device: str | None,
) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "缺少sentence-transformers，请安装项目的retrieval可选依赖"
        ) from error

    arguments: dict[str, Any] = {}
    if device:
        arguments["device"] = device
    model = SentenceTransformer(model_name_or_path, **arguments)
    instructed = [f"{instruction}{text}" for text in texts]
    return model.encode(
        instructed,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def ranked_candidates(
    indices: list[int], scores: Any, labels: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "label_path": labels[index]["label_path"],
            "score": round(float(scores[index]), 8),
        }
        for rank, index in enumerate(indices, start=1)
    ]


def fuse_candidates(
    bm25_candidates: list[dict[str, Any]],
    bge_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for source, candidates in (
        ("bm25", bm25_candidates),
        ("bge", bge_candidates),
    ):
        for candidate in candidates:
            label = candidate["label_path"]
            if label not in by_label:
                by_label[label] = {
                    "label_path": label,
                    "bm25_rank": None,
                    "bge_rank": None,
                }
                order.append(label)
            by_label[label][f"{source}_rank"] = candidate["rank"]
    return [by_label[label] for label in order]


def add_exact_region_matches(
    candidates: list[dict[str, Any]],
    exact_matches: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_label = {item["label_path"]: item for item in candidates}
    result = list(candidates)
    for match in exact_matches:
        label = match["label_path"]
        if label in by_label:
            by_label[label]["exact_name_match"] = True
            continue
        item = {
            "label_path": label,
            "bm25_rank": None,
            "bge_rank": None,
            "exact_name_match": True,
        }
        result.append(item)
        by_label[label] = item
    return result


def rank_fused_candidates(
    route_candidates: dict[str, list[dict[str, Any]]],
    exact_matches: list[dict[str, str]] | None = None,
    agreement_weight: float = 0.25,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Rank labels by their strongest route; other routes give bounded support."""
    if not 0 <= agreement_weight < 1:
        raise ValueError("agreement_weight必须大于等于0且小于1")
    by_label: dict[str, dict[str, Any]] = {}
    evidence_scores: dict[str, list[float]] = {}
    for route, candidates in route_candidates.items():
        for candidate in candidates:
            label = candidate["label_path"]
            item = by_label.setdefault(label, {"label_path": label})
            item[f"{route}_rank"] = candidate["rank"]
            item[f"{route}_raw_score"] = candidate["score"]
            evidence_scores.setdefault(label, []).append(1.0 / candidate["rank"])

    for match in exact_matches or []:
        label = match["label_path"]
        item = by_label.setdefault(label, {"label_path": label})
        item["exact_name_match"] = True
        item["matched_name"] = match["matched_name"]
        evidence_scores.setdefault(label, []).append(1.0)

    candidates = list(by_label.values())
    for candidate in candidates:
        scores = sorted(evidence_scores[candidate["label_path"]], reverse=True)
        best_score = scores[0]
        second_score = scores[1] if len(scores) > 1 else 0.0
        candidate["best_route_score"] = round(best_score, 8)
        candidate["second_route_score"] = round(second_score, 8)
        candidate["support_count"] = len(scores)
        candidate["fusion_score"] = round(
            best_score + agreement_weight * second_score,
            8,
        )
    candidates.sort(
        key=lambda item: (
            -item["fusion_score"],
            -item["best_route_score"],
            -item["second_route_score"],
            -item["support_count"],
            item["label_path"],
        )
    )
    if limit is not None:
        return candidates[:limit]
    return candidates


def rank_region_candidates(
    bm25_candidates: list[dict[str, Any]],
    bge_candidates: list[dict[str, Any]],
    exact_matches: list[dict[str, str]],
    limit: int,
    agreement_weight: float = 0.25,
) -> list[dict[str, Any]]:
    return rank_fused_candidates(
        {
            "region_bm25": bm25_candidates,
            "region_bge": bge_candidates,
        },
        exact_matches=exact_matches,
        agreement_weight=agreement_weight,
        limit=limit,
    )


def merge_candidate_lists(
    primary: list[dict[str, Any]], secondary: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result = list(primary)
    seen = {item["label_path"] for item in result}
    for item in secondary:
        if item["label_path"] not in seen:
            result.append(item)
            seen.add(item["label_path"])
    return result


def metric_record() -> dict[str, int]:
    return {
        "questions_with_gold": 0,
        "fully_covered_questions": 0,
        "gold_labels": 0,
        "recalled_gold_labels": 0,
    }


def update_metrics(
    metrics: dict[str, int], gold_labels: list[str], candidate_labels: set[str]
) -> None:
    if not gold_labels:
        return
    gold = set(gold_labels)
    recalled = gold & candidate_labels
    metrics["questions_with_gold"] += 1
    metrics["fully_covered_questions"] += int(gold <= candidate_labels)
    metrics["gold_labels"] += len(gold)
    metrics["recalled_gold_labels"] += len(recalled)


def finalize_metrics(metrics: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = dict(metrics)
    questions = metrics["questions_with_gold"]
    labels = metrics["gold_labels"]
    result["full_coverage_rate"] = (
        round(metrics["fully_covered_questions"] / questions, 6)
        if questions
        else None
    )
    result["label_recall"] = (
        round(metrics["recalled_gold_labels"] / labels, 6) if labels else None
    )
    return result


def run_retrieval(
    input_path: Path,
    index_dir: Path,
    output_path: Path,
    summary_path: Path,
    bm25_top_k: int,
    bge_top_k: int,
    batch_size: int,
    device: str | None,
    embedding_model: str | None,
    region_index_dir: Path | None = None,
    region_top_k: int = 10,
    max_candidates: int = 40,
    agreement_weight: float = 0.25,
    region_bm25_min_score: float = 0.0,
    region_bge_min_score: float = 0.4,
    region_recall_ks: tuple[int, ...] = DEFAULT_REGION_RECALL_KS,
    query_encoder: Callable[[list[str], str, str, int, str | None], Any] = encode_queries,
) -> dict[str, Any]:
    if (
        bm25_top_k <= 0
        or bge_top_k <= 0
        or region_top_k <= 0
        or max_candidates <= 0
    ):
        raise ValueError("各候选参数必须大于0")
    if not 0 <= agreement_weight < 1:
        raise ValueError("agreement_weight必须大于等于0且小于1")
    manifest, labels, bm25, label_embeddings = load_index(index_dir)
    region_manifest: dict[str, Any] | None = None
    region_labels: list[dict[str, Any]]
    region_bm25: dict[str, Any]
    region_embeddings: Any
    questions: list[dict[str, Any]] = []
    query_texts: list[str] = []
    for line_number, question in read_jsonl(input_path):
        query = question_query_text(question)
        if not query:
            raise ValueError(f"题目文件第{line_number}行没有可用于检索的文本")
        questions.append(question)
        query_texts.append(query)

    model_name = embedding_model or manifest["embedding"]["model"]
    instruction = manifest["embedding"].get("query_instruction", "")
    query_embeddings = query_encoder(
        query_texts, model_name, instruction, batch_size, device
    )
    if query_embeddings.shape[0] != len(questions):
        raise ValueError("题目向量数量与输入题目数量不一致")

    allowed_labels = {record["label_path"] for record in labels}
    global_region_indices = [
        index
        for index, record in enumerate(labels)
        if is_region_label_path(record["label_path"])
    ]
    if not global_region_indices:
        raise ValueError("标签索引中没有识别到中国地理或世界地理区域标签")
    expected_region_paths = {
        labels[index]["label_path"] for index in global_region_indices
    }
    nonregion_indices = [
        index for index in range(len(labels)) if index not in global_region_indices
    ]
    if region_index_dir is None:
        region_labels = labels
        region_bm25 = bm25
        region_embeddings = label_embeddings
        region_indices = global_region_indices
        region_query_embeddings = query_embeddings
    else:
        (
            region_manifest,
            region_labels,
            region_bm25,
            region_embeddings,
        ) = load_index(region_index_dir)
        actual_region_paths = {record["label_path"] for record in region_labels}
        if actual_region_paths != expected_region_paths:
            missing = sorted(expected_region_paths - actual_region_paths)
            extra = sorted(actual_region_paths - expected_region_paths)
            raise ValueError(
                f"区域索引与全量索引中的区域标签不一致；缺少={missing}；多出={extra}"
            )
        region_indices = list(range(len(region_labels)))
        region_model = region_manifest["embedding"]["model"]
        region_instruction = region_manifest["embedding"].get(
            "query_instruction", ""
        )
        if region_model == model_name and region_instruction == instruction:
            region_query_embeddings = query_embeddings
        else:
            region_query_embeddings = query_encoder(
                query_texts,
                region_model,
                region_instruction,
                batch_size,
                device,
            )
        if region_query_embeddings.shape[1] != region_embeddings.shape[1]:
            raise ValueError("区域题目向量与区域标签向量维度不一致")
    effective_region_ks = tuple(
        sorted({value for value in region_recall_ks if 0 < value <= region_top_k})
    )
    if region_top_k not in effective_region_ks:
        effective_region_ks += (region_top_k,)
        effective_region_ks = tuple(sorted(effective_region_ks))

    metrics = {
        name: metric_record()
        for name in ("bm25", "bge", "fusion", "combined")
    }
    regional_metrics = {
        route: {value: metric_record() for value in effective_region_ks}
        for route in ("bm25", "bge", "fusion")
    }
    regional_exact_metrics = metric_record()
    regional_final_metrics = metric_record()
    unknown_gold_labels: dict[str, int] = {}
    fused_counts: list[int] = []
    regional_counts: list[int] = []
    combined_counts: list[int] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for index, (question, query) in enumerate(zip(questions, query_texts)):
            sparse_scores = bm25_scores(query, bm25)
            dense_scores = label_embeddings @ query_embeddings[index]
            sparse = ranked_candidates(
                top_indices_from_pool(
                    sparse_scores,
                    nonregion_indices,
                    min(bm25_top_k, len(nonregion_indices)),
                ),
                sparse_scores,
                labels,
            )
            dense = ranked_candidates(
                top_indices_from_pool(
                    dense_scores,
                    nonregion_indices,
                    min(bge_top_k, len(nonregion_indices)),
                ),
                dense_scores,
                labels,
            )
            fused = rank_fused_candidates(
                {"bm25": sparse, "bge": dense},
                agreement_weight=agreement_weight,
            )
            regional_sparse_scores = (
                sparse_scores
                if region_index_dir is None
                else bm25_scores(query, region_bm25)
            )
            regional_dense_scores = (
                dense_scores
                if region_index_dir is None
                else region_embeddings @ region_query_embeddings[index]
            )
            regional_sparse_full = ranked_candidates(
                top_indices_from_pool(
                    regional_sparse_scores,
                    region_indices,
                    min(region_top_k, len(region_indices)),
                ),
                regional_sparse_scores,
                region_labels,
            )
            regional_dense_full = ranked_candidates(
                top_indices_from_pool(
                    regional_dense_scores,
                    region_indices,
                    min(region_top_k, len(region_indices)),
                ),
                regional_dense_scores,
                region_labels,
            )
            regional_sparse = [
                item
                for item in regional_sparse_full
                if item["score"] > region_bm25_min_score
            ]
            regional_dense = [
                item
                for item in regional_dense_full
                if item["score"] >= region_bge_min_score
            ]
            regional_exact = exact_region_candidates(
                question_region_exact_text(question),
                region_labels,
                region_indices,
            )
            regional_fused = rank_fused_candidates(
                {
                    "region_bm25": regional_sparse,
                    "region_bge": regional_dense,
                },
                exact_matches=regional_exact,
                agreement_weight=agreement_weight,
            )
            combined = rank_fused_candidates(
                {
                    "bm25": sparse,
                    "bge": dense,
                    "region_bm25": regional_sparse,
                    "region_bge": regional_dense,
                },
                exact_matches=regional_exact,
                agreement_weight=agreement_weight,
                limit=max_candidates,
            )
            for global_rank, candidate in enumerate(combined, start=1):
                candidate["global_rank"] = global_rank
            gold = question_gold_labels(question)
            for label in gold:
                if label not in allowed_labels:
                    unknown_gold_labels[label] = unknown_gold_labels.get(label, 0) + 1
            known_gold = [label for label in gold if label in allowed_labels]
            nonregional_gold = [
                label for label in known_gold if not is_region_label_path(label)
            ]
            sparse_labels = {item["label_path"] for item in sparse}
            dense_labels = {item["label_path"] for item in dense}
            fused_labels = {item["label_path"] for item in fused}
            combined_labels = {item["label_path"] for item in combined}
            update_metrics(metrics["bm25"], nonregional_gold, sparse_labels)
            update_metrics(metrics["bge"], nonregional_gold, dense_labels)
            update_metrics(metrics["fusion"], nonregional_gold, fused_labels)
            update_metrics(metrics["combined"], known_gold, combined_labels)

            regional_gold = [
                label for label in known_gold if is_region_label_path(label)
            ]
            exact_labels = {item["label_path"] for item in regional_exact}
            regional_fused_labels = {
                item["label_path"] for item in regional_fused
            }
            regional_final = [
                item
                for item in combined
                if is_region_label_path(item["label_path"])
            ]
            regional_final_labels = {
                item["label_path"] for item in regional_final
            }
            update_metrics(
                regional_exact_metrics, regional_gold, exact_labels
            )
            update_metrics(
                regional_final_metrics, regional_gold, regional_final_labels
            )
            for value in effective_region_ks:
                regional_sparse_labels = {
                    item["label_path"] for item in regional_sparse_full[:value]
                }
                regional_dense_labels = {
                    item["label_path"] for item in regional_dense_full[:value]
                }
                update_metrics(
                    regional_metrics["bm25"][value],
                    regional_gold,
                    regional_sparse_labels,
                )
                update_metrics(
                    regional_metrics["bge"][value],
                    regional_gold,
                    regional_dense_labels,
                )
                update_metrics(
                    regional_metrics["fusion"][value],
                    regional_gold,
                    regional_sparse_labels | regional_dense_labels | exact_labels,
                )
            fused_counts.append(len(fused))
            regional_counts.append(len(regional_final))
            combined_counts.append(len(combined))
            result = {
                "question_id": as_text(question.get("question_id")),
                "parent_id": as_text(question.get("parent_id")),
                "knw_labels": gold,
                "unmapped_gold_labels": [
                    label for label in gold if label not in allowed_labels
                ],
                "bm25_candidates": sparse,
                "bm25_missing_labels": [
                    label for label in nonregional_gold if label not in sparse_labels
                ],
                "bge_candidates": dense,
                "bge_missing_labels": [
                    label for label in nonregional_gold if label not in dense_labels
                ],
                "fused_candidates": fused,
                "fusion_missing_labels": [
                    label for label in nonregional_gold if label not in fused_labels
                ],
                "regional_exact_matches": regional_exact,
                "regional_bm25_candidates": regional_sparse_full,
                "regional_bge_candidates": regional_dense_full,
                "regional_fused_candidates": regional_fused,
                "regional_fused_missing_labels": [
                    label
                    for label in regional_gold
                    if label not in regional_fused_labels
                ],
                "regional_final_candidates": regional_final,
                "regional_final_missing_labels": [
                    label
                    for label in regional_gold
                    if label not in regional_final_labels
                ],
                "combined_candidates": combined,
                "combined_missing_labels": [
                    label for label in known_gold if label not in combined_labels
                ],
                "candidate_count": len(combined),
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {
        "input_questions": len(questions),
        "label_count": len(labels),
        "nonregional_label_count": len(nonregion_indices),
        "bm25_top_k": bm25_top_k,
        "bge_top_k": bge_top_k,
        "regional_label_count": len(region_indices),
        "regional_index": str(region_index_dir) if region_index_dir else None,
        "region_top_k": region_top_k,
        "max_candidates": max_candidates,
        "agreement_weight": agreement_weight,
        "region_bm25_min_score": region_bm25_min_score,
        "region_bge_min_score": region_bge_min_score,
        "embedding_model": model_name,
        "metrics": {name: finalize_metrics(value) for name, value in metrics.items()},
        "regional_metrics": {
            "exact_name_match": finalize_metrics(regional_exact_metrics),
            "final_candidates": finalize_metrics(regional_final_metrics),
            **{
                route: {
                    str(value): finalize_metrics(route_metrics[value])
                    for value in effective_region_ks
                }
                for route, route_metrics in regional_metrics.items()
            },
        },
        "fused_candidate_count": {
            "average": round(sum(fused_counts) / len(fused_counts), 6)
            if fused_counts
            else None,
            "minimum": min(fused_counts) if fused_counts else None,
            "maximum": max(fused_counts) if fused_counts else None,
        },
        "regional_candidate_count": {
            "average": round(sum(regional_counts) / len(regional_counts), 6)
            if regional_counts
            else None,
            "minimum": min(regional_counts) if regional_counts else None,
            "maximum": max(regional_counts) if regional_counts else None,
        },
        "combined_candidate_count": {
            "average": round(sum(combined_counts) / len(combined_counts), 6)
            if combined_counts
            else None,
            "minimum": min(combined_counts) if combined_counts else None,
            "maximum": max(combined_counts) if combined_counts else None,
        },
        "unknown_gold_label_counts": unknown_gold_labels,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cached BM25 and BGE retrieval for complete questions"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--region-index-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--bm25-top-k", type=int, default=12)
    parser.add_argument("--bge-top-k", type=int, default=23)
    parser.add_argument("--region-top-k", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=40)
    parser.add_argument("--agreement-weight", type=float, default=0.25)
    parser.add_argument("--region-bm25-min-score", type=float, default=0.0)
    parser.add_argument("--region-bge-min-score", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--embedding-model")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size必须大于0")
    summary = run_retrieval(
        input_path=args.input,
        index_dir=args.index_dir,
        region_index_dir=args.region_index_dir,
        output_path=args.output,
        summary_path=args.summary_output,
        bm25_top_k=args.bm25_top_k,
        bge_top_k=args.bge_top_k,
        batch_size=args.batch_size,
        device=args.device,
        embedding_model=args.embedding_model,
        region_top_k=args.region_top_k,
        max_candidates=args.max_candidates,
        agreement_weight=args.agreement_weight,
        region_bm25_min_score=args.region_bm25_min_score,
        region_bge_min_score=args.region_bge_min_score,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
