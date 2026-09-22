"""Retrieve geography label candidates with cached BM25 and BGE indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .build_retrieval_index import normalize_phrase, tokenize_char_ngrams


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
    "analysis",
    "explanation",
    "image_description",
)

CHINA_REGION_BRANCHES = {"中国地理分区", "中国地理微区域"}
WORLD_REGION_BRANCHES = {
    "世界主要的大洲",
    "世界重要的地区",
    "世界重要的国家",
    "世界地理微区域",
}
NONREGION_DIAGNOSTIC_LIMIT = 50
NONREGION_BM25_PRIMARY_QUOTA = 21
REGION_DIAGNOSTIC_LIMIT = 20
REGION_REPRESENTATIVE_BGE_MAX_RANK = 10
REGION_BGE_ONLY_MAX_RANK = 5
COMPREHENSIVE_DIAGNOSTIC_LIMIT = 20

# These labels contain “综合” in the leaf name, but describe a concrete topic or
# question type rather than an umbrella label. They stay in the original V3 pool.
NON_UMBRELLA_COMPREHENSIVE_LABELS = {
    "知识点@区域发展@区域发展@生态脆弱区的综合治理",
    "知识点@区域发展@区域发展@北方农牧交错带土地退化的综合治理",
    "知识点@区域发展@区域协调@流域综合开发",
    "知识点@选修地理（旧）@旅游地理综合题",
    "知识点@选修地理（旧）@环境保护综合题",
}


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
    """Return fields where a place-name occurrence can support regional retrieval."""
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
    tokenizer = index.get("tokenizer")
    if tokenizer == "unicode_char_ngram":
        terms = set(tokenize_char_ngrams(query, tuple(index["ngram_sizes"])))
    elif tokenizer == "region_phrase":
        normalized_query = normalize_phrase(query)
        terms = {
            phrase for phrase in index["idf"] if phrase in normalized_query
        }
    else:
        raise ValueError(f"不支持的BM25分词器：{tokenizer}")
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


def is_strict_comprehensive_label_path(label_path: str) -> bool:
    """Return whether a label is an umbrella comprehensive label."""
    leaf_name = label_path.rsplit("@", 1)[-1]
    return (
        "综合" in leaf_name
        and label_path not in NON_UMBRELLA_COMPREHENSIVE_LABELS
    )


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


def region_phrase_evidence(
    query: str,
    labels: list[dict[str, Any]],
    region_indices: list[int],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index in region_indices:
        label = labels[index]
        label_path = label["label_path"]
        label_name = label_path.rsplit("@", 1)[-1]
        fields = {
            "direct_names": label.get("exact_names") or [label_name],
            "contained_places": label.get("contained_places") or [],
            "representative_places": label.get("representative_places") or [],
        }
        matched = {
            field: list(
                dict.fromkeys(
                    name for name in names if name and name in query
                )
            )
            for field, names in fields.items()
        }
        if any(matched.values()):
            evidence.append({"label_path": label_path, **matched})
    return evidence


def admitted_region_label_paths(
    bm25_candidates: list[dict[str, Any]],
    bge_candidates: list[dict[str, Any]],
    phrase_evidence: list[dict[str, Any]],
) -> set[str]:
    bm25_ranks = {
        item["label_path"]: item["rank"] for item in bm25_candidates
    }
    bge_ranks = {
        item["label_path"]: item["rank"] for item in bge_candidates
    }
    evidence_by_label = {
        item["label_path"]: item for item in phrase_evidence
    }
    admitted: set[str] = set()
    for label_path in bm25_ranks.keys() | bge_ranks.keys() | evidence_by_label.keys():
        item = evidence_by_label.get(label_path, {})
        has_direct_evidence = bool(item.get("direct_names"))
        has_contained_place = bool(item.get("contained_places"))
        has_representative_place = bool(item.get("representative_places"))
        bge_rank = bge_ranks.get(label_path)
        has_supported_representative_place = (
            has_representative_place
            and bge_rank is not None
            and bge_rank <= REGION_REPRESENTATIVE_BGE_MAX_RANK
        )
        has_strong_bge_support = (
            bge_rank is not None and bge_rank <= REGION_BGE_ONLY_MAX_RANK
        )
        if (
            has_direct_evidence
            or has_contained_place
            or has_supported_representative_place
            or has_strong_bge_support
        ):
            admitted.add(label_path)
    return admitted


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
    phrase_evidence: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Rank admitted regions by evidence type, then semantic rank."""
    bm25_by_label = {item["label_path"]: item for item in bm25_candidates}
    bge_by_label = {item["label_path"]: item for item in bge_candidates}
    evidence_by_label = {
        item["label_path"]: item for item in phrase_evidence
    }
    admitted = admitted_region_label_paths(
        bm25_candidates,
        bge_candidates,
        phrase_evidence,
    )
    ranked: list[dict[str, Any]] = []
    for label_path in admitted:
        bm25 = bm25_by_label.get(label_path)
        bge = bge_by_label.get(label_path)
        evidence = evidence_by_label.get(label_path, {})
        if evidence.get("direct_names"):
            evidence_tier = 1
            evidence_type = "direct_name"
        elif evidence.get("contained_places"):
            evidence_tier = 2
            evidence_type = "contained_place"
        elif evidence.get("representative_places"):
            evidence_tier = 3
            evidence_type = "representative_place_with_bge"
        else:
            evidence_tier = 4
            evidence_type = "bge_fallback"
        ranked.append(
            {
                "label_path": label_path,
                "region_evidence_type": evidence_type,
                "region_evidence_tier": evidence_tier,
                "matched_direct_names": evidence.get("direct_names") or [],
                "matched_contained_places": evidence.get("contained_places") or [],
                "matched_representative_places": evidence.get("representative_places") or [],
                "region_bm25_rank": bm25["rank"] if bm25 else None,
                "region_bm25_raw_score": bm25["score"] if bm25 else None,
                "region_bge_rank": bge["rank"] if bge else None,
                "region_bge_raw_score": bge["score"] if bge else None,
            }
        )
    ranked.sort(
        key=lambda item: (
            item["region_evidence_tier"],
            item["region_bge_rank"] if item["region_bge_rank"] is not None else 10**9,
            item["region_bm25_rank"] if item["region_bm25_rank"] is not None else 10**9,
            item["label_path"],
        )
    )
    return ranked[:limit]


def combine_nonregion_candidates(
    bm25_candidates: list[dict[str, Any]],
    bge_candidates: list[dict[str, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bm25_by_label = {item["label_path"]: item for item in bm25_candidates}
    bge_by_label = {item["label_path"]: item for item in bge_candidates}

    def selected(label: str, source: str) -> dict[str, Any]:
        item: dict[str, Any] = {
            "label_path": label,
            "selection_source": source,
        }
        bm25 = bm25_by_label.get(label)
        if bm25 is not None:
            item["bm25_rank"] = bm25["rank"]
            item["bm25_raw_score"] = bm25["score"]
        bge = bge_by_label.get(label)
        if bge is not None:
            item["bge_rank"] = bge["rank"]
            item["bge_raw_score"] = bge["score"]
        return item

    primary_limit = min(NONREGION_BM25_PRIMARY_QUOTA, limit)
    primary = [
        selected(item["label_path"], "bm25_primary")
        for item in bm25_candidates[:primary_limit]
    ]
    seen = {item["label_path"] for item in primary}
    supplements: list[dict[str, Any]] = []
    for item in bge_candidates:
        if len(primary) + len(supplements) >= limit:
            break
        label = item["label_path"]
        if label in seen:
            continue
        supplements.append(selected(label, "bge_supplement"))
        seen.add(label)
    return primary + supplements, supplements


def combine_comprehensive_candidates(
    bm25_candidates: list[dict[str, Any]],
    bge_candidates: list[dict[str, Any]],
    agreement_weight: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Rank only strict comprehensive labels and keep a small separate quota."""
    return rank_fused_candidates(
        {"bm25": bm25_candidates, "bge": bge_candidates},
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
    nonregion_candidate_limit: int,
    region_candidate_limit: int,
    batch_size: int,
    device: str | None,
    embedding_model: str | None,
    comprehensive_candidate_limit: int = 3,
    region_index_dir: Path | None = None,
    agreement_weight: float = 0.25,
    region_bm25_min_score: float = 0.0,
    region_bge_min_score: float = 0.4,
    query_encoder: Callable[[list[str], str, str, int, str | None], Any] = encode_queries,
) -> dict[str, Any]:
    if (
        nonregion_candidate_limit <= 0
        or region_candidate_limit <= 0
        or comprehensive_candidate_limit <= 0
    ):
        raise ValueError("候选数量限制必须大于0")
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
    strict_comprehensive_indices = [
        index
        for index, record in enumerate(labels)
        if is_strict_comprehensive_label_path(record["label_path"])
    ]
    strict_comprehensive_index_set = set(strict_comprehensive_indices)
    global_region_index_set = set(global_region_indices)
    nonregion_indices = [
        index
        for index in range(len(labels))
        if index not in global_region_index_set
        and index not in strict_comprehensive_index_set
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
        region_tokenizer = region_bm25.get("tokenizer")
        if region_tokenizer != "region_phrase":
            raise ValueError(
                "区域索引必须使用完整短语BM25；"
                f"当前分词器为{region_tokenizer}"
            )
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
    metrics = {
        name: metric_record()
        for name in (
            "nonregional_final",
            "regional_final",
            "comprehensive_final",
            "combined",
        )
    }
    unknown_gold_labels: dict[str, int] = {}
    fused_counts: list[int] = []
    regional_counts: list[int] = []
    comprehensive_counts: list[int] = []
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
                    len(nonregion_indices),
                ),
                sparse_scores,
                labels,
            )
            dense = ranked_candidates(
                top_indices_from_pool(
                    dense_scores,
                    nonregion_indices,
                    len(nonregion_indices),
                ),
                dense_scores,
                labels,
            )
            nonregional_final, nonregional_supplements = combine_nonregion_candidates(
                sparse,
                dense,
                nonregion_candidate_limit,
            )
            comprehensive_sparse = ranked_candidates(
                top_indices_from_pool(
                    sparse_scores,
                    strict_comprehensive_indices,
                    len(strict_comprehensive_indices),
                ),
                sparse_scores,
                labels,
            )
            comprehensive_dense = ranked_candidates(
                top_indices_from_pool(
                    dense_scores,
                    strict_comprehensive_indices,
                    len(strict_comprehensive_indices),
                ),
                dense_scores,
                labels,
            )
            comprehensive_final = combine_comprehensive_candidates(
                comprehensive_sparse,
                comprehensive_dense,
                agreement_weight,
                comprehensive_candidate_limit,
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
                    len(region_indices),
                ),
                regional_sparse_scores,
                region_labels,
            )
            regional_dense_full = ranked_candidates(
                top_indices_from_pool(
                    regional_dense_scores,
                    region_indices,
                    len(region_indices),
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
            regional_phrase_evidence = region_phrase_evidence(
                question_region_exact_text(question),
                region_labels,
                region_indices,
            )
            regional_fused = rank_region_candidates(
                regional_sparse,
                regional_dense,
                regional_phrase_evidence,
                limit=region_candidate_limit,
            )
            combined = merge_candidate_lists(nonregional_final, regional_fused)
            combined = merge_candidate_lists(combined, comprehensive_final)
            gold = question_gold_labels(question)
            for label in gold:
                if label not in allowed_labels:
                    unknown_gold_labels[label] = unknown_gold_labels.get(label, 0) + 1
            known_gold = [label for label in gold if label in allowed_labels]
            nonregional_gold = [
                label
                for label in known_gold
                if not is_region_label_path(label)
                and not is_strict_comprehensive_label_path(label)
            ]
            nonregional_final_labels = {
                item["label_path"] for item in nonregional_final
            }
            combined_labels = {item["label_path"] for item in combined}
            update_metrics(
                metrics["nonregional_final"],
                nonregional_gold,
                nonregional_final_labels,
            )
            update_metrics(metrics["combined"], known_gold, combined_labels)

            regional_gold = [
                label for label in known_gold if is_region_label_path(label)
            ]
            regional_fused_labels = {
                item["label_path"] for item in regional_fused
            }
            update_metrics(
                metrics["regional_final"],
                regional_gold,
                regional_fused_labels,
            )
            comprehensive_gold = [
                label
                for label in known_gold
                if is_strict_comprehensive_label_path(label)
            ]
            comprehensive_final_labels = {
                item["label_path"] for item in comprehensive_final
            }
            update_metrics(
                metrics["comprehensive_final"],
                comprehensive_gold,
                comprehensive_final_labels,
            )
            fused_counts.append(len(nonregional_final))
            regional_counts.append(len(regional_fused))
            comprehensive_counts.append(len(comprehensive_final))
            combined_counts.append(len(combined))
            result = {
                "question_id": as_text(question.get("question_id")),
                "parent_id": as_text(question.get("parent_id")),
                "knw_labels": gold,
                "unmapped_gold_labels": [
                    label for label in gold if label not in allowed_labels
                ],
                "bm25_candidates": sparse[:NONREGION_DIAGNOSTIC_LIMIT],
                "bge_candidates": dense[:NONREGION_DIAGNOSTIC_LIMIT],
                "nonregional_bge_supplement_candidates": nonregional_supplements,
                "nonregional_final_candidates": nonregional_final,
                "nonregional_final_missing_labels": [
                    label
                    for label in nonregional_gold
                    if label not in nonregional_final_labels
                ],
                "fused_candidates": nonregional_final,
                "fusion_missing_labels": [
                    label
                    for label in nonregional_gold
                    if label not in nonregional_final_labels
                ],
                "regional_exact_matches": regional_exact,
                "regional_phrase_evidence": regional_phrase_evidence,
                "regional_bm25_candidates": regional_sparse_full[
                    :REGION_DIAGNOSTIC_LIMIT
                ],
                "regional_bge_candidates": regional_dense_full[
                    :REGION_DIAGNOSTIC_LIMIT
                ],
                "regional_fused_candidates": regional_fused,
                "regional_fused_missing_labels": [
                    label
                    for label in regional_gold
                    if label not in regional_fused_labels
                ],
                "regional_final_candidates": regional_fused,
                "regional_final_missing_labels": [
                    label
                    for label in regional_gold
                    if label not in regional_fused_labels
                ],
                "comprehensive_bm25_candidates": comprehensive_sparse[
                    :COMPREHENSIVE_DIAGNOSTIC_LIMIT
                ],
                "comprehensive_bge_candidates": comprehensive_dense[
                    :COMPREHENSIVE_DIAGNOSTIC_LIMIT
                ],
                "comprehensive_final_candidates": comprehensive_final,
                "comprehensive_final_missing_labels": [
                    label
                    for label in comprehensive_gold
                    if label not in comprehensive_final_labels
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
        "strict_comprehensive_label_count": len(strict_comprehensive_indices),
        "nonregion_primary_route": "bm25",
        "nonregion_bm25_primary_quota": NONREGION_BM25_PRIMARY_QUOTA,
        "nonregion_candidate_limit": nonregion_candidate_limit,
        "regional_label_count": len(region_indices),
        "regional_index": str(region_index_dir) if region_index_dir else None,
        "region_candidate_limit": region_candidate_limit,
        "comprehensive_candidate_limit": comprehensive_candidate_limit,
        "maximum_combined_candidates": (
            nonregion_candidate_limit
            + region_candidate_limit
            + comprehensive_candidate_limit
        ),
        "agreement_weight": agreement_weight,
        "region_bm25_min_score": region_bm25_min_score,
        "region_bge_min_score": region_bge_min_score,
        "embedding_model": model_name,
        "metrics": {name: finalize_metrics(value) for name, value in metrics.items()},
        "nonregional_candidate_count": {
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
        "comprehensive_candidate_count": {
            "average": round(
                sum(comprehensive_counts) / len(comprehensive_counts), 6
            )
            if comprehensive_counts
            else None,
            "minimum": min(comprehensive_counts) if comprehensive_counts else None,
            "maximum": max(comprehensive_counts) if comprehensive_counts else None,
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
    parser.add_argument("--region-index-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--nonregion-candidate-limit", type=int, default=30)
    parser.add_argument("--region-candidate-limit", type=int, default=5)
    parser.add_argument("--comprehensive-candidate-limit", type=int, default=3)
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
        nonregion_candidate_limit=args.nonregion_candidate_limit,
        region_candidate_limit=args.region_candidate_limit,
        comprehensive_candidate_limit=args.comprehensive_candidate_limit,
        batch_size=args.batch_size,
        device=args.device,
        embedding_model=args.embedding_model,
        agreement_weight=args.agreement_weight,
        region_bm25_min_score=args.region_bm25_min_score,
        region_bge_min_score=args.region_bge_min_score,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
