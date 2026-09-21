"""Retrieve geography label candidates with cached BM25 and BGE indexes."""

from __future__ import annotations

import argparse
import json
import math
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


def top_indices(scores: Any, limit: int) -> list[int]:
    return sorted(
        range(len(scores)), key=lambda index: (-float(scores[index]), index)
    )[:limit]


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
    query_encoder: Callable[[list[str], str, str, int, str | None], Any] = encode_queries,
) -> dict[str, Any]:
    if bm25_top_k <= 0 or bge_top_k <= 0:
        raise ValueError("两路top-k必须大于0")
    manifest, labels, bm25, label_embeddings = load_index(index_dir)
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
    metrics = {name: metric_record() for name in ("bm25", "bge", "fusion")}
    unknown_gold_labels: dict[str, int] = {}
    fused_counts: list[int] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for index, (question, query) in enumerate(zip(questions, query_texts)):
            sparse_scores = bm25_scores(query, bm25)
            dense_scores = label_embeddings @ query_embeddings[index]
            sparse = ranked_candidates(
                top_indices(sparse_scores, min(bm25_top_k, len(labels))),
                sparse_scores,
                labels,
            )
            dense = ranked_candidates(
                top_indices(dense_scores, min(bge_top_k, len(labels))),
                dense_scores,
                labels,
            )
            fused = fuse_candidates(sparse, dense)
            gold = question_gold_labels(question)
            for label in gold:
                if label not in allowed_labels:
                    unknown_gold_labels[label] = unknown_gold_labels.get(label, 0) + 1
            known_gold = [label for label in gold if label in allowed_labels]
            sparse_labels = {item["label_path"] for item in sparse}
            dense_labels = {item["label_path"] for item in dense}
            fused_labels = {item["label_path"] for item in fused}
            update_metrics(metrics["bm25"], known_gold, sparse_labels)
            update_metrics(metrics["bge"], known_gold, dense_labels)
            update_metrics(metrics["fusion"], known_gold, fused_labels)
            fused_counts.append(len(fused))
            result = {
                "question_id": as_text(question.get("question_id")),
                "parent_id": as_text(question.get("parent_id")),
                "knw_labels": gold,
                "unmapped_gold_labels": [
                    label for label in gold if label not in allowed_labels
                ],
                "bm25_candidates": sparse,
                "bm25_missing_labels": [
                    label for label in known_gold if label not in sparse_labels
                ],
                "bge_candidates": dense,
                "bge_missing_labels": [
                    label for label in known_gold if label not in dense_labels
                ],
                "fused_candidates": fused,
                "fusion_missing_labels": [
                    label for label in known_gold if label not in fused_labels
                ],
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {
        "input_questions": len(questions),
        "label_count": len(labels),
        "bm25_top_k": bm25_top_k,
        "bge_top_k": bge_top_k,
        "embedding_model": model_name,
        "metrics": {name: finalize_metrics(value) for name, value in metrics.items()},
        "fused_candidate_count": {
            "average": round(sum(fused_counts) / len(fused_counts), 6)
            if fused_counts
            else None,
            "minimum": min(fused_counts) if fused_counts else None,
            "maximum": max(fused_counts) if fused_counts else None,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--bm25-top-k", type=int, default=50)
    parser.add_argument("--bge-top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--embedding-model")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size必须大于0")
    summary = run_retrieval(
        input_path=args.input,
        index_dir=args.index_dir,
        output_path=args.output,
        summary_path=args.summary_output,
        bm25_top_k=args.bm25_top_k,
        bge_top_k=args.bge_top_k,
        batch_size=args.batch_size,
        device=args.device,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
