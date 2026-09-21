"""Build cached BM25 and BGE indexes for geography label retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


INDEX_VERSION = 1
TEXT_SEGMENT_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[a-z0-9°%≈.+/\-]+")


def parse_ngram_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("字符n-gram必须是逗号分隔的正整数") from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("字符n-gram必须是逗号分隔的正整数")
    return sizes


def tokenize_char_ngrams(text: str, ngram_sizes: tuple[int, ...]) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for segment in TEXT_SEGMENT_RE.findall(normalized):
        if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", segment):
            for size in ngram_sizes:
                tokens.extend(
                    segment[start : start + size]
                    for start in range(0, len(segment) - size + 1)
                )
        else:
            tokens.append(segment)
    return tokens


def load_retrieval_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            label_path = str(value.get("label_path") or "").strip()
            bm25_text = str(value.get("bm25_text") or "").strip()
            embedding_text = str(value.get("embedding_text") or "").strip()
            if not label_path:
                raise ValueError(f"第{line_number}行缺少label_path")
            if label_path in seen_paths:
                raise ValueError(f"第{line_number}行标签路径重复：{label_path}")
            if not bm25_text:
                raise ValueError(f"第{line_number}行缺少bm25_text：{label_path}")
            if not embedding_text:
                raise ValueError(f"第{line_number}行缺少embedding_text：{label_path}")
            seen_paths.add(label_path)
            record: dict[str, Any] = {
                "label_path": label_path,
                "bm25_text": bm25_text,
                "embedding_text": embedding_text,
            }
            exact_names = value.get("exact_names")
            if exact_names is not None:
                if not isinstance(exact_names, list) or not all(
                    isinstance(item, str) and item.strip() for item in exact_names
                ):
                    raise ValueError(
                        f"第{line_number}行exact_names必须是非空字符串数组"
                    )
                record["exact_names"] = list(
                    dict.fromkeys(item.strip() for item in exact_names)
                )
            records.append(record)
    if not records:
        raise ValueError("检索文本文件中没有可用标签")
    return records


def build_bm25_index(
    records: list[dict[str, str]],
    ngram_sizes: tuple[int, ...],
    k1: float,
    b: float,
) -> dict[str, Any]:
    if k1 <= 0:
        raise ValueError("BM25参数k1必须大于0")
    if not 0 <= b <= 1:
        raise ValueError("BM25参数b必须位于0到1之间")

    term_frequencies: list[Counter[str]] = []
    document_frequencies: Counter[str] = Counter()
    document_lengths: list[int] = []
    for record in records:
        tokens = tokenize_char_ngrams(record["bm25_text"], ngram_sizes)
        if not tokens:
            raise ValueError(f"BM25分词结果为空：{record['label_path']}")
        frequencies = Counter(tokens)
        term_frequencies.append(frequencies)
        document_frequencies.update(frequencies.keys())
        document_lengths.append(len(tokens))

    document_count = len(records)
    average_document_length = sum(document_lengths) / document_count
    vocabulary = sorted(document_frequencies)
    idf = {
        term: math.log(
            1
            + (document_count - document_frequencies[term] + 0.5)
            / (document_frequencies[term] + 0.5)
        )
        for term in vocabulary
    }
    postings: dict[str, list[list[int]]] = {term: [] for term in vocabulary}
    for document_index, frequencies in enumerate(term_frequencies):
        for term, frequency in frequencies.items():
            postings[term].append([document_index, frequency])

    return {
        "version": INDEX_VERSION,
        "tokenizer": "unicode_char_ngram",
        "ngram_sizes": list(ngram_sizes),
        "document_count": document_count,
        "k1": k1,
        "b": b,
        "average_document_length": average_document_length,
        "document_lengths": document_lengths,
        "idf": idf,
        "postings": postings,
    }


def encode_bge_embeddings(
    texts: list[str],
    model_name_or_path: str,
    batch_size: int,
    device: str | None,
) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "缺少sentence-transformers，请安装项目的retrieval可选依赖"
        ) from error

    model_arguments: dict[str, Any] = {}
    if device:
        model_arguments["device"] = device
    model = SentenceTransformer(model_name_or_path, **model_arguments)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def save_embeddings(path: Path, embeddings: Any, expected_rows: int) -> tuple[int, int]:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("缺少numpy，请安装项目的retrieval可选依赖") from error

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"BGE向量不是二维矩阵：shape={matrix.shape}")
    if matrix.shape[0] != expected_rows:
        raise ValueError(
            f"BGE向量数量与标签数量不一致：{matrix.shape[0]} != {expected_rows}"
        )
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("BGE向量包含NaN或无穷值")
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("BGE向量没有完成L2归一化")
    np.save(path, matrix, allow_pickle=False)
    return int(matrix.shape[0]), int(matrix.shape[1])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")


def write_label_mapping(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for document_index, record in enumerate(records):
            mapping = {
                "document_index": document_index,
                "label_path": record["label_path"],
                "bm25_text": record["bm25_text"],
                "embedding_text": record["embedding_text"],
            }
            if "exact_names" in record:
                mapping["exact_names"] = record["exact_names"]
            handle.write(json.dumps(mapping, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cached character-BM25 and BGE label indexes"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--embedding-model", default="BAAI/bge-large-zh-v1.5"
    )
    parser.add_argument(
        "--query-instruction",
        default="为这个句子生成表示以用于检索相关文章：",
    )
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--char-ngrams", type=parse_ngram_sizes, default=(1, 2))
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--expected-count", type=int, default=414)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size必须大于0")
    records = load_retrieval_records(args.input)
    if args.expected_count and len(records) != args.expected_count:
        raise ValueError(
            f"标签数量不是{args.expected_count}：{len(records)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / "labels.jsonl"
    bm25_path = args.output_dir / "bm25-index.json"
    embeddings_path = args.output_dir / "bge-embeddings.npy"
    manifest_path = args.output_dir / "manifest.json"

    write_label_mapping(labels_path, records)
    bm25_index = build_bm25_index(
        records,
        args.char_ngrams,
        args.bm25_k1,
        args.bm25_b,
    )
    write_json(bm25_path, bm25_index)

    embeddings = encode_bge_embeddings(
        [record["embedding_text"] for record in records],
        args.embedding_model,
        args.batch_size,
        args.device,
    )
    embedding_rows, embedding_dimension = save_embeddings(
        embeddings_path, embeddings, len(records)
    )

    manifest = {
        "version": INDEX_VERSION,
        "source": str(args.input),
        "source_sha256": sha256_file(args.input),
        "label_count": len(records),
        "labels_file": labels_path.name,
        "bm25": {
            "file": bm25_path.name,
            "tokenizer": "unicode_char_ngram",
            "ngram_sizes": list(args.char_ngrams),
            "k1": args.bm25_k1,
            "b": args.bm25_b,
        },
        "embedding": {
            "file": embeddings_path.name,
            "model": args.embedding_model,
            "query_instruction": args.query_instruction,
            "normalized": True,
            "dtype": "float32",
            "rows": embedding_rows,
            "dimension": embedding_dimension,
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
