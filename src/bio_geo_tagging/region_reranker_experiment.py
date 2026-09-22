"""Offline Cross-Encoder reranking experiment for regional label candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .hybrid_candidate_retrieval import (
    as_text,
    is_region_label_path,
    question_region_exact_text,
    read_jsonl,
)


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    for line_number, question in read_jsonl(path):
        question_id = as_text(question.get("question_id"))
        if not question_id:
            raise ValueError(f"题目文件第{line_number}行缺少question_id")
        if question_id in questions:
            raise ValueError(f"题目文件存在重复question_id：{question_id}")
        questions[question_id] = question
    return questions


def load_region_label_records(index_dir: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(
        (index_dir / "manifest.json").read_text(encoding="utf-8")
    )
    records: dict[str, dict[str, Any]] = {}
    for _, record in read_jsonl(index_dir / manifest["labels_file"]):
        label_path = as_text(record.get("label_path"))
        if not label_path:
            raise ValueError("区域索引存在缺少label_path的记录")
        records[label_path] = record
    return records


def region_label_passage(record: dict[str, Any]) -> str:
    label_path = as_text(record.get("label_path"))
    parts = [f"区域标签：{label_path}"]
    embedding_text = as_text(record.get("embedding_text"))
    if embedding_text:
        parts.append(embedding_text)
    for field, title in (
        ("exact_names", "标准名称和别名"),
        ("contained_places", "包含地点"),
        ("representative_places", "代表地点"),
    ):
        values = record.get(field) or []
        if values:
            parts.append(f"{title}：{'、'.join(dict.fromkeys(values))}")
    return "\n".join(parts)


def build_candidate_pool(
    result: dict[str, Any],
    bm25_top_k: int,
    bge_top_k: int,
) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, Any]] = {}

    def candidate(label_path: str) -> dict[str, Any]:
        return by_label.setdefault(
            label_path,
            {
                "label_path": label_path,
                "bm25_rank": None,
                "bge_rank": None,
                "direct_name_match": False,
                "matched_name": None,
                "original_selected": False,
            },
        )

    for item in result.get("regional_bm25_candidates", [])[:bm25_top_k]:
        current = candidate(item["label_path"])
        current["bm25_rank"] = item["rank"]
    for item in result.get("regional_bge_candidates", [])[:bge_top_k]:
        current = candidate(item["label_path"])
        current["bge_rank"] = item["rank"]
    for item in result.get("regional_exact_matches", []):
        current = candidate(item["label_path"])
        current["direct_name_match"] = True
        current["matched_name"] = item.get("matched_name")
    for item in result.get("regional_final_candidates", []):
        current = candidate(item["label_path"])
        current["original_selected"] = True
    return list(by_label.values())


def rerank_candidates(
    candidates: list[dict[str, Any]],
    scores: list[float],
    limit: int,
) -> list[dict[str, Any]]:
    if len(candidates) != len(scores):
        raise ValueError("区域候选数量与reranker分数数量不一致")
    ranked: list[dict[str, Any]] = []
    for candidate, score in zip(candidates, scores):
        ranked.append({**candidate, "reranker_score": round(float(score), 8)})
    ranked.sort(
        key=lambda item: (
            not item["direct_name_match"],
            -item["reranker_score"],
            item["label_path"],
        )
    )
    return ranked[:limit]


class TransformerReranker:
    def __init__(
        self,
        model_name_or_path: str,
        device: str | None,
        max_length: int,
        batch_size: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("缺少torch或transformers，无法运行reranker") from error

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        )
        self.model.to(self.device)
        self.model.eval()

    def compute(self, pairs: list[tuple[str, str]]) -> list[float]:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda values, **_: values  # type: ignore[assignment]

        scores: list[float] = []
        for start in tqdm(
            range(0, len(pairs), self.batch_size),
            desc="Reranking regional candidates",
        ):
            batch = pairs[start : start + self.batch_size]
            encoded = self.tokenizer(
                [item[0] for item in batch],
                [item[1] for item in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.no_grad():
                logits = self.model(**encoded, return_dict=True).logits
            if logits.shape[-1] != 1:
                raise ValueError(
                    f"reranker应输出单个相关性分数，实际维度为{tuple(logits.shape)}"
                )
            scores.extend(logits.reshape(-1).float().cpu().tolist())
        return scores


def metric_record() -> dict[str, int]:
    return {
        "questions_with_gold": 0,
        "fully_covered_questions": 0,
        "gold_labels": 0,
        "recalled_gold_labels": 0,
    }


def update_metrics(
    metrics: dict[str, int], gold_labels: list[str], selected: set[str]
) -> None:
    if not gold_labels:
        return
    gold = set(gold_labels)
    metrics["questions_with_gold"] += 1
    metrics["fully_covered_questions"] += int(gold <= selected)
    metrics["gold_labels"] += len(gold)
    metrics["recalled_gold_labels"] += len(gold & selected)


def finalize_metrics(metrics: dict[str, int]) -> dict[str, Any]:
    questions = metrics["questions_with_gold"]
    labels = metrics["gold_labels"]
    return {
        **metrics,
        "full_coverage_rate": round(
            metrics["fully_covered_questions"] / questions, 6
        )
        if questions
        else None,
        "label_recall": round(metrics["recalled_gold_labels"] / labels, 6)
        if labels
        else None,
    }


def run_experiment(
    input_path: Path,
    candidates_path: Path,
    region_index_dir: Path,
    output_path: Path,
    summary_path: Path,
    model_name_or_path: str,
    bm25_top_k: int,
    bge_top_k: int,
    final_limit: int,
    batch_size: int,
    max_length: int,
    device: str | None,
) -> dict[str, Any]:
    questions = load_questions(input_path)
    region_records = load_region_label_records(region_index_dir)
    candidate_results = [record for _, record in read_jsonl(candidates_path)]
    if len(candidate_results) != len(questions):
        raise ValueError(
            "题目数量与候选结果数量不一致："
            f"{len(questions)} != {len(candidate_results)}"
        )

    prepared: list[tuple[dict[str, Any], list[dict[str, Any]], int, int]] = []
    pairs: list[tuple[str, str]] = []
    seen_question_ids: set[str] = set()
    for result in candidate_results:
        question_id = as_text(result.get("question_id"))
        if question_id in seen_question_ids:
            raise ValueError(f"候选结果存在重复question_id：{question_id}")
        seen_question_ids.add(question_id)
        question = questions.get(question_id)
        if question is None:
            raise ValueError(f"候选结果中的题目不在输入文件中：{question_id}")
        query = question_region_exact_text(question)
        pool = build_candidate_pool(result, bm25_top_k, bge_top_k)
        start = len(pairs)
        for candidate in pool:
            label_path = candidate["label_path"]
            record = region_records.get(label_path)
            if record is None:
                raise ValueError(f"区域索引中缺少候选标签：{label_path}")
            pairs.append((query, region_label_passage(record)))
        prepared.append((result, pool, start, len(pairs)))

    reranker = TransformerReranker(
        model_name_or_path=model_name_or_path,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    all_scores = reranker.compute(pairs)
    if len(all_scores) != len(pairs):
        raise ValueError("reranker返回的分数数量与输入文本对数量不一致")

    baseline_metrics = metric_record()
    pool_metrics = metric_record()
    reranked_metrics = metric_record()
    recovered_counts: dict[str, int] = {}
    lost_counts: dict[str, int] = {}
    pool_counts: list[int] = []
    final_counts: list[int] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for result, pool, start, end in prepared:
            reranked = rerank_candidates(pool, all_scores[start:end], final_limit)
            gold = [
                label
                for label in result.get("knw_labels", [])
                if is_region_label_path(label)
            ]
            baseline_labels = {
                item["label_path"]
                for item in result.get("regional_final_candidates", [])
            }
            pool_labels = {item["label_path"] for item in pool}
            reranked_labels = {item["label_path"] for item in reranked}
            update_metrics(baseline_metrics, gold, baseline_labels)
            update_metrics(pool_metrics, gold, pool_labels)
            update_metrics(reranked_metrics, gold, reranked_labels)
            recovered = sorted((set(gold) & reranked_labels) - baseline_labels)
            lost = sorted((set(gold) & baseline_labels) - reranked_labels)
            for label in recovered:
                recovered_counts[label] = recovered_counts.get(label, 0) + 1
            for label in lost:
                lost_counts[label] = lost_counts.get(label, 0) + 1
            pool_counts.append(len(pool))
            final_counts.append(len(reranked))
            output.write(
                json.dumps(
                    {
                        "question_id": result["question_id"],
                        "knw_labels": result.get("knw_labels", []),
                        "regional_gold_labels": gold,
                        "baseline_candidates": result.get(
                            "regional_final_candidates", []
                        ),
                        "reranker_pool": reranked_candidates_with_all_scores(
                            pool, all_scores[start:end]
                        ),
                        "reranked_candidates": reranked,
                        "baseline_missing_labels": sorted(
                            set(gold) - baseline_labels
                        ),
                        "reranked_missing_labels": sorted(
                            set(gold) - reranked_labels
                        ),
                        "recovered_labels": recovered,
                        "lost_labels": lost,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    baseline = finalize_metrics(baseline_metrics)
    pool_result = finalize_metrics(pool_metrics)
    reranked_result = finalize_metrics(reranked_metrics)
    recovered_total = sum(recovered_counts.values())
    lost_total = sum(lost_counts.values())
    net_gain = reranked_metrics["recalled_gold_labels"] - baseline_metrics[
        "recalled_gold_labels"
    ]
    summary = {
        "input_questions": len(questions),
        "region_label_count": len(region_records),
        "model": model_name_or_path,
        "bm25_top_k": bm25_top_k,
        "bge_top_k": bge_top_k,
        "final_limit": final_limit,
        "metrics": {
            "baseline": baseline,
            "reranker_pool_oracle": pool_result,
            "reranked": reranked_result,
        },
        "change": {
            "recovered_gold_labels": recovered_total,
            "lost_gold_labels": lost_total,
            "net_recalled_gold_labels": net_gain,
            "recall_point_change": round(
                (reranked_result["label_recall"] or 0)
                - (baseline["label_recall"] or 0),
                6,
            ),
            "recovered_label_counts": recovered_counts,
            "lost_label_counts": lost_counts,
        },
        "candidate_count": {
            "pool_average": round(sum(pool_counts) / len(pool_counts), 6),
            "pool_minimum": min(pool_counts),
            "pool_maximum": max(pool_counts),
            "final_average": round(sum(final_counts) / len(final_counts), 6),
            "final_minimum": min(final_counts),
            "final_maximum": max(final_counts),
        },
        "adoption_rule": {
            "minimum_label_recall": 0.92,
            "minimum_net_gain": 15,
            "maximum_lost_gold_labels": 3,
            "passed": (
                (reranked_result["label_recall"] or 0) >= 0.92
                and net_gain >= 15
                and lost_total <= 3
            ),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def reranked_candidates_with_all_scores(
    candidates: list[dict[str, Any]], scores: list[float]
) -> list[dict[str, Any]]:
    return rerank_candidates(candidates, scores, len(candidates))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Cross-Encoder reranking of regional label candidates"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--region-index-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--bm25-top-k", type=int, default=10)
    parser.add_argument("--bge-top-k", type=int, default=10)
    parser.add_argument("--final-limit", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device")
    args = parser.parse_args()
    for name in (
        "bm25_top_k",
        "bge_top_k",
        "final_limit",
        "batch_size",
        "max_length",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')}必须大于0")
    summary = run_experiment(
        input_path=args.input,
        candidates_path=args.candidates,
        region_index_dir=args.region_index_dir,
        output_path=args.output,
        summary_path=args.summary_output,
        model_name_or_path=args.reranker_model,
        bm25_top_k=args.bm25_top_k,
        bge_top_k=args.bge_top_k,
        final_limit=args.final_limit,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
