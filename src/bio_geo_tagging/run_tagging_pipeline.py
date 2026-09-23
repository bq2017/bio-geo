"""Run geography tagging units, retrieval, adjudication, and diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from bio_geo_tagging.adjudication import run_adjudication
from bio_geo_tagging.adjudication_evaluation import evaluate_adjudication
from bio_geo_tagging.ds import DSClient
from bio_geo_tagging.hybrid_candidate_retrieval import run_retrieval
from bio_geo_tagging.question_tagging_units import process_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _emit(log_path: Path, stage: str, status: str, **details: Any) -> None:
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        **details,
    }
    message = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(message + "\n")
    print(message, flush=True)


def _log_only(log_path: Path, stage: str, status: str, **details: Any) -> None:
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        **details,
    }
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _ensure_manifest(path: Path, expected: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError(
                f"pipeline configuration changed for {path.parent}; use a new run-dir"
            )
        return
    _write_json(path, expected)


def run_pipeline(
    *,
    input_path: Path,
    labels_path: Path,
    index_dir: Path,
    region_index_dir: Path,
    run_dir: Path,
    client: Any,
    model: str,
    audited_exclusions_path: Path | None = None,
    limit: int | None = None,
    nonregion_candidate_limit: int = 30,
    region_candidate_limit: int = 5,
    comprehensive_candidate_limit: int = 5,
    region_bm25_min_score: float = 0.0,
    region_bge_min_score: float = 0.4,
    embedding_model: str | None = None,
    device: str | None = None,
    batch_size: int = 32,
    workers: int = 1,
    max_tokens: int = 1024,
    enable_thinking: bool | None = None,
    unit_runner: Callable[..., Any] = process_file,
    retrieval_runner: Callable[..., dict[str, Any]] = run_retrieval,
    adjudication_runner: Callable[..., dict[str, Any]] = run_adjudication,
    evaluation_runner: Callable[..., dict[str, Any]] = evaluate_adjudication,
) -> dict[str, Any]:
    for path, description in (
        (input_path, "input"),
        (labels_path, "labels"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} file not found: {path}")
    for path, description in (
        (index_dir, "index"),
        (region_index_dir, "region index"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{description} directory not found: {path}")
        if not (path / "manifest.json").is_file():
            raise FileNotFoundError(
                f"{description} manifest not found: {path / 'manifest.json'}"
            )
    if audited_exclusions_path is not None and not audited_exclusions_path.is_file():
        raise FileNotFoundError(
            f"audited exclusions file not found: {audited_exclusions_path}"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "pipeline.log"
    units_path = run_dir / "tagging-units.jsonl"
    units_complete_path = run_dir / "tagging-units-complete.json"
    units_log_path = run_dir / "tagging-units.log"
    candidates_path = run_dir / "candidates.jsonl"
    candidate_summary_path = run_dir / "candidate-summary.json"
    adjudication_dir = run_dir / "adjudication"
    evaluation_path = run_dir / "evaluation.json"
    evaluation_details_path = run_dir / "evaluation_details.jsonl"
    final_labels_path = run_dir / "final_labels.jsonl"

    manifest = {
        "pipeline_version": "geography-tagging-pipeline-v1",
        "input": {"path": str(input_path), "sha256": _sha256(input_path)},
        "labels": {"path": str(labels_path), "sha256": _sha256(labels_path)},
        "index": {
            "path": str(index_dir),
            "manifest_sha256": _sha256(index_dir / "manifest.json"),
        },
        "region_index": {
            "path": str(region_index_dir),
            "manifest_sha256": _sha256(region_index_dir / "manifest.json"),
        },
        "audited_exclusions": (
            {
                "path": str(audited_exclusions_path),
                "sha256": _sha256(audited_exclusions_path),
            }
            if audited_exclusions_path is not None
            else None
        ),
        "retrieval": {
            "nonregion_candidate_limit": nonregion_candidate_limit,
            "region_candidate_limit": region_candidate_limit,
            "comprehensive_candidate_limit": comprehensive_candidate_limit,
            "region_bm25_min_score": region_bm25_min_score,
            "region_bge_min_score": region_bge_min_score,
            "embedding_model": embedding_model,
            "device": device,
            "batch_size": batch_size,
        },
        "adjudication": {
            "model": model,
            "limit": limit,
            "workers": workers,
            "max_tokens": max_tokens,
            "enable_thinking": enable_thinking,
        },
    }
    _ensure_manifest(run_dir / "pipeline_manifest.json", manifest)

    try:
        if _is_nonempty_file(units_path) and _is_nonempty_file(units_complete_path):
            _emit(log_path, "tagging_units", "skipped", output=str(units_path))
        else:
            _emit(log_path, "tagging_units", "started")
            temporary_units_path = units_path.with_name(f".{units_path.name}.tmp")
            temporary_units_path.unlink(missing_ok=True)
            unit_runner(
                str(input_path), str(temporary_units_path), str(units_log_path)
            )
            if not _is_nonempty_file(temporary_units_path):
                raise RuntimeError("tagging unit generation produced no output")
            temporary_units_path.replace(units_path)
            _write_json(
                units_complete_path,
                {"input_sha256": manifest["input"]["sha256"]},
            )
            _emit(log_path, "tagging_units", "completed", output=str(units_path))

        if _is_nonempty_file(candidates_path) and _is_nonempty_file(
            candidate_summary_path
        ):
            _emit(log_path, "candidate_retrieval", "skipped", output=str(candidates_path))
        else:
            _emit(log_path, "candidate_retrieval", "started")
            retrieval_runner(
                input_path=input_path,
                index_dir=index_dir,
                region_index_dir=region_index_dir,
                output_path=candidates_path,
                summary_path=candidate_summary_path,
                nonregion_candidate_limit=nonregion_candidate_limit,
                region_candidate_limit=region_candidate_limit,
                comprehensive_candidate_limit=comprehensive_candidate_limit,
                region_bm25_min_score=region_bm25_min_score,
                region_bge_min_score=region_bge_min_score,
                embedding_model=embedding_model,
                device=device,
                batch_size=batch_size,
            )
            if not _is_nonempty_file(candidates_path) or not _is_nonempty_file(
                candidate_summary_path
            ):
                raise RuntimeError("candidate retrieval produced incomplete outputs")
            _emit(log_path, "candidate_retrieval", "completed", output=str(candidates_path))

        _emit(log_path, "candidate_adjudication", "started")
        adjudication_report = adjudication_runner(
            units_path,
            candidates_path,
            labels_path,
            adjudication_dir,
            client,
            model=model,
            limit=limit,
            max_tokens=max_tokens,
            workers=workers,
            audited_exclusions_path=audited_exclusions_path,
            enable_thinking=enable_thinking,
        )
        if (
            adjudication_report.get("error")
            or adjudication_report.get("success") != adjudication_report.get("input")
        ):
            raise RuntimeError(
                "candidate adjudication incomplete; rerun the same command to resume"
            )
        _emit(
            log_path,
            "candidate_adjudication",
            "completed",
            output=str(adjudication_dir),
        )

        _emit(log_path, "automatic_diagnostics", "started")
        diagnostics = evaluation_runner(
            adjudication_dir,
            candidates_path,
            labels_path,
            evaluation_path,
            evaluation_details_path,
            None,
            final_labels_path,
        )
        if not _is_nonempty_file(final_labels_path):
            raise RuntimeError("automatic diagnostics produced no final labels")
        _emit(
            log_path,
            "automatic_diagnostics",
            "completed",
            final_labels=str(final_labels_path),
        )
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        _emit(
            log_path,
            "pipeline",
            "error",
            error=error_message,
        )
        _log_only(
            log_path,
            "pipeline",
            "traceback",
            error=error_message,
            traceback=traceback.format_exc(),
        )
        raise

    result = {
        "run_dir": str(run_dir),
        "units": str(units_path),
        "candidates": str(candidates_path),
        "adjudication_dir": str(adjudication_dir),
        "evaluation": str(evaluation_path),
        "final_labels": str(final_labels_path),
        "adjudication": adjudication_report,
        "diagnostics": diagnostics,
    }
    _write_json(run_dir / "pipeline_report.json", result)
    _emit(log_path, "pipeline", "completed", final_labels=str(final_labels_path))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--region-index-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audited-exclusions", type=Path)
    parser.add_argument("--endpoint", action="append", dest="endpoints")
    parser.add_argument("--model", default=os.getenv("MODEL", "DeepSeek-V4-Flash"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--nonregion-candidate-limit", type=int, default=30)
    parser.add_argument("--region-candidate-limit", type=int, default=5)
    parser.add_argument("--comprehensive-candidate-limit", type=int, default=5)
    parser.add_argument("--region-bm25-min-score", type=float, default=0.0)
    parser.add_argument("--region-bge-min-score", type=float, default=0.4)
    parser.add_argument("--embedding-model")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument("--request-interval", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    thinking.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoints = args.endpoints or [
        value for value in (os.getenv("DS1"), os.getenv("DS2")) if value
    ]
    if not endpoints:
        raise SystemExit("provide --endpoint or set DS1/DS2")
    for name in (
        "limit",
        "nonregion_candidate_limit",
        "region_candidate_limit",
        "comprehensive_candidate_limit",
        "batch_size",
        "workers",
        "max_tokens",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.retry_delay < 0 or args.request_interval < 0:
        raise SystemExit("retry delay and request interval must be non-negative")

    client = DSClient(
        endpoints,
        args.model,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
        request_interval=args.request_interval,
        enable_thinking=args.enable_thinking,
    )
    try:
        result = run_pipeline(
            input_path=args.input,
            labels_path=args.labels,
            index_dir=args.index_dir,
            region_index_dir=args.region_index_dir,
            run_dir=args.run_dir,
            client=client,
            model=args.model,
            audited_exclusions_path=args.audited_exclusions,
            limit=args.limit,
            nonregion_candidate_limit=args.nonregion_candidate_limit,
            region_candidate_limit=args.region_candidate_limit,
            comprehensive_candidate_limit=args.comprehensive_candidate_limit,
            region_bm25_min_score=args.region_bm25_min_score,
            region_bge_min_score=args.region_bge_min_score,
            embedding_model=args.embedding_model,
            device=args.device,
            batch_size=args.batch_size,
            workers=args.workers,
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
        )
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        json.dumps(
            {
                "status": "completed",
                "run_dir": result["run_dir"],
                "final_labels": result["final_labels"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
