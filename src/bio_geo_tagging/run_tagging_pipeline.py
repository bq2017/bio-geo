"""Run geography tagging units, retrieval, adjudication, and diagnostics."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _write_unresolved_adjudication_failures(
    evidence_path: Path,
    output_path: Path,
) -> int:
    """Write the latest unresolved error for every adjudication unit."""
    latest_by_unit: dict[str, dict[str, Any]] = {}
    if evidence_path.is_file():
        with evidence_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid adjudication evidence JSON at line {line_number}"
                    ) from exc
                unit_key = str(record.get("unit_key") or "").strip()
                if unit_key:
                    latest_by_unit[unit_key] = record

    failures: list[dict[str, Any]] = []
    for unit_key, record in latest_by_unit.items():
        error = record.get("error")
        if not error:
            continue
        failures.append(
            {
                "stage": "candidate_adjudication",
                "unit_key": unit_key,
                "root_question_id": record.get("root_question_id"),
                "question_id": record.get("question_id"),
                "input_role": record.get("input_role"),
                "error": error,
                "endpoint": record.get("endpoint"),
                "attempts": record.get("attempts"),
                "retry_errors": record.get("retry_errors") or [],
                "created_at": record.get("created_at"),
            }
        )
    failures.sort(key=lambda item: item["unit_key"])

    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return len(failures)


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


def _release_retrieval_cuda_memory(device: str | None) -> None:
    if not device or not device.lower().startswith("cuda"):
        return
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_model_stage(
    *,
    output_root: Path,
    units_path: Path,
    candidates_path: Path,
    labels_path: Path,
    client: Any,
    model: str,
    limit: int | None,
    max_tokens: int,
    workers: int,
    audited_exclusions_path: Path | None,
    enable_thinking: bool | None,
    temperature: float,
    enable_vision: bool,
    adjudication_runner: Callable[..., dict[str, Any]],
    evaluation_runner: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    adjudication_dir = output_root / "adjudication"
    evaluation_path = output_root / "evaluation.json"
    evaluation_details_path = output_root / "evaluation_details.jsonl"
    final_labels_path = output_root / "final_labels.jsonl"
    failures_path = output_root / "pipeline-failures.jsonl"
    output_root.mkdir(parents=True, exist_ok=True)

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
        temperature=temperature,
        enable_vision=enable_vision,
    )
    incomplete = bool(
        adjudication_report.get("error")
        or adjudication_report.get("success") != adjudication_report.get("input")
    )
    unresolved_failure_units = _write_unresolved_adjudication_failures(
        adjudication_dir / "evidence.jsonl",
        failures_path,
    )
    diagnostics = evaluation_runner(
        adjudication_dir,
        candidates_path,
        labels_path,
        evaluation_path,
        evaluation_details_path,
        None,
        final_labels_path,
    )
    if not final_labels_path.is_file():
        raise RuntimeError("automatic diagnostics produced no final labels")
    return {
        "status": "completed_with_errors" if incomplete else "completed",
        "model": model,
        "output_root": str(output_root),
        "adjudication_dir": str(adjudication_dir),
        "evaluation": str(evaluation_path),
        "final_labels": str(final_labels_path),
        "failures": str(failures_path),
        "unresolved_failure_units": unresolved_failure_units,
        "adjudication": adjudication_report,
        "diagnostics": diagnostics,
    }


def run_pipeline(
    *,
    input_path: Path,
    labels_path: Path,
    index_dir: Path,
    region_index_dir: Path,
    run_dir: Path,
    client: Any,
    model: str,
    qwen_client: Any | None = None,
    qwen_model: str | None = None,
    qwen_workers: int | None = None,
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
    temperature: float = 0.0,
    enable_vision: bool = False,
    unit_runner: Callable[..., Any] = process_file,
    retrieval_runner: Callable[..., dict[str, Any]] = run_retrieval,
    adjudication_runner: Callable[..., dict[str, Any]] = run_adjudication,
    evaluation_runner: Callable[..., dict[str, Any]] = evaluate_adjudication,
) -> dict[str, Any]:
    dual_model = qwen_client is not None
    if dual_model and not qwen_model:
        raise ValueError("qwen_model is required when qwen_client is provided")
    if dual_model and (temperature != 0 or enable_vision):
        raise ValueError("dual-model pipeline requires temperature=0 and text-only mode")
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
        "adjudication": (
            {
                "mode": "dual_model_parallel",
                "limit": limit,
                "max_tokens": max_tokens,
                "enable_thinking": enable_thinking,
                "temperature": 0.0,
                "vision_enabled": False,
                "models": {
                    "deepseek": {"model": model, "workers": workers},
                    "qwen": {
                        "model": qwen_model,
                        "workers": qwen_workers or workers,
                    },
                },
            }
            if dual_model
            else {
                "mode": "single_model",
                "model": model,
                "limit": limit,
                "workers": workers,
                "max_tokens": max_tokens,
                "enable_thinking": enable_thinking,
                "temperature": temperature,
                "vision_enabled": enable_vision,
            }
        ),
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
            _release_retrieval_cuda_memory(device)

        profiles = {
            "deepseek": {
                "client": client,
                "model": model,
                "workers": workers,
                "output_root": run_dir / "deepseek" if dual_model else run_dir,
                "temperature": 0.0 if dual_model else temperature,
                "enable_vision": False if dual_model else enable_vision,
            }
        }
        if dual_model:
            profiles["qwen"] = {
                "client": qwen_client,
                "model": qwen_model,
                "workers": qwen_workers or workers,
                "output_root": run_dir / "qwen",
                "temperature": 0.0,
                "enable_vision": False,
            }
        for profile in profiles:
            _emit(log_path, "candidate_adjudication", "started", profile=profile)

        model_results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=len(profiles)) as executor:
            futures = {
                executor.submit(
                    _run_model_stage,
                    output_root=configuration["output_root"],
                    units_path=units_path,
                    candidates_path=candidates_path,
                    labels_path=labels_path,
                    client=configuration["client"],
                    model=configuration["model"],
                    limit=limit,
                    max_tokens=max_tokens,
                    workers=configuration["workers"],
                    audited_exclusions_path=audited_exclusions_path,
                    enable_thinking=enable_thinking,
                    temperature=configuration["temperature"],
                    enable_vision=configuration["enable_vision"],
                    adjudication_runner=adjudication_runner,
                    evaluation_runner=evaluation_runner,
                ): profile
                for profile, configuration in profiles.items()
            }
            for future in as_completed(futures):
                profile = futures[future]
                try:
                    model_results[profile] = future.result()
                except Exception as exc:
                    error_message = f"{type(exc).__name__}: {exc}"
                    output_root = profiles[profile]["output_root"]
                    output_root.mkdir(parents=True, exist_ok=True)
                    _write_json(
                        output_root / "model-error.json",
                        {"profile": profile, "error": error_message},
                    )
                    model_results[profile] = {
                        "status": "error",
                        "model": profiles[profile]["model"],
                        "output_root": str(output_root),
                        "error": error_message,
                    }
                _emit(
                    log_path,
                    "candidate_adjudication",
                    model_results[profile]["status"],
                    profile=profile,
                    output=model_results[profile]["output_root"],
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

    overall_status = (
        "completed"
        if all(value["status"] == "completed" for value in model_results.values())
        else "completed_with_errors"
    )
    result = {
        "status": overall_status,
        "run_dir": str(run_dir),
        "units": str(units_path),
        "candidates": str(candidates_path),
        "models": model_results,
    }
    if not dual_model:
        single_result = model_results["deepseek"]
        result.update(
            {
                "adjudication_dir": single_result["adjudication_dir"],
                "evaluation": single_result["evaluation"],
                "final_labels": single_result["final_labels"],
                "failures": single_result["failures"],
                "unresolved_failure_units": single_result[
                    "unresolved_failure_units"
                ],
                "adjudication": single_result["adjudication"],
                "diagnostics": single_result["diagnostics"],
            }
        )
    _write_json(run_dir / "pipeline_report.json", result)
    _emit(
        log_path,
        "pipeline",
        result["status"],
        models={name: value["status"] for name, value in model_results.items()},
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--region-index-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audited-exclusions", type=Path)
    parser.add_argument(
        "--deepseek-endpoint", action="append", dest="deepseek_endpoints"
    )
    parser.add_argument("--qwen-endpoint", action="append", dest="qwen_endpoints")
    parser.add_argument(
        "--deepseek-model",
        default=os.getenv("DEEPSEEK_MODEL", "DeepSeek-V4-Flash"),
    )
    parser.add_argument(
        "--qwen-model",
        default=os.getenv("QWEN_MODEL", "qwen3.8-27b-fp8"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--nonregion-candidate-limit", type=int, default=30)
    parser.add_argument("--region-candidate-limit", type=int, default=5)
    parser.add_argument("--comprehensive-candidate-limit", type=int, default=5)
    parser.add_argument("--region-bm25-min-score", type=float, default=0.0)
    parser.add_argument("--region-bge-min-score", type=float, default=0.4)
    parser.add_argument("--embedding-model")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--deepseek-workers", type=int, default=1)
    parser.add_argument("--qwen-workers", type=int, default=1)
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
    deepseek_endpoints = args.deepseek_endpoints or [
        value for value in (os.getenv("DS1"), os.getenv("DS2")) if value
    ]
    qwen_endpoints = args.qwen_endpoints or [
        value
        for value in (
            os.getenv("QWEN_LEGACY_ENDPOINT"),
            os.getenv("QWEN1"),
            os.getenv("QWEN2"),
        )
        if value
    ]
    if not deepseek_endpoints:
        raise SystemExit("provide --deepseek-endpoint or set DS1/DS2")
    if not qwen_endpoints:
        raise SystemExit(
            "provide --qwen-endpoint or set QWEN_LEGACY_ENDPOINT/QWEN1/QWEN2"
        )
    for name in (
        "limit",
        "nonregion_candidate_limit",
        "region_candidate_limit",
        "comprehensive_candidate_limit",
        "batch_size",
        "deepseek_workers",
        "qwen_workers",
        "max_tokens",
    ):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.retry_delay < 0 or args.request_interval < 0:
        raise SystemExit("retry delay and request interval must be non-negative")

    deepseek_client = DSClient(
        deepseek_endpoints,
        args.deepseek_model,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
        request_interval=args.request_interval,
        enable_thinking=args.enable_thinking,
        temperature=0.0,
    )
    qwen_client = DSClient(
        qwen_endpoints,
        args.qwen_model,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
        request_interval=args.request_interval,
        enable_thinking=args.enable_thinking,
        temperature=0.0,
    )
    try:
        result = run_pipeline(
            input_path=args.input,
            labels_path=args.labels,
            index_dir=args.index_dir,
            region_index_dir=args.region_index_dir,
            run_dir=args.run_dir,
            client=deepseek_client,
            model=args.deepseek_model,
            qwen_client=qwen_client,
            qwen_model=args.qwen_model,
            qwen_workers=args.qwen_workers,
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
            workers=args.deepseek_workers,
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
            temperature=0.0,
            enable_vision=False,
        )
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_dir": result["run_dir"],
                "models": {
                    name: {
                        "status": value["status"],
                        "output_root": value["output_root"],
                    }
                    for name, value in result["models"].items()
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
