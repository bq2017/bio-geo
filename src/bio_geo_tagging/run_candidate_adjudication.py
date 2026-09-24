"""Run geography candidate adjudication with an OpenAI-compatible model."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from bio_geo_tagging.adjudication import run_adjudication
from bio_geo_tagging.ds import DSClient


def parse_args(
    argv: list[str] | None = None,
    *,
    description: str = __doc__,
    default_model: str = "DeepSeek-V4-Flash",
    default_temperature: float = 0.0,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--audited-exclusions", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--endpoint", action="append", dest="endpoints")
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--limit", type=int, help="limit complete root questions")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=1.0)
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="minimum seconds between HTTP attempt starts across all workers",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=default_temperature)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--enable-vision",
        action="store_true",
        help="send question image URLs as OpenAI-compatible multimodal content",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        help="request thinking mode through chat_template_kwargs",
    )
    thinking.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="disable thinking mode through chat_template_kwargs",
    )
    parser.set_defaults(enable_thinking=None)
    return parser.parse_args(argv)


def _first_environment_value(names: tuple[str, ...]) -> str | None:
    return next((value for name in names if (value := os.getenv(name))), None)


def main(
    argv: list[str] | None = None,
    *,
    profile: str = "deepseek",
    default_model: str = "DeepSeek-V4-Flash",
    model_environment_names: tuple[str, ...] = ("MODEL",),
    endpoint_environment_names: tuple[str, ...] = ("DS1", "DS2"),
    default_endpoints: tuple[str, ...] = (),
    default_temperature: float = 0.0,
    default_run_suffix: str = "geography-candidate-adjudication-ds",
) -> int:
    resolved_default_model = (
        _first_environment_value(model_environment_names) or default_model
    )
    args = parse_args(
        argv,
        description=f"Run geography candidate adjudication with the {profile} profile.",
        default_model=resolved_default_model,
        default_temperature=default_temperature,
    )
    endpoints = args.endpoints or [
        value
        for name in endpoint_environment_names
        if (value := os.getenv(name))
    ] or list(default_endpoints)
    if not endpoints:
        environment_hint = "/".join(endpoint_environment_names)
        raise SystemExit(f"provide --endpoint or set {environment_hint}")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.retry_delay < 0:
        raise SystemExit("--retry-delay must be non-negative")
    if args.request_interval < 0:
        raise SystemExit("--request-interval must be non-negative")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be positive")
    if args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")
    run_dir = args.run_dir or Path("runs") / datetime.now().strftime(
        f"%Y%m%d-%H%M%S-{default_run_suffix}"
    )
    client = DSClient(
        endpoints,
        args.model,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
        request_interval=args.request_interval,
        enable_thinking=args.enable_thinking,
        temperature=args.temperature,
    )
    print(
        json.dumps(
            {
                "status": "started",
                "run_dir": str(run_dir),
                "profile": profile,
                "model": args.model,
                "temperature": args.temperature,
                "vision_enabled": args.enable_vision,
                "workers": args.workers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    report = run_adjudication(
        args.units,
        args.candidates,
        args.labels,
        run_dir,
        client,
        model=args.model,
        limit=args.limit,
        max_tokens=args.max_tokens,
        workers=args.workers,
        audited_exclusions_path=args.audited_exclusions,
        enable_thinking=args.enable_thinking,
        temperature=args.temperature,
        enable_vision=args.enable_vision,
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0 if report["success"] == report["input"] and report["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
