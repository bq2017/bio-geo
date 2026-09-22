"""Ask DS to select directly assessed geography labels from recalled candidates."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from bio_geo_tagging.adjudication import run_adjudication
from bio_geo_tagging.ds import DSClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--audited-exclusions", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--endpoint", action="append", dest="endpoints")
    parser.add_argument("--model", default=os.getenv("MODEL", "DeepSeek-V4-Flash"))
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
    parser.add_argument("--workers", type=int, default=1)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    endpoints = args.endpoints or [
        value for value in (os.getenv("DS1"), os.getenv("DS2")) if value
    ]
    if not endpoints:
        raise SystemExit("provide --endpoint or set DS1/DS2")
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
    run_dir = args.run_dir or Path("runs") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-geography-candidate-adjudication"
    )
    client = DSClient(
        endpoints,
        args.model,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
        request_interval=args.request_interval,
        enable_thinking=args.enable_thinking,
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
    )
    print(json.dumps({"run_dir": str(run_dir), **report}, ensure_ascii=False))
    return 0 if report["success"] == report["input"] and report["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
