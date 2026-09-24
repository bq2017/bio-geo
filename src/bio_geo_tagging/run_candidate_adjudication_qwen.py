"""Run geography candidate adjudication with qwen3.8-27b-fp8."""

from __future__ import annotations

from bio_geo_tagging.run_candidate_adjudication import main as run_main


def main() -> int:
    return run_main(
        profile="qwen",
        default_model="qwen3.8-27b-fp8",
        model_environment_names=("QWEN_MODEL",),
        endpoint_environment_names=(
            "QWEN_LEGACY_ENDPOINT",
            "QWEN1",
            "QWEN2",
        ),
        default_endpoints=("http://172.22.0.35:9204/v1/chat/completions",),
        default_temperature=0.6,
        default_run_suffix="geography-candidate-adjudication-qwen",
    )


if __name__ == "__main__":
    raise SystemExit(main())
