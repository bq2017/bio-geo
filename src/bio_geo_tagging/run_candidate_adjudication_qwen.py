"""Run geography candidate adjudication with Qwen3.8-27B."""

from __future__ import annotations

from bio_geo_tagging.run_candidate_adjudication import main as run_main


def main() -> int:
    return run_main(
        profile="qwen",
        default_model="Qwen3.8-27B",
        model_environment_names=("QWEN_MODEL",),
        endpoint_environment_names=(
            "QWEN_LEGACY_ENDPOINT",
            "QWEN1",
            "QWEN2",
        ),
        default_run_suffix="geography-candidate-adjudication-qwen",
    )


if __name__ == "__main__":
    raise SystemExit(main())
