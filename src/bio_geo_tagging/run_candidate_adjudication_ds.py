"""Run geography candidate adjudication with DeepSeek-V4-Flash."""

from __future__ import annotations

from bio_geo_tagging.run_candidate_adjudication import main as run_main


def main() -> int:
    return run_main(
        profile="deepseek",
        default_model="DeepSeek-V4-Flash",
        model_environment_names=("DEEPSEEK_MODEL",),
        endpoint_environment_names=("DS1", "DS2"),
        default_run_suffix="geography-candidate-adjudication-ds",
    )


if __name__ == "__main__":
    raise SystemExit(main())
