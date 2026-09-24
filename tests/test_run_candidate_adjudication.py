from __future__ import annotations

import json

import pytest

from bio_geo_tagging import run_candidate_adjudication as cli
from bio_geo_tagging import run_candidate_adjudication_ds as ds_cli
from bio_geo_tagging import run_candidate_adjudication_qwen as qwen_cli


@pytest.mark.parametrize(
    ("wrapper", "expected"),
    [
        (
            ds_cli,
            {
                "profile": "deepseek",
                "default_model": "DeepSeek-V4-Flash",
                "model_environment_names": ("DEEPSEEK_MODEL",),
                "endpoint_environment_names": ("DS1", "DS2"),
                "default_temperature": 0.0,
                "default_run_suffix": "geography-candidate-adjudication-ds",
            },
        ),
        (
            qwen_cli,
            {
                "profile": "qwen",
                "default_model": "qwen3.8-27b-fp8",
                "model_environment_names": ("QWEN_MODEL",),
                "endpoint_environment_names": (
                    "QWEN_LEGACY_ENDPOINT",
                    "QWEN1",
                    "QWEN2",
                ),
                "default_endpoints": (
                    "http://172.22.0.35:9204/v1/chat/completions",
                ),
                "default_temperature": 0.6,
                "default_run_suffix": "geography-candidate-adjudication-qwen",
            },
        ),
    ],
)
def test_model_specific_entrypoint_selects_expected_profile(
    monkeypatch, wrapper, expected
):
    captured = {}

    def fake_run_main(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(wrapper, "run_main", fake_run_main)

    assert wrapper.main() == 0
    assert captured == expected


def test_qwen_profile_uses_qwen_environment_and_shared_adjudication(
    monkeypatch, tmp_path, capsys
):
    captured = {}

    class FakeClient:
        def __init__(self, endpoints, model, **kwargs):
            captured["client"] = {
                "endpoints": endpoints,
                "model": model,
                **kwargs,
            }

    def fake_run_adjudication(*args, **kwargs):
        captured["run"] = {"args": args, "kwargs": kwargs}
        return {"input": 1, "success": 1, "error": 0}

    monkeypatch.delenv("QWEN_LEGACY_ENDPOINT", raising=False)
    monkeypatch.delenv("QWEN_MODEL", raising=False)
    monkeypatch.setattr(cli, "DSClient", FakeClient)
    monkeypatch.setattr(cli, "run_adjudication", fake_run_adjudication)

    exit_code = cli.main(
        [
            "--units",
            str(tmp_path / "units.jsonl"),
            "--candidates",
            str(tmp_path / "candidates.jsonl"),
            "--labels",
            str(tmp_path / "labels.jsonl"),
            "--run-dir",
            str(tmp_path / "qwen-run"),
            "--disable-thinking",
            "--workers",
            "3",
            "--max-tokens",
            "512",
        ],
        profile="qwen",
        default_model="qwen3.8-27b-fp8",
        model_environment_names=("QWEN_MODEL",),
        endpoint_environment_names=("QWEN_LEGACY_ENDPOINT", "QWEN1", "QWEN2"),
        default_endpoints=("http://172.22.0.35:9204/v1/chat/completions",),
        default_temperature=0.6,
        default_run_suffix="geography-candidate-adjudication-qwen",
    )

    assert exit_code == 0
    assert captured["client"]["endpoints"] == [
        "http://172.22.0.35:9204/v1/chat/completions"
    ]
    assert captured["client"]["model"] == "qwen3.8-27b-fp8"
    assert captured["client"]["enable_thinking"] is False
    assert captured["client"]["temperature"] == 0.6
    assert captured["run"]["kwargs"]["model"] == "qwen3.8-27b-fp8"
    assert captured["run"]["kwargs"]["temperature"] == 0.6
    assert captured["run"]["kwargs"]["workers"] == 3
    assert captured["run"]["kwargs"]["max_tokens"] == 512
    started = json.loads(capsys.readouterr().out.splitlines()[0])
    assert started["profile"] == "qwen"
    assert started["model"] == "qwen3.8-27b-fp8"
    assert started["temperature"] == 0.6


def test_profile_without_endpoint_names_the_expected_environment_variables(
    monkeypatch, tmp_path
):
    for name in ("QWEN_LEGACY_ENDPOINT", "QWEN1", "QWEN2"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(
        SystemExit,
        match="provide --endpoint or set QWEN_LEGACY_ENDPOINT/QWEN1/QWEN2",
    ):
        cli.main(
            [
                "--units",
                str(tmp_path / "units.jsonl"),
                "--candidates",
                str(tmp_path / "candidates.jsonl"),
                "--labels",
                str(tmp_path / "labels.jsonl"),
            ],
            profile="qwen",
            default_model="qwen3.8-27b-fp8",
            model_environment_names=("QWEN_MODEL",),
            endpoint_environment_names=(
                "QWEN_LEGACY_ENDPOINT",
                "QWEN1",
                "QWEN2",
            ),
        )
