from __future__ import annotations

import json

from bio_geo_tagging import ds


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(
            {
                "choices": [{"message": {"content": '{"selected": []}'}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        ).encode("utf-8")


def test_client_sends_profile_temperature_and_thinking_setting(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(ds, "urlopen", fake_urlopen)
    client = ds.DSClient(
        ["http://172.22.0.35:9204/v1/chat/completions"],
        "qwen3.8-27b-fp8",
        temperature=0.6,
        enable_thinking=False,
        timeout=600,
    )

    client.chat([{"role": "user", "content": "输出ok即可"}], max_tokens=1024)

    assert captured["payload"] == {
        "model": "qwen3.8-27b-fp8",
        "messages": [{"role": "user", "content": "输出ok即可"}],
        "temperature": 0.6,
        "max_tokens": 1024,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert captured["timeout"] == 600
