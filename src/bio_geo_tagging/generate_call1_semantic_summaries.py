"""Generate formal call-1 label summaries from the complete original definitions."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


SYSTEM_PROMPT = """你是高中地理知识点标签编辑。你的任务是把原释义压缩成供知识点召回模型阅读的简明释义。

必须遵守：
1. 只依据输入的原释义，不补充原释义没有的信息。
2. 同时阅读定义、关键词、常见考查方式、易混淆区分四部分，不得只截取某个字段的前半段。
3. 每个标签生成一段语义完整的中文释义，说明该标签考查什么、题目出现什么任务时应考虑该标签，以及最必要的相近标签边界。
4. 保留会影响标签选择的条件和边界；删除例题式罗列、重复表述和不影响判断的细节。
5. 不使用“含、词、界”等缩写，不机械截断，不写省略号，不输出主标签或次标签判断。
6. 建议每条80至140个汉字；内容较简单时可以更短，但必须是完整句子。
7. label_path必须原样返回。仅输出JSON数组，不要输出Markdown或说明文字。

输出格式：
[{"label_path":"原路径","summary":"一段式简明释义"}]
"""


def load_source(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = json.loads(line)
            original = source.get("existing_interpretation")
            if not isinstance(original, dict):
                raise ValueError(f"第{line_number}行缺少existing_interpretation")
            label_path = str(source.get("full_path") or "").replace("->", "@").strip()
            if not label_path or label_path in seen_paths:
                raise ValueError(f"第{line_number}行标签路径为空或重复：{label_path}")
            seen_paths.add(label_path)
            records.append(
                {
                    "label_path": label_path,
                    "original_definition": str(original.get("definition") or "").strip(),
                    "original_keywords": str(original.get("keywords") or "").strip(),
                    "original_exam_methods": str(original.get("exam_methods") or "").strip(),
                    "original_distinction": str(original.get("distinction") or "").strip(),
                }
            )
    return records


def parse_json_array(text: str) -> list[dict[str, str]]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError("模型输出不是JSON数组")
    return value


def call_model(
    endpoint: str,
    model: str,
    api_key: str,
    batch: list[dict[str, object]],
    timeout: int,
) -> list[dict[str, str]]:
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "请处理以下标签：\n" + json.dumps(batch, ensure_ascii=False),
            },
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    content = result["choices"][0]["message"]["content"]
    return parse_json_array(content)


def validate_batch(
    source_batch: list[dict[str, object]], generated: list[dict[str, str]]
) -> list[dict[str, str]]:
    expected = [str(item["label_path"]) for item in source_batch]
    by_path: dict[str, str] = {}
    for item in generated:
        path = str(item.get("label_path") or "").strip()
        summary = str(item.get("summary") or "").strip()
        if path in by_path or path not in expected:
            raise ValueError(f"模型返回未知或重复路径：{path}")
        if not summary or "…" in summary or "..." in summary:
            raise ValueError(f"释义为空或含机械省略号：{path}")
        if len(summary) < 35:
            raise ValueError(f"释义过短，可能未覆盖完整含义：{path}")
        if len(summary) > 180:
            raise ValueError(f"释义超过180字符：{path}")
        by_path[path] = summary
    if set(by_path) != set(expected):
        missing = set(expected) - set(by_path)
        raise ValueError(f"模型漏掉标签：{sorted(missing)}")
    return [{"label_path": path, "summary": by_path[path]} for path in expected]


def read_completed(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                completed[item["label_path"]] = item
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-catalog", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://172.22.0.35:9204/v1")
    parser.add_argument("--model", default="DeepSeek-V4-Flash")
    parser.add_argument("--api-key-env", default="DS_API_KEY")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    source_records = load_source(args.source)
    completed = read_completed(args.output_jsonl)
    pending = [item for item in source_records if item["label_path"] not in completed]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get(args.api_key_env, "")

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        last_error: Exception | None = None
        for attempt in range(1, args.retries + 1):
            try:
                generated = call_model(
                    args.endpoint, args.model, api_key, batch, args.timeout
                )
                validated = validate_batch(batch, generated)
                with args.output_jsonl.open("a", encoding="utf-8", newline="\n") as handle:
                    for item in validated:
                        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                        completed[item["label_path"]] = item
                print(f"completed={len(completed)}/{len(source_records)}")
                last_error = None
                break
            except (ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError) as error:
                last_error = error
                print(f"attempt={attempt} failed={error}")
                time.sleep(min(attempt * 2, 6))
        if last_error is not None:
            raise RuntimeError(f"批次生成失败：{batch[0]['label_path']}") from last_error

    ordered = [completed[str(item["label_path"])] for item in source_records]
    if len(ordered) != 414:
        raise ValueError(f"输出数量不是414：{len(ordered)}")
    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for item in ordered:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with args.output_catalog.open("w", encoding="utf-8", newline="\n") as handle:
        for item in ordered:
            handle.write(f"{item['label_path']}｜{item['summary']}\n")
    print(f"validated=414 output={args.output_catalog}")


if __name__ == "__main__":
    main()
