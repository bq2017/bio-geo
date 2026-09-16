"""Run the first DeepSeek call: retrieve candidate geography labels."""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from openai import OpenAI
from tqdm import tqdm


DEFAULT_MODEL = "DeepSeek-V4-Flash"
DEFAULT_BASE_URL = "http://172.22.0.35:9204/v1"
EXPECTED_LABEL_COUNT = 414

SYSTEM_PROMPT = """你是高中地理知识点候选标签召回器。

你的任务是独立理解当前打标对象，从本批给定的标签中找出所有可能成为最终答案的候选标签。本阶段只追求候选召回，不确定的相邻标签可以同时保留，不做最终取舍。另有其他批次的标签分别处理，你只判断本批标签。

判断要求：
1. 公共题干只用于理解当前打标对象。仅出现在公共题干中、但不参与当前设问解答的知识不能进入候选。
2. 结合题干、选项、已有答案和解析，判断完成当前设问实际需要的地理知识。
3. 错误选项涉及的知识、解析中的延伸知识和一般性背景知识不作为候选。
4. 只能返回标签目录中存在的完整标签路径，不补充父级、兄弟或层级近邻标签。
5. 候选最多20个，不要求凑满。

输出一个JSON对象：
{"candidate_labels":["完整标签路径"]}

【标签目录】
{catalog}
"""


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def load_catalog(
    catalog_path: Path, expected_count: int = EXPECTED_LABEL_COUNT
) -> tuple[str, set[str]]:
    """Load the prompt-ready label catalog and validate unique paths."""
    lines = [
        line.strip()
        for line in catalog_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    paths: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if "｜" not in line:
            raise ValueError(f"标签目录第{line_number}行缺少分隔符｜")
        label_path, summary = line.split("｜", 1)
        if not label_path.startswith("知识点@") or not summary.strip():
            raise ValueError(f"标签目录第{line_number}行格式无效")
        paths.append(label_path.strip())

    if len(lines) != expected_count:
        raise ValueError(f"标签目录应有{expected_count}条，实际为{len(lines)}条")
    if len(set(paths)) != len(paths):
        raise ValueError("标签目录存在重复路径")
    return "\n".join(lines), set(paths)


def split_catalog(catalog: str, parts: int = 3) -> list[tuple[str, set[str]]]:
    if parts <= 0:
        raise ValueError("标签目录分批数必须大于0")
    lines = catalog.splitlines()
    chunks = [lines[index::parts] for index in range(parts)]
    return [
        ("\n".join(chunk), {line.split("｜", 1)[0].strip() for line in chunk})
        for chunk in chunks
    ]


def make_unit_key(unit: dict[str, Any]) -> str:
    question_id = as_text(unit.get("question_id"))
    if not question_id:
        raise ValueError("题目缺少question_id")
    root_question_id = as_text(
        unit.get("root_question_id")
        or unit.get("parent_id")
        or question_id
    )
    input_role = as_text(unit.get("input_role")) or "root"
    return f"{root_question_id}|{question_id}|{input_role}"


def load_units(input_path: Path) -> tuple[list[dict[str, Any]], int]:
    units: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    skipped_empty_stem = 0
    with input_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                unit = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"题目文件第{line_number}行JSON无效：{error}") from error
            if not isinstance(unit, dict):
                raise ValueError(f"题目文件第{line_number}行不是JSON对象")
            unit_key = make_unit_key(unit)
            if unit_key in seen_keys:
                raise ValueError(f"题目文件存在重复打标单元：{unit_key}")
            seen_keys.add(unit_key)
            if not as_text(unit.get("stem")):
                skipped_empty_stem += 1
                logging.warning("跳过题干为空的打标单元：%s", unit_key)
                continue
            units.append(unit)
    return units, skipped_empty_stem


def build_question_text(unit: dict[str, Any]) -> str:
    """Render only fields allowed to reach DS; existing labels are excluded."""
    sections: list[str] = []
    context_stem = render_value(unit.get("context_stem"))
    if context_stem:
        sections.append(f"【公共题干，仅作为上下文】\n{context_stem}")

    sections.append(f"【当前打标对象】\n{render_value(unit.get('stem'))}")
    sections.append(f"【选项】\n{render_value(unit.get('options'))}")
    sections.append(f"【答案】\n{render_value(unit.get('answer'))}")
    analysis = unit.get("analysis")
    if analysis is None:
        analysis = unit.get("explanation")
    sections.append(f"【解析】\n{render_value(analysis)}")

    image_description = render_value(unit.get("image_description"))
    if image_description:
        sections.append(f"【图片描述】\n{image_description}")
    return "\n\n".join(sections)


def validate_result(
    result: Any, allowed_paths: set[str]
) -> tuple[list[str], str | None]:
    if not isinstance(result, dict):
        raise ValueError("DS输出不是JSON对象")
    labels = result.get("candidate_labels")
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        raise ValueError("candidate_labels必须是字符串数组")
    labels = [item.strip() for item in labels]
    if any(not item for item in labels):
        raise ValueError("candidate_labels包含空路径")
    if len(labels) != len(set(labels)):
        raise ValueError("candidate_labels包含重复路径")
    if len(labels) > 20:
        raise ValueError(f"候选标签超过20个：{len(labels)}")
    unknown = [item for item in labels if item not in allowed_paths]
    if unknown:
        raise ValueError(f"DS返回目录外标签：{unknown}")

    uncovered_topic = result.get("uncovered_topic")
    if uncovered_topic is not None and not isinstance(uncovered_topic, str):
        raise ValueError("uncovered_topic必须是字符串或null")
    if isinstance(uncovered_topic, str):
        uncovered_topic = uncovered_topic.strip() or None
    return labels, uncovered_topic


def normalize_shard_result(result: Any, allowed_paths: set[str]) -> list[str]:
    if not isinstance(result, dict):
        raise ValueError("DS输出不是JSON对象")
    raw_labels = result.get("candidate_labels")
    if not isinstance(raw_labels, list) or not all(
        isinstance(item, str) for item in raw_labels
    ):
        raise ValueError("candidate_labels必须是字符串数组")

    labels: list[str] = []
    seen: set[str] = set()
    ignored = 0
    for raw_label in raw_labels:
        label = raw_label.strip().split("｜", 1)[0].strip()
        if label and not label.startswith("知识点@"):
            label = f"知识点@{label}"
        if label not in allowed_paths:
            ignored += 1
            continue
        if label not in seen:
            seen.add(label)
            labels.append(label)

    if len(labels) > 20:
        raise ValueError(f"本批候选标签超过20个：{len(labels)}")
    if ignored:
        logging.info("忽略本批目录外候选：%s个", ignored)
    return labels


def parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()
    result = json.loads(cleaned)
    if not isinstance(result, dict):
        raise ValueError("DS输出不是JSON对象")
    return result


class DeepSeekCandidateRetriever:
    def __init__(
        self,
        catalog_parts: list[tuple[str, set[str]]],
        model: str,
        base_url: str,
        api_key: str | None,
        timeout: float,
    ) -> None:
        self.client = OpenAI(
            api_key=api_key or "not-required",
            base_url=base_url,
            timeout=timeout,
            max_retries=1,
        )
        self.catalog_parts = [
            (SYSTEM_PROMPT.replace("{catalog}", catalog), allowed_paths)
            for catalog, allowed_paths in catalog_parts
        ]
        self.model = model

    def retrieve(self, unit: dict[str, Any]) -> tuple[list[str], str | None]:
        question_text = build_question_text(unit)
        candidates: list[str] = []
        for part_number, (system_prompt, allowed_paths) in enumerate(
            self.catalog_parts, start=1
        ):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question_text},
                ],
                max_tokens=1024,
                stream=True,
                temperature=0,
            )
            content = "".join(
                chunk.choices[0].delta.content or ""
                for chunk in response
                if chunk.choices
            )
            if not content.strip():
                raise ValueError(f"第{part_number}批DS返回空内容")
            labels = normalize_shard_result(parse_json_object(content), allowed_paths)
            candidates.extend(labels)
        if len(candidates) > 20:
            raise ValueError(f"三批候选合并后超过20个：{len(candidates)}，未截断")
        return candidates, None


def load_completed(output_path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"输出文件第{line_number}行JSON无效：{error}") from error
            if record.get("status") == "completed":
                unit_key = as_text(record.get("unit_key"))
                if not unit_key:
                    raise ValueError(f"输出文件第{line_number}行缺少unit_key")
                if unit_key in completed:
                    raise ValueError(f"输出文件存在重复完成记录：{unit_key}")
                completed[unit_key] = record
    return completed


def run_retrieval(
    units: list[dict[str, Any]],
    catalog: str,
    allowed_paths: set[str],
    output_path: Path,
    model: str,
    base_url: str,
    api_key: str | None,
    concurrency: int,
    retries: int,
    timeout: float,
    limit: int | None,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_parts = split_catalog(catalog)
    completed = load_completed(output_path)
    unit_keys = {make_unit_key(unit) for unit in units}
    stale_keys = set(completed).difference(unit_keys)
    if stale_keys:
        raise ValueError(f"输出文件含有不属于当前输入的记录：{sorted(stale_keys)}")
    for record in completed.values():
        validate_result(record, allowed_paths)
    pending = [unit for unit in units if make_unit_key(unit) not in completed]
    if limit is not None:
        pending = pending[:limit]

    thread_local_retrievers: dict[int, DeepSeekCandidateRetriever] = {}
    retriever_lock = Lock()

    def get_retriever() -> DeepSeekCandidateRetriever:
        import threading

        thread_id = threading.get_ident()
        with retriever_lock:
            if thread_id not in thread_local_retrievers:
                thread_local_retrievers[thread_id] = DeepSeekCandidateRetriever(
                    catalog_parts,
                    model,
                    base_url,
                    api_key,
                    timeout,
                )
            return thread_local_retrievers[thread_id]

    def process(unit: dict[str, Any]) -> dict[str, Any]:
        unit_key = make_unit_key(unit)
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                candidate_labels, uncovered_topic = get_retriever().retrieve(unit)
                return {
                    "unit_key": unit_key,
                    "question_id": as_text(unit.get("question_id")),
                    "root_question_id": as_text(
                        unit.get("root_question_id")
                        or unit.get("parent_id")
                        or unit.get("question_id")
                    ),
                    "input_role": as_text(unit.get("input_role")) or "root",
                    "candidate_labels": candidate_labels,
                    "uncovered_topic": uncovered_topic,
                    "model": model,
                    "endpoint": base_url,
                    "status": "completed",
                }
            except Exception as error:
                last_error = error
                logging.warning(
                    "题目 %s 第 %s 次调用失败：%s", unit_key, attempt, error
                )
        raise RuntimeError(f"题目{unit_key}调用失败") from last_error

    errors = 0
    with (
        output_path.open("a", encoding="utf-8") as output_stream,
        ThreadPoolExecutor(max_workers=concurrency) as executor,
        tqdm(total=len(pending), desc="Retrieving candidates") as progress,
    ):
        futures = {executor.submit(process, unit): unit for unit in pending}
        for future in as_completed(futures):
            try:
                record = future.result()
                completed[record["unit_key"]] = record
                output_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_stream.flush()
            except Exception as error:
                errors += 1
                logging.error("%s", error)
            progress.update(1)

    if errors == 0 and limit is None and len(completed) == len(units):
        with output_path.open("w", encoding="utf-8", newline="\n") as stream:
            for unit in units:
                record = completed[make_unit_key(unit)]
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "total_units": len(units),
        "already_completed": len(completed) - (len(pending) - errors),
        "attempted": len(pending),
        "completed": len(pending) - errors,
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve candidate geography labels.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--base-url", default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    )
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.concurrency <= 0 or args.retries <= 0 or args.timeout <= 0:
        raise SystemExit("concurrency、retries和timeout必须大于0")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("limit必须大于0")

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
        force=True,
    )
    catalog, allowed_paths = load_catalog(args.catalog)
    units, skipped_empty_stem = load_units(args.input)
    summary = run_retrieval(
        units,
        catalog,
        allowed_paths,
        args.output,
        args.model,
        args.base_url,
        os.getenv("DEEPSEEK_API_KEY"),
        args.concurrency,
        args.retries,
        args.timeout,
        args.limit,
    )
    summary["skipped_empty_stem"] = skipped_empty_stem
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["errors"]:
        raise SystemExit("存在失败题目，请重复执行同一命令继续补跑")


if __name__ == "__main__":
    main()
