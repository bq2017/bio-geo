"""Build retrieval texts that describe regional labels by place names only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .hybrid_candidate_retrieval import is_region_label_path


LIST_FIELDS = (
    "aliases",
    "contained_places",
    "representative_places",
    "parent_regions",
)


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}第{line_number}行不是JSON对象")
            yield line_number, value


def normalize_string_list(value: object, field: str, line_number: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"第{line_number}行{field}必须是字符串数组")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = item.strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def load_taxonomy_region_paths(path: Path) -> set[str]:
    paths: set[str] = set()
    for line_number, record in read_jsonl(path):
        label_path = str(record.get("knw_label") or record.get("full_path") or "").replace(
            "->", "@"
        ).strip()
        if not label_path:
            raise ValueError(f"标签释义第{line_number}行缺少标签路径")
        if is_region_label_path(label_path):
            paths.add(label_path)
    return paths


def build_record(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    label_path = str(record.get("label_path") or "").replace("->", "@").strip()
    region_name = str(record.get("region_name") or "").strip()
    region_type = str(record.get("region_type") or "").strip()
    if not label_path or not is_region_label_path(label_path):
        raise ValueError(f"第{line_number}行不是区域标签：{label_path}")
    if not region_name:
        raise ValueError(f"第{line_number}行region_name为空")
    if not region_type:
        raise ValueError(f"第{line_number}行region_type为空")

    values = {
        field: normalize_string_list(record.get(field), field, line_number)
        for field in LIST_FIELDS
    }
    exact_names = list(dict.fromkeys([region_name, *values["aliases"]]))
    retrieval_names = list(
        dict.fromkeys(
            [
                *exact_names,
                *values["contained_places"],
                *values["representative_places"],
            ]
        )
    )
    if not retrieval_names:
        raise ValueError(f"第{line_number}行没有可用的区域名称")

    embedding_parts = [f"区域名称：{region_name}"]
    for title, field in (
        ("别称", "aliases"),
        ("包含或对应的地理范围", "contained_places"),
        ("代表性地理实体", "representative_places"),
        ("所属上级区域", "parent_regions"),
    ):
        if values[field]:
            embedding_parts.append(f"{title}：{'、'.join(values[field])}")

    return {
        "label_path": label_path,
        "label_name": label_path.rsplit("@", 1)[-1],
        "region_name": region_name,
        "region_type": region_type,
        **values,
        "exact_names": exact_names,
        "bm25_text": " ".join(retrieval_names),
        "embedding_text": "。".join(embedding_parts) + "。",
    }


def build_region_retrieval_texts(
    input_path: Path,
    taxonomy_path: Path,
    output_path: Path,
    expected_count: int = 78,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, record in read_jsonl(input_path):
        built = build_record(record, line_number)
        label_path = built["label_path"]
        if label_path in seen:
            raise ValueError(f"第{line_number}行标签路径重复：{label_path}")
        seen.add(label_path)
        records.append(built)

    taxonomy_paths = load_taxonomy_region_paths(taxonomy_path)
    if expected_count and len(taxonomy_paths) != expected_count:
        raise ValueError(f"标签体系中的区域标签不是{expected_count}个：{len(taxonomy_paths)}")
    if seen != taxonomy_paths:
        missing = sorted(taxonomy_paths - seen)
        extra = sorted(seen - taxonomy_paths)
        raise ValueError(f"区域名称信息与标签体系不一致；缺少={missing}；多出={extra}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "input_records": len(records),
        "taxonomy_region_labels": len(taxonomy_paths),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build place-name retrieval texts for geography regional labels"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=78)
    args = parser.parse_args()
    summary = build_region_retrieval_texts(
        args.input, args.taxonomy, args.output, args.expected_count
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
