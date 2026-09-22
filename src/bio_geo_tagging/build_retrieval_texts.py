"""Build deterministic BM25 and embedding texts from existing label definitions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


KEYWORD_SEPARATOR_RE = re.compile(r"[、，,；;\n]+")
LABEL_FIELD_WEIGHTS = {
    "label_name": 4,
    "label_path": 2,
    "definition": 2,
    "core_concepts": 2,
    "common_assessments": 1,
}
CHINA_REGION_BRANCHES = {"中国地理分区", "中国地理微区域"}
WORLD_REGION_BRANCHES = {
    "世界主要的大洲",
    "世界重要的地区",
    "世界重要的国家",
    "世界地理微区域",
}
NON_UMBRELLA_COMPREHENSIVE_LABELS = {
    "知识点@区域发展@区域发展@生态脆弱区的综合治理",
    "知识点@区域发展@区域发展@北方农牧交错带土地退化的综合治理",
    "知识点@区域发展@区域协调@流域综合开发",
    "知识点@选修地理（旧）@旅游地理综合题",
    "知识点@选修地理（旧）@环境保护综合题",
}


def uses_weighted_bm25(label_path: str) -> bool:
    parts = label_path.split("@")
    is_region = (
        len(parts) >= 4
        and (
            (parts[1] == "中国地理" and parts[2] in CHINA_REGION_BRANCHES)
            or (parts[1] == "世界地理" and parts[2] in WORLD_REGION_BRANCHES)
        )
    )
    is_strict_comprehensive = (
        "综合" in parts[-1]
        and label_path not in NON_UMBRELLA_COMPREHENSIVE_LABELS
    )
    return not is_region and not is_strict_comprehensive


def normalize_label_path(value: object) -> str:
    return str(value or "").replace("->", "@").strip()


def normalize_text(value: object) -> str:
    if isinstance(value, list):
        return "、".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def parse_keywords(value: object) -> list[str]:
    if isinstance(value, list):
        raw_items = [str(item) for item in value]
    else:
        raw_items = KEYWORD_SEPARATOR_RE.split(str(value or ""))

    keywords: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        keyword = raw_item.strip()
        if not keyword or keyword in seen:
            continue
        seen.add(keyword)
        keywords.append(keyword)
    return keywords


def load_source(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = json.loads(line)
            interpretation = source.get("existing_interpretation")
            if not isinstance(interpretation, dict):
                raise ValueError(f"第{line_number}行缺少existing_interpretation")

            label_path = normalize_label_path(
                source.get("full_path") or source.get("knw_label")
            )
            if not label_path:
                raise ValueError(f"第{line_number}行标签路径为空")
            if label_path in seen_paths:
                raise ValueError(f"第{line_number}行标签路径重复：{label_path}")
            seen_paths.add(label_path)

            records.append(
                {
                    "label_path": label_path,
                    "definition": normalize_text(interpretation.get("definition")),
                    "keywords": parse_keywords(interpretation.get("keywords")),
                    "exam_methods": normalize_text(
                        interpretation.get("exam_methods")
                    ),
                    "distinction": normalize_text(
                        interpretation.get("distinction")
                    ),
                }
            )
    return records


def build_record(source: dict[str, Any]) -> dict[str, Any]:
    label_path = source["label_path"]
    label_name = label_path.rsplit("@", 1)[-1]
    definition = source["definition"]
    keywords = source["keywords"]
    assessment_scope = source["exam_methods"]
    keyword_text = " ".join(keywords)

    field_weights = LABEL_FIELD_WEIGHTS if uses_weighted_bm25(label_path) else {}
    if field_weights:
        weighted_fields = (
            (label_name, field_weights["label_name"]),
            (label_path, field_weights["label_path"]),
            (definition, field_weights["definition"]),
            (keyword_text, field_weights["core_concepts"]),
            (assessment_scope, field_weights["common_assessments"]),
        )
        bm25_text = " ".join(
            text
            for text, weight in weighted_fields
            if text
            for _ in range(weight)
        )
    else:
        bm25_text = " ".join(
            part
            for part in (label_path, keyword_text, definition, assessment_scope)
            if part
        )
    embedding_text = "。".join(
        part.rstrip("。")
        for part in (label_path, definition, assessment_scope)
        if part
    )
    if embedding_text:
        embedding_text += "。"

    return {
        "label_path": label_path,
        "label_name": label_name,
        "positive_definition": definition,
        "keywords": keywords,
        "assessment_scope": assessment_scope,
        "bm25_field_weights": field_weights,
        "bm25_text": bm25_text,
        "embedding_text": embedding_text,
    }


def build_review_record(source: dict[str, Any]) -> dict[str, str] | None:
    missing_fields: list[str] = []
    if not source["definition"]:
        missing_fields.append("definition")
    if not source["keywords"]:
        missing_fields.append("keywords")
    if not source["exam_methods"]:
        missing_fields.append("exam_methods")
    if not missing_fields:
        return None
    return {
        "label_path": source["label_path"],
        "review_reason": "原释义缺少字段：" + "、".join(missing_fields),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_output(
    source_records: list[dict[str, Any]], output_records: list[dict[str, Any]]
) -> None:
    expected_paths = [record["label_path"] for record in source_records]
    actual_paths = [record["label_path"] for record in output_records]
    if actual_paths != expected_paths:
        raise ValueError("输出标签路径、顺序或数量与输入不一致")

    for source, output in zip(source_records, output_records):
        expected_keywords = source["keywords"]
        expected_definition = source["definition"]
        expected_scope = source["exam_methods"]
        expected = build_record(source)
        if output != expected:
            raise ValueError(f"输出字段不是确定性生成结果：{source['label_path']}")
        if output["keywords"] != expected_keywords:
            raise ValueError(f"关键词校验失败：{source['label_path']}")
        if output["positive_definition"] != expected_definition:
            raise ValueError(f"definition校验失败：{source['label_path']}")
        if output["assessment_scope"] != expected_scope:
            raise ValueError(f"exam_methods校验失败：{source['label_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic retrieval texts from existing definitions"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--expected-count", type=int, default=0)
    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit不能为负数")
    source_records = load_source(args.input)
    if args.limit:
        source_records = source_records[: args.limit]
    if args.expected_count and len(source_records) != args.expected_count:
        raise ValueError(
            f"处理数量不是{args.expected_count}：{len(source_records)}"
        )

    output_records = [build_record(record) for record in source_records]
    review_records = [
        review
        for record in source_records
        if (review := build_review_record(record)) is not None
    ]
    validate_output(source_records, output_records)
    write_jsonl(args.output, output_records)
    write_jsonl(args.review_output, review_records)

    print(
        json.dumps(
            {
                "input_records": len(source_records),
                "output_records": len(output_records),
                "review_records": len(review_records),
                "output": str(args.output),
                "review_output": str(args.review_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
