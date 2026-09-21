"""Attach question image URLs by question_id without loading the full URL map."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


IMAGE_KEY_RE = re.compile(r'^\s*"(?P<question_id>[^"]+)"\s*:\s*')
IMAGE_FIELDS = {
    "stemImageUrl": "stem_image_url",
    "analysisImageUrl": "analysis_image_url",
}


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}第{line_number}行不是JSON对象")
            yield line_number, value


def question_id(record: dict[str, Any], location: str) -> str:
    value = str(record.get("question_id") or "").strip()
    if not value:
        raise ValueError(f"{location}缺少question_id")
    return value


def collect_question_ids(
    questions: list[dict[str, Any]],
) -> tuple[set[str], int]:
    identifiers: set[str] = set()
    subquestion_count = 0
    for row_number, question in enumerate(questions, start=1):
        root_id = question_id(question, f"题目文件第{row_number}行")
        if root_id in identifiers:
            raise ValueError(f"题目文件存在重复question_id：{root_id}")
        identifiers.add(root_id)
        sub_questions = question.get("sub_questions") or []
        if not isinstance(sub_questions, list) or not all(
            isinstance(item, dict) for item in sub_questions
        ):
            raise ValueError(f"题目文件第{row_number}行sub_questions必须是对象数组")
        for index, sub_question in enumerate(sub_questions, start=1):
            sub_id = question_id(
                sub_question, f"题目文件第{row_number}行第{index}个小题"
            )
            if sub_id in identifiers:
                raise ValueError(f"题目文件存在重复question_id：{sub_id}")
            identifiers.add(sub_id)
            subquestion_count += 1
    return identifiers, subquestion_count


def load_selected_image_records(
    path: Path, target_ids: set[str]
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = IMAGE_KEY_RE.match(line)
            if match is None:
                continue
            identifier = match.group("question_id")
            if identifier not in target_ids:
                continue
            payload = line[match.end() :].strip()
            if payload.endswith(","):
                payload = payload[:-1].rstrip()
            try:
                value = json.loads(payload)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"图片文件第{line_number}行题目{identifier}的记录无法解析；"
                    "图片文件必须每个题目占一行"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"图片文件第{line_number}行题目{identifier}的图片信息不是对象"
                )
            if identifier in selected:
                raise ValueError(f"图片文件存在重复question_id：{identifier}")
            selected[identifier] = value
    return selected


def attach_image_fields(
    record: dict[str, Any], image_record: dict[str, Any] | None
) -> int:
    if image_record is None:
        return 0
    attached = 0
    for source_field, output_field in IMAGE_FIELDS.items():
        value = image_record.get(source_field)
        if isinstance(value, str):
            value = value.strip()
        if value in (None, "", []):
            continue
        record[output_field] = value
        attached += 1
    return attached


def attach_question_images(
    input_path: Path,
    image_map_path: Path,
    output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    questions = [record for _, record in read_jsonl(input_path)]
    target_ids, subquestion_count = collect_question_ids(questions)
    image_records = load_selected_image_records(image_map_path, target_ids)

    matched_root_questions = 0
    matched_subquestions = 0
    attached_url_fields = 0
    matched_ids: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for question in questions:
            root_id = str(question["question_id"]).strip()
            root_images = image_records.get(root_id)
            root_attached = attach_image_fields(question, root_images)
            if root_attached:
                matched_root_questions += 1
                attached_url_fields += root_attached
                matched_ids.add(root_id)
            for sub_question in question.get("sub_questions") or []:
                sub_id = str(sub_question["question_id"]).strip()
                sub_images = image_records.get(sub_id)
                sub_attached = attach_image_fields(sub_question, sub_images)
                if sub_attached:
                    matched_subquestions += 1
                    attached_url_fields += sub_attached
                    matched_ids.add(sub_id)
            output.write(json.dumps(question, ensure_ascii=False) + "\n")

    summary = {
        "input_questions": len(questions),
        "input_subquestions": subquestion_count,
        "target_question_ids": len(target_ids),
        "image_records_found": len(image_records),
        "matched_root_questions": matched_root_questions,
        "matched_subquestions": matched_subquestions,
        "matched_question_ids": len(matched_ids),
        "attached_url_fields": attached_url_fields,
        "image_records_without_usable_url": len(set(image_records) - matched_ids),
        "question_ids_without_image_record": len(target_ids - set(image_records)),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach stem and analysis image URLs to questions by question_id"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    summary = attach_question_images(
        input_path=args.input,
        image_map_path=args.image_map,
        output_path=args.output,
        summary_path=args.summary_output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
