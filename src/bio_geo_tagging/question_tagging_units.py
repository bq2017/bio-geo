# -*- coding: utf-8 -*-
"""Expand merged questions into independent knowledge-tagging units."""

import argparse
import json
import logging
import os
from typing import Any, Dict, Iterable


QUESTION_FIELDS = {
    "parent_id",
    "question_id",
    "stem",
    "options",
    "answer",
    "answer_text",
    "analysis",
    "explanation",
    "image_description",
    "stem_image_url",
    "analysis_image_url",
    "flags",
}


def _safe_question_fields(question: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only question content fields that may be used for adjudication."""
    return {key: question[key] for key in QUESTION_FIELDS if key in question}


def configure_logging(log_file: str) -> None:
    """Configure the processing log at the requested local path."""
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
        force=True,
    )


def expand_question(question: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield an ordinary unit, or sub-questions plus one comprehensive pass."""
    root_question_id = question.get("question_id", "")
    root_stem = question.get("stem", "")
    sub_questions = question.get("sub_questions", [])
    if not isinstance(sub_questions, list):
        logging.warning("题目 %s 的 sub_questions 不是列表", root_question_id)
        sub_questions = []
    valid_sub_questions = [
        sub_question for sub_question in sub_questions if isinstance(sub_question, dict)
    ]
    if len(valid_sub_questions) != len(sub_questions):
        logging.warning("题目 %s 包含无效小题记录", root_question_id)

    root_unit = _safe_question_fields(question)
    root_unit["root_question_id"] = root_question_id
    root_unit["context_stem"] = ""
    if valid_sub_questions:
        root_unit["sub_questions"] = [
            _safe_question_fields(sub_question) for sub_question in valid_sub_questions
        ]
        root_unit["input_role"] = "whole_question_comprehensive"
    else:
        root_unit["input_role"] = "root"
    yield root_unit

    for sub_question in valid_sub_questions:
        sub_unit = _safe_question_fields(sub_question)
        sub_unit["input_role"] = "subquestion"
        sub_unit["root_question_id"] = root_question_id
        sub_unit["context_stem"] = root_stem
        image_description = question.get("image_description")
        if image_description:
            sub_unit["context_image_description"] = image_description
        for field in ("stem_image_url", "analysis_image_url"):
            value = question.get(field)
            if value:
                sub_unit[f"context_{field}"] = value
        yield sub_unit


def process_file(input_file: str, output_file: str, log_file: str) -> None:
    """Expand every merged JSONL question into tagging units."""
    configure_logging(log_file)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    root_count = 0
    comprehensive_count = 0
    sub_question_count = 0

    with open(input_file, "r", encoding="utf-8") as input_stream, open(
        output_file, "w", encoding="utf-8"
    ) as output_stream:
        for line_number, line in enumerate(input_stream, start=1):
            try:
                question = json.loads(line)
            except json.JSONDecodeError as error:
                logging.error("第 %s 行JSON解析失败: %s", line_number, error)
                continue

            for unit in expand_question(question):
                output_stream.write(json.dumps(unit, ensure_ascii=False) + "\n")
                if unit["input_role"] == "root":
                    root_count += 1
                elif unit["input_role"] == "whole_question_comprehensive":
                    comprehensive_count += 1
                else:
                    sub_question_count += 1

    print(f"普通题打标单元: {root_count}")
    print(f"整题综合标签打标单元: {comprehensive_count}")
    print(f"小题打标单元: {sub_question_count}")
    print(
        "打标单元总数: "
        f"{root_count + comprehensive_count + sub_question_count}"
    )
    print(f"处理完成，输出文件: {output_file}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Expand merged questions into independent tagging units."
    )
    parser.add_argument("--input", required=True, help="Merged JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--log-file", required=True, help="Processing log file")
    return parser


def main() -> None:
    """Run the command-line interface."""
    arguments = build_parser().parse_args()
    process_file(arguments.input, arguments.output, arguments.log_file)


if __name__ == "__main__":
    main()
