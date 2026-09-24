# -*- coding: utf-8 -*-
"""Clean question content and merge sub-questions into parent questions."""

import argparse
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from tqdm import tqdm


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


def convert_sup_sub_to_latex_robust(text: str) -> str:
    """Convert HTML sup and sub tags to LaTeX notation."""
    while "</sup><sup>" in text:
        text = re.sub(r"</sup>\s*<sup>", "", text)
    while "</sub><sub>" in text:
        text = re.sub(r"</sub>\s*<sub>", "", text)
    text = re.sub(r"<sup>\s*(.*?)\s*</sup>", r"^{\1}", text)
    text = re.sub(r"<sub>\s*(.*?)\s*</sub>", r"_{\1}", text)
    return text


def normalize_text(text: Any) -> str:
    """Clean and normalize question text using the mentor script rules."""
    if not text or not isinstance(text, str):
        return ""

    text = re.sub(r"(?:\$\s*)*<img[^>]*>(?:\s*<br\s*/?>)*", " ", text)
    text = convert_sup_sub_to_latex_robust(text)
    text = text.replace("$(\\qquad)$", "(   )")
    text = re.sub(
        r'<span class="underline fillblank".*?>.*?</span>', "______", text
    )

    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()

    cleaned_text = re.sub(r"(?:\$\s*)+", " ", cleaned_text)
    cleaned_text = re.sub(
        r"\s*(?:\\!\s*)*%\s*(?:\\!\s*)*(?:\\text{\s*})?", r"\%", cleaned_text
    )

    full_width_punctuations_to_keep = {"，", "。", "；", "：", "？", "！"}
    result = []
    for char in cleaned_text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E and char not in full_width_punctuations_to_keep:
            result.append(chr(code - 0xFEE0))
        else:
            result.append(char)
    normalized_str = "".join(result)

    replacements = {
        "﹣": "-",
        "−": "-",
        "～": "~",
        "≥": ">=",
        "≤": "<=",
        "≠": "!=",
        "\u3000": " ",
        "\u00A0": " ",
        "\n": " ",
        "\r": " ",
    }
    for old, new in replacements.items():
        normalized_str = normalized_str.replace(old, new)

    normalized_str = re.sub(r"\s+", " ", normalized_str)
    return normalized_str.strip()


def format_options(options: List[Dict[str, Any]]) -> str:
    """Format the original option list as human-readable text."""
    if not options or not isinstance(options, list):
        return ""

    formatted_parts = []
    for option_data in options:
        label = option_data.get("title")
        content_html = option_data.get("htmlCode")

        if label and content_html:
            cleaned_content = normalize_text(content_html)
            formatted_parts.append(f"{label.strip()}. {cleaned_content}")

    if not formatted_parts:
        return ""

    return "\n".join(formatted_parts)


def post_process_text(text: Optional[str]) -> Optional[str]:
    """Apply the final text adjustments from the mentor script."""
    if not text:
        return ""

    text = text.replace("()", "(   )")
    text = re.sub(r"\s+(?=[。，！？；：\"'）】])", "", text)
    return text.strip()


def clean_question_info(question_info: Dict[str, Any]) -> Dict[str, Any]:
    """Clean question text while preserving the source answer."""
    if not question_info or not isinstance(question_info, dict):
        return {"stem": "", "options": "", "answer": "", "analysis": ""}

    cleaned_stem = post_process_text(normalize_text(question_info.get("stem")))
    cleaned_analysis = post_process_text(normalize_text(question_info.get("analysis")))
    formatted_options = format_options(question_info.get("options"))

    return {
        "stem": cleaned_stem,
        "options": formatted_options,
        "answer": question_info.get("answer") or "",
        "analysis": cleaned_analysis,
    }


def load_knowledge_labels(taxonomy_file: str) -> Dict[str, str]:
    """Load the knowledge point id-to-label mapping."""
    with open(taxonomy_file, "r", encoding="utf-8") as taxonomy_stream:
        taxonomy = json.load(taxonomy_stream)

    if not isinstance(taxonomy, dict):
        raise ValueError("Knowledge taxonomy must be a JSON object")

    return {str(label_id): str(label) for label_id, label in taxonomy.items()}


def map_knowledge_labels(
    knowledge_ids: Any,
    knowledge_labels: Dict[str, str],
    question_id: str,
) -> List[str]:
    """Map source knw_ids to labels while omitting the ids from output."""
    if not isinstance(knowledge_ids, list):
        return []

    labels = []
    missing_ids = []
    for knowledge_id in knowledge_ids:
        normalized_id = str(knowledge_id)
        label = knowledge_labels.get(normalized_id)
        if label is None:
            missing_ids.append(normalized_id)
        elif label not in labels:
            labels.append(label)

    if missing_ids:
        logging.warning(
            "题目 %s 的知识点ID未在taxonomy中找到: %s",
            question_id,
            ", ".join(missing_ids),
        )

    return labels


def process_file(
    input_file: str,
    output_file: str,
    log_file: str,
    taxonomy_file: str,
) -> None:
    """Read JSONL records, clean their text, and merge them by parent_id."""
    configure_logging(log_file)
    print(f"开始处理文件: {input_file}")
    print(f"知识点映射文件: {taxonomy_file}")
    print(f"日志将写入到: {log_file}")

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    try:
        knowledge_labels = load_knowledge_labels(taxonomy_file)

        print("正在计算文件总行数...")
        with open(input_file, "r", encoding="utf-8") as file:
            total_lines = sum(1 for _ in file)

        if total_lines == 0:
            print("文件为空，无需处理。")
            open(output_file, "w", encoding="utf-8").close()
            return

        print(f"文件总计 {total_lines} 行。")
        parent_questions = {}

        with open(input_file, "r", encoding="utf-8") as input_stream:
            for line in tqdm(input_stream, total=total_lines, desc="正在聚合"):
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError as error:
                    logging.error("JSON解析失败: %s\n原始行: %s", error, line.strip())
                    continue

                question_id = data.get("question_id", "")
                parent_id = data.get("parent_id", "")

                question_info_raw = data.get("question_info", "{}")
                if isinstance(question_info_raw, str):
                    try:
                        question_info = json.loads(question_info_raw)
                    except json.JSONDecodeError:
                        question_info = {}
                else:
                    question_info = (
                        question_info_raw
                        if isinstance(question_info_raw, dict)
                        else {}
                    )

                cleaned_current = clean_question_info(question_info)
                group_key = parent_id
                structure_type = data.get("structure_type", "")
                answered_count = data.get("answered_count", 0)
                percent_correct = data.get("percent_correct", 0.0)
                difficulty = data.get("difficulty", "")
                current_knw_labels = map_knowledge_labels(
                    data.get("knw_ids", []),
                    knowledge_labels,
                    question_id,
                )

                if parent_id == question_id:
                    if group_key not in parent_questions:
                        parent_questions[group_key] = {
                            "parent_id": parent_id,
                            "question_id": question_id,
                            "stem": cleaned_current["stem"],
                            "options": cleaned_current["options"],
                            "answer": cleaned_current["answer"],
                            "analysis": cleaned_current["analysis"],
                            "structure_type": structure_type,
                            "answered_count": answered_count,
                            "percent_correct": percent_correct,
                            "difficulty": difficulty,
                            "knw_labels": current_knw_labels,
                            "sub_questions": [],
                        }
                    else:
                        parent_questions[group_key]["stem"] = cleaned_current["stem"]
                        parent_questions[group_key]["options"] = cleaned_current[
                            "options"
                        ]
                        parent_questions[group_key]["answer"] = cleaned_current[
                            "answer"
                        ]
                        parent_questions[group_key]["analysis"] = cleaned_current[
                            "analysis"
                        ]
                        parent_questions[group_key]["structure_type"] = structure_type
                        parent_questions[group_key]["answered_count"] = answered_count
                        parent_questions[group_key]["percent_correct"] = percent_correct
                        parent_questions[group_key]["difficulty"] = difficulty
                        parent_questions[group_key]["knw_labels"] = (
                            current_knw_labels
                        )
                else:
                    if group_key not in parent_questions:
                        parent_question_info = data.get("parent_question_info", "{}")
                        if parent_question_info and isinstance(
                            parent_question_info, dict
                        ):
                            parent_info_raw = parent_question_info.get(
                                "question_info", "{}"
                            )
                            if isinstance(parent_info_raw, str):
                                try:
                                    parent_info = json.loads(parent_info_raw)
                                except json.JSONDecodeError:
                                    parent_info = {}
                            else:
                                parent_info = (
                                    parent_info_raw
                                    if isinstance(parent_info_raw, dict)
                                    else {}
                                )
                            cleaned_parent = clean_question_info(parent_info)
                            parent_structure_type = parent_question_info.get(
                                "structure_type", ""
                            )
                            parent_answered_count = parent_question_info.get(
                                "answered_count", 0
                            )
                            parent_percent_correct = parent_question_info.get(
                                "percent_correct", 0.0
                            )
                            parent_difficulty = parent_question_info.get(
                                "difficulty", ""
                            )
                            parent_knw_labels = map_knowledge_labels(
                                parent_question_info.get("knw_ids", []),
                                knowledge_labels,
                                parent_id,
                            )
                        else:
                            cleaned_parent = {
                                "stem": "",
                                "options": "",
                                "answer": "",
                                "analysis": "",
                            }
                            parent_structure_type = structure_type
                            parent_answered_count = answered_count
                            parent_percent_correct = percent_correct
                            parent_difficulty = difficulty
                            parent_knw_labels = []

                        parent_questions[group_key] = {
                            "parent_id": parent_id,
                            "question_id": parent_id,
                            "stem": cleaned_parent["stem"],
                            "options": cleaned_parent["options"],
                            "answer": cleaned_parent["answer"],
                            "analysis": cleaned_parent["analysis"],
                            "structure_type": parent_structure_type,
                            "answered_count": parent_answered_count,
                            "percent_correct": parent_percent_correct,
                            "difficulty": parent_difficulty,
                            "knw_labels": parent_knw_labels,
                            "sub_questions": [],
                        }

                    parent_questions[group_key]["sub_questions"].append(
                        {
                            "parent_id": parent_id,
                            "question_id": question_id,
                            "stem": cleaned_current["stem"],
                            "options": cleaned_current["options"],
                            "answer": cleaned_current["answer"],
                            "analysis": cleaned_current["analysis"],
                            "knw_labels": current_knw_labels,
                        }
                    )

        print(f"聚合完成，共 {len(parent_questions)} 道大题。")
        with open(output_file, "w", encoding="utf-8") as output_stream:
            for parent_question in parent_questions.values():
                output_stream.write(
                    json.dumps(parent_question, ensure_ascii=False) + "\n"
                )

        print(f"处理完成，输出文件: {output_file}")
    except FileNotFoundError:
        print(f"文件不存在: {input_file}")
        logging.error("文件不存在: %s", input_file)
    except Exception as error:
        print(f"处理失败: {error}")
        logging.error("处理失败: %s", error)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Clean question content and merge sub-questions."
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--log-file", required=True, help="Processing log file")
    parser.add_argument(
        "--taxonomy",
        required=True,
        help="JSON mapping knowledge point ids to labels",
    )
    return parser


def main() -> None:
    """Run the command-line interface."""
    arguments = build_parser().parse_args()
    process_file(
        arguments.input,
        arguments.output,
        arguments.log_file,
        arguments.taxonomy,
    )


if __name__ == "__main__":
    main()
