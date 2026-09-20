"""Run the first DeepSeek call: retrieve candidate geography labels."""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Any

from openai import OpenAI
from tqdm import tqdm


DEFAULT_MODEL = "DeepSeek-V4-Flash"
DEFAULT_BASE_URL = "http://172.22.0.35:9204/v1"
EXPECTED_LABEL_COUNT = 414
PIPELINE_VERSION = "global-key-v2"
MAX_PRE_CANDIDATES = 100

EVIDENCE_PROMPT = """你负责整理高中地理题目中可用于知识点标签判断的标注依据。本步骤只整理原题信息，不选择知识点标签。

标注依据是原题中能够实际支持知识点标签判断的地理内容。它可以来自某个小题直接涉及的知识、原理、过程、判断或应用，也可以来自完整题目实际展开的区域、地点、研究对象、主题和案例，还可以来自公共题干与多个小题共同形成的综合内容。

按照题目结构整理：

一、普通题

综合题干、选项、答案、解析和图片描述，整理整道题实际提供的标注依据。

二、复合题

1. 整理公共题干自身提供的标注依据，包括公共题干明确提供的区域、地点、对象、主题、地理过程、案例和材料信息。

2. 依次整理每个小题的标注依据。处理当前小题时，应结合公共题干以及当前小题的题干、选项、答案、解析和图片描述，说明当前小题实际涉及的地理内容。

3. 检查公共题干与多个小题结合后，是否形成了单个小题不能独立表达的整题标注依据。例如，多个小题共同覆盖同一地理模块的不同内容，或者共同呈现某一区域、对象、主题或案例的完整特征。只有确实形成新的整体性依据时，才增加“整道题”标注依据。不要把公共题干和各小题的内容简单重复汇总为一项整题依据。

以下内容不作为标注依据：

1. 只出现在错误选项或干扰项文字中的附带知识；
2. 只用于解析扩展、没有参与说明题目实际内容的知识；
3. 与题目主体无关的装饰性信息；
4. 只有地理词语，但没有形成具体地理内容的信息。

每项标注依据使用唯一编号E1、E2……，最多20项。

只输出JSON对象，不要输出其他内容：
{"tagging_evidence":[{"evidence_id":"E1","question_part":"普通题、公共题干、小题1或整道题","content":"原题实际提供的具体地理内容"}]}
"""

GLOBAL_RETRIEVAL_PROMPT = """你负责为高中地理题目进行全局标签预召回。

本步骤看到的是全部414个标签的临时序号和完整路径，但暂时看不到标签的详细释义。你的任务不是确定最终标签，而是找出下一步需要读取完整释义并继续比较的预候选。

先阅读完整原题和标注依据，依次检查普通题或复合题中的公共材料、各小题以及它们共同形成的整题内容，再对照全部标签路径进行召回。

以下标签应进入预候选：

1. 标签路径直接对应题目涉及的知识、原理、过程、判断、计算、解释或应用；
2. 地点、区域、研究对象、主题或案例是题目实际展开的内容，并存在使用相应区域或对象标签的合理可能；
3. 公共材料和多个小题共同形成某一模块的综合内容，使相应综合标签存在合理可能；
4. 两个或多个相近标签仅凭名称无法可靠区分，需要读取完整释义后再判断。

不要因为标签属于同一父级、兄弟分支或相同大类而机械加入。与原题没有实际联系的标签不进入预候选。不要在本步骤过早处理标签边界；存在合理可能但不能仅凭路径排除时，应当保留到下一步。

预候选最多100个，不要求凑满。只能输出目录中存在的临时序号，不要重新书写标签路径，不要输出判断理由或其他内容。

只输出JSON对象：
{"pre_candidate_keys":["L001","L002"]}

【标注依据】
{tagging_evidence}

【全部标签临时序号和路径】
{label_paths}
"""

FINAL_SELECTION_PROMPT = """你负责从全局预召回结果中选择高中地理知识点候选标签。

本步骤已经为每个预候选提供完整路径和简明释义。请重新结合完整原题、标注依据和标签释义进行统一比较。最终输出的是供下一次调用继续判断的候选集合，不是强行确定唯一答案。

按照以下顺序处理：

第一步，阅读标签释义，明确标签描述的知识、过程、区域、对象、主题、案例或综合范围。

第二步，在完整原题和标注依据中寻找支持。支持可以来自某个小题，也可以来自公共材料实际展开的内容，或者来自公共材料与多个小题共同形成的完整题目内容。

第三步，判断标签与题目的关系：

1. 明确匹配：原题提供了直接、充分的依据，标签释义清楚覆盖题目的实际内容，match_type设为clear。
2. 可能匹配：原题已经提供与标签关键语义直接相关的具体内容，使其具有成为最终标签的合理可能，但仍需要下一次调用结合边界决定是否保留，match_type设为possible。
3. 不匹配：标签与题目没有实际联系，或者只有同类、层级、词语和一般背景上的联系，不输出。

区域或对象标签：当该区域或对象是题目的主要研究范围，并且题目实际展开了其地理内容时，可以保留；不能因为同时存在气候、农业、工业等具体标签就排除区域标签。

综合标签：当完整题目的内容落在该综合标签覆盖范围内，或者公共材料与多个小题共同形成相关模块内容时，可以保留；不能因为已经选择具体标签就自动排除综合标签。

具体知识或专题标签：当题目直接涉及其知识、原理、过程、判断或应用时，可以保留。

同一道题可以同时保留具体知识标签、区域或对象标签和综合标签。每个标签都必须能指出具体证据编号。只出现在错误选项、干扰项或解析扩展内容中的附带知识不作为依据。

最终候选最多20个，可以少于20个，也可以为空，不要凑满。只能返回预候选目录中的临时序号，不要重新书写标签路径。

只输出JSON对象：
{"candidates":[{"candidate_key":"C001","match_type":"clear","evidence_ids":["E1","E2"]},{"candidate_key":"C002","match_type":"possible","evidence_ids":["E3"]}]}

【标注依据】
{tagging_evidence}

【预候选标签及简明释义】
{candidate_catalog}
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


def make_unit_key(unit: dict[str, Any]) -> str:
    question_id = as_text(unit.get("question_id"))
    if not question_id:
        raise ValueError("题目缺少question_id")
    root_question_id = as_text(
        unit.get("root_question_id")
        or unit.get("parent_id")
        or question_id
    )
    input_role = get_input_role(unit)
    return f"{root_question_id}|{question_id}|{input_role}"


def get_input_role(unit: dict[str, Any]) -> str:
    explicit_role = as_text(unit.get("input_role"))
    if explicit_role:
        return explicit_role
    if "sub_questions" in unit:
        return "question_group"
    return "root"


def has_question_content(unit: dict[str, Any]) -> bool:
    if as_text(unit.get("stem")):
        return True
    sub_questions = unit.get("sub_questions")
    return isinstance(sub_questions, list) and any(
        isinstance(question, dict) and as_text(question.get("stem"))
        for question in sub_questions
    )


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
            if "sub_questions" in unit:
                sub_questions = unit.get("sub_questions")
                if not isinstance(sub_questions, list) or not all(
                    isinstance(question, dict) for question in sub_questions
                ):
                    raise ValueError(
                        f"题目文件第{line_number}行sub_questions必须是对象数组"
                    )
            unit_key = make_unit_key(unit)
            if unit_key in seen_keys:
                raise ValueError(f"题目文件存在重复打标单元：{unit_key}")
            seen_keys.add(unit_key)
            if not has_question_content(unit):
                skipped_empty_stem += 1
                logging.warning("跳过题干为空的打标单元：%s", unit_key)
                continue
            units.append(unit)
    return units, skipped_empty_stem


def build_question_text(unit: dict[str, Any]) -> str:
    """Render only fields allowed to reach DS; existing labels are excluded."""
    if "sub_questions" in unit:
        return build_group_question_text(unit)

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


def append_question_fields(
    sections: list[str], question: dict[str, Any], heading: str
) -> None:
    sections.append(f"【{heading}】\n{render_value(question.get('stem'))}")
    options = render_value(question.get("options"))
    if options:
        sections.append(f"【{heading}选项】\n{options}")
    answer = render_value(question.get("answer"))
    if answer:
        sections.append(f"【{heading}答案】\n{answer}")
    analysis = question.get("analysis")
    if analysis is None:
        analysis = question.get("explanation")
    rendered_analysis = render_value(analysis)
    if rendered_analysis:
        sections.append(f"【{heading}解析】\n{rendered_analysis}")
    image_description = render_value(question.get("image_description"))
    if image_description:
        sections.append(f"【{heading}图片描述】\n{image_description}")


def build_group_question_text(unit: dict[str, Any]) -> str:
    sub_questions = unit.get("sub_questions") or []
    sections: list[str] = []
    if sub_questions:
        shared_stem = render_value(unit.get("stem"))
        if shared_stem:
            sections.append(f"【整道题公共材料】\n{shared_stem}")
        root_image = render_value(unit.get("image_description"))
        if root_image:
            sections.append(f"【整道题公共图片描述】\n{root_image}")
        for index, question in enumerate(sub_questions, start=1):
            append_question_fields(sections, question, f"小题{index}")
    else:
        append_question_fields(sections, unit, "题目")
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


def normalize_pre_candidate_result(
    result: Any,
    key_to_label: dict[str, str],
) -> tuple[list[str], list[str]]:
    if not isinstance(result, dict):
        raise ValueError("全局预召回输出不是JSON对象")
    raw_keys = result.get("pre_candidate_keys")
    if not isinstance(raw_keys, list) or not all(
        isinstance(item, str) for item in raw_keys
    ):
        raise ValueError("pre_candidate_keys必须是字符串数组")

    labels: list[str] = []
    seen: set[str] = set()
    rejected_keys: list[str] = []
    for raw_key in raw_keys:
        key = raw_key.strip().split("｜", 1)[0].strip().upper()
        label = key_to_label.get(key)
        if label is None:
            rejected_keys.append(raw_key.strip())
            continue
        if label not in seen:
            seen.add(label)
            labels.append(label)

    if raw_keys and not labels:
        raise ValueError(
            f"全局预召回未返回任何有效临时序号：{rejected_keys}"
        )
    if len(labels) > MAX_PRE_CANDIDATES:
        raise ValueError(
            f"全局预候选超过{MAX_PRE_CANDIDATES}个：{len(labels)}"
        )
    return labels, rejected_keys


def normalize_tagging_evidence(result: Any) -> list[dict[str, str]]:
    if not isinstance(result, dict):
        raise ValueError("标注依据输出不是JSON对象")
    raw_evidence = result.get("tagging_evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise ValueError("tagging_evidence必须是非空数组")
    if len(raw_evidence) > 20:
        raise ValueError(f"标注依据超过20项：{len(raw_evidence)}")

    evidence: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for item in raw_evidence:
        if not isinstance(item, dict):
            raise ValueError("tagging_evidence中的元素必须是对象")
        evidence_id = as_text(item.get("evidence_id"))
        question_part = as_text(item.get("question_part"))
        content = as_text(item.get("content"))
        if not evidence_id or not question_part or not content:
            raise ValueError("标注依据缺少evidence_id、question_part或content")
        if evidence_id in seen_ids:
            raise ValueError(f"标注依据编号重复：{evidence_id}")
        seen_ids.add(evidence_id)
        evidence.append(
            {
                "evidence_id": evidence_id,
                "question_part": question_part,
                "content": content,
            }
        )
    return evidence


def normalize_match_result(
    result: Any,
    key_to_label: dict[str, str],
    known_evidence_ids: set[str],
    max_candidates: int = 20,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    if not isinstance(result, dict):
        raise ValueError("标签匹配输出不是JSON对象")
    raw_candidates = result.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("candidates必须是数组")

    labels: list[str] = []
    seen_labels: set[str] = set()
    normalized_candidates: list[dict[str, Any]] = []
    rejected_keys: list[str] = []

    for item in raw_candidates:
        if not isinstance(item, dict):
            raise ValueError("candidates中的元素必须是对象")
        raw_key = as_text(item.get("candidate_key"))
        key = raw_key.split("｜", 1)[0].strip().upper()
        label = key_to_label.get(key)
        if label is None:
            rejected_keys.append(raw_key)
            continue
        match_type = as_text(item.get("match_type"))
        if match_type not in {"clear", "possible"}:
            raise ValueError("match_type必须是clear或possible")
        raw_evidence_ids = item.get("evidence_ids")
        if not isinstance(raw_evidence_ids, list) or not raw_evidence_ids:
            raise ValueError("candidates中的evidence_ids必须是非空数组")
        evidence_ids = [as_text(value) for value in raw_evidence_ids]
        if any(not value for value in evidence_ids):
            raise ValueError("evidence_ids包含空编号")
        evidence_ids = list(dict.fromkeys(evidence_ids))
        unknown_ids = set(evidence_ids).difference(known_evidence_ids)
        if unknown_ids:
            raise ValueError(f"candidates引用未知标注依据：{sorted(unknown_ids)}")
        if label in seen_labels:
            continue
        seen_labels.add(label)
        labels.append(label)
        normalized_candidates.append(
            {
                "label": label,
                "match_type": match_type,
                "evidence_ids": evidence_ids,
            }
        )

    if raw_candidates and not labels:
        raise ValueError(
            f"最终判断未返回任何有效候选临时序号：{rejected_keys}"
        )
    if len(labels) > max_candidates:
        raise ValueError(f"最终候选超过{max_candidates}个：{len(labels)}")
    return labels, normalized_candidates, rejected_keys


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
        catalog: str,
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
        self.base_url = base_url
        self.catalog_lines = {
            line.split("｜", 1)[0].strip(): line
            for line in catalog.splitlines()
        }
        self.global_key_to_label = {
            f"L{index:03d}": label
            for index, label in enumerate(self.catalog_lines, start=1)
        }
        self.label_paths = "\n".join(
            f"{key}｜{label}"
            for key, label in self.global_key_to_label.items()
        )
        self.model = model

    def request(
        self,
        system_prompt: str,
        question_text: str,
        max_tokens: int = 2048,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text},
            ],
            max_tokens=max_tokens,
            stream=True,
            temperature=0,
        )
        content: list[str] = []
        finish_reason: str | None = None
        for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            content.append(choice.delta.content or "")
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        if finish_reason == "length":
            raise ValueError("DS输出达到长度限制，结果可能被截断")
        return "".join(content)

    def extract_tagging_evidence(self, question_text: str) -> list[dict[str, str]]:
        content = self.request(EVIDENCE_PROMPT, question_text)
        if not content.strip():
            raise ValueError("标注依据整理返回空内容")
        return normalize_tagging_evidence(parse_json_object(content))

    def retrieve_global_candidates(
        self,
        question_text: str,
        tagging_evidence: list[dict[str, str]],
    ) -> tuple[list[str], list[str]]:
        system_prompt = (
            GLOBAL_RETRIEVAL_PROMPT
            .replace(
                "{tagging_evidence}",
                json.dumps(tagging_evidence, ensure_ascii=False),
            )
            .replace("{label_paths}", self.label_paths)
        )
        content = self.request(system_prompt, question_text, max_tokens=4096)
        if not content.strip():
            raise ValueError("全局预召回返回空内容")
        return normalize_pre_candidate_result(
            parse_json_object(content), self.global_key_to_label
        )

    def select_final_candidates(
        self,
        question_text: str,
        pre_candidates: list[str],
        tagging_evidence: list[dict[str, str]],
    ) -> tuple[list[str], list[dict[str, Any]], list[str]]:
        key_to_label = {
            f"C{index:03d}": label
            for index, label in enumerate(pre_candidates, start=1)
        }
        candidate_catalog = "\n".join(
            f"{key}｜{self.catalog_lines[label]}"
            for key, label in key_to_label.items()
        )
        system_prompt = (
            FINAL_SELECTION_PROMPT
            .replace(
                "{tagging_evidence}",
                json.dumps(tagging_evidence, ensure_ascii=False),
            )
            .replace("{candidate_catalog}", candidate_catalog)
        )
        content = self.request(system_prompt, question_text, max_tokens=4096)
        if not content.strip():
            raise ValueError("统一候选判断返回空内容")
        return normalize_match_result(
            parse_json_object(content),
            key_to_label,
            {item["evidence_id"] for item in tagging_evidence},
        )

    def retrieve(self, unit: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        question_text = build_question_text(unit)
        tagging_evidence = self.extract_tagging_evidence(question_text)
        pre_candidates, rejected_global_keys = self.retrieve_global_candidates(
            question_text, tagging_evidence
        )
        if pre_candidates:
            candidates, final_matches, rejected_final_keys = self.select_final_candidates(
                question_text, pre_candidates, tagging_evidence
            )
        else:
            candidates, final_matches, rejected_final_keys = [], [], []

        covered_evidence_ids = {
            evidence_id
            for match in final_matches
            for evidence_id in match["evidence_ids"]
        }
        uncovered_evidence = [
            evidence
            for evidence in tagging_evidence
            if evidence["evidence_id"] not in covered_evidence_ids
        ]

        candidate_evidence = {
            match["label"]: {
                "match_type": match["match_type"],
                "evidence_ids": match["evidence_ids"],
            }
            for match in final_matches
        }
        return candidates, {
            "tagging_evidence": tagging_evidence,
            "global_pre_candidate_labels": pre_candidates,
            "rejected_global_candidate_keys": rejected_global_keys,
            "final_candidate_matches": final_matches,
            "rejected_final_candidate_keys": rejected_final_keys,
            "uncovered_evidence": uncovered_evidence,
            "candidate_evidence": candidate_evidence,
            "candidate_labels": candidates,
        }


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
    trace_output: Path | None = None,
    base_urls: list[str] | None = None,
    concurrency_per_endpoint: int | None = None,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_units = units[:limit] if limit is not None else units
    completed = load_completed(output_path)
    unit_keys = {make_unit_key(unit) for unit in selected_units}
    stale_keys = set(completed).difference(unit_keys)
    if stale_keys:
        raise ValueError(f"输出文件含有不属于当前输入的记录：{sorted(stale_keys)}")
    for record in completed.values():
        if record.get("pipeline_version") != PIPELINE_VERSION:
            raise ValueError(
                "已有候选结果由旧版召回流程生成，请为新版流程使用新的输出文件"
            )
        validate_result(record, allowed_paths)
    trace_keys: set[str] = set()
    if trace_output is not None:
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        if trace_output.exists():
            with trace_output.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        trace_keys.add(as_text(json.loads(line).get("unit_key")))
        missing_trace = set(completed).difference(trace_keys)
        if missing_trace:
            raise ValueError(
                "已有候选结果缺少诊断记录，请为本次测试使用新的输出文件："
                f"{sorted(missing_trace)[:3]}"
            )
    pending = [
        unit
        for unit in selected_units
        if make_unit_key(unit) not in completed
    ]

    endpoints = list(dict.fromkeys(base_urls or [base_url]))
    slots_per_endpoint = concurrency_per_endpoint or concurrency
    retriever_pool: Queue[DeepSeekCandidateRetriever] = Queue()
    for endpoint in endpoints:
        for _ in range(slots_per_endpoint):
            retriever_pool.put(
                DeepSeekCandidateRetriever(
                    catalog,
                    model,
                    endpoint,
                    api_key,
                    timeout,
                )
            )

    def process(unit: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        unit_key = make_unit_key(unit)
        retriever = retriever_pool.get()
        try:
            last_error: Exception | None = None
            for attempt in range(1, retries + 1):
                try:
                    candidate_labels, trace = retriever.retrieve(unit)
                    record = {
                        "unit_key": unit_key,
                        "question_id": as_text(unit.get("question_id")),
                        "root_question_id": as_text(
                            unit.get("root_question_id")
                            or unit.get("parent_id")
                            or unit.get("question_id")
                        ),
                        "input_role": get_input_role(unit),
                        "candidate_labels": candidate_labels,
                        "uncovered_topic": None,
                        "model": model,
                        "endpoint": retriever.base_url,
                        "pipeline_version": PIPELINE_VERSION,
                        "status": "completed",
                    }
                    return record, {
                        "unit_key": unit_key,
                        "pipeline_version": PIPELINE_VERSION,
                        **trace,
                    }
                except Exception as error:
                    last_error = error
                    logging.warning(
                        "题目 %s 在 %s 第 %s 次调用失败：%s",
                        unit_key,
                        retriever.base_url,
                        attempt,
                        error,
                    )
            raise RuntimeError(f"题目{unit_key}调用失败") from last_error
        finally:
            retriever_pool.put(retriever)

    errors = 0
    with (
        output_path.open("a", encoding="utf-8") as output_stream,
        (trace_output.open("a", encoding="utf-8") if trace_output else open(os.devnull, "w")) as trace_stream,
        ThreadPoolExecutor(max_workers=len(endpoints) * slots_per_endpoint) as executor,
        tqdm(total=len(pending), desc="Retrieving candidates") as progress,
    ):
        futures = {executor.submit(process, unit): unit for unit in pending}
        for future in as_completed(futures):
            try:
                record, trace = future.result()
                if trace_output is not None:
                    trace_stream.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    trace_stream.flush()
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
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL))
    endpoint_group = parser.add_mutually_exclusive_group()
    endpoint_group.add_argument(
        "--base-url", default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
    )
    endpoint_group.add_argument("--base-urls", nargs="+")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--concurrency-per-endpoint", type=int, default=5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.concurrency <= 0
        or args.concurrency_per_endpoint <= 0
        or args.retries <= 0
        or args.timeout <= 0
    ):
        raise SystemExit(
            "concurrency、concurrency-per-endpoint、retries和timeout必须大于0"
        )
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
        args.trace_output,
        args.base_urls,
        args.concurrency_per_endpoint if args.base_urls else None,
    )
    summary["skipped_empty_stem"] = skipped_empty_stem
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["errors"]:
        raise SystemExit("存在失败题目，请重复执行同一命令继续补跑")


if __name__ == "__main__":
    main()
