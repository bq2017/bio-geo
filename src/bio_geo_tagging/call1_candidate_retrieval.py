"""Run the first DeepSeek call: retrieve candidate geography labels."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Any

from openai import OpenAI
from tqdm import tqdm


DEFAULT_MODEL = "DeepSeek-V4-Flash"
DEFAULT_BASE_URL = "http://172.22.0.35:9204/v1"
EXPECTED_LABEL_COUNT = 414

EVIDENCE_PROMPT = """你负责整理高中地理题目中可用于知识点标签判断的标注依据，暂时不要选择知识点标签。

标注依据应完整保留原题实际涉及的地理内容，而不是只概括完成答案所需的解题步骤。只要原题中的信息可能支持知识点标签判断，就应纳入标注依据。

按照题目结构处理：

普通题：综合题干、选项、答案、解析和图片描述，整理整道题提供的标注依据。

复合题：
1. 整理公共题干自身提供的标注依据。
2. 依次整理每个小题提供的标注依据。处理每个小题时，应结合公共题干，以及当前小题的题干、选项、答案、解析和图片描述。
3. 整理公共题干与所有小题组成的完整题目共同支持的标注依据。综合完整题目中所有可用于知识点标签判断的信息，整理完整题目共同形成的地理内容。

只出现在错误选项中的附带内容、只用于解析扩展的知识以及与题目主体无关的装饰性信息不作为标注依据。

每项标注依据使用唯一编号E1、E2……，最多20项。只输出JSON对象：
{"tagging_evidence":[{"evidence_id":"E1","question_part":"公共题干、小题1、整道题或普通题","content":"可用于知识点标签判断的完整题目信息"}]}
"""

SYSTEM_PROMPT = """你是高中地理知识点候选标签召回器。完整标签目录被分成多个批次，其他批次会分别处理，你只判断本批标签。

本阶段的任务是找出所有可能成为本题最终知识点标签的候选标签，不是在本阶段确定最终标签。原题是判断标签的最终依据，标注依据清单用于保证普通题、公共题干、各小题和完整复合题都得到检查。

阅读完整原题和标注依据清单。

原题提供完整题目信息；标注依据用于整理题目结构并保证普通题、公共题干、各小题和完整复合题都得到覆盖。判断每个标签时，结合原题的具体内容和相关标注依据进行判断。一个标签可以由一项或多项标注依据共同支持。

按照以下标准进行判断：

1. 明确匹配：原题的具体内容和相关标注依据能够充分支持标签释义。列入clear_labels。

2. 可能匹配：原题的具体内容和相关标注依据与标签释义存在实质联系，使该标签存在成为最终标签的可能，但目前不足以确认。列入possible_labels。

3. 不匹配：原题的实际内容和标注依据均不能支持该标签；或者相关内容只出现在错误选项、干扰项或解析扩展中。不列入候选。

每个标签独立判断。已经选择某个标签，不影响其他标签按照自身释义继续判断；不能仅因标签之间存在父子、兄弟或其他层级关系而增加候选。

clear_labels和possible_labels都属于候选标签。只能返回本批目录中存在的完整标签路径，本批所有候选标签去重后最多20个。

只输出以下JSON对象，不要输出其他内容：
{"matches":[{"evidence_ids":["E1"],"clear_labels":["完整标签路径"],"possible_labels":["完整标签路径"]}]}

没有候选标签的标注依据可以不写入matches。不要输出不匹配标签及其排除理由。

【标注依据】
{tagging_evidence}

【标签目录】
{catalog}
"""

CONSOLIDATION_PROMPT = """你负责将已经召回的高中地理候选标签收敛为20个，不重新生成标签或重新拆解题目。

候选已经标明对应的标注依据和匹配类型。优先保留明确匹配标签，再保留可能匹配标签；在可能的情况下保持各项标注依据都有候选。每个候选根据自身依据独立判断，不能仅因为已有更具体、更概括或语义相近的标签，就删除另一个有独立题目依据的候选。

只能从下面的候选中选择，必须恰好保留20个。输出一个JSON对象：
{"candidate_labels":["完整标签路径"]}

【标注依据与候选对应关系】
{evidence}

【待收敛候选】
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
    semantic_groups: dict[str, list[tuple[int, str]]] = {}
    for index, line in enumerate(lines):
        label_path = line.split("｜", 1)[0].strip()
        parent_path = label_path.rsplit("@", 1)[0]
        semantic_groups.setdefault(parent_path, []).append((index, line))

    chunks: list[list[tuple[int, str]]] = [[] for _ in range(parts)]
    groups = sorted(
        semantic_groups.values(),
        key=lambda group: (-len(group), group[0][0]),
    )
    for group in groups:
        target = min(range(parts), key=lambda index: (len(chunks[index]), index))
        chunks[target].extend(group)

    ordered_chunks = [
        [line for _, line in sorted(chunk)]
        for chunk in chunks
    ]
    return [
        ("\n".join(chunk), {line.split("｜", 1)[0].strip() for line in chunk})
        for chunk in ordered_chunks
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
    allowed_paths: set[str],
    known_evidence_ids: set[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    if not isinstance(result, dict):
        raise ValueError("标签匹配输出不是JSON对象")
    raw_matches = result.get("matches")
    if not isinstance(raw_matches, list):
        raise ValueError("matches必须是数组")

    labels: list[str] = []
    seen_labels: set[str] = set()
    normalized_matches: list[dict[str, Any]] = []

    def normalize_labels(raw: Any) -> list[str]:
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError("clear_labels和possible_labels必须是字符串数组")
        normalized: list[str] = []
        local_seen: set[str] = set()
        ignored = 0
        for item in raw:
            label = item.strip().split("｜", 1)[0].strip()
            if label and not label.startswith("知识点@"):
                label = f"知识点@{label}"
            if label not in allowed_paths:
                ignored += 1
                continue
            if label not in local_seen:
                local_seen.add(label)
                normalized.append(label)
            if label not in seen_labels:
                seen_labels.add(label)
                labels.append(label)
        if ignored:
            logging.info("忽略本批目录外候选：%s个", ignored)
        return normalized

    for item in raw_matches:
        if not isinstance(item, dict):
            raise ValueError("matches中的元素必须是对象")
        raw_evidence_ids = item.get("evidence_ids")
        if not isinstance(raw_evidence_ids, list) or not raw_evidence_ids:
            raise ValueError("matches中的evidence_ids必须是非空数组")
        evidence_ids = [as_text(value) for value in raw_evidence_ids]
        if any(not value for value in evidence_ids):
            raise ValueError("evidence_ids包含空编号")
        evidence_ids = list(dict.fromkeys(evidence_ids))
        unknown_ids = set(evidence_ids).difference(known_evidence_ids)
        if unknown_ids:
            raise ValueError(f"matches引用未知标注依据：{sorted(unknown_ids)}")
        clear_labels = normalize_labels(item.get("clear_labels", []))
        possible_labels = [
            label
            for label in normalize_labels(item.get("possible_labels", []))
            if label not in set(clear_labels)
        ]
        normalized_matches.append(
            {
                "evidence_ids": evidence_ids,
                "clear_labels": clear_labels,
                "possible_labels": possible_labels,
            }
        )

    if len(labels) > 20:
        raise ValueError(f"本批候选标签超过20个：{len(labels)}")
    return labels, normalized_matches


def parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()
    result = json.loads(cleaned)
    if not isinstance(result, dict):
        raise ValueError("DS输出不是JSON对象")
    return result


def parse_or_recover_result(
    content: str, allowed_paths: set[str]
) -> dict[str, Any]:
    try:
        return parse_json_object(content)
    except (json.JSONDecodeError, ValueError):
        cleaned = content.strip().removesuffix("```").strip()
        if not cleaned.endswith("}"):
            raise
        matches: list[tuple[int, str]] = []
        for path in allowed_paths:
            variants = (path, path.removeprefix("知识点@"))
            positions = []
            for variant in variants:
                for match in re.finditer(re.escape(variant), content):
                    end = match.end()
                    if end == len(content) or content[end] in '\"｜,]}\n\r':
                        positions.append(match.start())
            if positions:
                matches.append((min(positions), path))
        if not matches:
            raise
        matches.sort()
        logging.warning("DS输出JSON无效，已从返回文本恢复%s个标签", len(matches))
        return {"candidate_labels": [path for _, path in matches]}


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
        self.base_url = base_url
        self.catalog_parts = catalog_parts
        self.catalog_lines = {
            line.split("｜", 1)[0].strip(): line
            for catalog, _ in catalog_parts
            for line in catalog.splitlines()
        }
        self.model = model

    def request(self, system_prompt: str, question_text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text},
            ],
            max_tokens=2048,
            stream=True,
            temperature=0,
        )
        return "".join(
            chunk.choices[0].delta.content or ""
            for chunk in response
            if chunk.choices
        )

    def extract_tagging_evidence(self, question_text: str) -> list[dict[str, str]]:
        content = self.request(EVIDENCE_PROMPT, question_text)
        if not content.strip():
            raise ValueError("标注依据整理返回空内容")
        return normalize_tagging_evidence(parse_json_object(content))

    def retrieve_from_catalog(
        self,
        question_text: str,
        tagging_evidence: list[dict[str, str]],
        catalog: str,
        allowed_paths: set[str],
        shard_number: int,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        system_prompt = (
            SYSTEM_PROMPT
            .replace(
                "{tagging_evidence}",
                json.dumps(tagging_evidence, ensure_ascii=False),
            )
            .replace("{catalog}", catalog)
        )
        content = self.request(system_prompt, question_text)
        if not content.strip():
            raise ValueError(f"第{shard_number}批DS返回空内容")
        result = parse_json_object(content)
        return normalize_match_result(
            result,
            allowed_paths,
            {item["evidence_id"] for item in tagging_evidence},
        )

    def consolidate(
        self,
        question_text: str,
        candidates: list[str],
        tagging_evidence: list[dict[str, str]],
        label_evidence: dict[str, dict[str, Any]],
    ) -> list[str]:
        candidate_catalog = "\n".join(
            self.catalog_lines[label] for label in candidates
        )
        evidence = {
            "tagging_evidence": tagging_evidence,
            "candidate_evidence": [
                {"label": label, **label_evidence[label]}
                for label in candidates
            ],
        }
        system_prompt = (
            CONSOLIDATION_PROMPT
            .replace("{evidence}", json.dumps(evidence, ensure_ascii=False))
            .replace("{catalog}", candidate_catalog)
        )
        content = self.request(system_prompt, question_text)
        if not content.strip():
            raise ValueError("候选收敛时DS返回空内容")
        result = parse_or_recover_result(content, set(candidates))
        labels = normalize_shard_result(result, set(candidates))
        if len(labels) != 20:
            raise ValueError(f"候选收敛结果必须恰好20个，实际为{len(labels)}个")
        return labels

    def retrieve(self, unit: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        question_text = build_question_text(unit)
        tagging_evidence = self.extract_tagging_evidence(question_text)
        candidates: list[str] = []
        shard_candidates: list[list[str]] = []
        shard_matches: list[list[dict[str, Any]]] = []
        all_matches: list[dict[str, Any]] = []
        for part_number, (catalog, allowed_paths) in enumerate(
            self.catalog_parts, start=1
        ):
            labels, matches = self.retrieve_from_catalog(
                question_text,
                tagging_evidence,
                catalog,
                allowed_paths,
                part_number,
            )
            shard_candidates.append(labels)
            shard_matches.append(matches)
            candidates.extend(labels)
            all_matches.extend(matches)

        candidates = list(dict.fromkeys(candidates))
        covered_evidence_ids = {
            evidence_id
            for match in all_matches
            if match["clear_labels"] or match["possible_labels"]
            for evidence_id in match["evidence_ids"]
        }
        uncovered_evidence = [
            evidence
            for evidence in tagging_evidence
            if evidence["evidence_id"] not in covered_evidence_ids
        ]

        label_evidence: dict[str, dict[str, Any]] = {}
        for match in all_matches:
            for match_type, field in (
                ("clear", "clear_labels"),
                ("possible", "possible_labels"),
            ):
                for label in match[field]:
                    evidence = label_evidence.setdefault(
                        label,
                        {
                            "match_type": match_type,
                            "evidence_ids": [],
                        },
                    )
                    if match_type == "clear":
                        evidence["match_type"] = "clear"
                    for evidence_id in match["evidence_ids"]:
                        if evidence_id not in evidence["evidence_ids"]:
                            evidence["evidence_ids"].append(evidence_id)

        before_consolidation = candidates.copy()
        if len(candidates) > 20:
            logging.info("三批合并得到%s个候选，执行候选收敛", len(candidates))
            candidates = self.consolidate(
                question_text,
                candidates,
                tagging_evidence,
                label_evidence,
            )
        return candidates, {
            "tagging_evidence": tagging_evidence,
            "shard_candidate_matches": shard_matches,
            "shard_candidate_labels": shard_candidates,
            "uncovered_evidence": uncovered_evidence,
            "candidate_evidence": label_evidence,
            "before_consolidation": before_consolidation,
            "consolidation_used": len(before_consolidation) > 20,
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
    catalog_parts = split_catalog(catalog)
    completed = load_completed(output_path)
    unit_keys = {make_unit_key(unit) for unit in units}
    stale_keys = set(completed).difference(unit_keys)
    if stale_keys:
        raise ValueError(f"输出文件含有不属于当前输入的记录：{sorted(stale_keys)}")
    for record in completed.values():
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
    pending = [unit for unit in units if make_unit_key(unit) not in completed]
    if limit is not None:
        pending = pending[:limit]

    endpoints = list(dict.fromkeys(base_urls or [base_url]))
    slots_per_endpoint = concurrency_per_endpoint or concurrency
    retriever_pool: Queue[DeepSeekCandidateRetriever] = Queue()
    for endpoint in endpoints:
        for _ in range(slots_per_endpoint):
            retriever_pool.put(
                DeepSeekCandidateRetriever(
                    catalog_parts,
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
                        "status": "completed",
                    }
                    return record, {"unit_key": unit_key, **trace}
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
