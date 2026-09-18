"""Run the first DeepSeek call: retrieve candidate geography labels."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from openai import OpenAI
from tqdm import tqdm


DEFAULT_MODEL = "DeepSeek-V4-Flash"
DEFAULT_BASE_URL = "http://172.22.0.35:9204/v1"
EXPECTED_LABEL_COUNT = 414

REQUIREMENT_PROMPT = """你负责分析高中地理题目的实际考查要求，暂时不要选择知识点标签。

按照原题中的题目部分依次处理：普通题处理整道题；大题处理整道题整体以及每个小题。公共材料用于理解各小题，只有确实存在跨小题或整题综合考查时，才单列整道题整体要求。

考查要求必须描述完成答案实际需要进行的地理判断、计算、解释、比较或知识应用。结合题干、选项、答案和解析判断：判断或排除选项实际需要的知识应列入；只出现在干扰项文字中的附带内容、解析延伸知识和一般背景不列入。

每项要求使用唯一编号R1、R2……，最多20项。只输出JSON对象：
{"requirements":[{"requirement_id":"R1","question_part":"小题1","requirement":"完成该部分答案实际需要的地理知识或能力"}]}
"""

SYSTEM_PROMPT = """你是高中地理知识点候选标签召回器。另有其他标签批次分别处理，你只判断本批标签。

原题是最终依据。下面的考查要求清单用于保证整题和各小题得到覆盖；如果原题中还有清单遗漏的实际考查要求，可以通过additional_matches补充。

对标签采用以下统一标准：
1. 明确匹配：标签定义直接对应完成某项考查要求所需的判断、计算、解释或知识应用，必须保留。
2. 可能匹配：标签与某项考查要求存在实质对应，但当前阶段无法可靠排除，保留给下一次调用判断。
3. 主题相关：只与材料主题、地点、对象或关键词相关，没有参与答案形成，不保留。
4. 明确无关：与考查要求不对应，或仅因父子、兄弟和其他层级关系被联想到，不保留。

判断或排除选项实际需要的知识可以保留；只出现在干扰项文字中的附带内容不保留。解析中参与答案推导的知识可以保留；仅用于扩展说明的知识不保留。

每个候选必须对应一项考查要求。分别列出明确匹配和可能匹配标签；两类都属于候选。不输出主题相关和明确无关标签。只能返回本批目录中的完整标签路径，本批所有候选合计最多20个。

只输出JSON对象：
{"matches":[{"requirement_id":"R1","clear_labels":["完整标签路径"],"possible_labels":[]}],"additional_matches":[{"question_part":"小题1","requirement":"清单遗漏的实际考查要求","clear_labels":["完整标签路径"],"possible_labels":[]}]}

没有补充要求时additional_matches输出空数组。没有匹配标签的已有要求可以不写入matches。

【考查要求】
{requirements}

【标签目录】
{catalog}
"""

RECOVERY_PROMPT = """你负责对高中地理题目中尚未获得候选标签的考查要求进行一次定向补召回，只判断本批标签。

原题是最终依据。对每项未覆盖要求重新核对本批全部标签：标签定义直接对应要求时列入clear_labels；存在实质对应但无法可靠排除时列入possible_labels。只与主题、地点、对象或关键词相关的标签不保留，不根据标签层级机械补充。

只能返回本批目录中的完整标签路径，本批所有候选合计最多20个。只输出JSON对象：
{"matches":[{"requirement_id":"R1","clear_labels":["完整标签路径"],"possible_labels":[]}]}

【尚未覆盖的考查要求】
{requirements}

【标签目录】
{catalog}
"""

CONSOLIDATION_PROMPT = """你负责将已经召回的高中地理候选标签收敛为20个，不重新生成标签或重新拆解题目。

候选已经标明对应的考查要求和匹配类型。优先保留明确匹配标签，再保留可能匹配标签；在可能的情况下保持各项考查要求都有候选，不能仅因为已有更具体、更概括或语义相近的标签就删除另一个有独立依据的候选。

只能从下面的候选中选择，必须恰好保留20个。输出一个JSON对象：
{"candidate_labels":["完整标签路径"]}

【考查要求与候选对应关系】
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


def normalize_requirements(result: Any) -> list[dict[str, str]]:
    if not isinstance(result, dict):
        raise ValueError("考查要求输出不是JSON对象")
    raw_requirements = result.get("requirements")
    if not isinstance(raw_requirements, list) or not raw_requirements:
        raise ValueError("requirements必须是非空数组")
    if len(raw_requirements) > 20:
        raise ValueError(f"考查要求超过20项：{len(raw_requirements)}")

    requirements: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for item in raw_requirements:
        if not isinstance(item, dict):
            raise ValueError("requirements中的元素必须是对象")
        requirement_id = as_text(item.get("requirement_id"))
        question_part = as_text(item.get("question_part"))
        requirement = as_text(item.get("requirement"))
        if not requirement_id or not question_part or not requirement:
            raise ValueError("考查要求缺少requirement_id、question_part或requirement")
        if requirement_id in seen_ids:
            raise ValueError(f"考查要求编号重复：{requirement_id}")
        seen_ids.add(requirement_id)
        requirements.append(
            {
                "requirement_id": requirement_id,
                "question_part": question_part,
                "requirement": requirement,
            }
        )
    return requirements


def normalize_match_result(
    result: Any,
    allowed_paths: set[str],
    known_requirement_ids: set[str],
    additional_prefix: str,
    allow_additional: bool,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, str]]]:
    if not isinstance(result, dict):
        raise ValueError("标签匹配输出不是JSON对象")
    raw_matches = result.get("matches")
    if not isinstance(raw_matches, list):
        raise ValueError("matches必须是数组")
    raw_additional = result.get("additional_matches", [])
    if not isinstance(raw_additional, list):
        raise ValueError("additional_matches必须是数组")
    if raw_additional and not allow_additional:
        raise ValueError("定向补召回不能新增考查要求")

    labels: list[str] = []
    seen_labels: set[str] = set()
    normalized_matches: list[dict[str, Any]] = []
    additional_requirements: list[dict[str, str]] = []

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

    def add_match(item: Any, requirement_id: str) -> None:
        if not isinstance(item, dict):
            raise ValueError("matches中的元素必须是对象")
        clear_labels = normalize_labels(item.get("clear_labels", []))
        possible_labels = [
            label
            for label in normalize_labels(item.get("possible_labels", []))
            if label not in set(clear_labels)
        ]
        normalized_matches.append(
            {
                "requirement_id": requirement_id,
                "clear_labels": clear_labels,
                "possible_labels": possible_labels,
            }
        )

    for item in raw_matches:
        if not isinstance(item, dict):
            raise ValueError("matches中的元素必须是对象")
        requirement_id = as_text(item.get("requirement_id"))
        if requirement_id not in known_requirement_ids:
            raise ValueError(f"matches引用未知考查要求：{requirement_id}")
        add_match(item, requirement_id)

    for index, item in enumerate(raw_additional, start=1):
        if not isinstance(item, dict):
            raise ValueError("additional_matches中的元素必须是对象")
        question_part = as_text(item.get("question_part"))
        requirement = as_text(item.get("requirement"))
        if not question_part or not requirement:
            raise ValueError("补充考查要求缺少question_part或requirement")
        requirement_id = f"{additional_prefix}{index}"
        additional_requirements.append(
            {
                "requirement_id": requirement_id,
                "question_part": question_part,
                "requirement": requirement,
            }
        )
        add_match(item, requirement_id)

    if len(labels) > 20:
        raise ValueError(f"本批候选标签超过20个：{len(labels)}")
    return labels, normalized_matches, additional_requirements


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

    def extract_requirements(self, question_text: str) -> list[dict[str, str]]:
        content = self.request(REQUIREMENT_PROMPT, question_text)
        if not content.strip():
            raise ValueError("考查要求分析返回空内容")
        return normalize_requirements(parse_json_object(content))

    def retrieve_from_catalog(
        self,
        question_text: str,
        requirements: list[dict[str, str]],
        catalog: str,
        allowed_paths: set[str],
        shard_number: int,
    ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, str]]]:
        system_prompt = (
            SYSTEM_PROMPT
            .replace("{requirements}", json.dumps(requirements, ensure_ascii=False))
            .replace("{catalog}", catalog)
        )
        content = self.request(system_prompt, question_text)
        if not content.strip():
            raise ValueError(f"第{shard_number}批DS返回空内容")
        result = parse_json_object(content)
        return normalize_match_result(
            result,
            allowed_paths,
            {item["requirement_id"] for item in requirements},
            f"A{shard_number}_",
            allow_additional=True,
        )

    def recover_uncovered(
        self,
        question_text: str,
        uncovered_requirements: list[dict[str, str]],
    ) -> tuple[list[str], list[dict[str, Any]], list[list[str]]]:
        recovered_labels: list[str] = []
        recovered_matches: list[dict[str, Any]] = []
        shard_labels: list[list[str]] = []
        requirement_ids = {
            item["requirement_id"] for item in uncovered_requirements
        }
        rendered_requirements = json.dumps(
            uncovered_requirements, ensure_ascii=False
        )
        for shard_number, (catalog, allowed_paths) in enumerate(
            self.catalog_parts, start=1
        ):
            system_prompt = (
                RECOVERY_PROMPT
                .replace("{requirements}", rendered_requirements)
                .replace("{catalog}", catalog)
            )
            content = self.request(system_prompt, question_text)
            if not content.strip():
                raise ValueError(f"第{shard_number}批定向补召回返回空内容")
            labels, matches, _ = normalize_match_result(
                parse_json_object(content),
                allowed_paths,
                requirement_ids,
                "",
                allow_additional=False,
            )
            shard_labels.append(labels)
            recovered_labels.extend(labels)
            recovered_matches.extend(matches)
        return list(dict.fromkeys(recovered_labels)), recovered_matches, shard_labels

    def consolidate(
        self,
        question_text: str,
        candidates: list[str],
        requirements: list[dict[str, str]],
        label_evidence: dict[str, dict[str, Any]],
    ) -> list[str]:
        candidate_catalog = "\n".join(
            self.catalog_lines[label] for label in candidates
        )
        evidence = {
            "requirements": requirements,
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
        requirements = self.extract_requirements(question_text)
        candidates: list[str] = []
        shard_candidates: list[list[str]] = []
        shard_matches: list[list[dict[str, Any]]] = []
        all_matches: list[dict[str, Any]] = []
        for part_number, (catalog, allowed_paths) in enumerate(
            self.catalog_parts, start=1
        ):
            labels, matches, additional_requirements = self.retrieve_from_catalog(
                question_text,
                requirements,
                catalog,
                allowed_paths,
                part_number,
            )
            requirements.extend(additional_requirements)
            shard_candidates.append(labels)
            shard_matches.append(matches)
            candidates.extend(labels)
            all_matches.extend(matches)

        candidates = list(dict.fromkeys(candidates))
        covered_requirement_ids = {
            match["requirement_id"]
            for match in all_matches
            if match["clear_labels"] or match["possible_labels"]
        }
        uncovered_before_recovery = [
            requirement
            for requirement in requirements
            if requirement["requirement_id"] not in covered_requirement_ids
        ]
        recovery_matches: list[dict[str, Any]] = []
        recovery_shard_candidates: list[list[str]] = []
        if uncovered_before_recovery:
            recovered, recovery_matches, recovery_shard_candidates = (
                self.recover_uncovered(question_text, uncovered_before_recovery)
            )
            candidates.extend(
                label for label in recovered if label not in set(candidates)
            )
            all_matches.extend(recovery_matches)

        covered_requirement_ids = {
            match["requirement_id"]
            for match in all_matches
            if match["clear_labels"] or match["possible_labels"]
        }
        uncovered_after_recovery = [
            requirement
            for requirement in requirements
            if requirement["requirement_id"] not in covered_requirement_ids
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
                        {"match_type": match_type, "requirement_ids": []},
                    )
                    if match_type == "clear":
                        evidence["match_type"] = "clear"
                    if match["requirement_id"] not in evidence["requirement_ids"]:
                        evidence["requirement_ids"].append(match["requirement_id"])

        before_consolidation = candidates.copy()
        if len(candidates) > 20:
            logging.info("三批合并得到%s个候选，执行候选收敛", len(candidates))
            candidates = self.consolidate(
                question_text,
                candidates,
                requirements,
                label_evidence,
            )
        return candidates, {
            "requirements": requirements,
            "shard_candidate_matches": shard_matches,
            "shard_candidate_labels": shard_candidates,
            "uncovered_before_recovery": uncovered_before_recovery,
            "recovery_used": bool(uncovered_before_recovery),
            "recovery_candidate_matches": recovery_matches,
            "recovery_shard_candidate_labels": recovery_shard_candidates,
            "uncovered_after_recovery": uncovered_after_recovery,
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

    def process(unit: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        unit_key = make_unit_key(unit)
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                candidate_labels, trace = get_retriever().retrieve(unit)
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
                    "endpoint": base_url,
                    "status": "completed",
                }
                return record, {"unit_key": unit_key, **trace}
            except Exception as error:
                last_error = error
                logging.warning(
                    "题目 %s 第 %s 次调用失败：%s", unit_key, attempt, error
                )
        raise RuntimeError(f"题目{unit_key}调用失败") from last_error

    errors = 0
    with (
        output_path.open("a", encoding="utf-8") as output_stream,
        (trace_output.open("a", encoding="utf-8") if trace_output else open(os.devnull, "w")) as trace_stream,
        ThreadPoolExecutor(max_workers=concurrency) as executor,
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
        args.trace_output,
    )
    summary["skipped_empty_stem"] = skipped_empty_stem
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["errors"]:
        raise SystemExit("存在失败题目，请重复执行同一命令继续补跑")


if __name__ == "__main__":
    main()
