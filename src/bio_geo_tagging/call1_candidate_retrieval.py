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

SYSTEM_PROMPT = """你是高中地理知识点候选标签召回器。

完整标签目录已经分成多个批次，其他批次会分别处理。你只判断当前批次中的标签。

本阶段的任务是根据完整原题、标注依据和标签释义，保留所有得到具体题目依据支持、具有成为最终标签合理可能的标签。

原题是标签判断的最终依据。标注依据清单用于呈现普通题、公共题干、各小题和完整复合题中能够支持标签判断的具体地理内容。标注依据中的重复表述不构成额外支持。

对当前批次中的每个标签，按照下面的顺序独立判断。

第一步：理解标签成立条件

阅读标签的完整路径和简明释义，明确这个标签成立时，原题需要实际提供什么知识、地理过程、区域内容、研究对象、主题内容、案例特征或综合内容。

判断依据是标签自身的释义，不能仅根据标签名称进行联想。

第二步：寻找原题中的具体支持

检查完整原题和标注依据，判断原题是否提供了符合该标签成立条件的具体内容。

标签可以由以下一种或多种题目信息支持：

1. 某个小题直接涉及的知识、原理、过程、判断、计算、比较、解释或应用；
2. 公共题干或完整题目实际展开的区域、地点、研究对象、主题或案例内容；
3. 公共题干与多个小题共同形成的综合内容。

区域、地点、对象、主题和案例不因为出现在公共题干中就自动成为候选，也不因为没有直接出现在答案中就被排除。应判断它们是否构成题目实际展开的内容，并且是否符合标签释义。

第三步：区分实际支持与表面联系

实际支持是指原题提供的具体内容已经符合标签释义中的关键内容，使该标签有合理机会成为最终标签。

以下情况只有表面联系，不能作为候选依据：

1. 标签与题目只属于相同的大类或主题；
2. 原题只出现了与标签相同或相近的名称、词语、地点、区域、主题或对象，但没有呈现标签释义要求的具体内容；
3. 标签只是已选标签的父级、子级、兄弟标签或其他层级近邻；
4. 标签只与错误选项、干扰项或解析扩展内容有关；
5. 标签描述的内容可能与题目背景有关，但原题没有实际展开这一内容。

第四步：给出判断结果

1. 明确匹配

原题提供了具体、充分的标注依据，已经清楚满足标签释义中的关键内容。将标签的match_type设为clear。

2. 可能匹配

原题已经为标签释义中的关键内容提供了具体依据，使该标签有合理可能成为最终标签，但由于题目信息完整度、图片缺失或相邻标签边界等原因，目前不能确认最终是否保留。将标签的match_type设为possible。

“可能匹配”不是“存在一般关联”。如果原题尚未支持标签释义中的关键内容，不能列入possible。

3. 不匹配

原题没有提供符合标签成立条件的具体依据，或者原题与标签之间只有词语、地点、主题、层级或一般背景上的联系。不输出该标签。

不同类型标签使用同一判断原则，但应按照各自释义判断其成立条件：

区域或对象标签：如果该区域或对象是完整题目实际展开的范围，并且题目呈现了符合标签释义的具体区域或对象内容，可以成为候选。仅出现名称不能成为候选。

综合标签：如果公共题干和多个小题共同覆盖了该综合标签要求的多个相关内容，可以成为候选。仅因为题目属于该章节或父级模块，不能成为候选。

具体知识标签：如果原题实际涉及标签描述的知识、原理、过程或应用，可以成为候选。不能因为它与题目中的某个知识点属于同一主题就成为候选。

每个标签独立判断。已经选择某个标签，不影响其他标签按照自身释义继续判断。不能仅根据父子、兄弟或其他层级关系机械增加或排除标签。

不要凑候选数量。只输出满足明确匹配或可能匹配标准的标签。只能输出当前批次目录中存在的完整标签路径。

只输出JSON对象，不要输出其他内容：
{"candidates":[{"label":"完整标签路径","match_type":"clear","evidence_ids":["E1","E2"]},{"label":"完整标签路径","match_type":"possible","evidence_ids":["E3"]}]}

没有任何标签满足条件时，输出：
{"candidates":[]}

【标注依据】
{tagging_evidence}

【标签目录】
{catalog}
"""

CONSOLIDATION_PROMPT = """你负责从已经召回的高中地理候选标签中，选出最有可能成为本题最终标签的20个候选。

这一步不能生成新标签，只能从给定候选中选择。

逐个检查候选标签的释义、原题内容和对应标注依据，优先保留：

1. 原题具体内容充分满足标签释义关键内容的标签；
2. 由某个小题直接支持的标签；
3. 由公共题干或完整题目实际展开的区域、对象、主题或案例内容支持的标签；
4. 由公共题干与多个小题共同形成的综合内容支持的标签；
5. 虽然存在边界不确定性，但已经得到具体题目依据支持的标签。

优先删除：

1. 只有相同词语、地点、区域或主题联系的标签；
2. 只有父子、兄弟或其他层级联系的标签；
3. 题目没有实际展开其释义关键内容的标签；
4. 只由错误选项、干扰项或解析扩展内容支持的标签。

具体知识标签、区域或对象标签、综合标签使用同一个“是否得到具体题目依据支持”标准，不能仅因标签类型不同而优先保留或排除。

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
    raw_candidates = result.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("candidates必须是数组")

    labels: list[str] = []
    seen_labels: set[str] = set()
    normalized_candidates: list[dict[str, Any]] = []
    ignored = 0

    for item in raw_candidates:
        if not isinstance(item, dict):
            raise ValueError("candidates中的元素必须是对象")
        label = as_text(item.get("label")).split("｜", 1)[0].strip()
        if label and not label.startswith("知识点@"):
            label = f"知识点@{label}"
        if label not in allowed_paths:
            ignored += 1
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

    if ignored:
        logging.info("忽略本批目录外候选：%s个", ignored)
    return labels, normalized_candidates


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


def parse_or_recover_match_result(
    content: str,
    allowed_paths: set[str],
    known_evidence_ids: set[str],
) -> dict[str, Any]:
    try:
        return parse_json_object(content)
    except (json.JSONDecodeError, ValueError):
        recovered_labels: list[tuple[int, str]] = []
        for path in allowed_paths:
            positions: list[int] = []
            for variant in (path, path.removeprefix("知识点@")):
                for match in re.finditer(re.escape(variant), content):
                    end = match.end()
                    if end == len(content) or content[end] in '\"｜,]}\n\r':
                        positions.append(match.start())
            if positions:
                recovered_labels.append((min(positions), path))
        if not recovered_labels:
            raise

        recovered_labels.sort()
        evidence_ids = [
            evidence_id
            for evidence_id in sorted(
                known_evidence_ids,
                key=lambda value: (
                    (0, int(value[1:]))
                    if value[1:].isdigit()
                    else (1, value)
                ),
            )
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(evidence_id)}(?![A-Za-z0-9_])", content)
        ]
        if not evidence_ids:
            evidence_ids = sorted(known_evidence_ids)
        logging.warning(
            "DS输出JSON无效，已从标签匹配文本恢复%s个候选",
            len(recovered_labels),
        )
        return {
            "candidates": [
                {
                    "label": path,
                    "match_type": "possible",
                    "evidence_ids": evidence_ids,
                }
                for _, path in recovered_labels
            ]
        }


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
        result = parse_or_recover_match_result(
            content,
            allowed_paths,
            {item["evidence_id"] for item in tagging_evidence},
        )
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
            for evidence_id in match["evidence_ids"]
        }
        uncovered_evidence = [
            evidence
            for evidence in tagging_evidence
            if evidence["evidence_id"] not in covered_evidence_ids
        ]

        label_evidence: dict[str, dict[str, Any]] = {}
        for match in all_matches:
            label = match["label"]
            evidence = label_evidence.setdefault(
                label,
                {
                    "match_type": match["match_type"],
                    "evidence_ids": [],
                },
            )
            if match["match_type"] == "clear":
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
    selected_units = units[:limit] if limit is not None else units
    completed = load_completed(output_path)
    unit_keys = {make_unit_key(unit) for unit in selected_units}
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
