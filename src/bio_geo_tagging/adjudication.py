"""DS precision adjudication over geography candidate labels."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from bio_geo_tagging.ds import DSRequestError, append_evidence, parse_json_content


PROMPT_VERSION = "geography-candidate-adjudication-v1.5-complete-units"
CANDIDATE_ORDER_VERSION = "geography-candidate-order-v2"
IMAGE_REFERENCE_RE = re.compile(r"(?:读图|据图|下图|上图|图中|该图|如图|示意图|图示)")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            records.append(value)
    return records


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_run_log(path: Path, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{timestamp} {message}\n")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if "temperature" not in existing and manifest.get("temperature") == 0:
            existing["temperature"] = 0
        if existing != manifest:
            raise ValueError("run manifest mismatch; use a new run directory")
        return
    _write_json_atomic(path, manifest)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).strip()


def make_unit_key(unit: dict[str, Any]) -> str:
    question_id = _as_text(unit.get("question_id"))
    if not question_id:
        raise ValueError("every unit must have question_id")
    root_question_id = _as_text(
        unit.get("root_question_id") or unit.get("parent_id") or question_id
    )
    input_role = _as_text(unit.get("input_role") or unit.get("unit_type") or "root")
    return f"{root_question_id}|{question_id}|{input_role}"


def limit_units_by_root(
    units: list[dict[str, Any]], limit: int | None
) -> list[dict[str, Any]]:
    """Limit complete root-question groups without splitting a big question."""
    if limit is None:
        return units
    selected_roots: list[str] = []
    selected_set: set[str] = set()
    for unit in units:
        question_id = _as_text(unit.get("question_id"))
        root_question_id = _as_text(
            unit.get("root_question_id") or unit.get("parent_id") or question_id
        )
        if root_question_id not in selected_set:
            if len(selected_roots) >= limit:
                continue
            selected_roots.append(root_question_id)
            selected_set.add(root_question_id)
    return [
        unit
        for unit in units
        if _as_text(
            unit.get("root_question_id")
            or unit.get("parent_id")
            or unit.get("question_id")
        )
        in selected_set
    ]


def load_labels(path: str | Path) -> dict[str, dict[str, str]]:
    """Load geography labels and key them by the candidate-facing label path."""
    labels: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(_read_jsonl(path), 1):
        interpretation = row.get("existing_interpretation")
        if not isinstance(interpretation, dict):
            interpretation = {}
        label_path = _as_text(row.get("label_path") or row.get("knw_label"))
        taxonomy_label_id = _as_text(row.get("label_id"))
        if not label_path:
            raise ValueError(f"label line {line_number} lacks label_path or knw_label")
        if label_path in labels:
            raise ValueError(f"duplicate label_path: {label_path}")
        label_name = _as_text(row.get("label_name")) or label_path.rsplit("@", 1)[-1]
        definition = _as_text(
            row.get("definition")
            or row.get("positive_definition")
            or row.get("knowledge_scope")
            or interpretation.get("definition")
        )
        core_concepts = _as_text(
            row.get("core_concepts")
            or row.get("assessment_scope")
            or row.get("common_exam_content")
            or interpretation.get("keywords")
        )
        distinctions = _as_text(
            row.get("distinctions")
            or row.get("distinction_from_similar_labels")
            or interpretation.get("distinction")
        )
        assessment_scope = _as_text(
            row.get("exam_methods") or interpretation.get("exam_methods")
        )
        if not definition:
            raise ValueError(f"label {label_path} lacks a usable definition")
        labels[label_path] = {
            "label_id": label_path,
            "taxonomy_label_id": taxonomy_label_id,
            "label_name": label_name,
            "label_path": label_path,
            "definition": definition,
            "core_concepts": core_concepts,
            "distinctions": distinctions,
            "assessment_scope": assessment_scope,
        }
    if not labels:
        raise ValueError("labels file is empty")
    return labels


def normalize_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract candidate label paths without retaining retrieval metadata."""
    raw_candidates: Any
    if "combined_candidates" in row:
        raw_candidates = row.get("combined_candidates")
    elif "candidates" in row:
        raw_candidates = row.get("candidates")
    elif "candidate_labels" in row:
        raw_candidates = row.get("candidate_labels")
    else:
        raise ValueError("candidate row lacks candidates/combined_candidates/candidate_labels")
    if not isinstance(raw_candidates, list):
        raise ValueError("candidate collection must be a list")

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_candidates, 1):
        if isinstance(raw, str):
            label_id = raw.strip()
        elif isinstance(raw, dict):
            label_id = _as_text(raw.get("label_path") or raw.get("label_id"))
        else:
            raise ValueError(f"candidate {index} must be a string or object")
        if not label_id:
            raise ValueError(f"candidate {index} lacks label_id or label_path")
        if label_id in seen:
            raise ValueError(f"duplicate candidate label: {label_id}")
        seen.add(label_id)
        candidates.append({"label_id": label_id})
    return candidates


class CandidateIndex:
    """Resolve exact unit candidates first, then shared root-question candidates."""

    def __init__(self, rows: Iterable[dict[str, Any]]) -> None:
        self.by_unit_key: dict[str, list[dict[str, Any]]] = {}
        by_question_id: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
        for row in rows:
            candidates = normalize_candidates(row)
            unit_key = _as_text(row.get("unit_key"))
            if unit_key:
                if unit_key in self.by_unit_key:
                    raise ValueError(f"duplicate candidate unit_key: {unit_key}")
                self.by_unit_key[unit_key] = candidates
            question_id = _as_text(row.get("question_id") or row.get("parent_id"))
            if not question_id and not unit_key:
                raise ValueError("candidate row lacks question_id and unit_key")
            if question_id:
                by_question_id[question_id].append(candidates)
        self.by_question_id = dict(by_question_id)

    def resolve(self, unit: dict[str, Any]) -> list[dict[str, Any]]:
        unit_key = make_unit_key(unit)
        if unit_key in self.by_unit_key:
            return self.by_unit_key[unit_key]
        question_id = _as_text(unit.get("question_id"))
        root_question_id = _as_text(
            unit.get("root_question_id") or unit.get("parent_id") or question_id
        )
        for identifier in dict.fromkeys((question_id, root_question_id)):
            matches = self.by_question_id.get(identifier, [])
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    f"ambiguous candidates for {identifier}; provide matching unit_key"
                )
        raise ValueError(f"missing candidates for unit: {unit_key}")


def _is_comprehensive_label(label: dict[str, str]) -> bool:
    return "综合" in label["label_name"]


def _is_comprehensive_candidate(
    candidate: dict[str, Any], label: dict[str, str]
) -> bool:
    return _is_comprehensive_label(label) or (
        candidate.get("support_count") is not None
        and candidate.get("fusion_score") is not None
    )


def candidates_for_unit(
    unit: dict[str, Any],
    candidates: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Keep regional labels with ordinary labels; reserve comprehensive labels."""
    input_role = _as_text(unit.get("input_role") or unit.get("unit_type"))
    if input_role not in {"subquestion", "whole_question_comprehensive"}:
        return list(candidates)
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        label = labels_by_id[candidate["label_id"]]
        is_comprehensive = _is_comprehensive_candidate(candidate, label)
        if input_role == "whole_question_comprehensive" and is_comprehensive:
            selected.append(candidate)
        elif input_role == "subquestion" and not is_comprehensive:
            selected.append(candidate)
    return selected


def load_audited_exclusions(
    path: str | Path,
) -> dict[tuple[str, str], dict[str, str]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("rules"), list):
        raise ValueError("audited exclusions must contain a rules list")
    rules: dict[tuple[str, str], dict[str, str]] = {}
    allowed_actions = {"remove_label", "remove_label_and_exclude_training"}
    for index, raw_rule in enumerate(value["rules"], 1):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"audited exclusion rule {index} must be an object")
        question_id = _as_text(raw_rule.get("question_id"))
        label_id = _as_text(raw_rule.get("label_id") or raw_rule.get("label_path"))
        action = _as_text(raw_rule.get("action"))
        reason = _as_text(raw_rule.get("reason"))
        if not question_id or not label_id or not reason:
            raise ValueError(f"audited exclusion rule {index} has an empty field")
        if action not in allowed_actions:
            raise ValueError(f"audited exclusion rule {index} has invalid action")
        key = (question_id, label_id)
        if key in rules:
            raise ValueError(f"duplicate audited exclusion: {question_id}::{label_id}")
        rules[key] = {"action": action, "reason": reason}
    return rules


def apply_audited_exclusions(
    question_id: str,
    selected_labels: list[dict[str, Any]],
    rules: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], bool]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, str]] = []
    exclude_training = False
    for label in selected_labels:
        label_id = _as_text(label.get("label_id"))
        rule = rules.get((question_id, label_id))
        if not rule:
            kept.append(label)
            continue
        removed.append(
            {
                "label_id": label_id,
                "label_path": _as_text(label.get("label_path")),
                "action": rule["action"],
                "reason": rule["reason"],
            }
        )
        exclude_training = exclude_training or rule["action"].endswith(
            "exclude_training"
        )
    return kept, removed, exclude_training


def _image_context_missing(unit: dict[str, Any]) -> bool:
    flags = unit.get("flags") if isinstance(unit.get("flags"), dict) else {}
    if flags.get("image_context_missing") is not None:
        return bool(flags["image_context_missing"])
    sub_questions = unit.get("sub_questions")
    valid_sub_questions = (
        [value for value in sub_questions if isinstance(value, dict)]
        if isinstance(sub_questions, list)
        else []
    )
    text = "\n".join(
        [
            _as_text(unit.get("context_stem") or unit.get("parent_stem")),
            _as_text(unit.get("stem")),
            _as_text(unit.get("options")),
            *(
                _as_text(value.get("stem")) + "\n" + _as_text(value.get("options"))
                for value in valid_sub_questions
            ),
        ]
    )
    if not IMAGE_REFERENCE_RE.search(text):
        return False
    descriptions = [
        _as_text(unit.get("image_description")),
        _as_text(unit.get("context_image_description")),
        _as_text(unit.get("parent_image_description")),
        *(
            _as_text(value.get("image_description"))
            for value in valid_sub_questions
        ),
    ]
    return not any(descriptions)


def build_adjudication_inputs(
    unit: dict[str, Any],
    candidates: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    question_id = _as_text(unit.get("question_id"))
    shuffled_candidates = sorted(
        candidates,
        key=lambda candidate: hashlib.sha256(
            f"{question_id}\0{candidate['label_id']}\0{CANDIDATE_ORDER_VERSION}".encode(
                "utf-8"
            )
        ).digest(),
    )
    code_map = {
        f"C{index:02d}": _as_text(candidate["label_id"])
        for index, candidate in enumerate(shuffled_candidates, 1)
    }
    candidate_cards = []
    for code, label_id in code_map.items():
        if label_id not in labels_by_id:
            raise ValueError(f"candidate uses unknown label_id: {label_id}")
        label = labels_by_id[label_id]
        candidate_cards.append(
            {
                "code": code,
                "label_name": label["label_name"],
                "label_path": label["label_path"],
                "definition": label["definition"],
                "core_concepts": label["core_concepts"],
                "assessment_scope": label.get("assessment_scope", ""),
                "distinctions": label["distinctions"],
            }
        )
    analysis = unit.get("analysis")
    if analysis is None:
        analysis = unit.get("explanation")
    input_role = _as_text(unit.get("input_role") or unit.get("unit_type"))
    question: dict[str, Any] = {
        "question_id": question_id,
        "root_question_id": _as_text(
            unit.get("root_question_id") or unit.get("parent_id") or question_id
        ),
        "unit_type": input_role,
        "parent_stem": _as_text(
            unit.get("context_stem") or unit.get("parent_stem")
        )[:3000],
        "stem": _as_text(unit.get("stem"))[:5000],
        "options": _as_text(unit.get("options"))[:3000],
        "answer_text": _as_text(unit.get("answer") or unit.get("answer_text"))[:2000],
        "analysis": _as_text(analysis)[:6000],
        "image_description": _as_text(unit.get("image_description"))[:3000],
        "parent_image_description": _as_text(
            unit.get("context_image_description")
            or unit.get("parent_image_description")
        )[:3000],
        "parent_context_missing": bool(
            (unit.get("flags") or {}).get("parent_context_missing")
        ),
        "image_context_missing": _image_context_missing(unit),
    }
    if input_role == "whole_question_comprehensive":
        raw_sub_questions = unit.get("sub_questions")
        if not isinstance(raw_sub_questions, list) or not raw_sub_questions:
            raise ValueError(
                f"{input_role} unit must contain sub_questions"
            )
        question["sub_questions"] = []
        for sub_question in raw_sub_questions:
            if not isinstance(sub_question, dict):
                raise ValueError("sub_questions must contain JSON objects")
            sub_analysis = sub_question.get("analysis")
            if sub_analysis is None:
                sub_analysis = sub_question.get("explanation")
            question["sub_questions"].append(
                {
                    "question_id": _as_text(sub_question.get("question_id")),
                    "stem": _as_text(sub_question.get("stem"))[:5000],
                    "options": _as_text(sub_question.get("options"))[:3000],
                    "answer_text": _as_text(
                        sub_question.get("answer") or sub_question.get("answer_text")
                    )[:2000],
                    "analysis": _as_text(sub_analysis)[:6000],
                    "image_description": _as_text(
                        sub_question.get("image_description")
                    )[:3000],
                }
            )
    return question, candidate_cards, code_map


def build_adjudication_prompt(
    unit: dict[str, Any],
    candidates: list[dict[str, Any]],
    labels_by_id: dict[str, dict[str, str]],
) -> tuple[str, dict[str, str], dict[str, Any]]:
    question, candidate_cards, code_map = build_adjudication_inputs(
        unit, candidates, labels_by_id
    )
    if question["unit_type"] == "whole_question_comprehensive":
        scope_instruction = """本次是整道大题的综合Label专项判定。你可以阅读公共题干和全部小题，但候选中只提供综合Label。只有多个小题或同一小题中的知识必须跨模块联动、共同形成一个不可拆分的综合判断，并且符合综合Label定义时才选择。仅仅因为整道题包含多个独立知识点、多个小题或同一章节内容，不得选择综合Label。若题目只考查普通Label或区域Label、不构成综合考查，即使存在明确地理考点，也必须返回selected=[]、need_expand_recall=false。只有题目确实形成综合考查、但正确的综合Label不在候选中时，才返回selected=[]、need_expand_recall=true。"""
    elif question["unit_type"] == "root":
        scope_instruction = """本次判断一道不含小题的完整普通题。可以从候选中选择直接考查的普通Label、区域Label或综合Label，但每个Label都必须独立支持答案中的关键判断。区域仅作为材料发生地时不选区域Label；仅仅涉及多个知识点但不要求联动时不选综合Label。"""
    else:
        scope_instruction = """本次只判断当前小题。大题公共题干只用于补足当前小题明确指代的对象、区域和语境，不得引入兄弟小题的知识。可以同时选择直接考查的普通Label和区域Label；区域只作为材料发生地、案例载体或定位信息时不得选择。综合Label由独立的整题专项判定处理，本次候选中不应选择综合Label。"""
    prompt = f"""你是严谨的高中地理知识点判标器。本任务高精度优先：错标的代价远高于漏标。可以少选、selected=[]或要求扩召；不得为提高覆盖率加入只是相关、同章节、上下位邻近、同一因果链或常见伴随出现的Label。

任务是判断当前题目或当前小题是否直接考查候选Label所定义的知识范围，而不是寻找所有相关知识。只输出简短结论，不输出详细思考过程。

{scope_instruction}

一、先界定Label
Label有效范围由label_name、label_path、definition、assessment_scope和distinctions共同确定。distinctions是硬否决边界。core_concepts和assessment_scope只解释范围内的概念、规律、方法和常见考查方式，不能扩大Label范围；只命中一个地名、现象、材料、关键词或底层机制不足以选中Label。

二、硬否决：任意一项成立就拒绝，后续不得翻回
1. 区域不一致：地区或地名只是材料载体时，不自动选择该区域Label。区域Label只有在该区域的位置、环境特征、空间差异、区域联系或区域发展本身被直接考查时才能选择。不得把区域A的特有知识横向迁移到区域B。
2. 空间或时间尺度不一致：全球、国家、区域、城市和局地尺度不能互相替代；日变化、季节变化、年际变化和长期演化不能互相替代。
3. 任务维度不一致：分布、特征、成因、条件、过程、影响、措施、评价、预测、计算和判读不能互相替代。处于同一因果链不等于全部都是考点。
4. 自然与人文机制不一致：自然条件作为材料背景不等于直接考查自然地理机制；人类活动作为现象背景也不等于直接考查相应人文地理Label。
5. 判标范围不一致：普通题或小题判定只判断当前对象；parent_stem和父题图片描述只能补足当前小题明确指代的对象、时空和图表语境，不能单独制造考点，兄弟小题的知识不选。整题综合专项只判断综合Label，不得借机补选普通Label或区域Label。
6. 与distinctions冲突：题目落在distinctions排除的一侧时立即拒绝。

三、还原当前任务
仅根据当前stem、options、answer_text、analysis以及必要的父题语境，判断学生为了得出正确答案必须完成哪些具体地理判断。材料中出现的地名、现象和解析中的延伸背景不自动算考点。错误选项只有在辨别其错误必须直接调用该知识，且该辨别构成题目的实质性考查时才能支持Label；孤立干扰信息不能支持Label。

四、正向选中：必须同时满足
1. 范围命中：当前认知任务本身落在Label有效范围内，而非仅有关键词、同章节、上下位或因果链关联。
2. 直接考查：该Label必须直接支持答案中的一个关键判断。仅作为背景、材料对象、一般前置知识或更深层解释不算直接考查。
3. 独立作用：多Label时，每个Label都必须独立解释一个真实存在的判断任务。不得顺带加入父级、子级、同链条或常见搭配Label。
4. 可指证：必须能指出该Label直接支持了答案中的哪一个关键判断。如果只能说“有关”或“有助于理解”，就拒绝。

五、特殊Label
1. 区域Label：在某地区应用通用规律时可选真正被考查的通用Label；只有地区本身的区域特征或区域联系被直接考查时才选区域Label。两者分别直接考查时才可同选。
2. 地图、图表、遥感、调查、计算和方法类Label：只有当前设问真正考查相应判读规则、计算方法、数据解读、操作或评价时才选择。地图或图表仅作为信息载体时，不自动选择工具方法Label。
3. 综合Label：只在整题综合专项中判断。只有多个知识必须联动形成一个不可拆分的联合判断，且符合该Label定义时才选择。大题包含多个彼此独立的小问不等于考查综合Label。

六、evidence与最终复核
每个selected Label必须提供一条不超过60字的evidence，逐字复制自当前stem、options、answer_text、analysis、image_description，或在当前小题存在明确指代时复制自parent_stem和parent_image_description。不得改写或推理补写。evidence必须支持直接考查，而不只是证明二者相关。
生成selected前，对每个暂定Label反证复核：若它实际只是地区材料、共享条件、同一因果链、不同尺度、不同考查维度、上下位相关、背景补充或孤立干扰项，就删除。

七、状态判断
轻微错别字或OCR异常若可由答案、解析和其他信息唯一消除，context_insufficient=false。DS无法查看图片或图片URL；题目明确依赖图片且没有可靠的文字图片描述时，必须设context_insufficient=true，不得根据答案或解析反推图片内容。其他缺父题或信息冲突导致连一个可靠Label都无法确定时，也设context_insufficient=true。
普通题或小题若有明确高中地理考点、但所有候选都无法成立：selected=[]、need_expand_recall=true。整题综合专项严格遵循前述专项规则：不构成综合考查时need_expand_recall=false，确实构成综合考查但正确综合Label缺失时才为true。若已有可靠Label，只是怀疑存在不确定次要Label，保留可靠结果且need_expand_recall=false。非有效高中地理考查：selected=[]、need_expand_recall=false、context_insufficient=false。

只能返回C01等短代码，不能抄写label_path。

题目：
{json.dumps(question, ensure_ascii=False)}

候选Label（顺序不代表最终正确性）：
{json.dumps(candidate_cards, ensure_ascii=False)}

多Label按候选出现顺序输出。reason只说明最终选中或置空的核心原因，1至2句话、不超过120字，不输出详细分析。

只输出一个JSON对象：
{{
  "selected": ["C01", "C05"],
  "evidence": {{"C01": "题目原文", "C05": "题目原文"}},
  "context_insufficient": false,
  "need_expand_recall": false,
  "reason": "当前设问直接考查……"
}}
不要输出Markdown或JSON之外的内容。"""
    return prompt, code_map, question


def validate_adjudication_result(
    value: dict[str, Any],
    known_codes: set[str],
    question: dict[str, Any],
) -> dict[str, Any]:
    required = (
        "reason",
        "selected",
        "evidence",
        "need_expand_recall",
        "context_insufficient",
    )
    for field in required:
        if field not in value:
            raise ValueError(f"missing {field}")
    for field in ("need_expand_recall", "context_insufficient"):
        if not isinstance(value[field], bool):
            raise ValueError(f"{field} must be boolean")

    selected = value["selected"]
    if not isinstance(selected, list):
        raise ValueError("selected must be a list")
    normalized: list[str] = []
    unknown_codes: list[str] = []
    for item in selected:
        if not isinstance(item, str):
            raise ValueError("selected items must be short codes")
        code = item.strip()
        if code not in known_codes:
            if code and code not in unknown_codes:
                unknown_codes.append(code)
            continue
        if code not in normalized:
            normalized.append(code)

    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    reason = reason.strip()
    if len(reason) > 300:
        raise ValueError("reason is too long")

    raw_evidence = value["evidence"]
    if not isinstance(raw_evidence, dict):
        raise ValueError("evidence must be an object keyed by selected code")
    normalized_evidence: dict[str, str] = {}
    for code in normalized:
        evidence = raw_evidence.get(code)
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(f"missing non-empty evidence for {code}")
        evidence = evidence.strip()
        if len(evidence) > 300:
            raise ValueError(f"evidence for {code} exceeds 300 characters")
        normalized_evidence[code] = evidence

    return {
        "reason": reason,
        "selected": normalized,
        "evidence": normalized_evidence,
        "unknown_selected_codes_dropped": unknown_codes,
        "context_insufficient": value["context_insufficient"],
        "need_expand_recall": value["need_expand_recall"] or bool(unknown_codes),
        "none_of_candidates": not bool(normalized),
    }


def _latest_success(
    evidence_path: Path, *, prompt_version: str
) -> tuple[dict[str, dict[str, Any]], int]:
    completed: dict[str, dict[str, Any]] = {}
    evidence_rows = 0
    if not evidence_path.exists():
        return completed, evidence_rows
    for record in _read_jsonl(evidence_path):
        evidence_rows += 1
        if record.get("prompt_version") != prompt_version:
            continue
        unit_key = _as_text(record.get("unit_key"))
        if (
            unit_key
            and not record.get("error")
            and isinstance(record.get("parsed_response"), dict)
        ):
            completed[unit_key] = record
    return completed, evidence_rows


def write_question_predictions(
    path: Path,
    predictions: list[dict[str, Any]],
    *,
    expected_units: list[dict[str, Any]] | None = None,
    model: str = "",
) -> dict[str, int]:
    """Union sub-question labels with the whole-question comprehensive decision."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    prediction_root_order: list[str] = []
    for prediction in predictions:
        root_question_id = _as_text(prediction.get("root_question_id"))
        if root_question_id not in grouped:
            prediction_root_order.append(root_question_id)
        grouped[root_question_id].append(prediction)

    expected_by_root: dict[str, list[dict[str, str]]] = defaultdict(list)
    if expected_units is not None:
        for unit in expected_units:
            question_id = _as_text(unit.get("question_id"))
            root_question_id = _as_text(
                unit.get("root_question_id") or unit.get("parent_id") or question_id
            )
            expected_by_root[root_question_id].append(
                {
                    "unit_key": make_unit_key(unit),
                    "question_id": question_id,
                    "input_role": _as_text(
                        unit.get("input_role") or unit.get("unit_type") or "root"
                    ),
                }
            )
        root_order = list(expected_by_root)
    else:
        root_order = prediction_root_order

    temporary = path.with_name(f".{path.name}.tmp")
    usable_count = 0
    review_count = 0
    incomplete_count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for root_question_id in root_order:
            components = grouped.get(root_question_id, [])
            selected_by_id: dict[str, dict[str, Any]] = {}
            for component in components:
                for selected in component["selected_labels"]:
                    label_id = _as_text(selected.get("label_id"))
                    if label_id not in selected_by_id:
                        selected_by_id[label_id] = {
                            key: value
                            for key, value in selected.items()
                            if key != "evidence"
                        }
                        selected_by_id[label_id]["evidence_by_unit"] = []
                    selected_by_id[label_id]["evidence_by_unit"].append(
                        {
                            "unit_key": component["unit_key"],
                            "question_id": component["question_id"],
                            "input_role": component["input_role"],
                            "evidence": selected["evidence"],
                        }
                    )
            selected_labels = sorted(
                selected_by_id.values(),
                key=lambda label: (
                    int(label.get("prompt_position", 0)),
                    label["label_id"],
                ),
            )
            completed_unit_keys = {component["unit_key"] for component in components}
            expected_components = expected_by_root.get(root_question_id)
            if expected_components is None:
                expected_components = [
                    {
                        "unit_key": component["unit_key"],
                        "question_id": component["question_id"],
                        "input_role": component["input_role"],
                    }
                    for component in components
                ]
            missing_components = [
                component
                for component in expected_components
                if component["unit_key"] not in completed_unit_keys
            ]
            components_complete = not missing_components
            needs_review = bool(
                not components_complete
                or any(component["needs_review"] for component in components)
            )
            usable_for_training = bool(selected_labels and not needs_review)
            usable_count += int(usable_for_training)
            review_count += int(needs_review)
            incomplete_count += int(not components_complete)
            record = {
                "root_question_id": root_question_id,
                "selected_labels": selected_labels,
                "component_units": [
                    {
                        "unit_key": component["unit_key"],
                        "question_id": component["question_id"],
                        "input_role": component["input_role"],
                        "none_of_candidates": component["none_of_candidates"],
                        "need_expand_recall": component["need_expand_recall"],
                        "context_insufficient": component["context_insufficient"],
                    }
                    for component in components
                ],
                "expected_component_count": len(expected_components),
                "completed_component_count": len(completed_unit_keys),
                "components_complete": components_complete,
                "missing_component_units": missing_components,
                "needs_review": needs_review,
                "usable_for_training": usable_for_training,
                "model": components[0]["model"] if components else model,
                "prompt_version": (
                    components[0]["prompt_version"] if components else PROMPT_VERSION
                ),
            }
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)
    return {
        "whole_questions": len(root_order),
        "whole_questions_usable_for_training": usable_count,
        "whole_questions_needing_review": review_count,
        "whole_questions_with_incomplete_components": incomplete_count,
    }


def run_adjudication(
    units_path: str | Path,
    candidates_path: str | Path,
    labels_path: str | Path,
    run_dir: str | Path,
    client: Any,
    *,
    model: str,
    limit: int | None = None,
    max_tokens: int = 1024,
    workers: int = 1,
    audited_exclusions_path: str | Path | None = None,
    enable_thinking: bool | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    run_started = time.monotonic()
    run_started_at = datetime.now(timezone.utc).isoformat()
    if workers < 1:
        raise ValueError("workers must be positive")
    units = _read_jsonl(units_path)
    units = limit_units_by_root(units, limit)
    unit_keys = [make_unit_key(unit) for unit in units]
    if len(unit_keys) != len(set(unit_keys)):
        raise ValueError("units contain duplicate unit keys")
    candidate_index = CandidateIndex(_read_jsonl(candidates_path))
    labels_by_id = load_labels(labels_path)
    candidates_by_unit: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        resolved_candidates = candidate_index.resolve(unit)
        for candidate in resolved_candidates:
            if candidate["label_id"] not in labels_by_id:
                raise ValueError(
                    f"candidate uses unknown label_id: {candidate['label_id']}"
                )
        candidates_by_unit[make_unit_key(unit)] = candidates_for_unit(
            unit, resolved_candidates, labels_by_id
        )
    audited_exclusions = (
        load_audited_exclusions(audited_exclusions_path)
        if audited_exclusions_path is not None
        else {}
    )

    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "evidence.jsonl"
    candidate_count_distribution = dict(
        sorted(
            Counter(str(len(value)) for value in candidates_by_unit.values()).items(),
            key=lambda item: int(item[0]),
        )
    )
    manifest: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION,
        "candidate_order_version": CANDIDATE_ORDER_VERSION,
        "model": model,
        "limit": limit,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "input_paths": {
            "units": str(Path(units_path)),
            "candidates": str(Path(candidates_path)),
            "labels": str(Path(labels_path)),
        },
        "input_sha256": {
            "units": _file_sha256(units_path),
            "candidates": _file_sha256(candidates_path),
            "labels": _file_sha256(labels_path),
        },
        "candidate_count_distribution": candidate_count_distribution,
        "audited_exclusions": (
            {
                "path": str(Path(audited_exclusions_path)),
                "sha256": _file_sha256(audited_exclusions_path),
            }
            if audited_exclusions_path is not None
            else None
        ),
    }
    if enable_thinking is not None:
        manifest["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    _ensure_run_manifest(output_dir / "run_manifest.json", manifest)

    completed, evidence_rows = _latest_success(
        evidence_path, prompt_version=PROMPT_VERSION
    )
    pending_units = [
        (index, unit)
        for index, unit in enumerate(units, 1)
        if make_unit_key(unit) not in completed
    ]
    run_log_path = output_dir / "run.log"
    _append_run_log(
        run_log_path,
        "START "
        f"input={len(units)} resumed={len(completed)} pending={len(pending_units)} "
        f"workers={workers} model={model} prompt_version={PROMPT_VERSION}",
    )
    requests_succeeded = 0
    requests_failed = 0

    def adjudicate(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        index, unit = item
        unit_key = make_unit_key(unit)
        question_id = _as_text(unit.get("question_id"))
        candidates = candidates_by_unit[unit_key]
        prompt, code_map, question = build_adjudication_prompt(
            unit, candidates, labels_by_id
        )
        record: dict[str, Any] = {
            "stage": "geography_candidate_adjudication",
            "prompt_version": PROMPT_VERSION,
            "unit_key": unit_key,
            "question_id": question_id,
            "root_question_id": question["root_question_id"],
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
            "candidate_code_map": code_map,
            "model": model,
            "temperature": temperature,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_response": None,
            "parsed_response": None,
            "endpoint": None,
            "attempts": 0,
            "latency_seconds": None,
            "usage": None,
            "reasoning": None,
            "response_message_keys": [],
            "retry_errors": [],
            "error": None,
        }
        try:
            response = client.chat(
                [
                    {
                        "role": "system",
                        "content": "你是严谨的高中地理知识点判标器，只输出JSON。",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
            )
            record.update(
                {
                    "raw_response": response.content,
                    "endpoint": response.endpoint,
                    "attempts": response.attempts,
                    "latency_seconds": response.latency_seconds,
                    "usage": getattr(response, "usage", None),
                    "reasoning": getattr(response, "reasoning", None),
                    "response_message_keys": list(
                        getattr(response, "response_message_keys", ())
                    ),
                    "retry_errors": list(getattr(response, "retry_errors", ())),
                }
            )
            record["parsed_response"] = validate_adjudication_result(
                parse_json_content(response.content), set(code_map), question
            )
        except DSRequestError as exc:
            record.update(
                {
                    "endpoint": exc.endpoint,
                    "attempts": exc.attempts,
                    "latency_seconds": exc.latency_seconds,
                    "retry_errors": list(exc.retry_errors),
                }
            )
            record["error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        return index, record

    def persist(original_index: int, record: dict[str, Any], finished: int) -> None:
        nonlocal evidence_rows, requests_succeeded, requests_failed
        unit_key = _as_text(record["unit_key"])
        requests_failed += int(bool(record["error"]))
        requests_succeeded += int(not record["error"])
        append_evidence(evidence_path, record)
        evidence_rows += 1
        if not record["error"]:
            completed[unit_key] = record
        interim = {
            "input": len(units),
            "success": len(completed),
            "error": requests_failed,
            "pending": len(units) - len(completed),
            "evidence_rows": evidence_rows,
            "requests_succeeded_this_run": requests_succeeded,
            "requests_failed_this_run": requests_failed,
            "workers": workers,
            "model": model,
            "prompt_version": PROMPT_VERSION,
        }
        _write_json_atomic(output_dir / "report.json", interim)
        progress_message = (
            f"[{finished}/{len(pending_units)}; source={original_index}/{len(units)}] "
            f"{unit_key} {'ERROR' if record['error'] else 'OK'}"
        )
        if record["error"]:
            progress_message += f" error={record['error']}"
        _append_run_log(run_log_path, progress_message)

    if workers == 1:
        for finished, item in enumerate(map(adjudicate, pending_units), 1):
            persist(*item, finished)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(adjudicate, item) for item in pending_units]
            for finished, future in enumerate(as_completed(futures), 1):
                original_index, record = future.result()
                persist(original_index, record, finished)

    completed, evidence_rows = _latest_success(
        evidence_path, prompt_version=PROMPT_VERSION
    )
    predictions_path = output_dir / "predictions.jsonl"
    predictions_temporary = predictions_path.with_name(f".{predictions_path.name}.tmp")
    selected_count_distribution: Counter[str] = Counter()
    training_filter_reasons: Counter[str] = Counter()
    need_expand = 0
    none_count = 0
    context_insufficient_count = 0
    usable_for_training_count = 0
    audited_exclusion_questions = 0
    audited_excluded_labels = 0
    unknown_selected_codes_dropped_count = 0
    materialized_predictions: list[dict[str, Any]] = []

    with predictions_temporary.open("w", encoding="utf-8", newline="\n") as output:
        for unit in units:
            unit_key = make_unit_key(unit)
            record = completed.get(unit_key)
            if not record:
                continue
            question_id = _as_text(unit.get("question_id"))
            parsed = record["parsed_response"]
            code_map = record["candidate_code_map"]
            candidates = candidates_by_unit[unit_key]
            prompt_position_by_label = {
                label_id: int(code.removeprefix("C"))
                for code, label_id in code_map.items()
            }
            selected_labels: list[dict[str, Any]] = []
            for code in parsed["selected"]:
                label_id = code_map[code]
                label = labels_by_id[label_id]
                selected_labels.append(
                    {
                        "label_id": label_id,
                        "taxonomy_label_id": label.get("taxonomy_label_id", ""),
                        "label_name": label["label_name"],
                        "label_path": label["label_path"],
                        "prompt_position": prompt_position_by_label[label_id],
                        "evidence": parsed["evidence"][code],
                    }
                )
            selected_labels, removed_by_audit, exclude_from_training = (
                apply_audited_exclusions(
                    question_id, selected_labels, audited_exclusions
                )
            )
            selected_labels.sort(
                key=lambda item: (item["prompt_position"], item["label_id"])
            )
            if removed_by_audit:
                audited_exclusion_questions += 1
                audited_excluded_labels += len(removed_by_audit)

            selected_count_distribution[str(len(selected_labels))] += 1
            need_expand += int(parsed["need_expand_recall"])
            none_count += int(parsed["none_of_candidates"])
            context_insufficient_count += int(parsed["context_insufficient"])
            dropped_unknown = parsed.get("unknown_selected_codes_dropped", [])
            unknown_selected_codes_dropped_count += len(dropped_unknown)
            text_content_missing = not _as_text(unit.get("stem")) and not _as_text(
                unit.get("context_stem") or unit.get("parent_stem")
            )
            needs_review = bool(
                parsed["need_expand_recall"]
                or parsed["context_insufficient"]
                or text_content_missing
                or exclude_from_training
            )
            usable_for_training = bool(
                selected_labels
                and not parsed["need_expand_recall"]
                and not parsed["context_insufficient"]
                and not text_content_missing
                and not exclude_from_training
            )
            usable_for_training_count += int(usable_for_training)
            if not selected_labels:
                training_filter_reasons["empty_selected"] += 1
            if parsed["need_expand_recall"]:
                training_filter_reasons["need_expand_recall"] += 1
            if parsed["context_insufficient"]:
                training_filter_reasons["context_insufficient"] += 1
            if text_content_missing:
                training_filter_reasons["missing_question_text"] += 1
            if exclude_from_training:
                training_filter_reasons["audited_exclusion"] += 1

            prediction = {
                "unit_key": unit_key,
                "question_id": question_id,
                "root_question_id": _as_text(
                    unit.get("root_question_id")
                    or unit.get("parent_id")
                    or question_id
                ),
                "input_role": _as_text(
                    unit.get("input_role") or unit.get("unit_type")
                ),
                "reason": parsed["reason"],
                "selected_labels": selected_labels,
                "audited_excluded_labels": removed_by_audit,
                "unknown_selected_codes_dropped": dropped_unknown,
                "none_of_candidates": parsed["none_of_candidates"],
                "need_expand_recall": parsed["need_expand_recall"],
                "context_insufficient": parsed["context_insufficient"],
                "text_content_missing": text_content_missing,
                "needs_review": needs_review,
                "usable_for_training": usable_for_training,
                "candidate_count": len(candidates),
                "model": model,
                "prompt_version": PROMPT_VERSION,
            }
            materialized_predictions.append(prediction)
            output.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
    predictions_temporary.replace(predictions_path)
    question_prediction_summary = write_question_predictions(
        output_dir / "question_predictions.jsonl",
        materialized_predictions,
        expected_units=units,
        model=model,
    )

    success = len(completed)
    latencies = sorted(
        float(record["latency_seconds"])
        for record in completed.values()
        if isinstance(record.get("latency_seconds"), (int, float))
    )

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        index = max(0, math.ceil(len(values) * fraction) - 1)
        return round(values[index], 3)

    usages = [
        record["usage"]
        for record in completed.values()
        if isinstance(record.get("usage"), dict)
    ]
    run_wall_seconds = round(time.monotonic() - run_started, 3)
    requests_this_run = requests_succeeded + requests_failed
    report = {
        "input": len(units),
        "processed": success,
        "success": success,
        "error": len(units) - success,
        "pending": len(units) - success,
        "evidence_rows": evidence_rows,
        "requests_succeeded": requests_succeeded,
        "requests_failed": requests_failed,
        "workers": workers,
        "run_started_at": run_started_at,
        "run_wall_seconds": run_wall_seconds,
        "requests_per_second_this_run": (
            round(requests_this_run / run_wall_seconds, 4)
            if run_wall_seconds
            else None
        ),
        "request_latency_seconds": {
            "count": len(latencies),
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "max": round(latencies[-1], 3) if latencies else None,
        },
        "token_usage": {
            "requests_with_usage": len(usages),
            "total_prompt_tokens": sum(
                int(usage.get("prompt_tokens", 0)) for usage in usages
            ),
            "total_completion_tokens": sum(
                int(usage.get("completion_tokens", 0)) for usage in usages
            ),
            "total_tokens": sum(int(usage.get("total_tokens", 0)) for usage in usages),
        },
        "selected_count_distribution": dict(
            sorted(selected_count_distribution.items(), key=lambda item: int(item[0]))
        ),
        "need_expand_recall": need_expand,
        "none_of_candidates": none_count,
        "context_insufficient": context_insufficient_count,
        "unknown_selected_codes_dropped": unknown_selected_codes_dropped_count,
        "usable_for_training": usable_for_training_count,
        "filtered_from_training": success - usable_for_training_count,
        "training_filter_reasons": dict(sorted(training_filter_reasons.items())),
        "audited_exclusion_questions": audited_exclusion_questions,
        "audited_excluded_labels": audited_excluded_labels,
        "input_sha256": manifest["input_sha256"],
        "candidate_count_distribution": candidate_count_distribution,
        "model": model,
        "temperature": temperature,
        "prompt_version": PROMPT_VERSION,
        **question_prediction_summary,
    }
    _write_json_atomic(output_dir / "report.json", report)
    _append_run_log(
        run_log_path,
        "END "
        f"success={report['success']} error={report['error']} "
        f"pending={report['pending']} run_wall_seconds={report['run_wall_seconds']}",
    )
    return report
