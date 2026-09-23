"""Compare label-definition scoring with label-name-only scoring."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


MATCH_THRESHOLD = 0.70
LARGE_DELTA_THRESHOLD = 0.30
GRADE_NAMES = ("A", "B", "C", "D")


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield number, json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {number} 行不是有效JSON") from error


def score_grade(score: float) -> str:
    if score >= 0.80:
        return "A"
    if score >= 0.70:
        return "B"
    if score >= 0.40:
        return "C"
    return "D"


def load_results(path: str) -> tuple[dict[tuple[str, str], dict[str, Any]], Counter]:
    records = {}
    status_counts: Counter = Counter()
    for number, record in read_jsonl(path):
        question_id = str(record.get("question_id", ""))
        label_id = str(record.get("label_id", ""))
        if not question_id or not label_id:
            raise ValueError(f"{path} 第 {number} 行缺少question_id或label_id")
        key = (question_id, label_id)
        if key in records:
            raise ValueError(f"{path} 存在重复主键: {key}")
        records[key] = record
        status_counts[str(record.get("status"))] += 1
    return records, status_counts


def comparable_score(record: dict[str, Any]) -> float | None:
    if record.get("status") != "completed" or record.get("judgement") != "scored":
        return None
    score = record.get("score")
    if type(score) not in (float, int) or not 0 <= score <= 1:
        return None
    return float(score)


def comparison_type(definition_score: float, name_score: float) -> str:
    definition_match = definition_score >= MATCH_THRESHOLD
    name_match = name_score >= MATCH_THRESHOLD
    if definition_match and name_match:
        return "both_match"
    if not definition_match and name_match:
        return "definition_suppressed"
    if definition_match and not name_match:
        return "definition_expanded"
    return "both_mismatch"


def compact_question(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": str(question.get("question_id", "")),
        "stem": question.get("stem", ""),
        "options": question.get("options", ""),
        "analysis": question.get("analysis", ""),
        "sub_questions": [
            {
                "question_id": str(sub.get("question_id", "")),
                "stem": sub.get("stem", ""),
                "options": sub.get("options", ""),
                "analysis": sub.get("analysis", ""),
            }
            for sub in question.get("sub_questions", [])
            if isinstance(sub, dict)
        ],
    }


def load_selected_questions(path: str, wanted: set[str]) -> dict[str, dict[str, Any]]:
    questions = {}
    for number, record in read_jsonl(path):
        question_id = str(record.get("question_id", ""))
        if not question_id:
            raise ValueError(f"题目文件第 {number} 行缺少question_id")
        if question_id in wanted:
            questions[question_id] = compact_question(record)
    return questions


def load_definitions(path: str) -> dict[str, dict[str, Any]]:
    definitions = {}
    for number, record in read_jsonl(path):
        label = record.get("knw_label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"释义文件第 {number} 行缺少knw_label")
        definitions[label] = record.get("existing_interpretation", {})
    return definitions


def stable_sample(records: list[dict[str, Any]], limit: int, salt: str) -> list[dict[str, Any]]:
    def rank(record: dict[str, Any]) -> str:
        value = f"{salt}:{record['question_id']}:{record['label_id']}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()

    return sorted(records, key=rank)[:limit]


def select_evidence(records: list[dict[str, Any]], kind: str, limit: int) -> list[dict[str, Any]]:
    selected = [record for record in records if record["comparison_type"] == kind]
    if kind == "definition_suppressed":
        return sorted(selected, key=lambda item: (item["score_delta"], item["question_id"]))[:limit]
    if kind == "definition_expanded":
        return sorted(selected, key=lambda item: (-item["score_delta"], item["question_id"]))[:limit]
    return stable_sample(selected, limit, kind)


def candidate_reasons(statistic: dict[str, Any]) -> list[str]:
    count = statistic["evaluated_count"]
    suppressed = statistic["definition_suppressed_count"]
    expanded = statistic["definition_expanded_count"]
    reasons = []
    if count < 20 and statistic["large_flip_count"] > 0:
        reasons.append("小样本中存在明显跨阈值差异")
    if suppressed >= 5 and suppressed / count >= 0.05:
        reasons.append("释义抑制数量和占比达到复核阈值")
    if expanded >= 5 and expanded / count >= 0.05:
        reasons.append("释义扩张数量和占比达到复核阈值")
    if abs(statistic["mean_score_delta"]) >= 0.20:
        reasons.append("平均分差达到复核阈值")
    if statistic["large_flip_count"] >= 5:
        reasons.append("明显跨阈值差异数量达到复核阈值")
    return reasons


def compare(
    definition_jsonl: str,
    name_jsonl: str,
    questions_jsonl: str,
    definitions_jsonl: str,
    pairs_output: str,
    statistics_output: str,
    review_output: str,
    report_output: str,
    directional_examples: int = 10,
    agreement_examples: int = 5,
) -> dict[str, int]:
    if directional_examples < 1 or agreement_examples < 1:
        raise ValueError("证据题数量必须为正整数")

    definition_records, definition_status = load_results(definition_jsonl)
    name_records, name_status = load_results(name_jsonl)
    definition_keys = set(definition_records)
    name_keys = set(name_records)
    common_keys = definition_keys & name_keys
    label_mismatches = []
    non_comparable = Counter()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    type_totals: Counter = Counter()

    pairs_path = Path(pairs_output)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with pairs_path.open("w", encoding="utf-8") as stream:
        for key in sorted(common_keys):
            definition_record = definition_records[key]
            name_record = name_records[key]
            definition_label = definition_record.get("knw_label")
            name_label = name_record.get("knw_label")
            if definition_label != name_label:
                label_mismatches.append(key)
                continue
            definition_score = comparable_score(definition_record)
            name_score = comparable_score(name_record)
            if definition_score is None or name_score is None:
                if definition_score is None:
                    non_comparable["definition_not_scored"] += 1
                if name_score is None:
                    non_comparable["name_not_scored"] += 1
                continue
            delta = definition_score - name_score
            kind = comparison_type(definition_score, name_score)
            pair = {
                "question_id": key[0],
                "label_id": key[1],
                "knw_label": definition_label,
                "definition_score": definition_score,
                "definition_grade": score_grade(definition_score),
                "definition_match": definition_score >= MATCH_THRESHOLD,
                "definition_reason": definition_record.get("reason", ""),
                "name_score": name_score,
                "name_grade": score_grade(name_score),
                "name_match": name_score >= MATCH_THRESHOLD,
                "name_reason": name_record.get("reason", ""),
                "score_delta": round(delta, 6),
                "absolute_delta": round(abs(delta), 6),
                "large_delta": abs(delta) >= LARGE_DELTA_THRESHOLD,
                "comparison_type": kind,
            }
            stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
            grouped[(key[1], definition_label)].append(pair)
            type_totals[kind] += 1

    label_statistics = []
    for (label_id, label), records in grouped.items():
        counts = Counter(record["comparison_type"] for record in records)
        definition_grades = Counter(record["definition_grade"] for record in records)
        name_grades = Counter(record["name_grade"] for record in records)
        count = len(records)
        statistic = {
            "label_id": label_id,
            "knw_label": label,
            "evaluated_count": count,
            "definition_grade_counts": {grade: definition_grades[grade] for grade in GRADE_NAMES},
            "name_grade_counts": {grade: name_grades[grade] for grade in GRADE_NAMES},
            "definition_match_count": sum(record["definition_match"] for record in records),
            "name_match_count": sum(record["name_match"] for record in records),
            "definition_match_rate": sum(record["definition_match"] for record in records) / count,
            "name_match_rate": sum(record["name_match"] for record in records) / count,
            "mean_definition_score": sum(record["definition_score"] for record in records) / count,
            "mean_name_score": sum(record["name_score"] for record in records) / count,
            "mean_score_delta": sum(record["score_delta"] for record in records) / count,
            "both_match_count": counts["both_match"],
            "definition_suppressed_count": counts["definition_suppressed"],
            "definition_expanded_count": counts["definition_expanded"],
            "both_mismatch_count": counts["both_mismatch"],
            "large_delta_count": sum(record["large_delta"] for record in records),
            "large_flip_count": sum(
                record["large_delta"]
                and record["comparison_type"] in {"definition_suppressed", "definition_expanded"}
                for record in records
            ),
        }
        statistic["definition_suppressed_rate"] = statistic["definition_suppressed_count"] / count
        statistic["definition_expanded_rate"] = statistic["definition_expanded_count"] / count
        statistic["review_reasons"] = candidate_reasons(statistic)
        statistic["review_candidate"] = bool(statistic["review_reasons"])
        label_statistics.append(statistic)

    label_statistics.sort(
        key=lambda item: (-item["large_flip_count"], -item["evaluated_count"], item["label_id"])
    )
    candidates = [item for item in label_statistics if item["review_candidate"]]
    selected_by_label = {}
    wanted_question_ids = set()
    for statistic in candidates:
        key = (statistic["label_id"], statistic["knw_label"])
        records = grouped[key]
        selected = {
            "definition_suppressed_examples": select_evidence(
                records, "definition_suppressed", directional_examples
            ),
            "definition_expanded_examples": select_evidence(
                records, "definition_expanded", directional_examples
            ),
            "both_match_examples": select_evidence(records, "both_match", agreement_examples),
            "both_mismatch_examples": select_evidence(
                records, "both_mismatch", agreement_examples
            ),
        }
        selected_by_label[key] = selected
        for examples in selected.values():
            wanted_question_ids.update(example["question_id"] for example in examples)

    questions = load_selected_questions(questions_jsonl, wanted_question_ids)
    definitions = load_definitions(definitions_jsonl)
    review_path = Path(review_output)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    missing_questions = 0
    with review_path.open("w", encoding="utf-8") as stream:
        for statistic in candidates:
            key = (statistic["label_id"], statistic["knw_label"])
            evidence = selected_by_label[key]
            for examples in evidence.values():
                for example in examples:
                    question = questions.get(example["question_id"])
                    example["question"] = question
                    if question is None:
                        missing_questions += 1
            review = {
                **statistic,
                "existing_interpretation": definitions.get(statistic["knw_label"], {}),
                **evidence,
            }
            stream.write(json.dumps(review, ensure_ascii=False) + "\n")

    quality = {
        "definition_total": len(definition_records),
        "name_total": len(name_records),
        "common_keys": len(common_keys),
        "definition_only": len(definition_keys - name_keys),
        "name_only": len(name_keys - definition_keys),
        "label_mismatch_count": len(label_mismatches),
        "label_mismatch_examples": [list(key) for key in label_mismatches[:20]],
        "non_comparable_counts": dict(non_comparable),
        "definition_status_counts": dict(definition_status),
        "name_status_counts": dict(name_status),
    }
    statistics = {
        "data_quality": quality,
        "thresholds": {
            "match": MATCH_THRESHOLD,
            "large_delta": LARGE_DELTA_THRESHOLD,
            "A": "score >= 0.80",
            "B": "0.70 <= score < 0.80",
            "C": "0.40 <= score < 0.70",
            "D": "score < 0.40",
        },
        "comparable_count": sum(type_totals.values()),
        "comparison_type_counts": dict(type_totals),
        "covered_labels": len(label_statistics),
        "review_candidate_count": len(candidates),
        "missing_evidence_questions": missing_questions,
        "labels": label_statistics,
    }
    statistics_path = Path(statistics_output)
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_path.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path = Path(report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(statistics), encoding="utf-8")
    return {
        "definition_records": len(definition_records),
        "name_records": len(name_records),
        "comparable_records": statistics["comparable_count"],
        "covered_labels": len(label_statistics),
        "review_candidates": len(candidates),
        "missing_evidence_questions": missing_questions,
    }


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_report(statistics: dict[str, Any]) -> str:
    quality = statistics["data_quality"]
    counts = statistics["comparison_type_counts"]
    comparable = statistics["comparable_count"]
    lines = [
        "# 地理标签释义影响对比分析",
        "",
        "## 一、数据对齐情况",
        "",
        "| 指标 | 数量 |",
        "| --- | ---: |",
        f"| 原释义评分记录 | {quality['definition_total']} |",
        f"| 标签名称评分记录 | {quality['name_total']} |",
        f"| 共同主键 | {quality['common_keys']} |",
        f"| 可比较记录 | {comparable} |",
        f"| 仅原释义文件存在 | {quality['definition_only']} |",
        f"| 仅标签名称文件存在 | {quality['name_only']} |",
        f"| 标签名称不一致 | {quality['label_mismatch_count']} |",
        "",
        "## 二、逐题判断变化",
        "",
        "| 类型 | 数量 | 占比 |",
        "| --- | ---: | ---: |",
    ]
    descriptions = {
        "both_match": "两种依据均匹配",
        "definition_suppressed": "名称匹配、释义不匹配",
        "definition_expanded": "名称不匹配、释义匹配",
        "both_mismatch": "两种依据均不匹配",
    }
    for kind in descriptions:
        count = counts.get(kind, 0)
        share = count / comparable if comparable else 0
        lines.append(f"| {descriptions[kind]} | {count} | {percent(share)} |")
    lines.extend(
        [
            "",
            "## 三、需要进一步诊断的标签",
            "",
            "| 知识点 | 样本数 | 释义match率 | 名称match率 | 释义抑制 | 释义扩张 | 大幅翻转 | 平均分差 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    candidates = [item for item in statistics["labels"] if item["review_candidate"]]
    for item in candidates:
        lines.append(
            f"| {item['knw_label']} | {item['evaluated_count']} | "
            f"{percent(item['definition_match_rate'])} | {percent(item['name_match_rate'])} | "
            f"{item['definition_suppressed_count']} | {item['definition_expanded_count']} | "
            f"{item['large_flip_count']} | {item['mean_score_delta']:.3f} |"
        )
    if not candidates:
        lines.append("| 暂无 | 0 | - | - | 0 | 0 | 0 | - |")
    lines.extend(
        [
            "",
            "说明：候选标签只表示原释义可能系统性改变模型判断，需要结合证据题继续区分释义问题、历史误标和模型波动。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare definition-based and name-only scores")
    parser.add_argument("--definition-jsonl", required=True)
    parser.add_argument("--name-jsonl", required=True)
    parser.add_argument("--questions-jsonl", required=True)
    parser.add_argument("--definitions-jsonl", required=True)
    parser.add_argument("--pairs-output", required=True)
    parser.add_argument("--statistics-output", required=True)
    parser.add_argument("--review-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--directional-examples", type=int, default=10)
    parser.add_argument("--agreement-examples", type=int, default=5)
    args = parser.parse_args()
    result = compare(
        definition_jsonl=args.definition_jsonl,
        name_jsonl=args.name_jsonl,
        questions_jsonl=args.questions_jsonl,
        definitions_jsonl=args.definitions_jsonl,
        pairs_output=args.pairs_output,
        statistics_output=args.statistics_output,
        review_output=args.review_output,
        report_output=args.report_output,
        directional_examples=args.directional_examples,
        agreement_examples=args.agreement_examples,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
