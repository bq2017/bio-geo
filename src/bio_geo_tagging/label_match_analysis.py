"""Aggregate question-label scores and build a review report with examples."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


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


def load_questions(path: str) -> dict[str, dict[str, Any]]:
    questions = {}
    for number, record in read_jsonl(path):
        question_id = str(record.get("question_id", ""))
        if not question_id:
            raise ValueError(f"题目文件第 {number} 行缺少question_id")
        if question_id in questions:
            raise ValueError(f"题目文件中重复的question_id: {question_id}")
        questions[question_id] = record
    return questions


def load_definitions(path: str) -> dict[str, dict[str, Any]]:
    definitions = {}
    for number, record in read_jsonl(path):
        label = record.get("knw_label")
        if not isinstance(label, str):
            raise ValueError(f"释义文件第 {number} 行缺少knw_label")
        definitions[label] = record.get("existing_interpretation", {})
    return definitions


def anomaly_reason(
    evaluated_count: int,
    grade_counts: Counter,
    min_samples: int,
    min_d_count: int,
    min_d_rate: float,
    all_d_min: int,
) -> str | None:
    d_count = grade_counts["D"]
    d_rate = d_count / evaluated_count
    if evaluated_count >= all_d_min and d_count == evaluated_count:
        return "当前样本全部为D级，知识点释义或原标签需要优先复核。"
    if (
        evaluated_count >= min_samples
        and d_count >= min_d_count
        and d_rate >= min_d_rate
    ):
        if grade_counts["A"] + grade_counts["B"] > 0:
            return "高匹配与低匹配题目并存，知识点边界或部分原标签需要复核。"
        return "D级题目数量和占比较高，知识点释义或原标签需要复核。"
    return None


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


def analyze(
    match_jsonl: str,
    questions_jsonl: str,
    definitions_jsonl: str,
    statistics_json: str,
    anomalies_jsonl: str,
    report_md: str,
    scope_label: str = "40%阶段性快照",
    min_samples: int = 5,
    min_d_count: int = 3,
    min_d_rate: float = 0.30,
    all_d_min: int = 3,
    low_examples: int = 3,
    high_examples: int = 2,
) -> dict[str, int]:
    if min_samples < 1 or min_d_count < 1 or all_d_min < 1:
        raise ValueError("样本数阈值必须为正整数")
    if not 0 <= min_d_rate <= 1:
        raise ValueError("min_d_rate必须在0到1之间")

    questions = load_questions(questions_jsonl)
    definitions = load_definitions(definitions_jsonl)
    status_counts: Counter = Counter()
    judgement_counts: Counter = Counter()
    grade_totals: Counter = Counter()
    unique_questions: set[str] = set()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for _, record in read_jsonl(match_jsonl):
        status_counts[str(record.get("status"))] += 1
        judgement_counts[str(record.get("judgement"))] += 1
        question_id = str(record.get("question_id", ""))
        if question_id:
            unique_questions.add(question_id)
        if record.get("status") != "completed" or record.get("judgement") != "scored":
            continue
        score = record.get("score")
        if type(score) not in (float, int) or not 0 <= score <= 1:
            raise ValueError(f"question_id={question_id} 的score无效")
        label_id = str(record.get("label_id", ""))
        label = record.get("knw_label")
        if not label_id or not isinstance(label, str):
            raise ValueError(f"question_id={question_id} 缺少标签信息")
        enriched = dict(record)
        enriched["grade"] = score_grade(score)
        grade_totals[enriched["grade"]] += 1
        grouped[(label_id, label)].append(enriched)

    label_statistics = []
    anomalies = []
    missing_question_examples = 0
    for (label_id, label), records in grouped.items():
        grades = Counter(record["grade"] for record in records)
        evaluated_count = len(records)
        d_rate = grades["D"] / evaluated_count
        reason = anomaly_reason(
            evaluated_count,
            grades,
            min_samples,
            min_d_count,
            min_d_rate,
            all_d_min,
        )
        statistic = {
            "label_id": label_id,
            "knw_label": label,
            "evaluated_count": evaluated_count,
            "grade_counts": {grade: grades[grade] for grade in GRADE_NAMES},
            "match_count": grades["A"] + grades["B"],
            "match_rate": (grades["A"] + grades["B"]) / evaluated_count,
            "d_count": grades["D"],
            "d_rate": d_rate,
            "zero_count": sum(record["score"] == 0 for record in records),
            "anomaly_candidate": reason is not None,
            "anomaly_reason": reason,
        }
        label_statistics.append(statistic)
        if reason is None:
            continue

        low_records = sorted(
            (record for record in records if record["grade"] == "D"),
            key=lambda record: (record["score"], record["question_id"]),
        )[:low_examples]
        high_records = sorted(
            (record for record in records if record["grade"] in {"A", "B"}),
            key=lambda record: (-record["score"], record["question_id"]),
        )[:high_examples]

        def make_examples(selected):
            nonlocal missing_question_examples
            examples = []
            for record in selected:
                question = questions.get(str(record["question_id"]))
                if question is None:
                    missing_question_examples += 1
                    question_data = None
                else:
                    question_data = compact_question(question)
                examples.append(
                    {
                        "question_id": str(record["question_id"]),
                        "score": record["score"],
                        "grade": record["grade"],
                        "model_reason": record.get("reason", ""),
                        "question": question_data,
                    }
                )
            return examples

        anomalies.append(
            {
                **statistic,
                "existing_interpretation": definitions.get(label, {}),
                "low_score_examples": make_examples(low_records),
                "high_score_examples": make_examples(high_records),
            }
        )

    label_statistics.sort(key=lambda item: (-item["d_rate"], -item["evaluated_count"], item["label_id"]))
    anomalies.sort(key=lambda item: (-item["d_rate"], -item["evaluated_count"], item["label_id"]))
    total_records = sum(status_counts.values())
    statistics = {
        "scope": scope_label,
        "total_records": total_records,
        "unique_questions": len(unique_questions),
        "covered_labels": len(grouped),
        "status_counts": dict(status_counts),
        "judgement_counts": dict(judgement_counts),
        "grade_counts": {grade: grade_totals[grade] for grade in GRADE_NAMES},
        "anomaly_label_count": len(anomalies),
        "missing_question_examples": missing_question_examples,
        "thresholds": {
            "A": "score >= 0.80",
            "B": "0.70 <= score < 0.80",
            "C": "0.40 <= score < 0.70",
            "D": "score < 0.40",
            "min_samples": min_samples,
            "min_d_count": min_d_count,
            "min_d_rate": min_d_rate,
            "all_d_min": all_d_min,
        },
        "labels": label_statistics,
    }

    statistics_path = Path(statistics_json)
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_path.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    anomalies_path = Path(anomalies_jsonl)
    anomalies_path.parent.mkdir(parents=True, exist_ok=True)
    with anomalies_path.open("w", encoding="utf-8") as stream:
        for anomaly in anomalies:
            stream.write(json.dumps(anomaly, ensure_ascii=False) + "\n")

    report_path = Path(report_md)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(statistics, anomalies),
        encoding="utf-8",
    )
    return {
        "total_records": total_records,
        "covered_labels": len(grouped),
        "anomaly_labels": len(anomalies),
        "missing_question_examples": missing_question_examples,
    }


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def text_block(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "（空）"


def render_question(example: dict[str, Any], heading: str) -> list[str]:
    lines = [
        f"#### {heading}",
        "",
        f"- question_id：`{example['question_id']}`",
        f"- 级别：{example['grade']}",
        f"- 匹配分数：{example['score']:.2f}",
        f"- 模型理由：{text_block(example['model_reason'])}",
        "",
    ]
    question = example.get("question")
    if question is None:
        lines.extend(["题目数据未找到。", ""])
        return lines
    lines.extend(
        [
            "题干：",
            "",
            "~~~text",
            text_block(question.get("stem")),
            "~~~",
            "",
        ]
    )
    if question.get("options"):
        lines.extend(["选项：", "", "~~~text", text_block(question["options"]), "~~~", ""])
    if question.get("analysis"):
        lines.extend(["解析：", "", "~~~text", text_block(question["analysis"]), "~~~", ""])
    for index, sub in enumerate(question.get("sub_questions", []), 1):
        lines.extend(
            [
                f"小题{index}（`{sub.get('question_id', '')}`）：",
                "",
                "~~~text",
                text_block(sub.get("stem")),
                "~~~",
                "",
            ]
        )
        if sub.get("options"):
            lines.extend(["选项：", "", "~~~text", text_block(sub["options"]), "~~~", ""])
        if sub.get("analysis"):
            lines.extend(["解析：", "", "~~~text", text_block(sub["analysis"]), "~~~", ""])
    return lines


def render_report(statistics: dict[str, Any], anomalies: list[dict[str, Any]]) -> str:
    grades = statistics["grade_counts"]
    scored_total = sum(grades.values())
    lines = [
        "# 地理知识点释义匹配阶段性分析",
        "",
        f"> 本报告基于{statistics['scope']}，属于阶段性结果，不代表完整题库最终分布。",
        "",
        "## 一、总体概览",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 总记录数 | {statistics['total_records']} |",
        f"| 不重复题数 | {statistics['unique_questions']} |",
        f"| 覆盖标签数 | {statistics['covered_labels']} |",
        f"| 已评分记录数 | {scored_total} |",
        f"| 异常知识点候选数 | {statistics['anomaly_label_count']} |",
        f"| 未找到完整题目的示例数 | {statistics['missing_question_examples']} |",
        "",
        "## 二、ABCD分级",
        "",
        "| 级别 | 分数 | 数量 | 占比 |",
        "| --- | --- | ---: | ---: |",
    ]
    ranges = {"A": "≥0.80", "B": "0.70～0.79", "C": "0.40～0.69", "D": "<0.40"}
    for grade in GRADE_NAMES:
        share = grades[grade] / scored_total if scored_total else 0
        lines.append(f"| {grade} | {ranges[grade]} | {grades[grade]} | {percent(share)} |")
    lines.extend(
        [
            "",
            "## 三、异常知识点候选",
            "",
            "| 知识点 | label_id | 样本数 | A | B | C | D | match率 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for anomaly in anomalies:
        counts = anomaly["grade_counts"]
        lines.append(
            f"| {anomaly['knw_label']} | {anomaly['label_id']} | {anomaly['evaluated_count']} | "
            f"{counts['A']} | {counts['B']} | {counts['C']} | {counts['D']} | "
            f"{percent(anomaly['match_rate'])} |"
        )
    if not anomalies:
        lines.append("| 暂无 | - | 0 | 0 | 0 | 0 | 0 | - |")
    lines.extend(["", "## 四、异常知识点及题目示例", ""])
    for index, anomaly in enumerate(anomalies, 1):
        counts = anomaly["grade_counts"]
        lines.extend(
            [
                f"### {index}. {anomaly['knw_label']}",
                "",
                "| 指标 | 数值 |",
                "| --- | --- |",
                f"| label_id | `{anomaly['label_id']}` |",
                f"| 当前样本数 | {anomaly['evaluated_count']} |",
                f"| A/B/C/D | {counts['A']} / {counts['B']} / {counts['C']} / {counts['D']} |",
                f"| match率 | {percent(anomaly['match_rate'])} |",
                f"| 阶段性判断 | {anomaly['anomaly_reason']} |",
                "",
                "原释义：",
                "",
            ]
        )
        interpretation = anomaly.get("existing_interpretation", {})
        lines.extend(
            [
                f"- 定义/核心内容：{text_block(interpretation.get('definition'))}",
                f"- 核心概念/关键术语：{text_block(interpretation.get('keywords'))}",
                f"- 常见考查方式：{text_block(interpretation.get('exam_methods'))}",
                f"- 易混淆区分：{text_block(interpretation.get('distinction'))}",
                "",
                "#### D级题目示例",
                "",
            ]
        )
        for example_index, example in enumerate(anomaly["low_score_examples"], 1):
            lines.extend(render_question(example, f"D级示例{example_index}"))
        if anomaly["high_score_examples"]:
            lines.extend(["#### A/B级对照题", ""])
            for example_index, example in enumerate(anomaly["high_score_examples"], 1):
                lines.extend(render_question(example, f"高匹配对照示例{example_index}"))
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate label-match scores and build a review report")
    parser.add_argument("--match-jsonl", required=True)
    parser.add_argument("--questions-jsonl", required=True)
    parser.add_argument("--definitions-jsonl", required=True)
    parser.add_argument("--statistics-json", required=True)
    parser.add_argument("--anomalies-jsonl", required=True)
    parser.add_argument("--report-md", required=True)
    parser.add_argument("--scope-label", default="40%阶段性快照")
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--min-d-count", type=int, default=3)
    parser.add_argument("--min-d-rate", type=float, default=0.30)
    parser.add_argument("--all-d-min", type=int, default=3)
    parser.add_argument("--low-examples", type=int, default=3)
    parser.add_argument("--high-examples", type=int, default=2)
    args = parser.parse_args()
    result = analyze(
        args.match_jsonl,
        args.questions_jsonl,
        args.definitions_jsonl,
        args.statistics_json,
        args.anomalies_jsonl,
        args.report_md,
        args.scope_label,
        args.min_samples,
        args.min_d_count,
        args.min_d_rate,
        args.all_d_min,
        args.low_examples,
        args.high_examples,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
