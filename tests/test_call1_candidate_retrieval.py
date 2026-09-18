import json

import pytest

from bio_geo_tagging.call1_candidate_retrieval import (
    DeepSeekCandidateRetriever,
    REQUIREMENT_PROMPT,
    build_question_text,
    get_input_role,
    load_catalog,
    load_units,
    normalize_match_result,
    normalize_requirements,
    normalize_shard_result,
    parse_or_recover_result,
    run_retrieval,
    split_catalog,
    validate_result,
)


def test_load_catalog_returns_exact_paths(tmp_path):
    catalog = tmp_path / "catalog.txt"
    catalog.write_text(
        "知识点@自然地理@标签一｜释义一\n知识点@人文地理@标签二｜释义二\n",
        encoding="utf-8",
    )

    text, paths = load_catalog(catalog, expected_count=2)

    assert text.splitlines() == [
        "知识点@自然地理@标签一｜释义一",
        "知识点@人文地理@标签二｜释义二",
    ]
    assert paths == {"知识点@自然地理@标签一", "知识点@人文地理@标签二"}


def test_split_catalog_keeps_siblings_together_and_balances_parts():
    catalog = "\n".join([
        "知识点@自然地理@水@水循环｜释义0",
        "知识点@自然地理@水@河流｜释义1",
        "知识点@自然地理@大气@气候｜释义2",
        "知识点@自然地理@大气@天气｜释义3",
        "知识点@人文地理@农业@农业区位｜释义4",
        "知识点@人文地理@农业@农业类型｜释义5",
        "知识点@人文地理@工业@工业区位｜释义6",
        "知识点@人文地理@工业@工业地域｜释义7",
        "知识点@世界地理@亚洲@东亚｜释义8",
        "知识点@世界地理@亚洲@东南亚｜释义9",
        "知识点@世界地理@欧洲@西欧｜释义10",
        "知识点@世界地理@欧洲@北欧｜释义11",
    ])

    parts = split_catalog(catalog, parts=3)

    assert [len(paths) for _, paths in parts] == [4, 4, 4]
    all_paths = [path for _, paths in parts for path in paths]
    assert len(all_paths) == len(set(all_paths)) == 12

    part_by_path = {
        path: part_index
        for part_index, (_, paths) in enumerate(parts)
        for path in paths
    }
    sibling_pairs = [
        ("知识点@自然地理@水@水循环", "知识点@自然地理@水@河流"),
        ("知识点@自然地理@大气@气候", "知识点@自然地理@大气@天气"),
        ("知识点@人文地理@农业@农业区位", "知识点@人文地理@农业@农业类型"),
        ("知识点@人文地理@工业@工业区位", "知识点@人文地理@工业@工业地域"),
        ("知识点@世界地理@亚洲@东亚", "知识点@世界地理@亚洲@东南亚"),
        ("知识点@世界地理@欧洲@西欧", "知识点@世界地理@欧洲@北欧"),
    ]
    for first, second in sibling_pairs:
        assert part_by_path[first] == part_by_path[second]


def test_build_question_text_separates_context_and_excludes_existing_labels():
    unit = {
        "question_id": "child-1",
        "input_role": "subquestion",
        "context_stem": "公共材料",
        "stem": "当前小题",
        "options": [{"key": "A", "text": "选项A"}],
        "answer": "A",
        "analysis": "解析内容",
        "image_description": "等高线图",
        "knw_labels": ["知识点@不应泄漏"],
    }

    rendered = build_question_text(unit)

    assert "【公共题干，仅作为上下文】\n公共材料" in rendered
    assert "【当前打标对象】\n当前小题" in rendered
    assert json.dumps(unit["options"], ensure_ascii=False) in rendered
    assert "【答案】\nA" in rendered
    assert "【解析】\n解析内容" in rendered
    assert "【图片描述】\n等高线图" in rendered
    assert "不应泄漏" not in rendered


def test_build_question_text_renders_complete_group_without_labels():
    unit = {
        "question_id": "root-1",
        "parent_id": "root-1",
        "stem": "公共材料",
        "knw_labels": ["知识点@整题旧标签"],
        "sub_questions": [
            {
                "question_id": "child-1",
                "stem": "小题一",
                "options": "A. 甲\nB. 乙",
                "analysis": "小题一解析",
                "knw_labels": ["知识点@小题一旧标签"],
            },
            {
                "question_id": "child-2",
                "stem": "小题二",
                "options": "A. 丙\nB. 丁",
                "answer": "B",
                "analysis": "小题二解析",
                "knw_labels": ["知识点@小题二旧标签"],
            },
        ],
    }

    rendered = build_question_text(unit)

    assert "【整道题公共材料】\n公共材料" in rendered
    assert "【小题1】\n小题一" in rendered
    assert "【小题1选项】\nA. 甲\nB. 乙" in rendered
    assert "【小题1解析】\n小题一解析" in rendered
    assert "【小题2】\n小题二" in rendered
    assert "【小题2答案】\nB" in rendered
    assert "整题旧标签" not in rendered
    assert "小题一旧标签" not in rendered
    assert "小题二旧标签" not in rendered
    assert get_input_role(unit) == "question_group"


def test_load_units_skips_empty_root_but_keeps_its_subquestion(tmp_path):
    input_path = tmp_path / "units.jsonl"
    units = [
        {
            "question_id": "root-1",
            "root_question_id": "root-1",
            "input_role": "root",
            "stem": "",
        },
        {
            "question_id": "child-1",
            "root_question_id": "root-1",
            "input_role": "subquestion",
            "context_stem": "",
            "stem": "当前小题",
        },
    ]
    input_path.write_text(
        "".join(json.dumps(unit, ensure_ascii=False) + "\n" for unit in units),
        encoding="utf-8",
    )

    valid_units, skipped_empty_stem = load_units(input_path)

    assert valid_units == [units[1]]
    assert skipped_empty_stem == 1


def test_validate_result_accepts_known_unique_labels():
    allowed = {"知识点@标签一", "知识点@标签二"}

    labels, uncovered = validate_result(
        {
            "candidate_labels": ["知识点@标签一", "知识点@标签二"],
            "uncovered_topic": None,
        },
        allowed,
    )

    assert labels == ["知识点@标签一", "知识点@标签二"]
    assert uncovered is None


def test_normalize_shard_result_repairs_format_and_deduplicates():
    allowed = {"知识点@自然地理@地球仪", "知识点@自然地理@经纬网"}

    labels = normalize_shard_result(
        {
            "candidate_labels": [
                "自然地理@地球仪",
                "知识点@自然地理@地球仪｜标签释义",
                "知识点@自然地理@经纬网",
                "知识点@其他批次@标签",
            ]
        },
        allowed,
    )

    assert labels == ["知识点@自然地理@地球仪", "知识点@自然地理@经纬网"]


def test_normalize_requirements_validates_unique_complete_items():
    requirements = normalize_requirements(
        {
            "requirements": [
                {
                    "requirement_id": "R1",
                    "question_part": "小题1",
                    "requirement": "判断河流补给类型",
                },
                {
                    "requirement_id": "R2",
                    "question_part": "小题2",
                    "requirement": "分析径流季节变化",
                },
            ]
        }
    )

    assert [item["requirement_id"] for item in requirements] == ["R1", "R2"]


def test_normalize_match_result_keeps_clear_possible_and_additional_matches():
    allowed = {"知识点@标签一", "知识点@标签二", "知识点@标签三"}

    labels, matches, additional = normalize_match_result(
        {
            "matches": [
                {
                    "requirement_id": "R1",
                    "clear_labels": ["知识点@标签一"],
                    "possible_labels": ["知识点@标签二"],
                }
            ],
            "additional_matches": [
                {
                    "question_part": "小题2",
                    "requirement": "补充要求",
                    "clear_labels": ["知识点@标签三"],
                    "possible_labels": [],
                }
            ],
        },
        allowed,
        {"R1"},
        "A1_",
        allow_additional=True,
    )

    assert labels == ["知识点@标签一", "知识点@标签二", "知识点@标签三"]
    assert matches[1]["requirement_id"] == "A1_1"
    assert additional == [
        {
            "requirement_id": "A1_1",
            "question_part": "小题2",
            "requirement": "补充要求",
        }
    ]


def test_normalize_match_result_keeps_whole_question_matches():
    allowed = {"知识点@具体标签", "知识点@综合标签"}

    labels, matches, additional = normalize_match_result(
        {
            "matches": [
                {
                    "requirement_id": "R1",
                    "clear_labels": ["知识点@具体标签"],
                    "possible_labels": [],
                }
            ],
            "whole_question_matches": [
                {
                    "basis": "小题1和小题2共同形成综合考查",
                    "clear_labels": [],
                    "possible_labels": ["知识点@综合标签"],
                }
            ],
            "additional_matches": [],
        },
        allowed,
        {"R1"},
        "A1_",
        allow_additional=True,
    )

    assert labels == ["知识点@具体标签", "知识点@综合标签"]
    assert matches[1] == {
        "requirement_id": None,
        "evidence_scope": "whole_question",
        "basis": "小题1和小题2共同形成综合考查",
        "clear_labels": [],
        "possible_labels": ["知识点@综合标签"],
    }
    assert additional == []


def test_parse_or_recover_result_recovers_labels_from_broken_json():
    allowed = {
        "知识点@自然地理@地球仪",
        "知识点@自然地理@地球仪@经纬网",
        "知识点@人文地理@人口",
    }
    broken = (
        '{"candidate_labels":["自然地理@地球仪@经纬网",'
        '"知识点@人文地理@人口" "unexpected"]}'
    )

    result = parse_or_recover_result(broken, allowed)

    assert result == {
        "candidate_labels": [
            "知识点@自然地理@地球仪@经纬网",
            "知识点@人文地理@人口",
        ]
    }


def test_parse_or_recover_result_rejects_truncated_json():
    allowed = {"知识点@自然地理@地球仪"}

    with pytest.raises(json.JSONDecodeError):
        parse_or_recover_result(
            '{"candidate_labels":["知识点@自然地理@地球仪"', allowed
        )


@pytest.mark.parametrize(
    "candidate_labels",
    [
        ["知识点@不存在"],
        ["知识点@标签一", "知识点@标签一"],
        [f"知识点@标签{i}" for i in range(21)],
    ],
)
def test_validate_result_rejects_invalid_candidates(candidate_labels):
    allowed = {"知识点@标签一"} | {f"知识点@标签{i}" for i in range(21)}

    with pytest.raises(ValueError):
        validate_result(
            {"candidate_labels": candidate_labels, "uncovered_topic": None},
            allowed,
        )


def test_trace_records_each_shard_and_resume(monkeypatch, tmp_path):
    labels = [f"知识点@分类{i}@标签{i}" for i in range(3)]
    catalog = "\n".join(
        f"{label}｜释义{index}" for index, label in enumerate(labels)
    )
    unit = {"parent_id": "q1", "question_id": "q1", "stem": "题目", "sub_questions": []}
    output = tmp_path / "candidates.jsonl"
    trace_output = tmp_path / "trace.jsonl"

    def fake_request(self, system_prompt, question_text):
        if system_prompt == REQUIREMENT_PROMPT:
            return json.dumps(
                {
                    "requirements": [
                        {
                            "requirement_id": "R1",
                            "question_part": "题目",
                            "requirement": "完成题目",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        label = next(
            line.split("｜", 1)[0]
            for line in system_prompt.splitlines()
            if line.startswith("知识点@")
        )
        return json.dumps(
            {
                "matches": [
                    {
                        "requirement_id": "R1",
                        "clear_labels": [label],
                        "possible_labels": [],
                    }
                ],
                "additional_matches": [],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(DeepSeekCandidateRetriever, "request", fake_request)
    arguments = (
        [unit], catalog, set(labels),
        output, "test-model", "http://example.test/v1", None, 1, 1, 10.0, None,
        trace_output,
    )

    first = run_retrieval(*arguments)
    second = run_retrieval(*arguments)

    assert first["completed"] == 1
    assert second["completed"] == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1
    traces = [json.loads(line) for line in trace_output.read_text(encoding="utf-8").splitlines()]
    assert len(traces) == 1
    assert traces[0]["shard_candidate_labels"] == [
        [labels[0]], [labels[1]], [labels[2]]
    ]
    assert traces[0]["before_consolidation"] == traces[0]["candidate_labels"]
    assert traces[0]["consolidation_used"] is False
    assert traces[0]["recovery_used"] is False
    assert traces[0]["uncovered_after_recovery"] == []


def test_run_retrieval_shares_work_across_multiple_endpoints(monkeypatch, tmp_path):
    import threading

    labels = [f"知识点@分类{i}@标签{i}" for i in range(3)]
    catalog = "\n".join(
        f"{label}｜释义{index}" for index, label in enumerate(labels)
    )
    units = [
        {"parent_id": f"q{i}", "question_id": f"q{i}", "stem": f"题目{i}"}
        for i in range(4)
    ]
    endpoints = ["http://endpoint-1/v1", "http://endpoint-2/v1"]
    output = tmp_path / "candidates.jsonl"
    barrier = threading.Barrier(2)

    def fake_request(self, system_prompt, question_text):
        if system_prompt == REQUIREMENT_PROMPT:
            barrier.wait(timeout=2)
            return json.dumps(
                {
                    "requirements": [
                        {
                            "requirement_id": "R1",
                            "question_part": "题目",
                            "requirement": "完成题目",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        label = next(
            line.split("｜", 1)[0]
            for line in system_prompt.splitlines()
            if line.startswith("知识点@")
        )
        return json.dumps(
            {
                "matches": [
                    {
                        "requirement_id": "R1",
                        "clear_labels": [label],
                        "possible_labels": [],
                    }
                ],
                "whole_question_matches": [],
                "additional_matches": [],
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(DeepSeekCandidateRetriever, "request", fake_request)

    summary = run_retrieval(
        units,
        catalog,
        set(labels),
        output,
        "test-model",
        endpoints[0],
        None,
        1,
        1,
        10.0,
        None,
        base_urls=endpoints,
        concurrency_per_endpoint=1,
    )

    records = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert summary["completed"] == 4
    assert {record["endpoint"] for record in records} == set(endpoints)


def test_retrieve_only_recovers_requirements_left_uncovered(monkeypatch):
    labels = [f"知识点@分类{i}@标签{i}" for i in range(3)]
    catalog_parts = [
        (f"{label}｜释义{i}", {label}) for i, label in enumerate(labels)
    ]
    retriever = DeepSeekCandidateRetriever(
        catalog_parts,
        "test-model",
        "http://example.test/v1",
        None,
        10.0,
    )

    def fake_request(self, system_prompt, question_text):
        if system_prompt == REQUIREMENT_PROMPT:
            return json.dumps(
                {
                    "requirements": [
                        {
                            "requirement_id": "R1",
                            "question_part": "小题1",
                            "requirement": "要求一",
                        },
                        {
                            "requirement_id": "R2",
                            "question_part": "小题2",
                            "requirement": "要求二",
                        },
                    ]
                },
                ensure_ascii=False,
            )
        label = next(
            line.split("｜", 1)[0]
            for line in system_prompt.splitlines()
            if line.startswith("知识点@")
        )
        if "定向补召回" in system_prompt:
            matches = []
            if label == labels[2]:
                matches = [
                    {
                        "requirement_id": "R2",
                        "clear_labels": [],
                        "possible_labels": [label],
                    }
                ]
            return json.dumps({"matches": matches}, ensure_ascii=False)
        matches = []
        if label == labels[0]:
            matches = [
                {
                    "requirement_id": "R1",
                    "clear_labels": [label],
                    "possible_labels": [],
                }
            ]
        return json.dumps(
            {"matches": matches, "additional_matches": []},
            ensure_ascii=False,
        )

    monkeypatch.setattr(DeepSeekCandidateRetriever, "request", fake_request)

    candidates, trace = retriever.retrieve(
        {"question_id": "q1", "stem": "题目", "sub_questions": []}
    )

    assert candidates == [labels[0], labels[2]]
    assert trace["recovery_used"] is True
    assert [item["requirement_id"] for item in trace["uncovered_before_recovery"]] == ["R2"]
    assert trace["uncovered_after_recovery"] == []
    assert trace["recovery_shard_candidate_labels"] == [[], [], [labels[2]]]


def test_consolidation_requires_exactly_twenty_labels(monkeypatch):
    labels = [f"知识点@分类@标签{i}" for i in range(21)]
    catalog = "\n".join(f"{label}｜释义{i}" for i, label in enumerate(labels))
    retriever = DeepSeekCandidateRetriever(
        [(catalog, set(labels))],
        "test-model",
        "http://example.test/v1",
        None,
        10.0,
    )
    requirements = [
        {"requirement_id": "R1", "question_part": "题目", "requirement": "要求"}
    ]
    evidence = {
        label: {"match_type": "clear", "requirement_ids": ["R1"]}
        for label in labels
    }

    monkeypatch.setattr(
        DeepSeekCandidateRetriever,
        "request",
        lambda self, system_prompt, question_text: json.dumps(
            {"candidate_labels": labels[:19]}, ensure_ascii=False
        ),
    )
    with pytest.raises(ValueError, match="必须恰好20个"):
        retriever.consolidate("题目", labels, requirements, evidence)

    monkeypatch.setattr(
        DeepSeekCandidateRetriever,
        "request",
        lambda self, system_prompt, question_text: json.dumps(
            {"candidate_labels": labels[:20]}, ensure_ascii=False
        ),
    )
    assert retriever.consolidate("题目", labels, requirements, evidence) == labels[:20]
