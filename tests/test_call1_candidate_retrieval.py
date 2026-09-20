import json

import pytest

from bio_geo_tagging.call1_candidate_retrieval import (
    DeepSeekCandidateRetriever,
    EVIDENCE_PROMPT,
    ResponseGenerationError,
    StageResponseError,
    build_question_text,
    get_input_role,
    load_catalog,
    load_units,
    make_branch_key,
    make_label_key,
    normalize_branch_result,
    normalize_match_result,
    normalize_tagging_evidence,
    parse_json_object,
    run_retrieval,
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


def test_normalize_branch_result_maps_keys_and_deduplicates():
    key_to_branch = {
        "L001": "知识点@自然地理@地球仪",
        "L002": "知识点@自然地理@经纬网",
    }

    branches, matches, rejected = normalize_branch_result(
        {
            "branches": [
                {"branch_key": "L001", "evidence_ids": ["E1"]},
                {"branch_key": "l001｜附加内容", "evidence_ids": ["E2"]},
                {"branch_key": "L002", "evidence_ids": ["E2"]},
            ]
        },
        key_to_branch,
        {"E1", "E2"},
    )

    assert branches == ["知识点@自然地理@地球仪", "知识点@自然地理@经纬网"]
    assert matches == [
        {
            "branch": "知识点@自然地理@地球仪",
            "evidence_ids": ["E1", "E2"],
        },
        {"branch": "知识点@自然地理@经纬网", "evidence_ids": ["E2"]},
    ]
    assert rejected == []


def test_normalize_branch_result_rejects_only_unknown_keys():
    with pytest.raises(ValueError, match="任何有效临时序号"):
        normalize_branch_result(
            {
                "branches": [
                    {"branch_key": "L999", "evidence_ids": ["E1"]}
                ]
            },
            {"L001": "知识点@自然地理@地球仪"},
            {"E1"},
        )


def test_normalize_branch_result_keeps_valid_and_records_unknown_keys():
    branches, matches, rejected = normalize_branch_result(
        {
            "branches": [
                {"branch_key": "L001", "evidence_ids": ["E1"]},
                {"branch_key": "L999", "evidence_ids": ["E1"]},
            ]
        },
        {"L001": "知识点@自然地理@地球仪"},
        {"E1"},
    )

    assert branches == ["知识点@自然地理@地球仪"]
    assert matches == [
        {"branch": "知识点@自然地理@地球仪", "evidence_ids": ["E1"]}
    ]
    assert rejected == ["L999"]


def test_normalize_branch_result_allows_twelve_but_rejects_more():
    key_to_branch = {
        f"L{index:03d}": f"知识点@标签{index}"
        for index in range(1, 14)
    }

    branches_input = [
        {"branch_key": key, "evidence_ids": ["E1"]}
        for key in key_to_branch
    ]
    branches, matches, rejected = normalize_branch_result(
        {"branches": branches_input[:12]},
        key_to_branch,
        {"E1"},
    )

    assert len(branches) == 12
    assert len(matches) == 12
    assert rejected == []
    with pytest.raises(ValueError, match="超过12个"):
        normalize_branch_result(
            {"branches": branches_input},
            key_to_branch,
            {"E1"},
        )


def test_normalize_branch_result_requires_evidence():
    with pytest.raises(ValueError, match="非空evidence_ids"):
        normalize_branch_result(
            {
                "branches": [
                    {"branch_key": "L001", "evidence_ids": []}
                ]
            },
            {"L001": "知识点@自然地理@地球仪"},
            {"E1"},
        )


def test_normalize_branch_result_rejects_unknown_evidence():
    with pytest.raises(ValueError, match="不存在的标注依据"):
        normalize_branch_result(
            {
                "branches": [
                    {"branch_key": "L001", "evidence_ids": ["E9"]}
                ]
            },
            {"L001": "知识点@自然地理@地球仪"},
            {"E1"},
        )


def test_recall_pool_includes_branch_and_top_level_comprehensive_labels():
    labels = [
        "知识点@自然地理@地球的运动@地球公转特征",
        "知识点@自然地理@地球的运动@地球的运动综合",
        "知识点@自然地理@自然地理综合",
        "知识点@自然地理@大气的运动@锋面天气系统",
        "知识点@自然地理@大气的运动@大气的运动综合",
        "知识点@人文地理@人文地理综合",
    ]
    retriever = DeepSeekCandidateRetriever(
        "\n".join(f"{label}｜释义" for label in labels),
        "test-model",
        "http://example.test/v1",
        None,
        10.0,
    )

    recall_pool = retriever.build_recall_pool([
        "知识点@自然地理@地球的运动"
    ])

    assert recall_pool == labels[:3]


def test_request_stage_preserves_partial_response(monkeypatch):
    label = "知识点@自然地理@标签"
    retriever = DeepSeekCandidateRetriever(
        f"{label}｜释义",
        "test-model",
        "http://example.test/v1",
        None,
        10.0,
    )

    def fail_request(*args, **kwargs):
        raise ResponseGenerationError("输出达到限制", '{"partial":')

    monkeypatch.setattr(retriever, "request", fail_request)

    with pytest.raises(StageResponseError) as error_info:
        retriever.request_stage("分支定位", "提示词", "题目", 100, 10.0)

    assert error_info.value.stage == "分支定位"
    assert error_info.value.raw_response == '{"partial":'


def test_normalize_tagging_evidence_validates_unique_complete_items():
    evidence = normalize_tagging_evidence(
        {
            "tagging_evidence": [
                {
                    "evidence_id": "E1",
                    "question_part": "公共题干",
                    "content": "整道题以洞庭湖变化为研究内容",
                },
                {
                    "evidence_id": "E2",
                    "question_part": "小题2",
                    "content": "分析湖泊面积变化原因",
                },
            ]
        }
    )

    assert [item["evidence_id"] for item in evidence] == ["E1", "E2"]


def test_normalize_match_result_keeps_clear_possible_and_multiple_evidence():
    labels, matches, rejected = normalize_match_result(
        {
            "candidates": [
                {
                    "candidate_key": "C001",
                    "match_type": "clear",
                    "evidence_ids": ["E1", "E2"],
                },
                {
                    "candidate_key": "C002",
                    "match_type": "possible",
                    "evidence_ids": ["E1", "E2"],
                },
            ],
        },
        {"C001": "知识点@标签一", "C002": "知识点@标签二"},
        {"E1", "E2"},
    )

    assert labels == ["知识点@标签一", "知识点@标签二"]
    assert matches == [
        {
            "label": "知识点@标签一",
            "match_type": "clear",
            "evidence_ids": ["E1", "E2"],
        },
        {
            "label": "知识点@标签二",
            "match_type": "possible",
            "evidence_ids": ["E1", "E2"],
        },
    ]
    assert rejected == []


def test_normalize_match_result_rejects_more_than_twenty_candidates():
    allowed = {f"知识点@标签{i}" for i in range(21)}

    with pytest.raises(ValueError, match="最终候选超过20个"):
        normalize_match_result(
            {
                "candidates": [
                    {
                        "candidate_key": f"C{index:03d}",
                        "match_type": "clear",
                        "evidence_ids": ["E1"],
                    }
                    for index, label in enumerate(sorted(allowed), start=1)
                ]
            },
            {
                f"C{index:03d}": label
                for index, label in enumerate(sorted(allowed), start=1)
            },
            {"E1"},
        )


def test_normalize_match_result_rejects_only_unknown_keys():
    with pytest.raises(ValueError, match="任何有效候选临时序号"):
        normalize_match_result(
            {
                "candidates": [
                    {
                        "candidate_key": "C999",
                        "match_type": "possible",
                        "evidence_ids": ["E1"],
                    }
                ]
            },
            {"C001": "知识点@标签一"},
            {"E1"},
        )


def test_normalize_match_result_keeps_valid_and_records_unknown_keys():
    labels, matches, rejected = normalize_match_result(
        {
            "candidates": [
                {
                    "candidate_key": "C001",
                    "match_type": "clear",
                    "evidence_ids": ["E1"],
                },
                {"candidate_key": "C999"},
            ]
        },
        {"C001": "知识点@标签一"},
        {"E1"},
    )

    assert labels == ["知识点@标签一"]
    assert matches[0]["label"] == "知识点@标签一"
    assert rejected == ["C999"]


def test_parse_json_object_rejects_malformed_candidate_json():
    broken = (
        '{"candidates":[{"label":"知识点@自然地理@地球仪",'
        '"match_type":"clear","evidence_ids":["E2"]},'
        '{"label":"知识点@自然地理@经纬网"'
    )

    with pytest.raises(json.JSONDecodeError):
        parse_json_object(broken)


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


def test_trace_records_global_and_final_candidates_and_resume(monkeypatch, tmp_path):
    labels = [f"知识点@分类{i}@标签{i}" for i in range(3)]
    keys = [make_label_key(label) for label in labels]
    branch_keys = [make_branch_key(label) for label in labels]
    catalog = "\n".join(
        f"{label}｜释义{index}" for index, label in enumerate(labels)
    )
    unit = {"parent_id": "q1", "question_id": "q1", "stem": "题目", "sub_questions": []}
    output = tmp_path / "candidates.jsonl"
    trace_output = tmp_path / "trace.jsonl"

    def fake_request(
        self, system_prompt, question_text, max_tokens=2048, max_seconds=90.0
    ):
        if system_prompt == EVIDENCE_PROMPT:
            return json.dumps(
                {
                    "tagging_evidence": [
                        {
                            "evidence_id": "E1",
                            "question_part": "普通题",
                            "content": "题目提供的标注依据",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "定位高中地理题目可能涉及" in system_prompt:
            return json.dumps(
                {
                    "branches": [
                        {"branch_key": key, "evidence_ids": ["E1"]}
                        for key in branch_keys
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "candidates": [
                    {
                        "candidate_key": keys[0],
                        "match_type": "clear",
                        "evidence_ids": ["E1"],
                    }
                ],
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
    traces = [
        json.loads(line)
        for line in trace_output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(traces) == 1
    assert traces[0]["global_pre_candidate_labels"] == labels
    assert traces[0]["selected_branches"] == labels
    assert traces[0]["selected_branch_matches"] == [
        {"branch": label, "evidence_ids": ["E1"]}
        for label in labels
    ]
    assert traces[0]["rejected_branch_keys"] == []
    assert traces[0]["candidate_labels"] == [labels[0]]
    assert traces[0]["final_candidate_matches"] == [
        {
            "label": labels[0],
            "match_type": "clear",
            "evidence_ids": ["E1"],
        }
    ]
    assert traces[0]["uncovered_evidence"] == []
    assert traces[0]["rejected_final_candidate_keys"] == []


def test_limit_is_applied_before_resume_filter(monkeypatch, tmp_path):
    label = "知识点@自然地理@标签"
    catalog = f"{label}｜释义"
    units = [
        {"parent_id": f"q{i}", "question_id": f"q{i}", "stem": f"题目{i}"}
        for i in range(3)
    ]
    output = tmp_path / "candidates.jsonl"
    failed_once = False

    def fake_retrieve(self, unit):
        nonlocal failed_once
        if unit["question_id"] == "q0" and not failed_once:
            failed_once = True
            raise ValueError("模拟首次失败")
        return [label], {"candidate_labels": [label]}

    monkeypatch.setattr(DeepSeekCandidateRetriever, "retrieve", fake_retrieve)
    arguments = (
        units,
        catalog,
        {label},
        output,
        "test-model",
        "http://example.test/v1",
        None,
        1,
        1,
        10.0,
        2,
    )

    first = run_retrieval(*arguments)
    second = run_retrieval(*arguments)

    records = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert first["attempted"] == 2
    assert first["errors"] == 1
    assert second["attempted"] == 1
    assert {record["question_id"] for record in records} == {"q0", "q1"}


def test_run_retrieval_shares_work_across_multiple_endpoints(monkeypatch, tmp_path):
    import threading

    labels = [f"知识点@分类{i}@标签{i}" for i in range(3)]
    keys = [make_label_key(label) for label in labels]
    branch_keys = [make_branch_key(label) for label in labels]
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

    def fake_request(
        self, system_prompt, question_text, max_tokens=2048, max_seconds=90.0
    ):
        if system_prompt == EVIDENCE_PROMPT:
            barrier.wait(timeout=2)
            return json.dumps(
                {
                    "tagging_evidence": [
                        {
                            "evidence_id": "E1",
                            "question_part": "普通题",
                            "content": "题目提供的标注依据",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "定位高中地理题目可能涉及" in system_prompt:
            return json.dumps(
                {
                    "branches": [
                        {"branch_key": key, "evidence_ids": ["E1"]}
                        for key in branch_keys
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "candidates": [
                    {
                        "candidate_key": keys[0],
                        "match_type": "clear",
                        "evidence_ids": ["E1"],
                    }
                ],
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


def test_retrieve_records_uncovered_evidence_without_recovery(monkeypatch):
    labels = [f"知识点@分类{i}@标签{i}" for i in range(3)]
    keys = [make_label_key(label) for label in labels]
    branch_keys = [make_branch_key(label) for label in labels]
    catalog = "\n".join(
        f"{label}｜释义{i}" for i, label in enumerate(labels)
    )
    retriever = DeepSeekCandidateRetriever(
        catalog,
        "test-model",
        "http://example.test/v1",
        None,
        10.0,
    )

    request_count = 0

    def fake_request(
        self, system_prompt, question_text, max_tokens=2048, max_seconds=90.0
    ):
        nonlocal request_count
        request_count += 1
        if system_prompt == EVIDENCE_PROMPT:
            return json.dumps(
                {
                    "tagging_evidence": [
                        {
                            "evidence_id": "E1",
                            "question_part": "小题1",
                            "content": "依据一",
                        },
                        {
                            "evidence_id": "E2",
                            "question_part": "小题2",
                            "content": "依据二",
                        },
                    ]
                },
                ensure_ascii=False,
            )
        if "定位高中地理题目可能涉及" in system_prompt:
            return json.dumps(
                {
                    "branches": [
                        {"branch_key": key, "evidence_ids": ["E1"]}
                        for key in branch_keys
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "candidates": [
                    {
                        "candidate_key": keys[0],
                        "match_type": "clear",
                        "evidence_ids": ["E1"],
                    }
                ]
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(DeepSeekCandidateRetriever, "request", fake_request)

    candidates, trace = retriever.retrieve(
        {"question_id": "q1", "stem": "题目", "sub_questions": []}
    )

    assert candidates == [labels[0]]
    assert [item["evidence_id"] for item in trace["uncovered_evidence"]] == [
        "E2"
    ]
    assert request_count == 3


def test_retrieve_rejects_malformed_global_json(monkeypatch):
    label = "知识点@自然地理@标签"
    retriever = DeepSeekCandidateRetriever(
        f"{label}｜释义",
        "test-model",
        "http://example.test/v1",
        None,
        10.0,
    )

    def fake_request(
        self, system_prompt, question_text, max_tokens=2048, max_seconds=90.0
    ):
        if system_prompt == EVIDENCE_PROMPT:
            return json.dumps(
                {
                    "tagging_evidence": [
                        {
                            "evidence_id": "E1",
                            "question_part": "普通题",
                            "content": "依据",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        return '{"branches":[{"branch_key":"B001"'

    monkeypatch.setattr(DeepSeekCandidateRetriever, "request", fake_request)

    with pytest.raises(StageResponseError, match="分支定位阶段"):
        retriever.retrieve(
            {"question_id": "q1", "stem": "题目", "sub_questions": []}
        )


def test_run_retrieval_saves_stage_and_raw_failed_response(monkeypatch, tmp_path):
    label = "知识点@自然地理@标签"
    unit = {"question_id": "q1", "stem": "题目"}
    output = tmp_path / "candidates.jsonl"
    failure_trace = tmp_path / "failures.jsonl"
    broken = '{"branches":[{"branch_key":"B12345678"'

    def fake_request(
        self, system_prompt, question_text, max_tokens=2048, max_seconds=90.0
    ):
        if system_prompt == EVIDENCE_PROMPT:
            return json.dumps(
                {
                    "tagging_evidence": [
                        {
                            "evidence_id": "E1",
                            "question_part": "普通题",
                            "content": "依据",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        return broken

    monkeypatch.setattr(DeepSeekCandidateRetriever, "request", fake_request)

    summary = run_retrieval(
        [unit],
        f"{label}｜释义",
        {label},
        output,
        "test-model",
        "http://example.test/v1",
        None,
        1,
        1,
        10.0,
        None,
        failure_trace_output=failure_trace,
    )

    failure = json.loads(failure_trace.read_text(encoding="utf-8"))
    assert summary["errors"] == 1
    assert failure["status"] == "failed"
    assert failure["attempts"][0]["stage"] == "分支定位"
    assert failure["attempts"][0]["raw_response"] == broken


def test_candidate_recall_allows_fewer_than_twenty_labels(monkeypatch):
    labels = [f"知识点@分类@标签{i}" for i in range(21)]
    keys = [make_label_key(label) for label in labels]
    catalog = "\n".join(f"{label}｜释义{i}" for i, label in enumerate(labels))
    retriever = DeepSeekCandidateRetriever(
        catalog,
        "test-model",
        "http://example.test/v1",
        None,
        10.0,
    )
    tagging_evidence = [
        {"evidence_id": "E1", "question_part": "普通题", "content": "依据"}
    ]

    monkeypatch.setattr(
        DeepSeekCandidateRetriever,
        "request",
            lambda self, system_prompt, question_text, max_tokens=2048,
            max_seconds=90.0: json.dumps(
            {
                "candidates": [
                    {
                            "candidate_key": keys[i],
                        "match_type": "clear",
                        "evidence_ids": ["E1"],
                    }
                    for i, label in enumerate(labels[:3])
                ]
            },
            ensure_ascii=False,
        ),
    )
    selected, matches, rejected = retriever.recall_candidates(
        "题目", labels, tagging_evidence
    )

    assert selected == labels[:3]
    assert len(matches) == 3
    assert rejected == []
