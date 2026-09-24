import json

from bio_geo_tagging.run_tagging_pipeline import run_pipeline


def test_pipeline_connects_stages_writes_final_labels_and_resumes(tmp_path):
    input_path = tmp_path / "questions.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    index_dir = tmp_path / "index"
    region_index_dir = tmp_path / "region-index"
    run_dir = tmp_path / "run"
    input_path.write_text('{"question_id":"q1","stem":"题干"}\n', encoding="utf-8")
    labels_path.write_text('{"label_path":"知识点@甲"}\n', encoding="utf-8")
    index_dir.mkdir()
    region_index_dir.mkdir()
    (index_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (region_index_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    calls = {"units": 0, "retrieval": 0, "adjudication": 0, "evaluation": 0}

    def unit_runner(input_file, output_file, log_file):
        calls["units"] += 1
        with open(output_file, "w", encoding="utf-8") as handle:
            handle.write('{"question_id":"q1","input_role":"root"}\n')
        with open(log_file, "w", encoding="utf-8") as handle:
            handle.write("ok\n")

    def retrieval_runner(**kwargs):
        calls["retrieval"] += 1
        kwargs["output_path"].write_text(
            '{"question_id":"q1","knw_labels":["知识点@旧"],'
            '"combined_candidates":[{"label_path":"知识点@甲"}]}\n',
            encoding="utf-8",
        )
        kwargs["summary_path"].write_text("{}\n", encoding="utf-8")
        return {}

    def adjudication_runner(*args, **kwargs):
        calls["adjudication"] += 1
        return {"input": 1, "success": 1, "error": 0}

    def evaluation_runner(
        run_path,
        candidates_path,
        taxonomy_path,
        output_path,
        details_path,
        stability_run_dir,
        final_labels_path,
    ):
        calls["evaluation"] += 1
        output_path.write_text('{"report_kind":"automatic_adjudication_diagnostics"}\n', encoding="utf-8")
        details_path.write_text("{}\n", encoding="utf-8")
        final_labels_path.write_text(
            '{"root_question_id":"q1","legacy_labels":["知识点@旧"],'
            '"ds_added_labels":["知识点@甲"],'
            '"final_labels":["知识点@旧","知识点@甲"]}\n',
            encoding="utf-8",
        )
        return {"report_kind": "automatic_adjudication_diagnostics"}

    arguments = {
        "input_path": input_path,
        "labels_path": labels_path,
        "index_dir": index_dir,
        "region_index_dir": region_index_dir,
        "run_dir": run_dir,
        "client": object(),
        "model": "test-model",
        "unit_runner": unit_runner,
        "retrieval_runner": retrieval_runner,
        "adjudication_runner": adjudication_runner,
        "evaluation_runner": evaluation_runner,
    }
    result = run_pipeline(**arguments)

    assert calls == {"units": 1, "retrieval": 1, "adjudication": 1, "evaluation": 1}
    assert result["final_labels"] == str(run_dir / "final_labels.jsonl")
    final_row = json.loads((run_dir / "final_labels.jsonl").read_text(encoding="utf-8"))
    assert final_row["final_labels"] == ["知识点@旧", "知识点@甲"]

    run_pipeline(**arguments)
    assert calls == {"units": 1, "retrieval": 1, "adjudication": 2, "evaluation": 2}
    log_text = (run_dir / "pipeline.log").read_text(encoding="utf-8")
    assert '"stage": "tagging_units", "status": "skipped"' in log_text
    assert '"stage": "candidate_retrieval", "status": "skipped"' in log_text


def test_pipeline_continues_diagnostics_when_adjudication_is_incomplete(tmp_path):
    input_path = tmp_path / "questions.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    index_dir = tmp_path / "index"
    region_index_dir = tmp_path / "region-index"
    run_dir = tmp_path / "run"
    input_path.write_text("{}\n", encoding="utf-8")
    labels_path.write_text("{}\n", encoding="utf-8")
    index_dir.mkdir()
    region_index_dir.mkdir()
    (index_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (region_index_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    evaluation_called = False

    def units(input_file, output_file, log_file):
        with open(output_file, "w", encoding="utf-8") as handle:
            handle.write("{}\n")

    def retrieval(**kwargs):
        kwargs["output_path"].write_text("{}\n", encoding="utf-8")
        kwargs["summary_path"].write_text("{}\n", encoding="utf-8")
        return {}

    def incomplete(*args, **kwargs):
        run_path = args[3]
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "evidence.jsonl").write_text(
            '{"unit_key":"q2|q2|root","root_question_id":"q2",'
            '"question_id":"q2","input_role":"root",'
            '"error":"DSRequestError: timeout","endpoint":"test"}\n',
            encoding="utf-8",
        )
        return {"input": 2, "success": 1, "error": 1}

    def evaluation(*args, **kwargs):
        nonlocal evaluation_called
        evaluation_called = True
        output_path = args[3]
        details_path = args[4]
        final_labels_path = args[6]
        output_path.write_text("{}\n", encoding="utf-8")
        details_path.write_text("", encoding="utf-8")
        final_labels_path.write_text("", encoding="utf-8")
        return {"completed_question_predictions": 1}

    result = run_pipeline(
        input_path=input_path,
        labels_path=labels_path,
        index_dir=index_dir,
        region_index_dir=region_index_dir,
        run_dir=run_dir,
        client=object(),
        model="test-model",
        unit_runner=units,
        retrieval_runner=retrieval,
        adjudication_runner=incomplete,
        evaluation_runner=evaluation,
    )

    assert evaluation_called is True
    assert result["status"] == "completed_with_errors"
    assert result["unresolved_failure_units"] == 1
    failures = [
        json.loads(line)
        for line in (run_dir / "pipeline-failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert failures == [
        {
            "attempts": None,
            "created_at": None,
            "endpoint": "test",
            "error": "DSRequestError: timeout",
            "input_role": "root",
            "question_id": "q2",
            "retry_errors": [],
            "root_question_id": "q2",
            "stage": "candidate_adjudication",
            "unit_key": "q2|q2|root",
        }
    ]
    log_text = (run_dir / "pipeline.log").read_text(encoding="utf-8")
    assert '"status": "completed_with_errors"' in log_text
