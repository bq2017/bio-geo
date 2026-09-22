from bio_geo_tagging.bge_length_diagnosis import diagnose


def candidate_result(question_id, bge_labels, final_labels):
    return {
        "question_id": question_id,
        "bge_candidates": [
            {"label_path": label} for label in bge_labels
        ],
        "nonregional_final_candidates": [
            {"label_path": label} for label in final_labels
        ],
    }


def test_diagnosis_separates_truncated_and_short_question_recall():
    questions = [
        {
            "question_id": "short",
            "parent_id": "short",
            "stem": "短题",
            "knw_labels": ["知识点@自然地理@短题标签"],
            "sub_questions": [],
        },
        {
            "question_id": "long",
            "parent_id": "long",
            "stem": "很长的公共材料",
            "knw_labels": [],
            "sub_questions": [
                {
                    "stem": "很长的小题题干",
                    "knw_labels": ["知识点@自然地理@长题标签"],
                }
            ],
        },
    ]
    candidates = {
        "short": candidate_result(
            "short",
            ["知识点@自然地理@短题标签"],
            ["知识点@自然地理@短题标签"],
        ),
        "long": candidate_result(
            "long",
            [],
            ["知识点@自然地理@长题标签"],
        ),
    }

    details, summary = diagnose(
        questions,
        candidates,
        token_counter=lambda text: len(text),
        max_seq_length=10,
        instruction="",
        bge_top_k=23,
    )

    assert [item["question_id"] for item in details] == ["long"]
    assert details[0]["bge_missing_labels"] == ["知识点@自然地理@长题标签"]
    assert summary["length"]["truncated_questions"] == 1
    assert summary["metrics"]["within_limit"]["bge_label_recall"] == 1.0
    assert summary["metrics"]["truncated"]["bge_label_recall"] == 0.0
    assert summary["metrics"]["truncated"]["final_label_recall"] == 1.0
