from __future__ import annotations

from tests.strict_qa_helpers import seeded_client


def test_preset_answers_are_visibly_different(tmp_path) -> None:
    client = seeded_client(tmp_path)
    question = "Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?"
    answers = {}
    for preset_id in ["expert_max", "strict_audit", "offline_reliable"]:
        response = client.post("/ask", json={"question": question, "top_k": 8, "preset_id": preset_id})
        assert response.status_code == 200, response.text
        answers[preset_id] = response.json()["answer"]
    assert len(set(answers.values())) == 3
    assert "Ограничения" in answers["expert_max"]
    assert "Вывод" in answers["expert_max"]
    assert "Статус проверки" in answers["strict_audit"]
    assert "Проверенная цепочка" in answers["strict_audit"]
    assert "офлайн-режиме" in answers["offline_reliable"].lower()
