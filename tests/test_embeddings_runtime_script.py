from __future__ import annotations

import json

from scripts.check_embeddings_runtime import check_embeddings_runtime


class _FakeSentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_check_embeddings_runtime_reports_missing_dependency(monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    monkeypatch.setenv("ENABLE_LOCAL_EMBEDDINGS", "true")

    report = check_embeddings_runtime(sentence_transformer_cls=None)

    assert report["retrieval_mode"] == "hybrid"
    assert report["enable_local_embeddings"] is True
    assert report["sentence_transformers_import_ok"] is False
    assert report["model_load_ok"] is False
    assert "dependency missing" in report["error"]


def test_check_embeddings_runtime_loads_mocked_model(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    report = check_embeddings_runtime(sentence_transformer_cls=_FakeSentenceTransformer)

    assert report["sentence_transformers_import_ok"] is True
    assert report["model_load_ok"] is True
    assert report["short_embedding_ok"] is True
    assert report["vector_dimension"] == 3


def test_check_embeddings_runtime_does_not_print_secrets(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "secret-mistral")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-openrouter")

    report = check_embeddings_runtime(sentence_transformer_cls=None)
    rendered = json.dumps(report, ensure_ascii=False)

    assert "secret-mistral" not in rendered
    assert "secret-openrouter" not in rendered


def test_check_embeddings_runtime_can_skip_model_load(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDINGS_SKIP_MODEL_LOAD", "true")

    report = check_embeddings_runtime(sentence_transformer_cls=_FakeSentenceTransformer)

    assert report["sentence_transformers_import_ok"] is True
    assert report["model_load_skipped"] is True
    assert report["model_load_ok"] is False
