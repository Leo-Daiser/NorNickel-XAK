from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from app.config import settings  # noqa: E402
from app.retrieval.retrieval import PersistentVectorCache  # noqa: E402
from app.storage.catalog import SQLiteCatalog  # noqa: E402


_AUTO = object()


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def check_embeddings_runtime(sentence_transformer_cls: Any = _AUTO) -> dict[str, Any]:
    """Return a secret-free local embeddings runtime report.

    The function is intentionally small and import-injected for tests. By
    default it tries to load and encode with the configured model. Set
    EMBEDDINGS_SKIP_MODEL_LOAD=true to verify only that the dependency imports.
    """

    model_name = os.getenv("EMBEDDING_MODEL") or getattr(
        settings,
        "embedding_model",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    model_path = str(os.getenv("EMBEDDING_MODEL_PATH") or getattr(settings, "embedding_model_path", "") or "").strip()
    model_path_exists = bool(model_path and Path(model_path).exists())
    model_cache_id = model_path or model_name
    retrieval_mode = os.getenv("RETRIEVAL_MODE") or getattr(settings, "retrieval_mode", "bm25")
    local_embeddings_enabled = _bool_env(
        "ENABLE_LOCAL_EMBEDDINGS",
        bool(getattr(settings, "enable_local_embeddings", False)),
    )
    offline_mode = _bool_env("TRANSFORMERS_OFFLINE", False) or _bool_env("HF_HUB_OFFLINE", False)
    cache_path = Path(getattr(settings, "data_dir", "data")) / "cache" / "embedding_vectors.sqlite3"
    vector_stats = _vector_cache_stats(cache_path=cache_path, model_cache_id=model_cache_id, dimensions=None)
    report: dict[str, Any] = {
        "retrieval_mode": retrieval_mode,
        "enable_local_embeddings": local_embeddings_enabled,
        "embedding_model": model_name,
        "embedding_model_id": model_name,
        "embedding_model_path": model_path,
        "offline_mode": offline_mode,
        "model_path_exists": model_path_exists,
        "sentence_transformers_import_ok": False,
        "dependency_available": False,
        "model_load_ok": False,
        "model_load_error": "",
        "short_embedding_ok": False,
        "vector_dimension": None,
        "dimensions": None,
        "test_vector_shape": [],
        "cache_path": str(cache_path),
        "vectors_total": vector_stats["vectors_total"],
        "vectors_cached": vector_stats["vectors_cached"],
        "vectors_missing": vector_stats["vectors_missing"],
        "effective_retrieval_mode": _effective_retrieval_mode(
            retrieval_mode,
            local_embeddings_enabled=local_embeddings_enabled,
            dependency_available=False,
            model_load_ok=False,
        ),
        "degraded_reason": _degraded_reason(
            retrieval_mode,
            local_embeddings_enabled=local_embeddings_enabled,
            dependency_available=False,
            model_load_ok=False,
        ),
        "model_load_skipped": False,
        "action": "",
        "error": "",
    }

    auto_runtime = sentence_transformer_cls is _AUTO
    if auto_runtime:
        try:
            with warnings.catch_warnings(), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                warnings.simplefilter("ignore")
                logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
                logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
                logging.getLogger("transformers").setLevel(logging.ERROR)
                from sentence_transformers import SentenceTransformer as sentence_transformer_cls  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            report["error"] = _safe_error(exc)
            report["degraded_reason"] = _degraded_reason(
                retrieval_mode,
                local_embeddings_enabled=local_embeddings_enabled,
                dependency_available=False,
                model_load_ok=False,
                error=report["error"],
            )
            return report
    elif sentence_transformer_cls is None:
        report["error"] = "sentence-transformers dependency missing"
        return report

    report["sentence_transformers_import_ok"] = True
    report["dependency_available"] = True
    report["effective_retrieval_mode"] = _effective_retrieval_mode(
        retrieval_mode,
        local_embeddings_enabled=local_embeddings_enabled,
        dependency_available=True,
        model_load_ok=False,
    )
    report["degraded_reason"] = _degraded_reason(
        retrieval_mode,
        local_embeddings_enabled=local_embeddings_enabled,
        dependency_available=True,
        model_load_ok=False,
    )
    if _bool_env("EMBEDDINGS_SKIP_MODEL_LOAD", False):
        report["model_load_skipped"] = True
        return report
    if auto_runtime and model_path and not model_path_exists:
        message = f"model path does not exist: {model_path}. Mount local model to /models/... or allow build-time download."
        report["model_load_error"] = message
        report["error"] = message
        report["action"] = "Mount local model to /models/... or allow build-time download."
        report["degraded_reason"] = _degraded_reason(
            retrieval_mode,
            local_embeddings_enabled=local_embeddings_enabled,
            dependency_available=True,
            model_load_ok=False,
            error=message,
        )
        return report

    if auto_runtime:
        timeout_seconds = _int_env("EMBEDDINGS_MODEL_LOAD_TIMEOUT_SECONDS", 180)
        model_report = _load_model_with_timeout(
            model_name,
            model_path=model_path,
            timeout_seconds=timeout_seconds,
            local_files_only=bool(model_path) or offline_mode or not _bool_env("EMBEDDINGS_ALLOW_DOWNLOAD", False),
        )
        report.update(model_report)
        ok = bool(report["model_load_ok"] and report["short_embedding_ok"])
        report["model_load_error"] = "" if ok else str(report.get("error") or "")
        if not ok and not report.get("action"):
            report["action"] = "Mount local model to /models/... or allow build-time download."
        vector_stats = _vector_cache_stats(
            cache_path=cache_path,
            model_cache_id=model_cache_id,
            dimensions=report.get("dimensions"),
        )
        report.update(vector_stats)
        report["effective_retrieval_mode"] = _effective_retrieval_mode(
            retrieval_mode,
            local_embeddings_enabled=local_embeddings_enabled,
            dependency_available=True,
            model_load_ok=ok,
        )
        report["degraded_reason"] = _degraded_reason(
            retrieval_mode,
            local_embeddings_enabled=local_embeddings_enabled,
            dependency_available=True,
            model_load_ok=ok,
            error=report.get("error", ""),
        )
        return report

    try:
        with warnings.catch_warnings(), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            warnings.simplefilter("ignore")
            model = sentence_transformer_cls(model_path or model_name)
            report["model_load_ok"] = True
            dimension_getter = getattr(
                model,
                "get_embedding_dimension",
                getattr(model, "get_sentence_embedding_dimension", lambda: None),
            )
            dimension = dimension_getter()
            encoded = model.encode(["test"], normalize_embeddings=True, show_progress_bar=False)
        vector = encoded[0] if len(encoded) else []
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        report["short_embedding_ok"] = bool(vector)
        report["vector_dimension"] = int(dimension or len(vector or [])) if (dimension or vector) else None
        report["dimensions"] = report["vector_dimension"]
        report["test_vector_shape"] = [1, report["vector_dimension"]] if report["vector_dimension"] else []
        report.update(
            _vector_cache_stats(
                cache_path=cache_path,
                model_cache_id=model_cache_id,
                dimensions=report.get("dimensions"),
            )
        )
        report["effective_retrieval_mode"] = _effective_retrieval_mode(
            retrieval_mode,
            local_embeddings_enabled=local_embeddings_enabled,
            dependency_available=True,
            model_load_ok=bool(report["model_load_ok"] and report["short_embedding_ok"]),
        )
        report["degraded_reason"] = _degraded_reason(
            retrieval_mode,
            local_embeddings_enabled=local_embeddings_enabled,
            dependency_available=True,
            model_load_ok=bool(report["model_load_ok"] and report["short_embedding_ok"]),
        )
    except Exception as exc:
        report["error"] = _safe_error(exc)
        report["model_load_error"] = report["error"]
        report["degraded_reason"] = _degraded_reason(
            retrieval_mode,
            local_embeddings_enabled=local_embeddings_enabled,
            dependency_available=True,
            model_load_ok=False,
            error=report["error"],
        )
    return report


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


def _vector_cache_stats(*, cache_path: Path, model_cache_id: str, dimensions: Any) -> dict[str, int]:
    try:
        catalog = SQLiteCatalog(getattr(settings, "catalog_db_path", "data/catalog.sqlite3"))
        chunks = catalog.list_chunks(active_only=True)
    except Exception:
        chunks = []
    total = len(chunks)
    if not dimensions:
        return {"vectors_total": total, "vectors_cached": 0, "vectors_missing": total}
    try:
        _, stats = PersistentVectorCache(cache_path).load_vectors(
            chunks,
            model_id=model_cache_id,
            dimensions=int(dimensions),
        )
        return {
            "vectors_total": int(stats.get("vectors_total", total)),
            "vectors_cached": int(stats.get("vectors_cached", 0)),
            "vectors_missing": int(stats.get("vectors_missing", total)),
        }
    except Exception:
        return {"vectors_total": total, "vectors_cached": 0, "vectors_missing": total}


def _load_model_with_timeout(
    model_name: str,
    *,
    model_path: str = "",
    timeout_seconds: int,
    local_files_only: bool,
) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue(maxsize=1)
    process = mp.Process(target=_model_worker, args=(model_name, model_path, local_files_only, queue), daemon=True)
    process.start()
    process.join(max(1, timeout_seconds))
    if process.is_alive():
        process.terminate()
        process.join(10)
        return {
            "model_load_ok": False,
            "short_embedding_ok": False,
            "vector_dimension": None,
            "dimensions": None,
            "test_vector_shape": [],
            "error": f"model load timed out after {timeout_seconds}s",
        }
    if not queue.empty():
        return dict(queue.get())
    return {
        "model_load_ok": False,
        "short_embedding_ok": False,
        "vector_dimension": None,
        "dimensions": None,
        "test_vector_shape": [],
        "error": f"model load worker exited with code {process.exitcode}",
    }


def _model_worker(model_name: str, model_path: str, local_files_only: bool, queue: mp.Queue) -> None:
    try:
        with warnings.catch_warnings(), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            warnings.simplefilter("ignore")
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            from sentence_transformers import SentenceTransformer

            target = model_path or model_name
            kwargs = {}
            if local_files_only:
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                kwargs["local_files_only"] = True
            try:
                model = SentenceTransformer(target, **kwargs)
            except TypeError:
                model = SentenceTransformer(target)
            dimension_getter = getattr(
                model,
                "get_embedding_dimension",
                getattr(model, "get_sentence_embedding_dimension", lambda: None),
            )
            dimension = dimension_getter()
            encoded = model.encode(["test"], normalize_embeddings=True, show_progress_bar=False)
        vector = encoded[0] if len(encoded) else []
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        vector_dimension = int(dimension or len(vector or [])) if (dimension or vector) else None
        queue.put(
            {
                "model_load_ok": True,
                "short_embedding_ok": bool(vector),
                "vector_dimension": vector_dimension,
                "dimensions": vector_dimension,
                "test_vector_shape": [1, vector_dimension] if vector_dimension else [],
                "error": "",
            }
        )
    except BaseException as exc:  # pragma: no cover - depends on optional runtime/model cache
        queue.put(
            {
                "model_load_ok": False,
                "short_embedding_ok": False,
                "vector_dimension": None,
                "dimensions": None,
                "test_vector_shape": [],
                "error": _safe_error(exc),
            }
        )


def _effective_retrieval_mode(
    retrieval_mode: str,
    *,
    local_embeddings_enabled: bool,
    dependency_available: bool,
    model_load_ok: bool,
) -> str:
    mode = str(retrieval_mode or "bm25").strip().lower()
    if mode != "hybrid":
        return mode
    if local_embeddings_enabled and dependency_available and model_load_ok:
        return "hybrid"
    return "hybrid_degraded_to_bm25"


def _degraded_reason(
    retrieval_mode: str,
    *,
    local_embeddings_enabled: bool,
    dependency_available: bool,
    model_load_ok: bool,
    error: str = "",
) -> str:
    mode = str(retrieval_mode or "bm25").strip().lower()
    if mode != "hybrid":
        return ""
    if not local_embeddings_enabled:
        return "disabled by config"
    if not dependency_available:
        return error or "dependency missing"
    if not model_load_ok:
        return error or "model not loaded"
    return ""


def main() -> int:
    report = check_embeddings_runtime()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["sentence_transformers_import_ok"]:
        return 2
    if report["model_load_skipped"]:
        return 0
    if not report["model_load_ok"] or not report["short_embedding_ok"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
