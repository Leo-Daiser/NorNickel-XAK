"""Diagnose fallback, retrieval, graph and optional LLM/embeddings layers.

Default mode is intentionally safe: it disables live LLM and local embeddings
before importing the FastAPI app, so the script cannot hang on model loading or
external providers. Use --live-optional when the demo machine is prepared and
you want to test the real configured optional stack.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


DEMO_QUESTIONS = [
    "Какие параметры указаны для клапана DN50?",
    "Какие артикулы относятся к насосу NPK-200?",
    "Какие установки или оборудование использовали в опытах с ВТ6?",
    "Какие лаборатории работали со сталью 12Х18Н10Т?",
    "Где не хватает данных по коррозионной стойкости 7075-T6?",
]


def _configure_env(args: argparse.Namespace) -> None:
    if args.live_optional:
        return
    if args.live_embeddings:
        os.environ["ENABLE_LLM"] = "false"
        os.environ["DIRECT_QDRANT_PROJECTION"] = "false"
        os.environ.setdefault("RETRIEVAL_MODE", "hybrid")
        os.environ["ENABLE_LOCAL_EMBEDDINGS"] = "true"
        return
    if args.live_llm:
        os.environ["ENABLE_LOCAL_EMBEDDINGS"] = "false"
        os.environ["DIRECT_QDRANT_PROJECTION"] = "false"
        os.environ["RETRIEVAL_MODE"] = "bm25"
        os.environ["LLM_TIMEOUT_SECONDS"] = str(args.llm_timeout)
        return
    os.environ["ENABLE_LLM"] = "false"
    os.environ["ENABLE_LOCAL_EMBEDDINGS"] = "false"
    os.environ["DIRECT_QDRANT_PROJECTION"] = "false"
    os.environ["RETRIEVAL_MODE"] = "bm25"


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact(val) for key, val in value.items() if "api_key" not in key.lower()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _ingest_demo(client: Any, root: Path) -> dict[str, Any]:
    demo_dir = root / "demo_data"
    files = []
    handles = []
    try:
        for path in sorted(demo_dir.iterdir()):
            if path.is_file() and path.suffix.lower() not in {".md"}:
                handle = path.open("rb")
                handles.append(handle)
                files.append(("files", (path.name, handle, "application/octet-stream")))
        response = client.post("/ingest/documents", files=files)
        return {"status_code": response.status_code, "body": response.json()}
    finally:
        for handle in handles:
            handle.close()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--live-optional", action="store_true", help="use configured embeddings/LLM/Qdrant together; can be slow")
    parser.add_argument("--live-embeddings", action="store_true", help="test configured local embeddings while disabling LLM")
    parser.add_argument("--live-llm", action="store_true", help="test configured LLM while disabling local embeddings")
    parser.add_argument("--ingest-demo", action="store_true", help="ingest demo_data before asking diagnostic questions")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--question-limit", type=int, default=None, help="limit diagnostic /ask calls")
    parser.add_argument("--llm-timeout", type=int, default=8, help="per-request LLM timeout for --live-llm")
    args = parser.parse_args()

    _configure_env(args)

    from fastapi.testclient import TestClient

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from app.api import app

    client = TestClient(app)

    mode = "safe_fallback"
    if args.live_optional:
        mode = "live_optional"
    elif args.live_embeddings:
        mode = "live_embeddings"
    elif args.live_llm:
        mode = "live_llm"
    output: dict[str, Any] = {"mode": mode}

    health = client.get("/health")
    output["health_status_code"] = health.status_code
    output["health"] = _redact(health.json())

    if args.ingest_demo:
        output["ingest_demo"] = _ingest_demo(client, root)
        health_after = client.get("/health")
        output["health_after_ingest"] = _redact(health_after.json())

    checks = []
    question_limit = args.question_limit
    if question_limit is None and args.live_llm:
        question_limit = 1
    questions = DEMO_QUESTIONS[:question_limit] if question_limit else DEMO_QUESTIONS
    for question in questions:
        response = client.post("/ask", params={"question": question, "top_k": args.top_k})
        body = response.json()
        checks.append(
            {
                "question": question,
                "status_code": response.status_code,
                "answer_mode": body.get("answer_mode"),
                "answer_preview": str(body.get("answer", ""))[:350],
                "facts": len(body.get("facts") or []),
                "sources": len(body.get("sources") or []),
                "gaps": len(body.get("gaps") or []),
                "graph_nodes": len(((body.get("subgraph") or {}).get("nodes") or [])),
                "graph_edges": len(((body.get("subgraph") or {}).get("edges") or [])),
                "retrieval": _redact(body.get("retrieval") or {}),
                "llm": _redact(body.get("llm") or {}),
            }
        )
    output["ask_checks"] = checks

    print(json.dumps(output, ensure_ascii=False, indent=2))
    failures = [item for item in checks if item["status_code"] != 200 or not item["answer_preview"]]
    return 1 if health.status_code != 200 or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
