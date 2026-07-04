from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug real /ask retrieval diagnostics for one query.")
    parser.add_argument("question", help="Natural-language query to send to /ask.")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL.")
    parser.add_argument("--top-k", type=int, default=8, help="Ask top_k.")
    args = parser.parse_args()

    url = args.api_url.rstrip("/") + "/ask"
    try:
        response = requests.post(
            url,
            json={"question": args.question, "top_k": args.top_k},
            timeout=180,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"API target is unavailable or failed: {exc}", file=sys.stderr)
        return 2

    payload = response.json()
    diagnostics = payload.get("diagnostics") or {}
    retrieval = payload.get("retrieval") or {}
    report: dict[str, Any] = {
        "question": args.question,
        "status": payload.get("status"),
        "selected_answer_mode": diagnostics.get("selected_answer_mode") or payload.get("answer_mode"),
        "retrieval_status": diagnostics.get("retrieval_status"),
        "partial_reason": diagnostics.get("partial_reason"),
        "normalized_query_terms": diagnostics.get("normalized_query_terms") or {},
        "effective_retrieval_mode": diagnostics.get("effective_retrieval_mode") or retrieval.get("effective_retrieval_mode"),
        "degraded_reason": diagnostics.get("degraded_reason") or retrieval.get("degraded_reason"),
        "embedding_status": diagnostics.get("embedding_status") or retrieval.get("embedding_status") or {},
        "bm25_top_k_count": int(diagnostics.get("chunks_found_bm25") or retrieval.get("chunks_found_bm25") or 0),
        "dense_top_k_count": int(diagnostics.get("chunks_found_dense") or retrieval.get("chunks_found_dense") or 0),
        "fused_top_k_count": int(diagnostics.get("chunks_after_fusion") or retrieval.get("chunks_after_fusion") or 0),
        "typed_facts_found": int(diagnostics.get("typed_facts_found") or 0),
        "top_fused_chunks": _trim_chunks(diagnostics.get("top_fused_chunks") or retrieval.get("top_fused_chunks") or []),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _trim_chunks(chunks: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not isinstance(chunks, list):
        return result
    for item in chunks[:8]:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "source": item.get("source_name"),
                "page": item.get("page"),
                "section_path": item.get("section_path"),
                "score": item.get("score"),
            }
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
