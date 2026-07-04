from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.schemas import Chunk  # noqa: E402
from app.retrieval.retrieval import RetrievalEngine  # noqa: E402
from app.config import settings  # noqa: E402


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"semantic_{chunk_id}",
        workspace_uid="semantic-eval",
        text=text,
        page_start=1,
        page_end=1,
        section_path="semantic-fixture",
        embedding_version="synthetic",
    )


CHUNKS = [
    _chunk("strength_aging_en", "7075-T6 tensile strength after aging treatment reached 520 MPa."),
    _chunk("corrosion_heat_en", "Corrosion resistance after heat treatment was not measured for 7075-T6."),
    _chunk("vt6_annealing_mixed", "Ti-6Al-4V alloy ВТ6 after annealing showed strength near 1120 MPa."),
    _chunk("strength_ru_exact", "Прочность после старения для контрольного образца указана в таблице."),
    _chunk("noise", "Pump valve DN50 pressure rating and catalog notes."),
]


CASES = [
    {
        "query": "прочность после старения",
        "expected": {"strength_aging_en"},
        "dense": ["strength_aging_en"],
    },
    {
        "query": "устойчивость к коррозии после обработки",
        "expected": {"corrosion_heat_en"},
        "dense": ["corrosion_heat_en"],
    },
    {
        "query": "титановый сплав после отжига",
        "expected": {"vt6_annealing_mixed"},
        "dense": ["vt6_annealing_mixed"],
    },
]

EXACT_RUSSIAN_CASE = {
    "query": "Прочность после старения",
    "expected": {"strength_ru_exact"},
    "dense": ["strength_aging_en"],
}


def _configure(mode: str) -> None:
    settings.retrieval_mode = mode
    settings.enable_local_embeddings = mode == "hybrid"
    settings.eager_local_embeddings = False
    settings.direct_qdrant_projection = False
    settings.retrieval_query_expansion = False


def _engine(mode: str, dense_map: dict[str, list[str]] | None = None) -> RetrievalEngine:
    _configure(mode)
    engine = RetrievalEngine()
    engine.index_chunks(CHUNKS)
    if dense_map is not None:
        def _dense(query: str, top_k: int = 20):
            result = [
                (chunk_id, 1.0 / (rank + 1))
                for rank, chunk_id in enumerate(dense_map.get(query, [])[:top_k])
            ]
            engine._last_dense_candidates = len(result)
            return result

        engine.dense_retrieve = _dense  # type: ignore[method-assign]
    return engine


def _run_case(engine: RetrievalEngine, case: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    ids = [chunk.chunk_id for chunk in engine.query(str(case["query"]), top_k=top_k)]
    expected = set(case["expected"])
    found = expected & set(ids)
    return {
        "query": case["query"],
        "ids": ids,
        "expected": sorted(expected),
        "found": sorted(found),
        "found_count": len(found),
        "passed": bool(found),
    }


def main() -> int:
    dense_map = {str(case["query"]): list(case["dense"]) for case in CASES + [EXACT_RUSSIAN_CASE]}
    bm25_engine = _engine("bm25")
    hybrid_engine = _engine("hybrid", dense_map=dense_map)

    bm25_rows = [_run_case(bm25_engine, case) for case in CASES]
    hybrid_rows = [_run_case(hybrid_engine, case) for case in CASES]
    exact_bm25 = _run_case(bm25_engine, EXACT_RUSSIAN_CASE)
    exact_hybrid = _run_case(hybrid_engine, EXACT_RUSSIAN_CASE)

    bm25_found = sum(row["found_count"] for row in bm25_rows)
    hybrid_found = sum(row["found_count"] for row in hybrid_rows)
    no_regression_exact = exact_hybrid["found_count"] >= exact_bm25["found_count"] and exact_bm25["found_count"] > 0
    metrics = {
        "bm25_expected_found": bm25_found,
        "hybrid_expected_found": hybrid_found,
        "hybrid_improvement": hybrid_found - bm25_found,
        "no_regression_exact_russian": no_regression_exact,
        "hybrid_dense_enabled": bool(hybrid_engine.stats().get("last_dense_candidates")),
    }
    passed = metrics["hybrid_improvement"] > 0 and no_regression_exact

    print("Semantic retrieval evaluation:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("PASS" if passed else "FAIL")
    print(
        "SUMMARY",
        json.dumps(
            {
                "metrics": metrics,
                "bm25_rows": bm25_rows,
                "hybrid_rows": hybrid_rows,
                "exact_bm25": exact_bm25,
                "exact_hybrid": exact_hybrid,
            },
            ensure_ascii=False,
        ),
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
