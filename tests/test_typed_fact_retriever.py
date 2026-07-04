from __future__ import annotations

from app.extraction.models import AcceptedFact, EvidenceSpan, ExtractionSource
from app.models.schemas import Chunk
from app.retrieval.typed_fact_retriever import TypedFactQuery, TypedFactRetriever


class _Repo:
    def find_accepted_facts(self, fact_types=None, limit: int = 100):
        source = ExtractionSource(document_id="doc1", chunk_id="chunk1", source_name="leaching.pdf", page=1)
        return [
            AcceptedFact(
                candidate_id="fact1",
                fact_type="TechnologySolutionFact",
                normalized_fact={
                    "subject": {"material": "никель", "process": "выщелачивание"},
                    "object": {"technology": "выщелачивание"},
                },
                evidence=[EvidenceSpan(source=source, quote="Технология выщелачивания никеля описана в таблице.")],
                score=0.9,
            )
        ]


class _Retrieval:
    def query(self, question: str, top_k: int = 12):
        return [
            Chunk(
                chunk_id="chunk2",
                doc_id="doc2",
                text="Для глубоких рудников описаны холодильные установки и вентиляция.",
                page_start=3,
                page_end=3,
                section_path="/",
                metadata={"filename": "cooling.pdf", "source_name": "cooling.pdf"},
            )
        ]

    def stats(self):
        return {
            "chunks_found_bm25": 1,
            "chunks_found_dense": 1,
            "chunks_after_fusion": 1,
            "effective_retrieval_mode": "hybrid",
            "embedding_status": {"vectors_cached": 1, "vectors_missing": 0},
        }


def test_typed_retriever_does_not_treat_fact_type_only_as_exact_match() -> None:
    query = TypedFactQuery(
        question="Какие способы охлаждения применяются для глубоких рудников?",
        target_fact_types=["TechnologySolutionFact"],
        answer_mode="technology_solution_search",
    )

    result = TypedFactRetriever(_Repo(), retrieval_engine=_Retrieval()).search(query, top_k=3)

    assert result.retrieval_status == "chunks_only_no_structured_facts"
    assert result.accepted_facts == []
    assert len(result.fallback_chunks) == 1
    assert result.diagnostics["missing_required_anchors"] == ["query_anchor"]
