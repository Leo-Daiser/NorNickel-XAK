from __future__ import annotations

from app.extraction.models import AcceptedFact, EvidenceSpan, ExtractionSource
from app.models.schemas import Chunk
from app.retrieval.typed_fact_retriever import TypedFactQuery, TypedFactRetriever, filter_source_grounded_chunks


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
    assert result.diagnostics["missing_required_anchors"] == []
    assert result.diagnostics["coverage_score"] is None
    assert result.diagnostics["evidence_coverage_score"] > 0


def test_source_grounded_filter_excludes_smoke_docs_and_low_relevance() -> None:
    query = TypedFactQuery(
        question="Какие способы охлаждения применяются для глубоких рудников?",
        target_fact_types=["TechnologySolutionFact"],
        answer_mode="technology_solution_search",
    )
    chunks = [
        Chunk(
            chunk_id="smoke",
            doc_id="doc_smoke",
            text="Тестовый технический документ: охлаждение глубоких рудников.",
            page_start=1,
            page_end=1,
            section_path="/",
            metadata={"source_name": "kg ui smoke deep mine cooling.txt", "filename": "kg_ui_smoke.txt"},
        ),
        Chunk(
            chunk_id="real",
            doc_id="doc_real",
            text="На глубоких рудниках применяются холодильные установки, вентиляция и охлаждение воздуха.",
            page_start=2,
            page_end=2,
            section_path="/",
            metadata={"source_name": "Глубокие рудники_2017.pdf", "filename": "Глубокие рудники_2017.pdf"},
        ),
        Chunk(
            chunk_id="irrelevant",
            doc_id="doc_irrelevant",
            text="Охлаждение раствора в технологическом баке медного производства.",
            page_start=3,
            page_end=3,
            section_path="/",
            metadata={"source_name": "tnk_solution_cooling.txt"},
        ),
    ]

    filtered, diagnostics = filter_source_grounded_chunks(chunks, query, top_k=5)

    assert [chunk.chunk_id for chunk in filtered] == ["real"]
    assert diagnostics["excluded_test_chunks_count"] == 1
    assert diagnostics["evidence_dropped_low_relevance_count"] == 1
    assert diagnostics["retrieval_filtered_for_production"] is True
    assert diagnostics["evidence_coverage_score"] > 0
