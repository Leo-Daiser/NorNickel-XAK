from __future__ import annotations

from app.extraction.extraction import EntityRelationExtractor
from app.models.schemas import Chunk


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        workspace_uid="test",
        text=text,
        page_start=1,
        page_end=1,
        section_path="/",
    )


def test_extraction_material_process_property_measurement() -> None:
    text = (
        "Experiment: EXP-VT6-AN. Material: сплав ВТ6. Process: отжиг at 750 C for 2 h. "
        "Property: прочность. Result: прочность decreased to 980 MPa. "
        "Equipment: Вакуумная печь SNOL-75. Laboratory: Лаборатория легких сплавов. "
        "Conclusion: отжиг ВТ6 снижает прочность."
    )

    result = EntityRelationExtractor().extract_from_chunk(_chunk(text))
    entities = {(entity.entity_type, entity.canonical_name) for entity in result.entities}
    predicates = {relation.predicate for relation in result.relations}

    assert ("Material", "ВТ6") in entities
    assert any(entity_type == "ProcessRegime" and "Отжиг" in name for entity_type, name in entities)
    assert ("Property", "Прочность") in entities
    assert any(entity_type == "PropertyValue" and "980 MPa" in name for entity_type, name in entities)
    assert "STUDIES" in predicates
    assert "USES_REGIME" in predicates
    assert "MEASURES" in predicates
    assert "OF_PROPERTY" in predicates
    assert "HAS_CHANGE" in predicates


def test_extraction_data_gap() -> None:
    text = (
        "Experiment: EXP-AL7075-AGE. Material: алюминиевый сплав 7075. "
        "Process: старение at 160 C for 8 h. Property: коррозионная стойкость. "
        "Data gap: нет данных по коррозионной стойкости алюминиевого сплава 7075 после старения."
    )

    result = EntityRelationExtractor().extract_from_chunk(_chunk(text))

    assert any(entity.entity_type == "DataGap" for entity in result.entities)
    assert any(relation.predicate == "MISSING_FOR" for relation in result.relations)
