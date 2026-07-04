from __future__ import annotations

from app.extraction.extraction import EntityRelationExtractor
from app.models.schemas import Chunk


def _chunk(text: str) -> Chunk:
    return Chunk(chunk_id="c1", doc_id="d1", text=text, page_start=1, page_end=1, section_path="/")


def test_extractor_finds_technical_entities() -> None:
    text = (
        "Клапан DN50 PN16. Артикул VALVE-DN50-PN16. Материал корпуса 12Х18Н10Т. "
        "Стандарт ГОСТ 33259 и ISO 5208. P=1.6 MPa. Рабочая температура от -40 до +120 °C."
    )
    result = EntityRelationExtractor().extract_from_chunk(_chunk(text))
    entities = {(e.entity_type, e.canonical_name) for e in result.entities}
    predicates = {r.predicate for r in result.relations}
    assert any(t == "TechnicalObject" and "клапан" in n.lower() for t, n in entities)
    assert ("ArticleNumber", "VALVE-DN50-PN16") in entities
    assert ("Material", "12Х18Н10Т") in entities
    assert ("Standard", "ГОСТ 33259") in entities
    assert ("Standard", "ISO 5208") in entities
    assert any(t == "Parameter" and "PN" in n for t, n in entities)
    assert "OBJECT_HAS_PARAMETER" in predicates
    assert "OBJECT_COMPLIES_WITH_STANDARD" in predicates


def test_extractor_finds_images_and_gaps() -> None:
    text = (
        "Image: url: images/pump.png | alt: схема монтажа | caption: Монтаж насоса | section_path: /Монтаж\n"
        "Data gap: нет данных по коррозионной стойкости 7075-T6 после старения."
    )
    result = EntityRelationExtractor().extract_from_chunk(_chunk(text))
    assert any(e.entity_type == "ImageArtifact" for e in result.entities)
    assert any(e.entity_type == "DataGap" for e in result.entities)
    assert any(r.predicate == "IMAGE_LINKED_TO_SECTION" for r in result.relations)
    assert any(r.predicate == "MISSING_FOR" for r in result.relations)


def test_extractor_reads_markdown_strength_table_values() -> None:
    text = (
        "| Material | Diameter | Tensile Strength min | | Yield Strength | | Elongation in 2\" |\n"
        "| | | ksi | MPa | ksi | MPa | |\n"
        "| Titanium Alpha-Beta Ti-6Al-4V Annealed Bar (min) | <2.00\" | 135 | 931 | 125 | 862 | 10 |\n"
    )
    result = EntityRelationExtractor().extract_from_chunk(_chunk(text))
    entities = {(e.entity_type, e.canonical_name) for e in result.entities}
    assert ("PropertyValue", "Tensile strength = 931 MPa") in entities
    assert ("PropertyValue", "Yield strength = 862 MPa") in entities
    assert ("PropertyValue", "Elongation = 10 %") in entities
    assert any(r.predicate == "HAS_MEASUREMENT" and "Ti-6Al-4V" in r.subject for r in result.relations)


def test_extractor_reads_ksi_strength_table_values() -> None:
    text = (
        "| Material | Temper | Size (\") | Tensile Strength (ksi) | Yield Strength (ksi) | Elongation in 2\" % |\n"
        "| Alloy 7075 Bar | T651 | - | 77 | 66 | 7 |\n"
    )
    result = EntityRelationExtractor().extract_from_chunk(_chunk(text))
    entities = {(e.entity_type, e.canonical_name) for e in result.entities}
    assert ("PropertyValue", "Tensile strength = 77 ksi") in entities
    assert ("PropertyValue", "Yield strength = 66 ksi") in entities
    assert ("PropertyValue", "Elongation = 7 %") in entities


def test_extractor_uses_table_experiment_id_not_generic_fragment() -> None:
    text = (
        "Table columns: experiment_id | material | process_regime | property | value | unit\n"
        "experiment_id: VT6-AN-01 | material: ВТ6 | process_regime: отжиг | property: прочность | value: 1120 | unit: MPa"
    )
    result = EntityRelationExtractor().extract_from_chunk(_chunk(text))
    assert any(e.entity_type == "Experiment" and e.canonical_name == "VT6-AN-01" for e in result.entities)
    assert not any(e.entity_type == "Experiment" and e.canonical_name in {"s", "_id"} for e in result.entities)


def test_extractor_does_not_treat_pressure_as_strength_measurement() -> None:
    text = "Технический объект: насос NPK-200. Параметры: P=10 MPa, напор 10 MPa, рабочая температура T=300 C."
    result = EntityRelationExtractor().extract_from_chunk(_chunk(text))
    assert any(e.entity_type == "Parameter" and "10 MPa" in e.canonical_name for e in result.entities)
    assert not any(e.entity_type == "PropertyValue" and "10 MPa" in e.canonical_name for e in result.entities)
