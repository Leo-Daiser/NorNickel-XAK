from __future__ import annotations

from app.parsing.source_metadata import infer_source_metadata


def test_source_metadata_detects_year_geography_type_and_reliability() -> None:
    metadata = infer_source_metadata(
        source_name="nickel_electrowinning_review_2024.pdf",
        source_type="file",
        parser_name="fallback",
        text=(
            "Рецензируемая статья 2024 года сравнивает российскую и зарубежную практику "
            "электроэкстракции никеля. Journal DOI: 10.000/demo."
        ),
    )["source_metadata"]

    assert metadata["publication_year"] == 2024
    assert "Россия" in metadata["geographies"]
    assert "зарубежная практика" in metadata["geographies"]
    assert metadata["practice_scope"] == "domestic_and_foreign"
    assert metadata["source_type_detected"] == "publication"
    assert metadata["reliability_level"] == "high"


def test_source_metadata_uses_extension_when_text_has_no_type_hint() -> None:
    metadata = infer_source_metadata(
        source_name="measurement_register.xlsx",
        source_type="file",
        text="Material; process; property; value",
    )["source_metadata"]

    assert metadata["source_type_detected"] == "catalog"
    assert metadata["type_basis"] == "extension"
    assert metadata["reliability_level"] == "medium"
