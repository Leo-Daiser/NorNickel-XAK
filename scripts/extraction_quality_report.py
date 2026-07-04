from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.domain.fact_normalization import (  # noqa: E402
    build_conflict_summary,
    canonical_fact_key,
    measurement_normalization_fields,
)
from app.extraction.pipeline import ExtractionPipeline  # noqa: E402
from app.storage.catalog import SQLiteCatalog  # noqa: E402


REPORT_PATH = ROOT / "artifacts" / "extraction_quality_report.json"
NEO4J_NOT_CONFIGURED_WARNING = "Neo4j scan skipped: Neo4j connection settings are not configured for this runtime."


def _bucket(confidence: float | None) -> str:
    value = float(confidence or 0)
    if value >= 0.85:
        return "0.85-1.00"
    if value >= 0.70:
        return "0.70-0.84"
    if value >= 0.55:
        return "0.55-0.69"
    return "0.00-0.54"


def _fact_key(experiment, measurement) -> str:
    material = experiment.materials[0].canonical_name if experiment.materials else ""
    regime = experiment.regimes[0].canonical_name if experiment.regimes else ""
    return canonical_fact_key(
        material=material,
        regime=regime,
        property_name=measurement.property_canonical,
        value=measurement.value,
        unit=measurement.unit,
        effect=measurement.effect,
    )


def _normalized_value(measurement) -> float | None:
    fields = measurement_normalization_fields(measurement.property_canonical, measurement.value, measurement.unit)
    value = fields["value_normalized"]
    return float(value) if value is not None else None


def _suspicious_numeric(measurement) -> bool:
    value = _normalized_value(measurement)
    if value is None:
        return False
    prop = measurement.property_canonical
    if prop == "прочность":
        return value <= 0 or value > 5000
    if prop == "твёрдость":
        return value <= 0 or value > 1500
    if prop == "пластичность":
        return value < 0 or value > 100
    return abs(value) > 1_000_000


def _has_complete_normalized_fields(fields: dict[str, Any]) -> bool:
    return bool(
        fields.get("value_original") is not None
        and fields.get("unit_original")
        and fields.get("value_normalized") is not None
        and fields.get("unit_normalized")
        and fields.get("normalization_family")
    )


@dataclass(frozen=True)
class Neo4jScanOptions:
    skip_neo4j: bool = False
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str | None = None


@dataclass(frozen=True)
class Neo4jScanResult:
    status: str
    missing_normalized_fields: int | None = None
    warning: str = ""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build extraction quality report for the current corpus.")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip persisted Neo4j normalized-field scan.")
    parser.add_argument("--neo4j-uri", default=None, help="Neo4j URI for optional persisted graph scan.")
    parser.add_argument("--neo4j-user", default=None, help="Neo4j user for optional persisted graph scan.")
    parser.add_argument("--neo4j-password", default=None, help="Neo4j password for optional persisted graph scan; never printed.")
    parser.add_argument("--neo4j-database", default=None, help="Neo4j database for optional persisted graph scan.")
    return parser.parse_args(argv)


def _neo4j_scan_options_from_args(args: argparse.Namespace) -> Neo4jScanOptions:
    return Neo4jScanOptions(
        skip_neo4j=bool(args.skip_neo4j),
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
    )


def _resolved_neo4j_scan_config(options: Neo4jScanOptions) -> dict[str, str] | None:
    uri = options.neo4j_uri or os.getenv("NEO4J_URI")
    user = options.neo4j_user or os.getenv("NEO4J_USER") or "neo4j"
    password = options.neo4j_password or os.getenv("NEO4J_PASSWORD")
    database = options.neo4j_database or os.getenv("NEO4J_DATABASE") or "neo4j"
    if not uri or not password:
        return None
    return {"uri": uri, "user": user, "password": password, "database": database}


def _legacy_neo4j_records_missing_normalized_fields(options: Neo4jScanOptions) -> Neo4jScanResult:
    if options.skip_neo4j:
        return Neo4jScanResult(status="skipped", warning="Neo4j scan skipped by --skip-neo4j.")
    config = _resolved_neo4j_scan_config(options)
    if config is None:
        return Neo4jScanResult(status="skipped", warning=NEO4J_NOT_CONFIGURED_WARNING)
    try:
        from app.graph.graph_db import GraphDB

        graph_db = GraphDB(
            uri=config["uri"],
            user=config["user"],
            password=config["password"],
            database=config["database"],
        )
    except Exception as exc:
        return Neo4jScanResult(status="error", warning=f"Neo4j scan failed: {type(exc).__name__}.")
    try:
        rows = graph_db.run(
            """
            MATCH (m:Measurement)
            WHERE m.value IS NOT NULL
              AND (
                m.value_original IS NULL OR
                m.unit_original IS NULL OR
                m.value_normalized IS NULL OR
                m.unit_normalized IS NULL OR
                m.normalization_family IS NULL
              )
            RETURN count(m) AS missing
            """
        )
        missing = int(rows[0]["missing"]) if rows else 0
        return Neo4jScanResult(status="ok", missing_normalized_fields=missing)
    except Exception as exc:
        return Neo4jScanResult(status="error", warning=f"Neo4j scan failed: {type(exc).__name__}.")
    finally:
        try:
            graph_db.close()
        except Exception:
            pass


def build_report(scan_options: Neo4jScanOptions | None = None) -> dict[str, Any]:
    scan_options = scan_options or Neo4jScanOptions()
    warnings: list[str] = []
    catalog = SQLiteCatalog(settings.catalog_db_path)
    counts = catalog.counts()
    chunks = catalog.list_chunks(active_only=True)
    if not chunks:
        warnings.append("No active chunks found in catalog.")

    pipeline = ExtractionPipeline(mode="deterministic", audit_enabled=False)
    entity_counter: Counter[str] = Counter()
    material_counter: Counter[str] = Counter()
    property_counter: Counter[str] = Counter()
    regime_counter: Counter[str] = Counter()
    confidence_counter: Counter[str] = Counter()
    rejected_counter: Counter[str] = Counter()
    suspicious: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate_counter: Counter[str] = Counter()
    fact_rows: list[dict[str, Any]] = []
    value_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    total_facts = 0
    numeric_facts = 0
    facts_with_units = 0
    facts_without_source = 0
    normalized_measurements_count = 0
    measurements_missing_normalized_fields = 0
    accepted_experiments = 0
    accepted_gaps = 0

    for chunk in chunks:
        try:
            bundle = pipeline.extract_from_chunk(chunk)
        except Exception as exc:
            warnings.append(f"Extraction failed for chunk {chunk.chunk_id}: {type(exc).__name__}: {exc}")
            continue
        for entity in bundle.entities:
            entity_counter[entity.entity_type] += 1
        for item in bundle.rejected_items:
            rejected_counter[f"{item.item_type}:{item.reason}"] += 1
        accepted_gaps += len(bundle.data_gaps)

        for experiment in bundle.experiments:
            accepted_experiments += 1
            if not experiment.materials:
                suspicious["missing_material"].append({"experiment_id": experiment.experiment_id, "chunk_id": chunk.chunk_id})
            for material in experiment.materials:
                material_counter[material.canonical_name] += 1
            for regime in experiment.regimes:
                regime_counter[regime.canonical_name] += 1
            confidence_counter[_bucket(experiment.confidence)] += 1

            for measurement in experiment.measurements:
                total_facts += 1
                property_counter[measurement.property_canonical] += 1
                confidence_counter[_bucket(measurement.confidence)] += 1
                if not measurement.property_canonical:
                    suspicious["missing_property"].append({"experiment_id": experiment.experiment_id, "chunk_id": chunk.chunk_id})
                if measurement.value is not None:
                    numeric_facts += 1
                    if not measurement.unit:
                        suspicious["missing_unit_for_numeric_value"].append(
                            {
                                "experiment_id": experiment.experiment_id,
                                "property": measurement.property_canonical,
                                "value": measurement.value,
                                "chunk_id": chunk.chunk_id,
                            }
                        )
                    if _suspicious_numeric(measurement):
                        suspicious["suspicious_numeric_values"].append(
                            {
                                "experiment_id": experiment.experiment_id,
                                "property": measurement.property_canonical,
                                "value": measurement.value,
                                "unit": measurement.unit,
                                "chunk_id": chunk.chunk_id,
                            }
                        )
                if measurement.unit:
                    facts_with_units += 1
                if not (measurement.evidence or experiment.evidence):
                    facts_without_source += 1
                key = _fact_key(experiment, measurement)
                duplicate_counter[key] += 1
                material = experiment.materials[0].canonical_name if experiment.materials else ""
                regime = experiment.regimes[0].canonical_name if experiment.regimes else ""
                normalized_fields = measurement_normalization_fields(measurement.property_canonical, measurement.value, measurement.unit)
                if measurement.value is not None:
                    if _has_complete_normalized_fields(normalized_fields):
                        normalized_measurements_count += 1
                    else:
                        measurements_missing_normalized_fields += 1
                fact_rows.append(
                    {
                        "material": material,
                        "regime": regime,
                        "property": measurement.property_canonical,
                        "value": measurement.value,
                        "raw_value": measurement.value,
                        "unit": measurement.unit,
                        "effect": measurement.effect,
                        "evidence": [
                            {
                                "document_id": span.source.document_id,
                                "chunk_id": span.source.chunk_id,
                                "source_name": span.source.source_name,
                                "quote": span.quote,
                            }
                            for span in (measurement.evidence or experiment.evidence)
                        ],
                        **normalized_fields,
                    }
                )
                group_key = (material, regime, measurement.property_canonical)
                normalized = _normalized_value(measurement)
                if normalized is not None:
                    value_groups[group_key].append(
                        {
                            "experiment_id": experiment.experiment_id,
                            "value": normalized,
                            "raw_value": measurement.value,
                            "unit": measurement.unit,
                            "chunk_id": chunk.chunk_id,
                        }
                    )

    for key, count in duplicate_counter.items():
        if count > 1:
            suspicious["duplicated_facts"].append(
                {"canonical_fact_key": key, "count": count}
            )
    for (material, regime, prop), rows in value_groups.items():
        values = [row["value"] for row in rows if row["value"] is not None]
        if len(values) < 2:
            continue
        spread = max(values) - min(values)
        ratio = max(values) / max(min(values), 1e-9)
        if spread >= 250 or ratio >= 1.8:
            suspicious["conflicting_values"].append(
                {
                    "material": material,
                    "regime": regime,
                    "property": prop,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                    "examples": rows[:5],
                }
            )

    conflict_summary = build_conflict_summary(fact_rows)
    neo4j_scan = _legacy_neo4j_records_missing_normalized_fields(scan_options)
    legacy_neo4j_missing = neo4j_scan.missing_normalized_fields
    raw_facts_count = total_facts
    canonical_facts_count = len(duplicate_counter)
    duplicate_groups_count = sum(1 for count in duplicate_counter.values() if count > 1)
    duplicate_facts_count = raw_facts_count - canonical_facts_count
    conflict_groups_count = len(conflict_summary)

    if conflict_groups_count > 0:
        warnings.append(f"Conflict groups detected: {conflict_groups_count}. Review conflict_summary before demo claims.")
    if measurements_missing_normalized_fields > 0:
        warnings.append(f"Measurements missing normalized fields: {measurements_missing_normalized_fields}.")
    if facts_without_source > 0:
        warnings.append(f"Facts without evidence: {facts_without_source}.")
    if neo4j_scan.warning:
        warnings.append(neo4j_scan.warning)
    if legacy_neo4j_missing not in {None, 0}:
        warnings.append(f"Neo4j Measurement records missing normalized fields: {legacy_neo4j_missing}.")

    return {
        "summary": {
            "total_documents": counts.get("active_documents", counts.get("documents", 0)),
            "total_chunks": counts.get("active_chunks", counts.get("chunks", 0)),
            "processed_chunks": len(chunks),
            "total_facts": total_facts,
            "raw_facts_count": raw_facts_count,
            "canonical_facts_count": canonical_facts_count,
            "duplicate_groups_count": duplicate_groups_count,
            "duplicate_facts_count": duplicate_facts_count,
            "conflict_groups_count": conflict_groups_count,
            "accepted_experiments": accepted_experiments,
            "accepted_gaps": accepted_gaps,
            "facts_with_numeric_values": numeric_facts,
            "facts_with_units": facts_with_units,
            "facts_without_evidence": facts_without_source,
            "facts_without_source_or_evidence": facts_without_source,
            "normalized_measurements_count": normalized_measurements_count,
            "measurements_missing_normalized_fields": measurements_missing_normalized_fields,
            "neo4j_scan_status": neo4j_scan.status,
            "neo4j_scan_warning": neo4j_scan.warning,
            "legacy_neo4j_records_missing_normalized_fields": legacy_neo4j_missing,
            "rejected_or_low_confidence_candidates": sum(rejected_counter.values()),
        },
        "neo4j_scan": {
            "status": neo4j_scan.status,
            "warning": neo4j_scan.warning,
            "legacy_neo4j_records_missing_normalized_fields": legacy_neo4j_missing,
        },
        "facts_by_entity_type": dict(entity_counter),
        "facts_by_property": dict(property_counter),
        "facts_by_material": dict(material_counter),
        "facts_by_regime": dict(regime_counter),
        "facts_by_confidence_bucket": dict(confidence_counter),
        "rejections_by_reason": dict(rejected_counter),
        "top_materials": material_counter.most_common(10),
        "top_properties": property_counter.most_common(10),
        "top_process_regimes": regime_counter.most_common(10),
        "suspicious_facts": {key: rows[:50] for key, rows in suspicious.items()},
        "suspicious_counts": {key: len(rows) for key, rows in suspicious.items()},
        "conflict_summary": conflict_summary,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args([] if argv is None else argv)
    report = build_report(_neo4j_scan_options_from_args(args))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    print("Extraction quality report")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nTop materials:", report["top_materials"][:5])
    print("Top properties:", report["top_properties"][:5])
    print("Top regimes:", report["top_process_regimes"][:5])
    if report["warnings"]:
        print("\nWarnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
    print("\nSuspicious facts:")
    for key, count in report["suspicious_counts"].items():
        print(f"{key}: {count}")
    print("Conflict groups:", summary["conflict_groups_count"])
    print(f"\nJSON report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
