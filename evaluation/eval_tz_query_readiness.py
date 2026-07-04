from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.retrieval.query_planner import QueryPlanner  # noqa: E402


ARTIFACT_PATH = ROOT / "artifacts" / "eval_tz_query_readiness.json"


CASES = [
    {
        "case_id": "desalination_water_constraints",
        "question": (
            "Какие методы обессоливания воды подходят для обогатительной фабрики, если исходная вода содержит "
            "сульфаты, хлориды, Ca, Mg, Na по 200–300 мг/л, а требуемый сухой остаток — ≤1000 мг/дм³?"
        ),
        "expected": {
            "materials": ["сульфаты", "хлориды", "Ca", "Mg", "Na"],
            "regimes": ["обессоливание"],
            "properties": ["сухой остаток"],
            "topic_tags": ["обогатительная фабрика"],
            "numeric": {
                "сульфаты": {"operator": "range", "value_min": 200.0, "value_max": 300.0, "unit": "mg/L"},
                "хлориды": {"operator": "range", "value_min": 200.0, "value_max": 300.0, "unit": "mg/L"},
                "Ca": {"operator": "range", "value_min": 200.0, "value_max": 300.0, "unit": "mg/L"},
                "Mg": {"operator": "range", "value_min": 200.0, "value_max": 300.0, "unit": "mg/L"},
                "Na": {"operator": "range", "value_min": 200.0, "value_max": 300.0, "unit": "mg/L"},
                "сухой остаток": {"operator": "<=", "value": 1000.0, "unit": "mg/L"},
            },
        },
    },
    {
        "case_id": "catholyte_circulation_world_practice",
        "question": "Какие технические решения организации циркуляции католита при электроэкстракции никеля описаны в мировой практике, и какая скорость потока считается оптимальной?",
        "expected": {
            "materials": ["никель", "католит"],
            "regimes": ["электроэкстракция", "циркуляция католита"],
            "properties": ["скорость потока"],
            "geographies": ["мировая практика"],
        },
    },
    {
        "case_id": "nickel_electrowinning_equipment",
        "question": "Какие схемы подачи электролита в ванны электроэкстракции никеля и диафрагменные ячейки описаны в мировой практике?",
        "expected": {
            "materials": ["никель", "электролит"],
            "regimes": ["электроэкстракция"],
            "equipment": ["ванна электроэкстракции", "диафрагменная ячейка"],
            "geographies": ["мировая практика"],
        },
    },
    {
        "case_id": "flash_smelting_gas_cleaning_equipment",
        "question": "Какие системы очистки газов и печи взвешенной плавки применяются для удаления SO2?",
        "expected": {
            "regimes": ["газоочистка", "удаление SO2", "ПВП"],
            "equipment": ["система газоочистки", "ПВП"],
        },
    },
    {
        "case_id": "matte_slag_distribution_last_5_years",
        "question": "Покажите все эксперименты и публикации по распределению Au, Ag и МПГ между медным/никелевым штейном и шлаком за последние 5 лет.",
        "expected": {
            "materials": ["Au", "Ag", "МПГ", "медь", "никель", "штейн", "шлак"],
            "properties": ["распределение"],
            "time_filters": [{"type": "relative_years", "years": 5}],
        },
    },
    {
        "case_id": "mine_water_injection_geo_economics",
        "question": "Какие способы закачки шахтных вод в глубокие горизонты применялись в России и за рубежом, и каковы их технико-экономические показатели?",
        "expected": {
            "materials": ["шахтные воды"],
            "regimes": ["закачка шахтных вод"],
            "properties": ["экономический показатель"],
            "geographies": ["Россия", "зарубежная практика"],
        },
    },
]


def _contains_all(actual: list[Any], expected: list[Any]) -> list[str]:
    missing = [item for item in expected if item not in actual]
    return [f"missing {item!r}" for item in missing]


def _numeric_errors(actual_rows: list[dict[str, Any]], expected: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    by_parameter = {row.get("parameter"): row for row in actual_rows}
    for parameter, expected_row in expected.items():
        actual = by_parameter.get(parameter)
        if actual is None:
            errors.append(f"missing numeric constraint for {parameter}")
            continue
        for key, value in expected_row.items():
            if actual.get(key) != value:
                errors.append(f"{parameter}.{key}: expected {value!r}, got {actual.get(key)!r}")
    return errors


def validate_case(case: dict[str, Any], constraints: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    errors: list[str] = []
    for key in ["materials", "regimes", "properties", "equipment", "topic_tags", "geographies"]:
        if key in expected:
            errors.extend(f"{key}: {item}" for item in _contains_all(constraints.get(key) or [], expected[key]))
    if "time_filters" in expected and constraints.get("time_filters") != expected["time_filters"]:
        errors.append(f"time_filters: expected {expected['time_filters']!r}, got {constraints.get('time_filters')!r}")
    if "numeric" in expected:
        errors.extend(_numeric_errors(constraints.get("numeric_constraints") or [], expected["numeric"]))
    if constraints.get("require_exact_match"):
        errors.append("broad TZ query should not force exact graph match")
    return {
        "case_id": case["case_id"],
        "passed": not errors,
        "errors": errors,
        "constraints": constraints,
    }


def run_eval() -> tuple[dict[str, Any], int]:
    planner = QueryPlanner()
    rows = [validate_case(case, planner.parse(case["question"]).model_dump(mode="json")) for case in CASES]
    failed = [row for row in rows if not row["passed"]]
    result = {
        "summary": "FAIL" if failed else "PASS",
        "cases_total": len(rows),
        "cases_passed": len(rows) - len(failed),
        "rows": rows,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result, 1 if failed else 0


if __name__ == "__main__":
    result, exit_code = run_eval()
    print(f"SUMMARY: {result['summary']}")
    for row in result["rows"]:
        status = "PASS" if row["passed"] else "FAIL"
        reason = "ok" if row["passed"] else "; ".join(row["errors"])
        print(f"[{status}] {row['case_id']}: {reason}")
    print(f"JSON report: {ARTIFACT_PATH}")
    raise SystemExit(exit_code)
