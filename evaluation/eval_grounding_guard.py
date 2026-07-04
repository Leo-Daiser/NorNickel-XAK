from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.answering.human_answer import enhance_answer_payload  # noqa: E402


ARTIFACT_PATH = ROOT / "artifacts" / "eval_grounding_guard.json"


def _base_payload(answer: str, facts: list[dict[str, Any]] | None = None, *, status: str = "ok") -> dict[str, Any]:
    return {
        "answer": answer,
        "status": status,
        "answer_mode": "comparison",
        "analytical_intent": "material_comparison",
        "constraints": {"materials": ["ВТ6", "7075-T6"], "regimes": [], "properties": ["прочность"]},
        "facts": facts
        if facts is not None
        else [
            {
                "material": "ВТ6",
                "regime": "отжиг",
                "property": "прочность",
                "value": 980.0,
                "unit": "MPa",
                "effect": "increase",
                "evidence": [{"source_name": "article_vt6.txt", "quote": "ВТ6 отжиг прочность 980 MPa"}],
            },
            {
                "material": "7075-T6",
                "regime": "старение",
                "property": "прочность",
                "value": 77.0,
                "unit": "ksi",
                "value_original": 77.0,
                "unit_original": "ksi",
                "value_normalized": 530.896289,
                "unit_normalized": "MPa",
                "normalization_family": "strength",
                "effect": "increase",
                "evidence": [{"source_name": "article_7075.txt", "quote": "tensile strength of 77 ksi"}],
            },
        ],
        "sources": [],
        "evidence": [],
        "subgraph": {"nodes": [], "edges": []},
        "graph_context": {},
        "retrieval": {},
        "diagnostics": {"llm_answer_polished": True},
    }


def _unsupported_number() -> dict[str, Any]:
    return _base_payload("ВТ6 подтверждён: прочность 980 MPa, обработка 1050 °C.")


def _supported_normalized() -> dict[str, Any]:
    return _base_payload("7075-T6 после старения имеет прочность 531 MPa после пересчёта из 77 ksi.")


def _unsupported_material() -> dict[str, Any]:
    return _base_payload(
        "ВТ6 и 7075-T6 сопоставимы, а 12Х18Н10Т показала 980 MPa.",
        facts=[
            {
                "material": "7075-T6",
                "regime": "старение",
                "property": "прочность",
                "value": 77.0,
                "unit": "ksi",
                "value_normalized": 530.896289,
                "unit_normalized": "MPa",
                "evidence": [{"source_name": "article_7075.txt", "quote": "tensile strength of 77 ksi"}],
            }
        ],
    )


def _no_data_answer() -> dict[str, Any]:
    payload = _base_payload("По X999 данных нет, но похожая закалка была при 1050 °C.", facts=[], status="partial")
    payload["answer_mode"] = "overview"
    payload["analytical_intent"] = "material_overview"
    payload["constraints"] = {"materials": ["X999"], "raw_question": "Что известно о X999 при лазерной обработке?"}
    return payload


def _case(
    name: str,
    payload_factory: Callable[[], dict[str, Any]],
    expected_status: str,
    forbidden: list[str],
    repairer: Callable[[dict[str, Any]], str | None] | None = None,
) -> dict[str, Any]:
    payload = enhance_answer_payload(payload_factory(), "expert_max", llm_repairer=repairer)
    guard = payload.get("diagnostics", {}).get("llm_grounding_guard") or {}
    answer = str(payload.get("answer") or "")
    errors = []
    if guard.get("status") != expected_status:
        errors.append(f"expected guard status {expected_status}, got {guard.get('status')}")
    for token in forbidden:
        if token in answer:
            errors.append(f"forbidden token remained in final answer: {token}")
    return {
        "case": name,
        "passed": not errors,
        "errors": errors,
        "guard_status": guard.get("status"),
        "repair_attempted": bool(guard.get("repair_attempted")),
        "repair_passed": bool(guard.get("repair_passed")),
        "violations_count": guard.get("violations_count", 0),
        "repaired_violations_count": guard.get("repaired_violations_count", 0),
    }


def main() -> int:
    rows = [
        _case("unsupported_number_injection", _unsupported_number, "fallback", ["1050"]),
        _case(
            "unsafe_first_polish_repaired",
            _unsupported_number,
            "repaired",
            ["1050"],
            repairer=lambda request: "ВТ6: 980 MPa. 7075-T6: 531 MPa после пересчёта из 77 ksi.",
        ),
        _case(
            "unsafe_first_polish_repair_still_unsafe",
            _unsupported_number,
            "fallback",
            ["1050", "9999"],
            repairer=lambda request: "ВТ6 якобы имеет 9999 MPa.",
        ),
        _case("supported_normalized_value", _supported_normalized, "pass", []),
        _case("unsupported_material_claim", _unsupported_material, "fallback", ["12Х18Н10Т"]),
        _case("no_data_numeric_claim", _no_data_answer, "fallback", ["1050"]),
    ]
    summary = "PASS" if all(row["passed"] for row in rows) else "FAIL"
    result = {"summary": summary, "rows": rows}
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY: {summary}")
    for row in rows:
        label = "PASS" if row["passed"] else "FAIL"
        print(
            f"[{label}] {row['case']}: guard={row['guard_status']} "
            f"repair={int(row['repair_attempted'])}/{int(row['repair_passed'])} "
            f"violations={row['violations_count']} repaired_violations={row['repaired_violations_count']} "
            f"errors={'; '.join(row['errors']) or 'ok'}"
        )
    print(f"JSON report: {ARTIFACT_PATH}")
    return 0 if summary == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
