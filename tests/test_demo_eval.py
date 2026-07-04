from __future__ import annotations

from evaluation.eval_demo import main


def test_demo_evaluation_passes() -> None:
    assert main() == 0
