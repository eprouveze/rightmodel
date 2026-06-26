"""Tests for spine.rightmodel_freshness — anti-staleness flags + item rotation (Module 10).

Covers cold-start safety (every flag fn returns [] on empty/young input), prediction_drift
firing when predicted and observed pass-rates diverge, and rotate_items determinism + holdout
preservation. (No version_stale: identity is model_id — a new model is a new cold cell, never a
version-based re-eval that would blend a stable id's data.)

Lightweight doubles stand in for the Module 4 matrix and Module 2 ledger so the test
exercises freshness without disk or live LLM calls.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rightmodel.rightmodel_types")
pytest.importorskip("rightmodel.rightmodel_freshness")

from rightmodel.rightmodel_freshness import (  # noqa: E402
    FreshnessFlag,
    prediction_drift,
    rotate_items,
    sealed_holdout,
)
from rightmodel.rightmodel_types import CellState  # noqa: E402


class FakeCell:
    def __init__(self, *, state=CellState.SETTLED.value, version_hash="v1", mu=70.0):
        self.state = state
        self.version_hash = version_hash
        self.mu = mu


class FakeMatrix:
    """Returns a configured cell per (task_class, model), else a cold default."""

    def __init__(self, cells: dict[tuple[str, str], FakeCell] | None = None):
        self._cells = cells or {}

    def cell(self, task_class, model):
        return self._cells.get(
            (task_class, model),
            FakeCell(state=CellState.COLD.value, version_hash="", mu=60.0),
        )


class FakeRow:
    def __init__(self, *, task_class="coding", model="m", outcome="pass",
                 source="production"):
        self.task_class = task_class
        self.model = model
        self.outcome = outcome
        self.source = source


class FakeLedger:
    def __init__(self, rows: list[FakeRow]):
        self._rows = rows

    def rows(self, *, task_class=None, model=None, source=None, since_iso=None):
        out = self._rows
        if task_class is not None:
            out = [r for r in out if r.task_class == task_class]
        if source is not None:
            out = [r for r in out if r.source == source]
        return list(out)


class FakeItem:
    def __init__(self, id, task_class):
        self.id = id
        self.task_class = task_class


# --------------------------------------------------------------------------- cold start


def test_prediction_drift_young_ledger_returns_empty() -> None:
    rows = [FakeRow() for _ in range(3)]  # below _MIN_DRIFT_ROWS
    assert prediction_drift(FakeLedger(rows), FakeMatrix(), classes=["coding"]) == []
    assert prediction_drift(FakeLedger([]), FakeMatrix()) == []


def test_no_version_stale_export() -> None:
    """Identity is model_id — there is deliberately no version-based re-eval trigger."""
    import rightmodel.rightmodel_freshness as fr
    assert not hasattr(fr, "version_stale")


# --------------------------------------------------------------------------- prediction_drift


def test_prediction_drift_fires_on_divergence() -> None:
    # matrix predicts ~0.90 success; observed pass-rate is 0/8 → large gap
    cells = {("coding", "m"): FakeCell(mu=90.0)}
    rows = [FakeRow(model="m", outcome="fail") for _ in range(8)]
    flags = prediction_drift(
        FakeLedger(rows), FakeMatrix(cells), window=30, threshold=0.10, classes=["coding"]
    )
    assert len(flags) == 1
    assert flags[0].kind == "prediction_drift"
    assert flags[0].task_class == "coding"


def test_prediction_drift_quiet_when_calibrated() -> None:
    cells = {("coding", "m"): FakeCell(mu=100.0)}
    rows = [FakeRow(model="m", outcome="pass") for _ in range(8)]
    flags = prediction_drift(
        FakeLedger(rows), FakeMatrix(cells), threshold=0.10, classes=["coding"]
    )
    assert flags == []


# --------------------------------------------------------------------------- rotate_items


def _pack() -> list[FakeItem]:
    return [
        FakeItem("code-a", "coding"),
        FakeItem("code-b", "coding"),
        FakeItem("code-c", "coding"),
        FakeItem("code-d", "coding"),
        FakeItem("code-e", "coding"),
    ]


def _samples() -> list[dict]:
    return [
        {"prompt": "real task one", "difficulty": 0.3, "lang": "en"},
        {"prompt": "real task two", "difficulty": 0.6, "lang": "ja"},
    ]


def test_sealed_holdout_stable_per_class() -> None:
    held = sealed_holdout(_pack(), k_per_class=2)
    assert held == {"code-a", "code-b"}  # lowest two ids
    # deterministic across calls
    assert sealed_holdout(_pack(), k_per_class=2) == held


def test_rotate_items_deterministic_and_preserves_holdout() -> None:
    pack = _pack()
    held = sealed_holdout(pack, k_per_class=2)
    out1 = rotate_items(pack, _samples(), frac=0.5, seed=7, holdout_ids=held)
    out2 = rotate_items(pack, _samples(), frac=0.5, seed=7, holdout_ids=held)
    # same length, same result for same seed
    assert len(out1) == len(pack)
    ids1 = [it["id"] if isinstance(it, dict) else it.id for it in out1]
    ids2 = [it["id"] if isinstance(it, dict) else it.id for it in out2]
    assert ids1 == ids2
    # holdout items survive untouched (still FakeItem, original ids)
    survivors = {it.id for it in out1 if not isinstance(it, dict)}
    assert held <= survivors
    # ~frac of the 3 non-holdout items replaced (int(3*0.5)=1)
    replaced = [it for it in out1 if isinstance(it, dict)]
    assert len(replaced) == 1
    assert replaced[0]["prompt"] in {"real task one", "real task two"}


def test_rotate_items_different_seed_can_differ() -> None:
    pack = _pack()
    held = sealed_holdout(pack, k_per_class=1)  # only code-a held → 4 eligible
    out_a = rotate_items(pack, _samples(), frac=0.5, seed=1, holdout_ids=held)
    out_b = rotate_items(pack, _samples(), frac=0.5, seed=999, holdout_ids=held)
    pos_a = [i for i, it in enumerate(out_a) if isinstance(it, dict)]
    pos_b = [i for i, it in enumerate(out_b) if isinstance(it, dict)]
    assert len(pos_a) == 2 and len(pos_b) == 2  # int(4*0.5)=2 replaced each


def test_rotate_items_no_samples_returns_unchanged() -> None:
    pack = _pack()
    out = rotate_items(pack, [], frac=0.5, seed=7)
    assert out == pack
