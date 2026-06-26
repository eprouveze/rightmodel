"""Tests for spine.rightmodel_antigoodhart — flag-only health analyzers.

Uses lightweight fakes for the ledger and matrix so these tests are self-
contained (they don't require the real Module 2/4 to be wired). The contract
guarantees these analyzers are safe on an empty/young ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from rightmodel.rightmodel_antigoodhart import AntiGoodhart, HealthFlag


@dataclass
class FakeRow:
    task_class: str
    model: str
    lang: str = "en"
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class FakeCell:
    task_class: str
    model: str
    mu: float
    n: int
    state: str
    version_hash: str = ""


class FakeLedger:
    def __init__(self, rows: list[FakeRow]) -> None:
        self._rows = rows

    def rows(self, *, task_class=None, model=None, source=None, since_iso=None):
        out = self._rows
        if task_class is not None:
            out = [r for r in out if r.task_class == task_class]
        if model is not None:
            out = [r for r in out if r.model == model]
        if since_iso is not None:
            out = [r for r in out if r.ts >= since_iso]
        return list(out)


class FakeMatrix:
    def __init__(self, cells: dict[tuple[str, str], FakeCell], bar: float = 60.0) -> None:
        self._cells = cells
        self._bar = bar

    def cell(self, task_class, model):
        return self._cells.get(
            (task_class, model),
            FakeCell(task_class, model, mu=70.0, n=0, state="cold"),
        )

    def rank(self, task_class, candidates):
        cells = [self.cell(task_class, m) for m in candidates]
        return sorted(cells, key=lambda c: c.mu, reverse=True)

    def min_bar(self, task_class):
        return self._bar


# --------------------------------------------------------------------------- cold start


def test_empty_ledger_returns_no_flags() -> None:
    ag = AntiGoodhart(FakeLedger([]), FakeMatrix({}))
    assert ag.report() == []
    assert ag.starvation_flags() == []
    assert ag.gold_spotcheck([]) == []
    assert ag.escrow_flags(None) == []
    assert ag.ja_bias_check() == []


# --------------------------------------------------------------------------- starvation


def test_starvation_flag_raised_for_dominant_underobserved_model() -> None:
    rows = [FakeRow("coding", "claude-haiku-4-5") for _ in range(8)]
    rows += [FakeRow("coding", "gpt-5.5") for _ in range(2)]
    cells = {("coding", "claude-haiku-4-5"): FakeCell("coding", "claude-haiku-4-5", mu=72, n=5, state="warming")}
    ag = AntiGoodhart(FakeLedger(rows), FakeMatrix(cells))
    flags = ag.starvation_flags()
    assert len(flags) == 1
    assert flags[0].kind == "starvation"
    assert flags[0].model == "claude-haiku-4-5"
    assert flags[0].severity == "warn"


def test_starvation_suppressed_once_cell_is_well_observed() -> None:
    rows = [FakeRow("coding", "claude-haiku-4-5") for _ in range(10)]
    cells = {("coding", "claude-haiku-4-5"): FakeCell("coding", "claude-haiku-4-5", mu=72, n=40, state="settled")}
    ag = AntiGoodhart(FakeLedger(rows), FakeMatrix(cells))
    assert ag.starvation_flags() == []


def test_starvation_ignores_old_rows_outside_window() -> None:
    old_ts = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    rows = [FakeRow("coding", "claude-haiku-4-5", ts=old_ts) for _ in range(8)]
    cells = {("coding", "claude-haiku-4-5"): FakeCell("coding", "claude-haiku-4-5", mu=72, n=5, state="warming")}
    ag = AntiGoodhart(FakeLedger(rows), FakeMatrix(cells))
    assert ag.starvation_flags() == []


# --------------------------------------------------------------------------- gold


def test_gold_regression_critical_for_settled_high_model() -> None:
    cells = {("coding", "gpt-5.5"): FakeCell("coding", "gpt-5.5", mu=88, n=40, state="settled")}
    ag = AntiGoodhart(FakeLedger([]), FakeMatrix(cells))
    flags = ag.gold_spotcheck([{"task_class": "coding", "model": "gpt-5.5", "passed": False, "id": "g1"}])
    assert len(flags) == 1
    assert flags[0].kind == "gold_regression"
    assert flags[0].severity == "critical"


def test_gold_pass_produces_no_flag() -> None:
    cells = {("coding", "gpt-5.5"): FakeCell("coding", "gpt-5.5", mu=88, n=40, state="settled")}
    ag = AntiGoodhart(FakeLedger([]), FakeMatrix(cells))
    assert ag.gold_spotcheck([{"task_class": "coding", "model": "gpt-5.5", "passed": True}]) == []


def test_gold_fail_on_cold_cell_is_not_a_regression() -> None:
    cells = {("coding", "gpt-5.5"): FakeCell("coding", "gpt-5.5", mu=88, n=2, state="cold")}
    ag = AntiGoodhart(FakeLedger([]), FakeMatrix(cells))
    assert ag.gold_spotcheck([{"task_class": "coding", "model": "gpt-5.5", "passed": False}]) == []


# --------------------------------------------------------------------------- escrow


def test_escrow_flags_degraded_downstream_metric() -> None:
    ag = AntiGoodhart(FakeLedger([]), FakeMatrix({}))
    flags = ag.escrow_flags({"coding::gpt-5.5": -0.4, "content_writing::claude-opus-4-8": 0.1})
    assert len(flags) == 1
    assert flags[0].kind == "escrow_divergence"
    assert flags[0].model == "gpt-5.5"
    assert flags[0].task_class == "coding"


def test_escrow_none_is_stub_safe() -> None:
    ag = AntiGoodhart(FakeLedger([]), FakeMatrix({}))
    assert ag.escrow_flags(None) == []


# --------------------------------------------------------------------------- ja bias


def test_ja_bias_flag_when_non_western_model_starved_on_ja() -> None:
    rows = [FakeRow("content_writing", "claude-opus-4-8", lang="ja") for _ in range(40)]
    rows += [FakeRow("content_writing", "kimi-k2.6", lang="en") for _ in range(20)]
    cells = {
        ("content_writing", "claude-opus-4-8"): FakeCell("content_writing", "claude-opus-4-8", mu=80, n=40, state="settled"),
        ("content_writing", "kimi-k2.6"): FakeCell("content_writing", "kimi-k2.6", mu=90, n=35, state="settled"),
    }
    ag = AntiGoodhart(FakeLedger(rows), FakeMatrix(cells))
    flags = ag.ja_bias_check()
    assert len(flags) == 1
    assert flags[0].kind == "ja_bias"
    assert flags[0].model == "kimi-k2.6"


def test_ja_bias_quiet_below_min_observations() -> None:
    rows = [FakeRow("content_writing", "claude-opus-4-8", lang="ja") for _ in range(10)]
    cells = {
        ("content_writing", "kimi-k2.6"): FakeCell("content_writing", "kimi-k2.6", mu=90, n=35, state="settled"),
    }
    ag = AntiGoodhart(FakeLedger(rows), FakeMatrix(cells))
    assert ag.ja_bias_check() == []


# --------------------------------------------------------------------------- report aggregation


def test_report_aggregates_all_analyzers() -> None:
    rows = [FakeRow("coding", "claude-haiku-4-5") for _ in range(8)]
    cells = {("coding", "claude-haiku-4-5"): FakeCell("coding", "claude-haiku-4-5", mu=72, n=5, state="warming")}
    ag = AntiGoodhart(FakeLedger(rows), FakeMatrix(cells))
    flags = ag.report(
        gold_results=[],
        downstream={"coding::claude-haiku-4-5": -0.2},
    )
    kinds = {f.kind for f in flags}
    assert "starvation" in kinds
    assert "escrow_divergence" in kinds
    assert all(isinstance(f, HealthFlag) for f in flags)
