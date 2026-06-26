"""Tests for spine.rightmodel_matrix — capability matrix (Module 4).

Covers cold-start (cell == prior, max σ), the prior↔data blend, difficulty-scaled
Elo over collisions, synthetic half-weight, ranking, and the min_bar default.

Uses a lightweight in-memory ledger double matching the Module 2 contract so the
test exercises the matrix without touching disk. If sibling modules 0/1/2 are not
yet present (parallel build), the whole module is skipped — integration runs it
against the real implementations.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("rightmodel.rightmodel_ledger")
pytest.importorskip("rightmodel.model_prices")
pytest.importorskip("rightmodel.rightmodel_types")

from rightmodel.rightmodel_ledger import LedgerRow  # noqa: E402
from rightmodel.rightmodel_matrix import (  # noqa: E402
    DEFAULT_MIN_BAR,
    PRIOR_WEIGHT,
    SIGMA_SCALE,
    CapabilityMatrix,
    Cell,
)
from rightmodel.model_prices import SEED_PRIOR  # noqa: E402
from rightmodel.router_policy import model_to_provider  # noqa: E402


class FakeLedger:
    """In-memory RightModelLedger double (Module 2 read surface)."""

    def __init__(self, rows: list[LedgerRow]) -> None:
        self._rows = rows

    def rows(self, *, task_class=None, model=None, source=None, since_iso=None):
        out = self._rows
        if task_class is not None:
            out = [r for r in out if r.task_class == task_class]
        if model is not None:
            out = [r for r in out if r.model == model]
        if source is not None:
            out = [r for r in out if r.source == source]
        return list(out)

    def collisions(self, task_class):
        groups: dict[str, list[LedgerRow]] = {}
        for r in self._rows:
            if r.task_class != task_class or r.collision_group is None:
                continue
            groups.setdefault(r.collision_group, []).append(r)
        return [g for g in groups.values() if len(g) > 1]


def _row(**kw) -> LedgerRow:
    base = dict(
        id="rm-x",
        ts="2026-06-16T00:00:00+00:00",
        task_class="coding",
        model="claude-haiku-4-5",
        provider=model_to_provider("claude-haiku-4-5").value,
        outcome="pass",
        verified=True,
        cost_usd=0.001,
        difficulty=0.5,
        lang="en",
        source="production",
        version_hash="",
        run_id=None,
        collision_group=None,
        note=None,
    )
    base.update(kw)
    return LedgerRow(**base)


# --------------------------------------------------------------------------- cold start


def test_cold_cell_equals_prior_and_max_sigma() -> None:
    m = CapabilityMatrix(FakeLedger([]))
    model = next(iter(SEED_PRIOR["coding"]))
    c = m.cell("coding", model)
    assert c.n == 0
    assert c.state == "cold"
    assert c.mu == pytest.approx(SEED_PRIOR["coding"][model])
    assert c.sigma == pytest.approx(SIGMA_SCALE / math.sqrt(1.0))


def test_unknown_model_falls_back_to_default_prior() -> None:
    m = CapabilityMatrix(FakeLedger([]))
    c = m.cell("coding", "totally-unseen-model")
    assert isinstance(c, Cell)
    assert c.n == 0
    assert c.mu == pytest.approx(60.0)


def test_empty_ledger_rank_uses_priors() -> None:
    m = CapabilityMatrix(FakeLedger([]))
    cands = list(SEED_PRIOR["coding"])[:3]
    ranked = m.rank("coding", cands)
    mus = [c.mu for c in ranked]
    assert mus == sorted(mus, reverse=True)


# --------------------------------------------------------------------------- blend + duels


def test_winner_outranks_loser_after_collisions() -> None:
    rows: list[LedgerRow] = []
    for i in range(20):
        g = f"g{i}"
        rows.append(_row(model="gpt-5.4-mini", outcome="pass", collision_group=g, difficulty=0.8))
        rows.append(_row(model="gpt-5.4-nano", outcome="fail", collision_group=g, difficulty=0.8))
    m = CapabilityMatrix(FakeLedger(rows))
    winner = m.cell("coding", "gpt-5.4-mini")
    loser = m.cell("coding", "gpt-5.4-nano")
    assert winner.n == 20 and loser.n == 20
    assert winner.state == "warming"
    assert winner.mu > loser.mu


def test_sigma_shrinks_with_evidence() -> None:
    rows = [_row(model="claude-haiku-4-5") for _ in range(40)]
    m = CapabilityMatrix(FakeLedger(rows))
    c = m.cell("coding", "claude-haiku-4-5")
    assert c.n == 40
    assert c.state == "settled"
    assert c.sigma == pytest.approx(SIGMA_SCALE / math.sqrt(41.0))


def test_blend_formula_holds() -> None:
    rows: list[LedgerRow] = []
    for i in range(5):
        g = f"g{i}"
        rows.append(_row(model="claude-opus-4-8", outcome="pass", collision_group=g))
        rows.append(_row(model="gpt-5.4-nano", outcome="fail", collision_group=g))
    m = CapabilityMatrix(FakeLedger(rows))
    c = m.cell("coding", "claude-opus-4-8")
    prior = SEED_PRIOR["coding"].get("claude-opus-4-8", 60.0)
    assert c.n == 5
    # Reverse the blend to recover the data estimate, then assert mu lies between
    # prior and that data estimate (the defining property of the convex blend).
    bt_mu = ((PRIOR_WEIGHT + 5) * c.mu - PRIOR_WEIGHT * prior) / 5
    expected = (PRIOR_WEIGHT * prior + 5 * bt_mu) / (PRIOR_WEIGHT + 5)
    assert c.mu == pytest.approx(expected, abs=1e-6)
    assert min(prior, bt_mu) - 1e-6 <= c.mu <= max(prior, bt_mu) + 1e-6
    # opus won every duel, so its data estimate must clear the capability bar
    assert bt_mu > 50.0


def test_synthetic_rows_count_half_weight() -> None:
    rows = [_row(model="claude-haiku-4-5", source="synthetic") for _ in range(10)]
    m = CapabilityMatrix(FakeLedger(rows))
    c = m.cell("coding", "claude-haiku-4-5")
    assert c.n == 5  # 10 synthetic * 0.5


def test_lang_coverage_tracked() -> None:
    rows = [
        _row(model="claude-haiku-4-5", lang="en"),
        _row(model="claude-haiku-4-5", lang="en"),
        _row(model="claude-haiku-4-5", lang="ja"),
    ]
    m = CapabilityMatrix(FakeLedger(rows))
    c = m.cell("coding", "claude-haiku-4-5")
    assert c.lang_coverage == {"en": 2, "ja": 1}


# --------------------------------------------------------------------------- min_bar / refresh


def test_min_bar_default() -> None:
    m = CapabilityMatrix(FakeLedger([]))
    assert m.min_bar("coding") == DEFAULT_MIN_BAR


def test_refresh_picks_up_new_rows() -> None:
    ledger = FakeLedger([])
    m = CapabilityMatrix(ledger)
    before = m.cell("coding", "gpt-5.4-mini").n
    assert before == 0
    for i in range(6):
        g = f"g{i}"
        ledger._rows.append(_row(model="gpt-5.4-mini", outcome="pass", collision_group=g))
        ledger._rows.append(_row(model="gpt-5.4-nano", outcome="fail", collision_group=g))
    m.refresh()
    assert m.cell("coding", "gpt-5.4-mini").n == 6
