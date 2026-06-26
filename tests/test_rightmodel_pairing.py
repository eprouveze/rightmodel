"""Tests for spine.rightmodel_pairing — deliberate duels (Module 11).

Covers the info-gain heuristic (overlapping high-σ cells score, SETTLED-confident
cells score ~0), the should_pair cost gate, and propose_pairs ranking/cap on both
a cold-start (empty ledger → priors) and a settled ledger.

Uses the same in-memory FakeLedger double as the matrix test (Module 2 read
surface). No live LLM calls. Skipped whole if sibling modules 0/1/4 are absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rightmodel.rightmodel_ledger")
pytest.importorskip("rightmodel.model_prices")
pytest.importorskip("rightmodel.rightmodel_matrix")

from rightmodel.rightmodel_ledger import LedgerRow  # noqa: E402
from rightmodel.rightmodel_matrix import CapabilityMatrix, Cell  # noqa: E402
from rightmodel.rightmodel_pairing import (  # noqa: E402
    PairProposal,
    expected_sigma_reduction,
    propose_pairs,
    should_pair,
)
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


def _cell(mu: float, sigma: float, *, n: int = 0, state: str = "cold") -> Cell:
    return Cell(
        task_class="coding",
        model="m",
        mu=mu,
        sigma=sigma,
        n=n,
        state=state,
        version_hash="",
    )


# --------------------------------------------------------------------------- heuristic


def test_overlapping_high_sigma_cells_have_positive_gain() -> None:
    a = _cell(70.0, 20.0)
    b = _cell(72.0, 20.0)
    assert expected_sigma_reduction(a, b) > 0.0


def test_settled_confident_cells_have_near_zero_gain() -> None:
    # Tight, far-apart CIs: no overlap → no discrimination value.
    a = _cell(90.0, 1.0, n=40, state="settled")
    b = _cell(60.0, 1.0, n=40, state="settled")
    assert expected_sigma_reduction(a, b) == pytest.approx(0.0)


def test_disjoint_cells_outscored_by_overlapping_cells() -> None:
    overlap = expected_sigma_reduction(_cell(70.0, 15.0), _cell(72.0, 15.0))
    disjoint = expected_sigma_reduction(_cell(95.0, 2.0), _cell(40.0, 2.0))
    assert overlap > disjoint


# --------------------------------------------------------------------------- should_pair


def test_should_pair_requires_gain_to_beat_cost() -> None:
    a = _cell(70.0, 20.0)
    b = _cell(72.0, 20.0)
    cheap = should_pair(a, b, second_inference_cost=0.001)
    pricey = should_pair(a, b, second_inference_cost=10_000.0)
    assert cheap is True
    assert pricey is False


def test_should_pair_false_for_settled_pair() -> None:
    a = _cell(90.0, 1.0, n=40, state="settled")
    b = _cell(60.0, 1.0, n=40, state="settled")
    assert should_pair(a, b, second_inference_cost=0.001) is False


# --------------------------------------------------------------------------- propose_pairs


def test_cold_start_proposes_from_overlapping_priors() -> None:
    m = CapabilityMatrix(FakeLedger([]))
    pairs = propose_pairs(m, "coding", max_pairs=3)
    assert pairs, "cold-start should still propose deliberate duels from priors"
    assert all(isinstance(p, PairProposal) for p in pairs)
    assert len(pairs) <= 3
    gains = [p.expected_info_gain for p in pairs]
    assert gains == sorted(gains, reverse=True)


def test_high_sigma_overlap_pair_proposed_two_settled_not() -> None:
    # Two high-σ overlapping cells (cold candidates) → a pair; two SETTLED-confident
    # cells (lots of decisive duels, tight + separated CIs) → no pair between them.
    rows: list[LedgerRow] = []
    for i in range(40):
        g = f"g{i}"
        rows.append(_row(model="claude-opus-4-8", outcome="pass", collision_group=g, difficulty=0.9))
        rows.append(_row(model="gpt-5.4-nano", outcome="fail", collision_group=g, difficulty=0.9))
    m = CapabilityMatrix(FakeLedger(rows))

    settled_a = m.cell("coding", "claude-opus-4-8")
    settled_b = m.cell("coding", "gpt-5.4-nano")
    assert settled_a.state == "settled" and settled_b.state == "settled"
    # The two settled, decisively-separated cells must not be paired.
    assert should_pair(
        settled_a, settled_b, second_inference_cost=blended_b_cost()
    ) is False

    # But two cold high-σ candidates that the matrix never saw collide DO pair.
    cold_a = m.cell("coding", "gpt-5.3-codex")
    cold_b = m.cell("coding", "claude-sonnet-4-6")
    assert cold_a.state == "cold" and cold_b.state == "cold"
    assert should_pair(cold_a, cold_b, second_inference_cost=0.5) is True

    # And propose_pairs surfaces a cold pair while never emitting the settled duel.
    pairs = propose_pairs(m, "coding", max_pairs=5)
    pair_sets = {frozenset((p.model_a, p.model_b)) for p in pairs}
    assert frozenset(("claude-opus-4-8", "gpt-5.4-nano")) not in pair_sets
    assert any("gpt-5.4-nano" not in fs for fs in pair_sets)


def test_max_pairs_cap_respected() -> None:
    m = CapabilityMatrix(FakeLedger([]))
    pairs = propose_pairs(m, "coding", max_pairs=1)
    assert len(pairs) <= 1


def test_custom_cost_fn_can_veto_all_pairs() -> None:
    m = CapabilityMatrix(FakeLedger([]))
    pairs = propose_pairs(m, "coding", second_inference_cost_fn=lambda _m: 1e9)
    assert pairs == []


def blended_b_cost() -> float:
    from rightmodel.model_prices import blended_cost

    return blended_cost("gpt-5.4-nano")
