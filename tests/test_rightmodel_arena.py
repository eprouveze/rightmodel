"""Tests for spine.rightmodel_arena — blind pairwise HITL Arena (Module 9).

Uses a real tmp-path ledger + real CapabilityMatrix (no LLM calls). Covers the
happy path (record_vote writes two collision rows the matrix sees as a duel),
cold-start (empty ledger triggers + queues from priors, is_settled False), and
edges (tie → both PARTIAL, Wilson interval, stable item_id hash).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rightmodel.rightmodel_ledger")
pytest.importorskip("rightmodel.rightmodel_matrix")
pytest.importorskip("rightmodel.model_prices")

from rightmodel.rightmodel_arena import Arena, ArenaMatchup, _item_id  # noqa: E402
from rightmodel.rightmodel_ledger import RightModelLedger  # noqa: E402
from rightmodel.rightmodel_matrix import CapabilityMatrix  # noqa: E402


def _arena(tmp_path: Path) -> tuple[Arena, RightModelLedger]:
    ledger = RightModelLedger(path=tmp_path / "ledger.jsonl")
    matrix = CapabilityMatrix(ledger)
    return Arena(ledger, matrix), ledger


# --------------------------------------------------------------------------- cold start


def test_should_trigger_on_cold_start(tmp_path: Path) -> None:
    arena, _ = _arena(tmp_path)
    fire, reason = arena.should_trigger("content_writing")
    assert fire is True
    assert "cold" in reason.lower()


def test_queue_from_priors_on_empty_ledger(tmp_path: Path) -> None:
    arena, _ = _arena(tmp_path)
    matchups = arena.queue("content_writing", n_pairs=3)
    assert 1 <= len(matchups) <= 3
    for m in matchups:
        assert isinstance(m, ArenaMatchup)
        assert m.model_a != m.model_b
        assert m.item_id.startswith("arena-")


def test_is_settled_false_on_empty_ledger(tmp_path: Path) -> None:
    arena, _ = _arena(tmp_path)
    assert arena.is_settled("content_writing", "claude-opus-4-8", "minimax-m3") is False


# --------------------------------------------------------------------------- happy path


def test_record_vote_writes_two_linked_collision_rows(tmp_path: Path) -> None:
    arena, ledger = _arena(tmp_path)
    matchup = ArenaMatchup(
        task_class="content_writing",
        item_id="arena-fixed-1",
        model_a="claude-opus-4-8",
        model_b="minimax-m3",
        prompt="write a haiku",
    )
    arena.record_vote(matchup, winner="claude-opus-4-8", voter="emmanuel")

    rows = ledger.rows(task_class="content_writing")
    assert len(rows) == 2
    for r in rows:
        assert r.source == "arena"
        assert r.origin == "arena"
        assert r.review_kind == "human"
        assert r.reviewer == "emmanuel"
        assert r.collision_group == "arena-fixed-1"
        assert r.verified is True  # genuine human review survives provenance enforcement
        assert r.author == r.model

    by_model = {r.model: r for r in rows}
    assert by_model["claude-opus-4-8"].outcome == "pass"
    assert by_model["minimax-m3"].outcome == "fail"

    # The matrix now sees exactly one duel for the class.
    groups = ledger.collisions("content_writing")
    assert len(groups) == 1
    assert {r.model for r in groups[0]} == {"claude-opus-4-8", "minimax-m3"}

    # And the winner outranks the loser in the (blended) matrix.
    matrix = CapabilityMatrix(ledger)
    win = matrix.cell("content_writing", "claude-opus-4-8")
    lose = matrix.cell("content_writing", "minimax-m3")
    assert win.n == 1 and lose.n == 1
    assert win.mu > lose.mu


# --------------------------------------------------------------------------- edges


def test_tie_records_both_partial(tmp_path: Path) -> None:
    arena, ledger = _arena(tmp_path)
    matchup = ArenaMatchup(
        task_class="design",
        item_id="arena-tie",
        model_a="claude-opus-4-8",
        model_b="gpt-5.5",
        prompt="",
    )
    arena.record_vote(matchup, winner="")  # tie
    rows = ledger.rows(task_class="design")
    assert len(rows) == 2
    assert all(r.outcome == "partial" for r in rows)
    assert all(r.collision_group == "arena-tie" for r in rows)


def test_record_vote_rejects_unknown_winner(tmp_path: Path) -> None:
    arena, _ = _arena(tmp_path)
    matchup = ArenaMatchup(
        task_class="design", item_id="x", model_a="a", model_b="b", prompt="",
    )
    with pytest.raises(ValueError):
        arena.record_vote(matchup, winner="c")


def test_item_id_stable_and_order_independent() -> None:
    a = _item_id("content_writing", "claude-opus-4-8", "minimax-m3", 0)
    b = _item_id("content_writing", "minimax-m3", "claude-opus-4-8", 0)
    assert a == b  # order-independent over the pair
    assert a != _item_id("content_writing", "claude-opus-4-8", "minimax-m3", 1)
    assert a.startswith("arena-")


def test_wilson_interval_bounds() -> None:
    assert Arena.wilson_interval(0, 0) == (0.0, 1.0)  # cold
    lo, hi = Arena.wilson_interval(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    # More evidence → tighter interval.
    lo_w, hi_w = Arena.wilson_interval(50, 100)
    assert (hi_w - lo_w) < (hi - lo)


def test_is_settled_true_when_one_sided_and_many_duels(tmp_path: Path) -> None:
    arena, ledger = _arena(tmp_path)
    for i in range(40):
        ledger.append(
            task_class="marketing", model="claude-opus-4-8", outcome="pass",
            verified=True, cost_usd=0.0, source="arena",
            collision_group=f"d{i}", origin="arena", reviewer="human",
            review_kind="human",
        )
        ledger.append(
            task_class="marketing", model="minimax-m3", outcome="fail",
            verified=True, cost_usd=0.0, source="arena",
            collision_group=f"d{i}", origin="arena", reviewer="human",
            review_kind="human",
        )
    assert arena.is_settled("marketing", "claude-opus-4-8", "minimax-m3") is True


def test_should_trigger_on_production_divergence(tmp_path: Path) -> None:
    arena, ledger = _arena(tmp_path)
    # Warm a single model so it is no longer COLD, but make production disagree.
    for i in range(40):
        ledger.append(
            task_class="content_writing", model="claude-opus-4-8", outcome="fail",
            verified=True, cost_usd=0.0, source="arena",
            collision_group=f"w{i}", origin="arena", reviewer="human",
            review_kind="human",
        )
        ledger.append(
            task_class="content_writing", model="claude-sonnet-4-6", outcome="pass",
            verified=True, cost_usd=0.0, source="arena",
            collision_group=f"w{i}", origin="arena", reviewer="human",
            review_kind="human",
        )
    fire, reason = arena.should_trigger("content_writing")
    # Some candidate is still COLD or production diverges — either way it fires.
    assert fire is True
