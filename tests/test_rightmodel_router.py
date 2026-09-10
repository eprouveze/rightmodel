"""Tests for spine.rightmodel_router — the cascade orchestrator (Module 5).

Determinism via rng_seed; tmp ledger + breaker. To stay robust against the
exact internal fit of the sibling matrix, these tests drive the cascade through
a lightweight stub matrix that returns hand-set Cells, while exercising the REAL
ledger, breaker, taint check, and price table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rightmodel.action_receipt import TaintClass
from rightmodel.circuit_breaker import CircuitBreaker
from rightmodel.rightmodel_ledger import RightModelLedger
from rightmodel.rightmodel_router import RightModelRouter, RouteDecision
from rightmodel.rightmodel_types import CellState, Outcome


# --------------------------------------------------------------------------- stub matrix


class _Cell:
    def __init__(self, mu: float, sigma: float, state: str) -> None:
        self.mu = mu
        self.sigma = sigma
        self.state = state


class _StubMatrix:
    """Hand-set capability cells so cascade behaviour is deterministic."""

    def __init__(self, cells: dict[str, _Cell], bar: float = 60.0) -> None:
        self._cells = cells
        self._bar = bar

    def cell(self, task_class: str, model: str) -> _Cell:
        return self._cells.get(model, _Cell(50.0, 30.0, CellState.COLD.value))

    def min_bar(self, task_class: str) -> float:
        return self._bar


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def ledger(tmp_path: Path) -> RightModelLedger:
    return RightModelLedger(path=tmp_path / "ledger.jsonl")


@pytest.fixture
def breaker(tmp_path: Path) -> CircuitBreaker:
    return CircuitBreaker(path=tmp_path / "breaker.json")


def _router(ledger, breaker, cells, *, bar=60.0, seed=0, explore_rate=0.10):
    return RightModelRouter(
        ledger=ledger,
        matrix=_StubMatrix(cells, bar=bar),
        breaker=breaker,
        explore_rate=explore_rate,
        rng_seed=seed,
    )


# --------------------------------------------------------------------------- happy path


def test_cost_min_among_qualified(ledger, breaker) -> None:
    cells = {
        "claude-haiku-4-5": _Cell(80.0, 4.0, CellState.SETTLED.value),
        "gpt-5.5": _Cell(85.0, 4.0, CellState.SETTLED.value),
    }
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "gpt-5.5"],
    )
    assert d.model == "claude-haiku-4-5"  # cheaper, mu gap <=15
    assert d.exploration is False
    assert d.rejected == {}


def test_bounded_quality_premium_prefers_pricier_settled(ledger, breaker) -> None:
    cells = {
        "claude-haiku-4-5": _Cell(60.0, 4.0, CellState.SETTLED.value),
        "gpt-5.5": _Cell(90.0, 4.0, CellState.SETTLED.value),  # +30 > 15, SETTLED
    }
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "gpt-5.5"],
    )
    assert d.model == "gpt-5.5"
    assert "quality-premium" in d.reason


def test_premium_ignored_when_not_settled(ledger, breaker) -> None:
    cells = {
        "claude-haiku-4-5": _Cell(60.0, 4.0, CellState.SETTLED.value),
        "gpt-5.5": _Cell(90.0, 4.0, CellState.WARMING.value),  # big mu but not SETTLED
    }
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "gpt-5.5"],
    )
    assert d.model == "claude-haiku-4-5"


# --------------------------------------------------------------------------- taint stage


def test_taint_drops_disallowed_provider(ledger, breaker) -> None:
    cells = {
        "gemini-2.5-flash": _Cell(85.0, 4.0, CellState.SETTLED.value),
        "claude-haiku-4-5": _Cell(80.0, 4.0, CellState.SETTLED.value),
    }
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.CUSTOMER_PII},  # gemini disallowed
        candidates=["gemini-2.5-flash", "claude-haiku-4-5"],
    )
    assert d.model == "claude-haiku-4-5"
    assert "gemini-2.5-flash" in d.rejected
    assert d.rejected["gemini-2.5-flash"].startswith("taint")


def test_all_taint_blocked_returns_none(ledger, breaker) -> None:
    cells = {"gemini-2.5-flash": _Cell(85.0, 4.0, CellState.SETTLED.value)}
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.AUTH_SECRETS},
        candidates=["gemini-2.5-flash"],
    )
    assert d.model is None
    assert "taint" in d.reason


# --------------------------------------------------------------------------- breaker stage


def test_breaker_open_drops_provider(ledger, breaker) -> None:
    cells = {
        "gpt-5.5": _Cell(90.0, 4.0, CellState.SETTLED.value),
        "claude-haiku-4-5": _Cell(80.0, 4.0, CellState.SETTLED.value),
    }
    breaker.record_failure("openai", kind="payment")  # OPEN immediately
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["gpt-5.5", "claude-haiku-4-5"],
    )
    assert d.model == "claude-haiku-4-5"
    assert "gpt-5.5" in d.rejected
    assert d.rejected["gpt-5.5"].startswith("breaker")


# --------------------------------------------------------------------------- capability bar + safety


def test_below_bar_safety_keeps_highest_mu(ledger, breaker) -> None:
    cells = {
        "claude-haiku-4-5": _Cell(40.0, 4.0, CellState.SETTLED.value),
        "gpt-5.4-mini": _Cell(55.0, 4.0, CellState.SETTLED.value),
    }
    # bar 60 empties the set → safety floor keeps the single highest-mu survivor
    r = _router(ledger, breaker, cells, bar=60.0, explore_rate=0.0)
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "gpt-5.4-mini"],
    )
    assert d.model == "gpt-5.4-mini"
    assert "claude-haiku-4-5" in d.rejected


# --------------------------------------------------------------------------- exploration / cold-start


def test_exploration_picks_highest_sigma(ledger, breaker) -> None:
    cells = {
        "claude-haiku-4-5": _Cell(70.0, 3.0, CellState.SETTLED.value),
        "kimi-k2.6": _Cell(65.0, 28.0, CellState.COLD.value),  # high sigma, COLD
    }
    # explore_rate=1.0 forces the exploration branch deterministically
    r = _router(ledger, breaker, cells, explore_rate=1.0, seed=1)
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "kimi-k2.6"],
    )
    assert d.model == "kimi-k2.6"
    assert d.exploration is True
    assert "exploration" in d.reason


def test_no_exploration_when_rate_zero(ledger, breaker) -> None:
    cells = {
        "claude-haiku-4-5": _Cell(70.0, 3.0, CellState.SETTLED.value),
        "kimi-k2.6": _Cell(65.0, 28.0, CellState.COLD.value),
    }
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "kimi-k2.6"],
    )
    assert d.exploration is False


def test_determinism_same_seed_same_choice(ledger, breaker) -> None:
    cells = {
        "claude-haiku-4-5": _Cell(70.0, 20.0, CellState.COLD.value),
        "kimi-k2.6": _Cell(65.0, 28.0, CellState.COLD.value),
    }
    a = _router(ledger, breaker, cells, explore_rate=0.5, seed=42).choose(
        task_class="coding", taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "kimi-k2.6"],
    )
    b = _router(ledger, breaker, cells, explore_rate=0.5, seed=42).choose(
        task_class="coding", taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "kimi-k2.6"],
    )
    assert a.model == b.model
    assert a.exploration == b.exploration


def test_empty_candidates_returns_none(ledger, breaker) -> None:
    r = _router(ledger, breaker, {}, explore_rate=0.0)
    d = r.choose(task_class="coding", taint_set={TaintClass.PUBLIC}, candidates=[])
    assert d.model is None


# --------------------------------------------------------------------------- execution-mode eligibility


def test_completion_mode_is_a_noop_eligibility(ledger, breaker) -> None:
    # default completion: every candidate eligible → routing unchanged.
    cells = {
        "claude-haiku-4-5": _Cell(80.0, 4.0, CellState.SETTLED.value),
        "deepseek-v4-pro": _Cell(85.0, 4.0, CellState.SETTLED.value),
    }
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="coding", taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "deepseek-v4-pro"],
    )
    assert d.model == "deepseek-v4-pro"  # cheaper, both eligible
    assert d.execution_mode == "completion"
    assert d.harness == "api"


def test_agentic_drops_completion_only_candidates(ledger, breaker) -> None:
    # deepseek has no agentic harness → dropped; gemini routes to the agy harness.
    cells = {
        "deepseek-v4-pro": _Cell(88.0, 4.0, CellState.SETTLED.value),
        "gemini-3.1-pro-preview": _Cell(80.0, 4.0, CellState.SETTLED.value),
    }
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="coding", taint_set={TaintClass.PUBLIC},
        candidates=["deepseek-v4-pro", "gemini-3.1-pro-preview"],
        execution_mode="agentic",
    )
    assert d.model == "gemini-3.1-pro-preview"
    assert d.harness == "agy"
    assert "deepseek-v4-pro" in d.rejected
    assert "execution-mode" in d.rejected["deepseek-v4-pro"]


def test_agentic_no_eligible_candidate_returns_none(ledger, breaker) -> None:
    cells = {
        "deepseek-v4-pro": _Cell(88.0, 4.0, CellState.SETTLED.value),
        "sonar-pro": _Cell(80.0, 4.0, CellState.SETTLED.value),
    }
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="coding", taint_set={TaintClass.PUBLIC},
        candidates=["deepseek-v4-pro", "sonar-pro"],
        execution_mode="agentic",
    )
    assert d.model is None
    assert "execution_mode=agentic" in d.reason


def test_skill_mode_routes_to_agentic_harness(ledger, breaker) -> None:
    cells = {"gemini-3.1-pro-preview": _Cell(80.0, 4.0, CellState.SETTLED.value)}
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="design", taint_set={TaintClass.PUBLIC},
        candidates=["gemini-3.1-pro-preview"],
        execution_mode="skill:geo",
    )
    assert d.model == "gemini-3.1-pro-preview" and d.harness == "agy"


# --------------------------------------------------------------------------- record_result


def test_record_result_pass_logs_and_succeeds_breaker(ledger, breaker) -> None:
    cells = {"claude-haiku-4-5": _Cell(80.0, 4.0, CellState.SETTLED.value)}
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="coding", taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5"],
    )
    r.record_result(d, outcome=Outcome.PASS.value, verified=True, cost_usd=0.002)
    rows = ledger.rows(model="claude-haiku-4-5")
    assert len(rows) == 1
    assert rows[0].outcome == Outcome.PASS.value
    assert breaker.is_available("claude")


def test_record_result_fail_trips_breaker_failure(ledger, breaker) -> None:
    cells = {"gpt-5.5": _Cell(90.0, 4.0, CellState.SETTLED.value)}
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="coding", taint_set={TaintClass.PUBLIC}, candidates=["gpt-5.5"],
    )
    for _ in range(3):
        r.record_result(d, outcome=Outcome.FAIL.value, verified=True, cost_usd=0.01)
    snap = breaker.snapshot()
    assert snap["openai"].consecutive_failures >= 3


def test_record_result_none_model_is_noop(ledger, breaker) -> None:
    d = RouteDecision(model=None, task_class="coding", reason="none")
    r = _router(ledger, breaker, {}, explore_rate=0.0)
    r.record_result(d, outcome=Outcome.FAIL.value, verified=False, cost_usd=0.0)
    assert ledger.rows() == []


# --------------------------------------------------------------------------- cold-start UCB bar


def test_cold_cell_survives_bar_via_ucb(ledger, breaker) -> None:
    # a COLD cell below the bar by true mu but whose optimistic bound (mu +
    # k*sigma) clears the bar survives; a SETTLED sub-bar cell is rejected.
    cells = {
        "claude-haiku-4-5": _Cell(55.0, 25.0, CellState.COLD.value),  # ucb 80 >= 60
        "gpt-5.5": _Cell(55.0, 4.0, CellState.SETTLED.value),         # ucb 55 < 60
    }
    r = _router(ledger, breaker, cells, bar=60.0, explore_rate=0.0)
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "gpt-5.5"],
    )
    assert d.model == "claude-haiku-4-5"  # cold cell admitted via UCB, sole survivor
    assert "gpt-5.5" in d.rejected
    assert "below-bar" in d.rejected["gpt-5.5"]


def test_settled_cell_below_bar_still_rejected(ledger, breaker) -> None:
    # control: a SETTLED cell gets no sigma bonus, so sub-bar stays rejected.
    cells = {
        "claude-haiku-4-5": _Cell(80.0, 4.0, CellState.SETTLED.value),
        "gpt-5.5": _Cell(55.0, 4.0, CellState.SETTLED.value),
    }
    r = _router(ledger, breaker, cells, bar=60.0, explore_rate=0.0)
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "gpt-5.5"],
    )
    assert d.model == "claude-haiku-4-5"
    assert "gpt-5.5" in d.rejected
    assert "below-bar" in d.rejected["gpt-5.5"]


# --------------------------------------------------------------------------- role & cross-family exclusion


def test_evaluator_role_excludes_same_family_by_name(ledger, breaker) -> None:
    cells = {
        "claude-haiku-4-5": _Cell(90.0, 4.0, CellState.SETTLED.value),
        "gpt-5.5": _Cell(88.0, 4.0, CellState.SETTLED.value),
    }
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "gpt-5.5"],
        role="evaluator",
        exclude_family="claude",
    )
    assert d.model == "gpt-5.5"
    assert d.role == "evaluator"
    assert "claude-haiku-4-5" in d.rejected
    assert "cross-family" in d.rejected["claude-haiku-4-5"]


def test_evaluator_role_infers_exclusion_from_executor_model(ledger, breaker) -> None:
    cells = {
        "claude-haiku-4-5": _Cell(90.0, 4.0, CellState.SETTLED.value),
        "gemini-3.1-pro-preview": _Cell(85.0, 4.0, CellState.SETTLED.value),
    }
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "gemini-3.1-pro-preview"],
        role="evaluator",
        executor_model="claude-opus-4-8",
    )
    assert d.model == "gemini-3.1-pro-preview"
    assert d.role == "evaluator"
    assert "claude-haiku-4-5" in d.rejected


def test_evaluator_returns_none_when_no_cross_family_survives(ledger, breaker) -> None:
    cells = {
        "claude-opus-4-8": _Cell(90.0, 4.0, CellState.SETTLED.value),
        "claude-haiku-4-5": _Cell(70.0, 4.0, CellState.SETTLED.value),
    }
    r = _router(ledger, breaker, cells, explore_rate=0.0)
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-opus-4-8", "claude-haiku-4-5"],
        role="evaluator",
        exclude_family="claude",
    )
    assert d.model is None
    assert d.role == "evaluator"
    assert "cross-family" in d.reason
