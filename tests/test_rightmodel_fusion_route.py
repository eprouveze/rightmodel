"""Tests for the Stage-0 stakes short-circuit in spine.rightmodel_router.

High-stakes callers escalate to the fusion panel; absent/None stakes route via
the normal cost-min cascade; an AUTH_SECRETS-tainted high-stakes task falls
through to in-house cost-min (the external panel is host-local forbidden). Every
routed call writes a kind="route" StakesDecision. `rightmodel_home()` is
monkeypatched to a tmp dir so the stakes-decisions sidecar never touches host state.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rightmodel import stakes_score
from rightmodel.action_receipt import TaintClass
from rightmodel.circuit_breaker import CircuitBreaker
from rightmodel.rightmodel_ledger import RightModelLedger
from rightmodel.rightmodel_router import FUSION_MODEL, RightModelRouter
from rightmodel.rightmodel_types import CellState


class _Cell:
    def __init__(self, mu: float, sigma: float, state: str) -> None:
        self.mu = mu
        self.sigma = sigma
        self.state = state


class _StubMatrix:
    def __init__(self, cells: dict[str, _Cell], bar: float = 60.0) -> None:
        self._cells = cells
        self._bar = bar

    def cell(self, task_class: str, model: str) -> _Cell:
        return self._cells.get(model, _Cell(50.0, 30.0, CellState.COLD.value))

    def min_bar(self, task_class: str) -> float:
        return self._bar


@pytest.fixture(autouse=True)
def _tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(stakes_score, "rightmodel_home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def ledger(tmp_path: Path) -> RightModelLedger:
    return RightModelLedger(path=tmp_path / "ledger.jsonl")


@pytest.fixture
def breaker(tmp_path: Path) -> CircuitBreaker:
    return CircuitBreaker(path=tmp_path / "breaker.json")


def _decisions(home: Path) -> list[dict]:
    path = home / "stakes-decisions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _router(ledger, breaker, cells, *, bar=60.0, seed=0, explore_rate=0.0):
    return RightModelRouter(
        ledger=ledger,
        matrix=_StubMatrix(cells, bar=bar),
        breaker=breaker,
        explore_rate=explore_rate,
        rng_seed=seed,
    )


def _cells() -> dict[str, _Cell]:
    return {
        "claude-haiku-4-5": _Cell(80.0, 4.0, CellState.SETTLED.value),
        "gpt-5.5": _Cell(85.0, 4.0, CellState.SETTLED.value),
    }


def test_high_stakes_escalates_to_fusion(ledger, breaker) -> None:
    r = _router(ledger, breaker, _cells())
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "gpt-5.5"],
        run_id="run-high",
        stakes="high",
    )
    assert d.model == FUSION_MODEL
    assert d.reason == "stakes-escalation"


def test_none_stakes_unchanged_cost_min(ledger, breaker, _tmp_home: Path) -> None:
    cells = _cells()
    baseline = _router(ledger, breaker, cells).choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "gpt-5.5"],
        run_id="run-baseline",
    )
    explicit_low = _router(ledger, breaker, cells).choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "gpt-5.5"],
        run_id="run-low",
        stakes="low",
    )
    assert baseline.model != FUSION_MODEL
    assert baseline.model == explicit_low.model


def test_auth_secrets_high_stakes_does_not_escalate(ledger, breaker) -> None:
    r = _router(ledger, breaker, _cells())
    d = r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC, TaintClass.AUTH_SECRETS},
        candidates=["claude-haiku-4-5", "gpt-5.5"],
        run_id="run-secrets",
        stakes="high",
    )
    assert d.model != FUSION_MODEL


def test_every_routed_call_writes_route_decision(ledger, breaker, _tmp_home: Path) -> None:
    r = _router(ledger, breaker, _cells())
    r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "gpt-5.5"],
        run_id="run-1",
        stakes="high",
    )
    r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC},
        candidates=["claude-haiku-4-5", "gpt-5.5"],
        run_id="run-2",
    )
    routes = [d for d in _decisions(_tmp_home) if d["kind"] == "route"]
    assert len(routes) == 2
    assert {d["action"] for d in routes} == {"escalated", "cost-min"}
    assert all(d["run_id"] in {"run-1", "run-2"} for d in routes)


def test_auth_secrets_fallthrough_writes_cost_min_route(ledger, breaker, _tmp_home: Path) -> None:
    r = _router(ledger, breaker, _cells())
    r.choose(
        task_class="coding",
        taint_set={TaintClass.PUBLIC, TaintClass.AUTH_SECRETS},
        candidates=["claude-haiku-4-5", "gpt-5.5"],
        run_id="run-secrets",
        stakes="high",
    )
    routes = [d for d in _decisions(_tmp_home) if d["kind"] == "route"]
    assert len(routes) == 1
    assert routes[0]["action"] == "cost-min"
    assert routes[0]["resolved_stakes"] == "high"
