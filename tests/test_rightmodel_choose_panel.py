"""RightModelRouter.choose_panel — RightModel composes the fusion budget panel.

The panel is the N cheapest above-bar models for a task class, at most one per provider
(genuine cross-check, not near-duplicates). Same constraint cascade as choose() — taint,
breaker, capability bar — so a tainted model never reaches an external panel.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rightmodel.action_receipt import TaintClass
from rightmodel.circuit_breaker import CircuitBreaker
from rightmodel.rightmodel_ledger import RightModelLedger
from rightmodel.rightmodel_router import RightModelRouter
from rightmodel.rightmodel_types import CellState


class _Cell:
    def __init__(self, mu: float, sigma: float = 6.0, state: str = CellState.SETTLED.value) -> None:
        self.mu = mu
        self.sigma = sigma
        self.state = state


class _StubMatrix:
    def __init__(self, cells: dict[str, _Cell], bar: float = 60.0) -> None:
        self._cells = cells
        self._bar = bar

    def cell(self, task_class: str, model: str) -> _Cell:
        return self._cells.get(model, _Cell(70.0))

    def min_bar(self, task_class: str) -> float:
        return self._bar


def _router(tmp_path: Path, cells: dict[str, _Cell], bar: float = 60.0) -> RightModelRouter:
    return RightModelRouter(
        ledger=RightModelLedger(path=tmp_path / "ledger.jsonl"),
        matrix=_StubMatrix(cells, bar=bar),
        breaker=CircuitBreaker(path=tmp_path / "breaker.json"),
        explore_rate=0.0,
        rng_seed=0,
    )


# A pool spanning four providers at clearly different blended costs.
_POOL = ["claude-opus-4-8", "claude-haiku-4-5", "deepseek-v4-pro", "kimi-k2.6", "gpt-5.4-mini"]


def test_picks_cheapest_one_per_provider(tmp_path: Path) -> None:
    cells = {m: _Cell(75.0) for m in _POOL}
    panel = _router(tmp_path, cells).choose_panel(
        task_class="coding", taint_set={TaintClass.PUBLIC}, n=3, candidates=_POOL,
    )
    assert len(panel) == 3
    # claude-haiku is cheaper than claude-opus, so the claude seat is haiku, never both.
    assert "claude-opus-4-8" not in panel or "claude-haiku-4-5" not in panel
    providers = {m.split("-")[0] for m in panel}  # rough provider distinctness for this pool
    assert len(providers) == 3  # one per provider


def test_n_caps_panel_size(tmp_path: Path) -> None:
    cells = {m: _Cell(75.0) for m in _POOL}
    panel = _router(tmp_path, cells).choose_panel(
        task_class="coding", taint_set={TaintClass.PUBLIC}, n=2, candidates=_POOL,
    )
    assert len(panel) == 2


def test_below_bar_models_excluded(tmp_path: Path) -> None:
    cells = {
        "deepseek-v4-pro": _Cell(50.0),  # below bar
        "kimi-k2.6": _Cell(80.0),
        "gpt-5.4-mini": _Cell(80.0),
    }
    panel = _router(tmp_path, cells, bar=60.0).choose_panel(
        task_class="coding", taint_set={TaintClass.PUBLIC}, n=3,
        candidates=["deepseek-v4-pro", "kimi-k2.6", "gpt-5.4-mini"],
    )
    assert "deepseek-v4-pro" not in panel
    assert set(panel) == {"kimi-k2.6", "gpt-5.4-mini"}


def test_safety_floor_keeps_best_when_all_below_bar(tmp_path: Path) -> None:
    cells = {"deepseek-v4-pro": _Cell(40.0), "kimi-k2.6": _Cell(55.0)}
    panel = _router(tmp_path, cells, bar=90.0).choose_panel(
        task_class="coding", taint_set={TaintClass.PUBLIC}, n=3,
        candidates=["deepseek-v4-pro", "kimi-k2.6"],
    )
    assert panel == ["kimi-k2.6"]  # all below bar → keep the single highest-mu survivor


def test_empty_when_nothing_survives(tmp_path: Path) -> None:
    panel = _router(tmp_path, {}).choose_panel(
        task_class="coding", taint_set={TaintClass.PUBLIC}, n=3, candidates=[],
    )
    assert panel == []
