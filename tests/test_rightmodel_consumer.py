"""Tests for spine.rightmodel_consumer — the route→run→review→record loop (Module 12).

Covers: a faked successful run records a production row; no reviewer leaves the row
unverified (the consumer's own bookkeeping); a distinct reviewer_model verifies it; live=False
skips both call and record; and the model=None path returns a clean fail. NO live
LLM calls — ``llm_runner.run`` is monkeypatched throughout. A tmp ledger + breaker
keep state off the host-local store.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rightmodel.rightmodel_ledger")
pytest.importorskip("rightmodel.rightmodel_router")
pytest.importorskip("rightmodel.llm_runner")

from rightmodel import llm_runner  # noqa: E402
from rightmodel import rightmodel_consumer  # noqa: E402
from rightmodel.circuit_breaker import CircuitBreaker  # noqa: E402
from rightmodel.rightmodel_consumer import DelegationResult, delegate  # noqa: E402
from rightmodel.rightmodel_ledger import RightModelLedger  # noqa: E402
from rightmodel.rightmodel_matrix import CapabilityMatrix  # noqa: E402
from rightmodel.rightmodel_router import RightModelRouter  # noqa: E402


def _router(tmp_path) -> RightModelRouter:
    ledger = RightModelLedger(tmp_path / "ledger.jsonl")
    matrix = CapabilityMatrix(ledger)
    breaker = CircuitBreaker(path=tmp_path / "breaker.json")
    return RightModelRouter(ledger=ledger, matrix=matrix, breaker=breaker, rng_seed=0)


def _fake_run(text: str = "an answer", *, ok: bool = True, cost: float = 0.0):
    def _run(model, prompt, **kw):  # noqa: ARG001
        return llm_runner.LLMResult(
            ok=ok, backend="claude", model=model, text=text if ok else None,
            cost_usd=cost, reason=None if ok else "boom",
        )
    return _run


# --------------------------------------------------------------------------- happy path


def test_faked_run_records_production_row(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(llm_runner, "run", _fake_run("hello", cost=0.002))
    router = _router(tmp_path)
    res = delegate("coding", "do a thing", set(), router=router, live=True)
    assert isinstance(res, DelegationResult)
    assert res.model is not None
    assert res.outcome == "pass"
    assert res.cost_usd == pytest.approx(0.002)
    rows = router.ledger.rows(source="production")
    assert len(rows) == 1
    assert rows[0].model == res.model
    assert rows[0].origin == "agent"
    assert rows[0].basis == "llm_runner"


def test_no_reviewer_leaves_row_unverified(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(llm_runner, "run", _fake_run())
    router = _router(tmp_path)
    res = delegate("coding", "task", set(), router=router, live=True)
    assert res.verified is False
    row = router.ledger.rows(source="production")[0]
    assert row.verified is False
    assert row.review_kind == "none"
    assert row.reviewer == ""


def test_distinct_reviewer_verifies(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(llm_runner, "run", _fake_run())
    router = _router(tmp_path)
    res = delegate(
        "coding", "task", set(), router=router, live=True,
        reviewer_model="claude-haiku-4-5",
    )
    # reviewer must differ from the chosen model for verification to stand
    assert res.model != "claude-haiku-4-5"
    assert res.verified is True
    row = router.ledger.rows(source="production")[0]
    assert row.verified is True
    assert row.review_kind == "ai-peer"
    assert row.reviewer == "claude-haiku-4-5"


# --------------------------------------------------------------------------- dry / cold paths


def test_live_false_skips_call_and_record(tmp_path, monkeypatch) -> None:
    called = {"n": 0}

    def _boom(model, prompt, **kw):  # noqa: ARG001
        called["n"] += 1
        raise AssertionError("llm_runner.run must not be called when live=False")

    monkeypatch.setattr(llm_runner, "run", _boom)
    router = _router(tmp_path)
    res = delegate("coding", "task", set(), router=router, live=False)
    assert called["n"] == 0
    assert res.model is not None  # routing still happened
    assert res.outcome == "fail"
    assert res.verified is False
    assert "dry run" in res.reason
    assert router.ledger.rows(source="production") == []


def test_model_none_path_returns_fail(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(llm_runner, "run", _fake_run())
    router = _router(tmp_path)
    # An empty candidate pool for an unknown class → cascade chooses no model.
    res = delegate("nonexistent_class", "task", set(), router=router, live=True)
    assert res.model is None
    assert res.outcome == "fail"
    assert res.verified is False
    assert res.cost_usd == 0.0
    assert router.ledger.rows(source="production") == []


def test_failed_run_records_fail_unverified(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(llm_runner, "run", _fake_run(ok=False))
    router = _router(tmp_path)
    res = delegate(
        "coding", "task", set(), router=router, live=True,
        reviewer_model="claude-haiku-4-5",
    )
    assert res.outcome == "fail"
    assert res.verified is False  # a failed run is never peer-verified
    row = router.ledger.rows(source="production")[0]
    assert row.outcome == "fail"
    assert row.verified is False


# --------------------------------------------------------------------------- execution mode


def test_execution_mode_threaded_to_router_and_run(tmp_path, monkeypatch) -> None:
    seen = {}

    def _run(model, prompt, **kw):  # noqa: ARG001
        seen["execution_mode"] = kw.get("execution_mode")
        seen["add_dirs"] = kw.get("add_dirs")
        return llm_runner.LLMResult(ok=True, backend="agy", model=model, text="done")

    monkeypatch.setattr(llm_runner, "run", _run)
    router = _router(tmp_path)
    res = delegate(
        "coding", "build the thing", set(), router=router, live=True,
        execution_mode="agentic", add_dirs=["spine", "docs"],
    )
    # agentic routing drops completion-only models; a real agentic model is chosen.
    assert res.model is not None
    assert seen["execution_mode"] == "agentic"
    assert seen["add_dirs"] == ["spine", "docs"]
    assert res.decision.execution_mode == "agentic"
    assert res.decision.harness in ("agy", "codex", "claude_cli")


def test_script_mode_short_circuits_without_routing(tmp_path, monkeypatch) -> None:
    def _boom(*a, **k):  # noqa: ARG001
        raise AssertionError("script mode must not call llm_runner.run")

    monkeypatch.setattr(llm_runner, "run", _boom)
    router = _router(tmp_path)
    res = delegate(
        "coding", "run a script", set(), router=router, live=True,
        execution_mode="script:scripts/build.sh",
    )
    assert res.model is None
    assert res.outcome == "pass"
    assert "script" in res.reason
    assert res.decision.execution_mode == "script:scripts/build.sh"
    assert router.ledger.rows(source="production") == []
