"""Tests for spine.rightmodel_shadow — observe-only shadow mode (Tier 4).

Shadow mode makes NO model calls; it compares the router's would-pick to what
actually ran and records both. A tmp ledger + tmp shadow log keep host state clean.
"""

from __future__ import annotations

from pathlib import Path

from rightmodel.action_receipt import TaintClass
from rightmodel.circuit_breaker import CircuitBreaker
from rightmodel.rightmodel_matrix import CapabilityMatrix
from rightmodel.rightmodel_ledger import RightModelLedger
from rightmodel.rightmodel_router import RightModelRouter
from rightmodel.rightmodel_shadow import ShadowRecorder


def _recorder(tmp: Path) -> ShadowRecorder:
    ledger = RightModelLedger(path=tmp / "ledger.jsonl")
    router = RightModelRouter(
        ledger=ledger, matrix=CapabilityMatrix(ledger),
        breaker=CircuitBreaker(path=tmp / "breaker.json"), explore_rate=0.0, rng_seed=0,
    )
    return ShadowRecorder(router=router, path=tmp / "shadow.jsonl")


def test_observe_records_both_ledger_and_shadow(tmp_path: Path) -> None:
    rec = _recorder(tmp_path)
    # The default coding cost-min pick is deepseek-v4-pro; we say a DIFFERENT model ran.
    r = rec.observe(task_class="coding", taint_set={TaintClass.PUBLIC},
                    actual_model="claude-opus-4-8", actual_outcome="pass", cost_usd=0.02)
    assert r.would_model == "deepseek-v4-pro"
    assert r.actual_model == "claude-opus-4-8"
    assert r.agree is False  # would-route diverged from actual

    # The ACTUAL outcome was recorded as real production data (for the model that ran).
    rows = rec.router.ledger.rows()
    assert len(rows) == 1
    assert rows[0].model == "claude-opus-4-8" and rows[0].source == "production"
    assert rows[0].origin == "shadow"

    # And a shadow comparison row was logged.
    assert len(rec.records()) == 1


def test_agreement_when_actual_matches_would(tmp_path: Path) -> None:
    rec = _recorder(tmp_path)
    r = rec.observe(task_class="coding", taint_set={TaintClass.PUBLIC},
                    actual_model="deepseek-v4-pro", actual_outcome="pass")
    assert r.agree is True


def test_agreement_report_aggregates(tmp_path: Path) -> None:
    rec = _recorder(tmp_path)
    # 2 agree, 1 diverge on coding.
    rec.observe(task_class="coding", taint_set={TaintClass.PUBLIC},
                actual_model="deepseek-v4-pro", actual_outcome="pass")
    rec.observe(task_class="coding", taint_set={TaintClass.PUBLIC},
                actual_model="deepseek-v4-pro", actual_outcome="fail")
    rec.observe(task_class="coding", taint_set={TaintClass.PUBLIC},
                actual_model="gpt-5.4-mini", actual_outcome="pass")
    report = rec.agreement_report()
    assert report["total"] == 3
    assert abs(report["agreement_rate"] - 2 / 3) < 1e-9
    assert report["by_class"]["coding"]["n"] == 3
    assert len(report["divergences"]) == 1
    assert report["divergences"][0]["actual"] == "gpt-5.4-mini"


def test_empty_shadow_log_is_safe(tmp_path: Path) -> None:
    rec = _recorder(tmp_path)
    assert rec.records() == []
    report = rec.agreement_report()
    assert report["total"] == 0 and report["agreement_rate"] == 0.0
