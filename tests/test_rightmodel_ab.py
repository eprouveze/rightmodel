"""Tests for spine.rightmodel_ab — the A/B sampler (Workstream E2).

Unlike shadow mode, the A/B sampler runs BOTH the cost-min route AND the fusion
route on the SAME task and tags both ledger rows with a shared collision_group, so
the counterfactual (does fusion beat cost-min on the same card) is observable. NO
live LLM calls — both runners are injected stubs that record their own ledger rows.
A tmp ledger keeps host-local state clean. report() buckets by stakes, never across.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("rightmodel.rightmodel_ledger")

from rightmodel.rightmodel_ab import AbSampler  # noqa: E402
from rightmodel.rightmodel_ledger import RightModelLedger  # noqa: E402


@dataclass
class _FakeResult:
    model: str
    outcome: str


def _runner(ledger: RightModelLedger, model: str, outcome: str, bucket: str):
    def _run(card, *, collision_group):  # noqa: ARG001
        ledger.append(
            task_class="coding", model=model, outcome=outcome, verified=False,
            cost_usd=0.0, collision_group=collision_group,
            basis=f"stakes={bucket}",
        )
        return _FakeResult(model=model, outcome=outcome)
    return _run


def _sampler(tmp: Path, *, costmin, fusion, rate_env=None) -> AbSampler:
    ledger = RightModelLedger(path=tmp / "ledger.jsonl")
    return AbSampler(
        costmin_runner=_runner(ledger, *costmin),
        fusion_runner=_runner(ledger, *fusion),
        ledger=ledger,
        rng=random.Random(0),
    )


def test_sampled_card_yields_two_linked_rows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FUSION_AB_RATE", "1")  # always sample
    s = _sampler(
        tmp_path,
        costmin=("deepseek-v4-pro", "fail", "high"),
        fusion=("council:budget-trio", "pass", "high"),
    )
    run = s.run_ab({"id": "c1"})
    assert run is not None
    assert run.costmin.model == "deepseek-v4-pro"
    assert run.fusion.model == "council:budget-trio"

    rows = s.ledger.rows()
    assert len(rows) == 2
    # Both rows share the collision_group -> a real duel the matrix can fit.
    assert {r.collision_group for r in rows} == {run.collision_group}
    assert run.collision_group.startswith("ab-")


def test_report_buckets_by_stakes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FUSION_AB_RATE", "1")
    s = _sampler(
        tmp_path,
        costmin=("deepseek-v4-pro", "fail", "high"),
        fusion=("council:budget-trio", "pass", "high"),
    )
    s.run_ab({"id": "c1"})
    s.run_ab({"id": "c2"})

    rep = s.report()
    assert "high" in rep
    hi = rep["high"]
    assert hi["n_fusion"] == 2 and hi["n_costmin"] == 2
    assert hi["fusion_winrate"] == pytest.approx(1.0)
    assert hi["costmin_winrate"] == pytest.approx(0.0)
    assert hi["marginal_lift"] == pytest.approx(1.0)


def test_report_never_aggregates_across_buckets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FUSION_AB_RATE", "1")
    ledger = RightModelLedger(path=tmp_path / "ledger.jsonl")
    # high bucket: fusion wins; low bucket: fusion loses. Must stay separated.
    s_hi = AbSampler(
        costmin_runner=_runner(ledger, "deepseek-v4-pro", "fail", "high"),
        fusion_runner=_runner(ledger, "council:budget-trio", "pass", "high"),
        ledger=ledger, rng=random.Random(0),
    )
    s_lo = AbSampler(
        costmin_runner=_runner(ledger, "deepseek-v4-pro", "pass", "low"),
        fusion_runner=_runner(ledger, "council:budget-trio", "fail", "low"),
        ledger=ledger, rng=random.Random(0),
    )
    s_hi.run_ab({"id": "h1"})
    s_lo.run_ab({"id": "l1"})

    rep = AbSampler(ledger=ledger).report()
    assert set(rep) == {"high", "low"}
    assert rep["high"]["marginal_lift"] == pytest.approx(1.0)
    assert rep["low"]["marginal_lift"] == pytest.approx(-1.0)


def test_rate_zero_fires_no_extra_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FUSION_AB_RATE", "0")
    s = _sampler(
        tmp_path,
        costmin=("deepseek-v4-pro", "fail", "high"),
        fusion=("council:budget-trio", "pass", "high"),
    )
    assert s.should_sample() is False
    assert s.run_ab({"id": "c1"}) is None
    assert s.ledger.rows() == []


def test_sample_index_is_deterministic(monkeypatch) -> None:
    monkeypatch.setenv("FUSION_AB_RATE", "0.1")
    s = AbSampler()
    # 1-in-10 cadence: index 0, 10, 20 sampled; others not.
    sampled = [i for i in range(30) if s.should_sample(sample_index=i)]
    assert sampled == [0, 10, 20]
