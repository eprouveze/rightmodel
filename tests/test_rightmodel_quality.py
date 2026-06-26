"""Tests for rightmodel_quality + rightmodel_feedback — the quality signal.

No live LLM calls: the judge path monkeypatches llm_runner.run.
"""

from __future__ import annotations

from pathlib import Path

from rightmodel import rightmodel_quality as q
from rightmodel import rightmodel_feedback as fb
from rightmodel.rightmodel_ledger import RightModelLedger


# ---------------------------------------------------------------- auto-grade path

def test_grade_uses_deterministic_grader() -> None:
    v = q.grade("5", task_class="coding", expected="5", grader="exact")
    assert v.outcome == "pass" and v.score == 100.0 and v.method == "grader" and v.verified
    v2 = q.grade("6", task_class="coding", expected="5", grader="exact")
    assert v2.outcome == "fail" and v2.score == 0.0


# ------------------------------------------------------------------- judge path

class _FakeParsed:
    def __init__(self, score, verdict, rationale=""):
        self.score, self.verdict, self.rationale = score, verdict, rationale


class _FakeRes:
    def __init__(self, ok=True, parsed=None, reason=None):
        self.ok, self.parsed, self.reason = ok, parsed, reason


def test_judge_independent_when_distinct_model(monkeypatch) -> None:
    monkeypatch.setattr(q.llm_runner, "run",
                        lambda *a, **k: _FakeRes(parsed=_FakeParsed(85, "pass", "solid")))
    v = q.judge("content_writing", "write x", "some output",
                judge_model="claude-sonnet-4-6", producer_model="minimax-m3")
    assert v.outcome == "pass" and v.score == 85.0 and v.verified is True
    assert v.reviewer == "claude-sonnet-4-6"


def test_judge_not_verified_when_self(monkeypatch) -> None:
    monkeypatch.setattr(q.llm_runner, "run",
                        lambda *a, **k: _FakeRes(parsed=_FakeParsed(90, "pass")))
    v = q.judge("content_writing", "p", "o", judge_model="m", producer_model="m")
    assert v.verified is False  # a model cannot independently verify its own work


def test_judge_failure_is_fail_not_silent_pass(monkeypatch) -> None:
    monkeypatch.setattr(q.llm_runner, "run", lambda *a, **k: _FakeRes(ok=False, reason="boom"))
    v = q.judge("coding", "p", "o", judge_model="j", producer_model="p")
    assert v.outcome == "fail" and v.score == 0.0 and v.verified is False


def test_capture_prefers_grader_over_judge(monkeypatch) -> None:
    # If a grader applies, the judge must not even be consulted.
    monkeypatch.setattr(q.llm_runner, "run",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("judge called")))
    v = q.capture("coding", "p", "5", expected="5", grader="exact")
    assert v.method == "grader" and v.outcome == "pass"


# --------------------------------------------------------------- HITL + battle

def test_rate_lands_human_verified_row(tmp_path: Path) -> None:
    led = RightModelLedger(path=tmp_path / "l.jsonl")
    row = fb.rate(task_class="content_writing", model="minimax-m3", score="pass", ledger=led)
    assert row.verified is True and row.review_kind == "human" and row.source == "production"
    assert row.outcome == "pass" and row.origin == "hitl-rating"
    # numeric score maps too
    assert fb.rate(task_class="coding", model="x", score=55, ledger=led).outcome == "partial"


def test_ingest_battle_writes_one_duel(tmp_path: Path) -> None:
    led = RightModelLedger(path=tmp_path / "l.jsonl")
    rows = fb.ingest_battle("battle-1", [
        {"model": "claude-opus-4-8", "passed": True},
        {"model": "gpt-5.4-mini", "passed": False},
    ], ledger=led)
    assert len(rows) == 2
    assert {r.collision_group for r in rows} == {"battle-1"}  # one duel
    assert all(r.source == "battle" for r in rows)
    assert led.collisions("coding")  # the matrix sees it as a duel
