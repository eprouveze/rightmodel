"""RightModel quality capture — turn a delegated output into an outcome + score.

The router can pick and the ledger can track, but without a QUALITY signal the matrix
never self-improves: it would only learn "did the call return", not "was the work good".
This module produces that signal from three sources, in priority order:

  1. grade()  — deterministic auto-grade when the card supplies a checkable answer
                (reuses the eval-harness graders). The cheapest, most trustworthy signal.
  2. judge()  — an LLM-judge scores free-form output against the task. Fast, automatic,
                and (when the judge model differs from the producer) an INDEPENDENT check,
                so it also satisfies provenance verification. This is the default for the
                open-ended cards real users create.
  3. HITL     — Emmanuel rates / Arena votes (handled in the CLI + rightmodel_arena), the
                gold signal, recorded verified=True/human and weighted above the judge.

A QualityVerdict maps to an Outcome (pass/partial/fail) the matrix consumes. The judge is a
known Goodhart risk (council round 2), which is exactly why HITL outranks it and the
anti-Goodhart analyzers watch for judge-vs-human divergence.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from . import llm_runner
from .rightmodel_eval import EvalItem, score as grader_score
from .rightmodel_types import Outcome

DEFAULT_JUDGE = "claude-sonnet-4-6"
_PASS_AT = 70.0
_PARTIAL_AT = 40.0


@dataclass
class QualityVerdict:
    """The quality of one delegated output: an outcome + 0-100 score + how it was judged."""

    outcome: str       # Outcome value (pass|partial|fail)
    score: float       # 0-100
    method: str        # "grader" | "judge" | "hitl"
    verified: bool     # True when an independent party (distinct model / human) judged it
    reviewer: str      # judge model id or human id ("" for an auto-grader)
    rationale: str


class _JudgeScore(BaseModel):
    score: int = Field(ge=0, le=100)
    verdict: str  # pass | partial | fail
    rationale: str = ""


def _outcome_for(score: float) -> str:
    if score >= _PASS_AT:
        return Outcome.PASS.value
    if score >= _PARTIAL_AT:
        return Outcome.PARTIAL.value
    return Outcome.FAIL.value


def grade(output: str, *, task_class: str, expected: str, grader: str,
          lang: str = "en") -> QualityVerdict:
    """Deterministic auto-grade via the eval-harness graders (exact/contains/python_assert)."""
    item = EvalItem(id="rt", task_class=task_class, prompt="", difficulty=0.5,
                    lang=lang, grader=grader, expected=expected)
    passed = grader_score(item, output)
    return QualityVerdict(
        outcome=Outcome.PASS.value if passed else Outcome.FAIL.value,
        score=100.0 if passed else 0.0,
        method="grader",
        verified=True,  # a deterministic check is an independent verification
        reviewer="auto-grader",
        rationale=f"{grader} grader: {'matched' if passed else 'did not match'} expected",
    )


def judge(task_class: str, prompt: str, output: str, *,
          judge_model: str = DEFAULT_JUDGE, producer_model: str = "") -> QualityVerdict:
    """LLM-judge: score the output for the task. INDEPENDENT iff judge != producer.

    Returns a fail verdict (score 0) if the judge call itself fails, so a flaky judge never
    silently inflates quality. ``verified`` is True only when the judge is a different model
    than the one that produced the output (an AI cannot independently verify its own work).
    """
    judge_prompt = (
        f"You are grading how well an AI response accomplishes a task.\n"
        f"Task class: {task_class}\n"
        f"--- TASK ---\n{prompt}\n"
        f"--- RESPONSE ---\n{output}\n"
        f"--- END ---\n"
        f"Score 0-100 on correctness, completeness, and following the instructions. "
        f"verdict is 'pass' (>=70), 'partial' (40-69), or 'fail' (<40)."
    )
    res = llm_runner.run(judge_model, judge_prompt, schema=_JudgeScore, timeout=120)
    independent = bool(judge_model and judge_model != producer_model)
    if not res.ok or res.parsed is None:
        return QualityVerdict(
            outcome=Outcome.FAIL.value, score=0.0, method="judge",
            verified=False, reviewer=judge_model,
            rationale=f"judge unavailable: {res.reason or 'no structured verdict'}",
        )
    parsed: _JudgeScore = res.parsed
    score = float(max(0, min(100, parsed.score)))
    return QualityVerdict(
        outcome=_outcome_for(score),
        score=score,
        method="judge",
        verified=independent,
        reviewer=judge_model,
        rationale=parsed.rationale[:300],
    )


def capture(task_class: str, prompt: str, output: str, *,
            expected: str | None = None, grader: str | None = None,
            judge_model: str | None = None, producer_model: str = "",
            lang: str = "en") -> QualityVerdict:
    """Best available quality signal: auto-grade if checkable, else LLM-judge.

    HITL (Emmanuel / Arena) is layered on separately via the rate/vote path and outranks
    whatever this returns, because it lands as a verified=human row the matrix weights above.
    """
    if expected is not None and grader:
        return grade(output, task_class=task_class, expected=expected, grader=grader, lang=lang)
    return judge(task_class, prompt, output,
                 judge_model=judge_model or DEFAULT_JUDGE, producer_model=producer_model)
