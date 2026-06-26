"""RightModel consumer — the route→run→review→record loop (Module 12).

THE first real consumer of the routing substrate: the loop the heartbeat and
Foreman call to actually delegate a task to a model and record the outcome.

    1. router.choose(...)            — pick a model via the lexicographic cascade.
    2. llm_runner.run(model, prompt) — execute it (honoring the kill-switch).
    3. outcome = pass/fail from result.ok.
    4. optional peer review          — a distinct reviewer_model verifies; verified
       is set True only when an independent reviewer (!= the producing model) passes.
       This is the consumer's own honest bookkeeping of whether a real second check
       happened — NOT Proof's 'AI cannot approve its own work' governance gate, which
       is Proof-specific and not enforced on the capability ledger.
    5. router.record_result(...)     — append a production row with provenance.

Like ``llm_runner.run``, ``delegate`` NEVER raises — every failure path returns a
``DelegationResult`` the caller can branch on. A ``live=False`` (or kill-switch-off)
run is a dry, non-recorded probe.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import fusion_guard
from . import llm_runner
from . import rightmodel_harness as rm_harness
from .model_prices import CANDIDATES
from .rightmodel_router import RightModelRouter, RouteDecision
from .rightmodel_types import Outcome
from .router_policy import model_to_provider

# Cost-min fallback when the C rails refuse the panel (nested fusion forbidden, or the
# panel could not be run). A cheap in-house model so a refused fusion still produces an
# answer instead of nothing.
_FUSION_REFUSE_FALLBACK = "claude-haiku-4-5"
# Self-fusion-lite: 2x the same cheap model at differing temperature/seed + the same A1 judge.
_SELF_LITE_MODEL = "council:self-lite"
_SELF_LITE_BASE = "deepseek-v4-pro"

# Maps a RightModel provider to its council.sh seat key. RightModel selects the budget
# panel from CANDIDATES (choose_panel); these are the providers that have a council seat
# to actually run them (perplexity/sonar and xAI/grok have no seat, so are never picked).
# NOTE: claude is deliberately EXCLUDED. Its council seat (LexO) runs `claude -p`, which is
# (a) the future-metered Agent-SDK path the cost rule forbids for autonomous work — fusion
# must stay on subscription CLIs + pay-per-use APIs — and (b) deadlock-prone when nested
# inside an active Claude Code process tree (it hangs on its own lifecycle hooks). A budget
# panel draws diversity from the other providers; claude is what you escalate FROM, not a seat.
_PROVIDER_TO_SEAT = {
    "gemini": "lexi",
    "openai": "lext",
    "kimi": "kimi",
    "deepseek": "lek",
    "minimax": "lem",
}
# council.sh env var that overrides each seat's default (premium) model, so the seat runs the
# EXACT model RightModel picked — otherwise lext would run its default gpt-5.5 instead of the
# cheap gpt-5.4-mini selected, breaking the budget property. kimi has a single model (no var);
# agy ignores LEXI_GEMINI_MODEL (gemini stuck at the settings default), set for the record.
_SEAT_MODEL_ENV = {
    "lek": "LEK_MODEL",
    "lem": "LEM_MODEL",
    "lext": "LEXT_CODEX_MODEL",
    "lexi": "LEXI_GEMINI_MODEL",
}


def _panel_models(
    router: RightModelRouter, task_class: str, taint_set: set, run_id: str | None, n: int = 3
) -> list[str]:
    """RightModel-selected models for a budget panel (cheapest above-bar, 1 per provider).
    Only seatable-provider candidates are offered to the router, so every pick maps to a
    runnable council seat."""
    seatable = [
        m for m in CANDIDATES.get(task_class, [])
        if model_to_provider(m).value in _PROVIDER_TO_SEAT
    ]
    return router.choose_panel(
        task_class=task_class, taint_set=taint_set, n=n, candidates=seatable, run_id=run_id,
    )


def _seats_for(models: list[str]) -> list[str]:
    """The council seat keys for the selected panel models (1 per provider, preserving order)."""
    return [_PROVIDER_TO_SEAT[model_to_provider(m).value] for m in models]


@dataclass
class DelegationResult:
    """Outcome of one delegate() call — model, outcome, verification, audit trace."""

    model: str | None
    outcome: str
    verified: bool
    cost_usd: float
    text: str | None
    reason: str
    decision: RouteDecision


def _peer_review(reviewer_model: str, model: str, prompt: str, text: str) -> tuple[bool, str]:
    """Run a cheap second-model check. Returns (verified, trace).

    A genuine independent pass requires reviewer != model AND a successful
    reviewer run — the consumer records verified accordingly.
    """
    if reviewer_model == model:
        return False, "review skipped: reviewer == model (self-review)"
    review_prompt = (
        "You are reviewing another model's output. Reply PASS if it adequately "
        f"addresses the task, otherwise FAIL.\n\nTASK:\n{prompt}\n\nOUTPUT:\n{text}"
    )
    review = llm_runner.run(reviewer_model, review_prompt)
    if not review.ok:
        return False, f"review run failed: {review.reason}"
    return True, f"peer-reviewed by {reviewer_model}: ok"


_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class FusionRun:
    """Outcome of one fusion panel + synthesis: model, outcome, cost, synthesis trailer."""

    model: str
    outcome: str
    cost_usd: float
    text: str | None
    synthesis: dict
    reason: str


def _card_id_from(origin: str, run_id: str | None) -> str:
    """Derive a lease key. Prefer the card id embedded in origin (heartbeat-card:<id>)."""
    if origin and origin.startswith("heartbeat-card:"):
        return origin.split(":", 1)[1] or (run_id or "fusion")
    return run_id or origin or "fusion"


def _panel_env(self_lite: bool, models: list[str] | None = None) -> dict:
    """Subprocess env for the panel — seats can never re-escalate (RIGHTMODEL_NO_FUSION=1)."""
    env = dict(os.environ)
    env["RIGHTMODEL_NO_FUSION"] = "1"
    if self_lite:
        # A single-provider 2-seat panel: same cheap model twice at differing temperature/seed.
        # The council seats that map to this base model run; the heterogeneous CLI seats are off.
        env["RIGHTMODEL_SELF_LITE"] = "1"
        env["RIGHTMODEL_SELF_LITE_MODEL"] = _SELF_LITE_BASE
    elif models:
        # RightModel-selected budget panel: restrict council.sh to exactly these seats AND
        # pin each seat to the specific model RightModel picked (so a seat runs the cheap
        # variant, not its default premium model — the budget property depends on this).
        seats = _seats_for(models)
        env["COUNCIL_SEATS"] = ",".join(seats)
        for seat, model in zip(seats, models):
            var = _SEAT_MODEL_ENV.get(seat)
            if var:
                env[var] = model
    return env


def _run_panel(
    session_dir: Path, prompt: str, *, self_lite: bool, models: list[str] | None = None
) -> tuple[bool, str]:
    """Run scripts/council.sh headless into session_dir. Returns (ok, reason)."""
    script = _REPO_ROOT / "scripts" / "council.sh"
    env = _panel_env(self_lite, models)
    env["LEX_COUNCIL_DIR"] = str(session_dir)
    try:
        proc = subprocess.run(
            ["bash", str(script), prompt],
            capture_output=True, text=True, timeout=900, env=env,
            cwd=str(_REPO_ROOT),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"panel run failed: {e}"
    if proc.returncode != 0:
        return False, f"panel exit {proc.returncode}: {(proc.stderr or '')[-300:]}"
    return True, "panel ok"


def _run_synthesis(session_dir: Path, prompt: str) -> tuple[bool, dict, str]:
    """Run scripts/council-synthesize.py (A1) over session_dir; read synthesis.json."""
    script = _REPO_ROOT / "scripts" / "council-synthesize.py"
    try:
        proc = subprocess.run(
            ["python3", str(script), str(session_dir), prompt],
            capture_output=True, text=True, timeout=600, env=dict(os.environ),
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, {}, f"synthesis failed: {e}"
    out = session_dir / "synthesis.json"
    if not out.exists():
        return False, {}, f"synthesis produced no synthesis.json: {(proc.stderr or '')[-300:]}"
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return False, {}, f"synthesis.json unreadable: {e}"
    return True, data, "synthesis ok"


def _panel_cost(session_dir: Path) -> float:
    """Sum the real marginal cost (A2) of a council session — $0 for subscription seats."""
    script = _REPO_ROOT / "scripts" / "council-cost.py"
    try:
        proc = subprocess.run(
            ["python3", str(script), str(session_dir)],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0.0
    if proc.returncode != 0:
        return 0.0
    try:
        return float((proc.stdout or "0").strip())
    except ValueError:
        return 0.0


def _run_fusion(
    model: str, prompt: str, *, self_lite: bool, models: list[str] | None = None
) -> FusionRun:
    """Run a fusion panel (RightModel-selected budget-trio or self-fusion-lite) + A1 judge
    + A2 cost.

    Gated by the FOREMAN_LLM_LIVE kill-switch (same gate the card executor uses): when off,
    no panel runs and the result is an inert FAIL the caller records as a dry probe.
    """
    if not llm_runner.is_live():
        return FusionRun(
            model=model, outcome=Outcome.FAIL.value, cost_usd=0.0, text=None,
            synthesis={}, reason="kill_switch_off: no panel, no record",
        )
    session_dir = Path(tempfile.mkdtemp(prefix="fusion-"))
    ok, reason = _run_panel(session_dir, prompt, self_lite=self_lite, models=models)
    if not ok:
        return FusionRun(
            model=model, outcome=Outcome.FAIL.value, cost_usd=0.0, text=None,
            synthesis={}, reason=reason,
        )
    ok, synthesis, syn_reason = _run_synthesis(session_dir, prompt)
    cost = _panel_cost(session_dir)
    if not ok:
        return FusionRun(
            model=model, outcome=Outcome.FAIL.value, cost_usd=cost, text=None,
            synthesis={}, reason=syn_reason,
        )
    recommendation = synthesis.get("recommendation")
    confidence = synthesis.get("confidence")
    # Confidence informs the outcome: a confident recommendation is a PASS; a present-but-low
    # or unparseable confidence is PARTIAL; a missing recommendation is a FAIL.
    if recommendation is None:
        outcome = Outcome.FAIL.value
    elif isinstance(confidence, (int, float)) and confidence >= 0.5:
        outcome = Outcome.PASS.value
    else:
        outcome = Outcome.PARTIAL.value
    return FusionRun(
        model=model, outcome=outcome, cost_usd=cost,
        text=recommendation if isinstance(recommendation, str) else None,
        synthesis=synthesis,
        reason=f"fusion panel + A1 judge ({syn_reason}); cost=${cost:.4f}",
    )


def _delegate_fusion(
    router: RightModelRouter,
    decision: RouteDecision,
    prompt: str,
    *,
    origin: str,
    run_id: str | None,
    difficulty: float,
    lang: str,
    collision_group: str | None,
    taint_set: set,
) -> DelegationResult:
    """Run the fusion panel for a council:* decision under the C rails, and record one row.

    Rails: nested-fusion forbidden → refuse (single cost-min fallback); per-card lease →
    skip if not acquired; over budget → DOWNGRADE to self-fusion-lite. The fusion RUN itself
    is NOT earned-autonomy gated (it is deliberation) — only an irreversible/outward-facing
    ACTION the synthesis informs is gated, by returning the synthesis as a recommendation.
    """
    # Rail C — nested fusion forbidden: a panel seat must never re-escalate.
    if fusion_guard.nested_fusion_forbidden():
        fb = RouteDecision(
            model=_FUSION_REFUSE_FALLBACK, task_class=decision.task_class,
            reason="fusion refused (nested fusion forbidden) — single cost-min fallback",
        )
        result = llm_runner.run(_FUSION_REFUSE_FALLBACK, prompt)
        outcome = Outcome.PASS.value if result.ok else Outcome.FAIL.value
        router.record_result(
            fb, outcome=outcome, verified=False, cost_usd=result.cost_usd,
            difficulty=difficulty, lang=lang, origin=origin,
            basis="fusion-refused-nested", collision_group=collision_group,
        )
        return DelegationResult(
            model=_FUSION_REFUSE_FALLBACK, outcome=outcome, verified=False,
            cost_usd=result.cost_usd, text=result.text,
            reason=fb.reason + f" | run: {'ok' if result.ok else result.reason}",
            decision=fb,
        )

    # Rail B — per-card lease: a duplicate concurrent run on the same card is blocked.
    card_id = _card_id_from(origin, run_id)
    if not fusion_guard.acquire_fusion_lease(card_id):
        return DelegationResult(
            model=decision.model, outcome=Outcome.FAIL.value, verified=False,
            cost_usd=0.0, text=None,
            reason="fusion lease not acquired — another run holds this card; skipped",
            decision=decision,
        )

    try:
        # Rail A — budget: over cap → DOWNGRADE to self-fusion-lite (cheaper single-provider panel).
        self_lite = not fusion_guard.fusion_budget_ok()
        downgrade_reason = "over budget" if self_lite else ""
        # RightModel composes the budget panel (cheapest above-bar model per provider for
        # this task class) — not a hard-coded roster. Too few distinct providers to reach
        # quorum (<2) → downgrade to self-fusion-lite rather than run a degenerate panel.
        models: list[str] | None = None
        if not self_lite:
            models = _panel_models(router, decision.task_class, taint_set, run_id)
            if len(models) < 2:
                self_lite, downgrade_reason, models = True, "panel quorum < 2", None
        model = _SELF_LITE_MODEL if self_lite else decision.model
        run = _run_fusion(model, prompt, self_lite=self_lite, models=models)
        if not llm_runner.is_live():
            return DelegationResult(
                model=model, outcome=run.outcome, verified=False, cost_usd=0.0,
                text=None, reason=f"{decision.reason} | {run.reason}",
                decision=RouteDecision(model=model, task_class=decision.task_class,
                                       reason=decision.reason),
            )
        fusion_guard.record_fusion_run()
        record_decision = RouteDecision(
            model=model, task_class=decision.task_class, reason=decision.reason,
        )
        router.record_result(
            record_decision, outcome=run.outcome, verified=False,
            cost_usd=run.cost_usd, difficulty=difficulty, lang=lang, origin=origin,
            basis="fusion:self-lite" if self_lite else "fusion:budget-trio",
            collision_group=collision_group,
        )
        panel_note = (
            f" | panel=[{','.join(f'{s}:{m}' for s, m in zip(_seats_for(models), models))}]"
            if models else ""
        )
        downgrade_note = (
            f" | DOWNGRADED to self-fusion-lite ({downgrade_reason})" if self_lite else panel_note
        )
        return DelegationResult(
            model=model, outcome=run.outcome, verified=False, cost_usd=run.cost_usd,
            text=run.text, reason=f"{decision.reason} | {run.reason}{downgrade_note}",
            decision=record_decision,
        )
    finally:
        fusion_guard.release_fusion_lease(card_id)


def delegate(
    task_class: str,
    prompt: str,
    taint_set: set,
    *,
    router: RightModelRouter | None = None,
    run_id: str | None = None,
    lang: str = "en",
    origin: str = "agent",
    reviewer_model: str | None = None,
    difficulty: float = 0.5,
    live: bool | None = None,
    quality: bool = False,
    judge_model: str | None = None,
    expected: str | None = None,
    grader: str | None = None,
    execution_mode: str = "completion",
    add_dirs: list[str] | None = None,
    candidates: list[str] | None = None,
    collision_group: str | None = None,
    timeout: int | None = None,
    stakes: str | None = None,
) -> DelegationResult:
    """Route, run, score quality, and record one delegation. Never raises.

    ``execution_mode`` (``completion`` | ``agentic`` | ``skill:<name>`` |
    ``script:<path>``) is threaded into routing (an eligibility filter over the
    (model × harness) registry) and into the run (the agentic harness drives the
    model, e.g. Gemini via Antigravity ``agy``, with ``add_dirs`` as tool scope).
    A ``script:`` mode needs no LLM — it short-circuits with no route and no record.

    When ``quality`` is set, the recorded outcome reflects HOW GOOD the output is — an
    auto-grade (if ``expected``+``grader`` given) else an LLM-judge (``judge_model``, default
    a model distinct from the producer) — not merely whether the call returned. This is the
    self-improvement signal: the matrix learns measured quality, not liveness. HITL/Arena
    still outranks the judge via a later verified=human row.
    """
    router = router if router is not None else RightModelRouter()

    # script: is deterministic — no model, no route, no ledger row.
    if rm_harness.is_script(execution_mode):
        return DelegationResult(
            model=None,
            outcome=Outcome.PASS.value,
            verified=False,
            cost_usd=0.0,
            text=None,
            reason=f"script execution_mode={execution_mode}: no LLM routing",
            decision=RouteDecision(
                model=None, task_class=task_class,
                reason="script mode: deterministic, not an LLM route",
                execution_mode=execution_mode,
            ),
        )

    decision = router.choose(
        task_class=task_class,
        taint_set=taint_set,
        run_id=run_id,
        lang=lang,
        execution_mode=execution_mode,
        candidates=candidates,
        stakes=stakes,
    )
    if decision.model is None:
        return DelegationResult(
            model=None,
            outcome=Outcome.FAIL.value,
            verified=False,
            cost_usd=0.0,
            text=None,
            reason=f"no model chosen: {decision.reason}",
            decision=decision,
        )

    run_live = llm_runner.is_live() if live is None else bool(live)
    if not run_live:
        return DelegationResult(
            model=decision.model,
            outcome=Outcome.FAIL.value,
            verified=False,
            cost_usd=0.0,
            text=None,
            reason=f"{decision.reason} | dry run (not live): no call, no record",
            decision=decision,
        )

    # Fusion path — decision.model is a council pseudo-model ("council:budget-trio" from the
    # Stage-0 stakes short-circuit). Run the panel under the C rails instead of a single model.
    if decision.model.startswith("council:"):
        return _delegate_fusion(
            router, decision, prompt,
            origin=origin, run_id=run_id, difficulty=difficulty, lang=lang,
            collision_group=collision_group, taint_set=taint_set,
        )

    result = llm_runner.run(
        decision.model, prompt, execution_mode=execution_mode, add_dirs=add_dirs,
        timeout=timeout,
    )
    outcome = Outcome.PASS.value if result.ok else Outcome.FAIL.value

    verified = False
    review_kind = "none"
    reviewer = reviewer_model or ""
    review_trace = ""
    basis = "llm_runner"

    if result.ok and quality:
        # Quality scoring sets the outcome (and, when the judge is independent, verifies it).
        # Grade the ARTIFACT an agentic run wrote (the diff it produced), not its narration —
        # the agent's last stdout message ("I've opened the PR…") is not the deliverable and
        # judging it produced a bogus fail for a real PR (Bug 1, c-1b70c9e4 / PR #111).
        from .rightmodel_quality import capture
        graded_output = (result.meta or {}).get("artifact") or result.text or ""
        verdict = capture(
            task_class, prompt, graded_output,
            expected=expected, grader=grader,
            judge_model=judge_model, producer_model=decision.model, lang=lang,
        )
        outcome = verdict.outcome
        verified = verdict.verified
        review_kind = verdict.method if verdict.verified else "none"
        reviewer = verdict.reviewer if verdict.verified else reviewer
        basis = f"quality:{verdict.method}"
        review_trace = f" | quality({verdict.method}): {verdict.score:.0f}/100 {verdict.outcome}"
    elif result.ok and reviewer_model and reviewer_model != decision.model:
        verified, review_trace = _peer_review(
            reviewer_model, decision.model, prompt, result.text or ""
        )
        if verified:
            review_kind = "ai-peer"
        review_trace = " | " + review_trace

    router.record_result(
        decision,
        outcome=outcome,
        verified=verified,
        cost_usd=result.cost_usd,
        difficulty=difficulty,
        lang=lang,
        origin=origin,
        reviewer=reviewer,
        review_kind=review_kind,
        basis=basis,
        collision_group=collision_group,
    )

    reason = f"{decision.reason} | run: {'ok' if result.ok else result.reason}{review_trace}"
    return DelegationResult(
        model=decision.model,
        outcome=outcome,
        verified=verified,
        cost_usd=result.cost_usd,
        text=result.text,
        reason=reason,
        decision=decision,
    )
