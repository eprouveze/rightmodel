"""RightModel router — the lexicographic cascade orchestrator (Module 5).

THE public entry point of the routing substrate. Given a task class + taint set
+ candidate models, choose a model via the constraints-first cascade:

    1. taint     — reuse router_policy.check(); drop blocked candidates.
    2. breaker   — drop providers whose circuit breaker is unavailable (OPEN).
    3. capability-bar — drop candidates whose matrix μ < class min_bar; SAFETY:
       if the bar empties the survivor set, keep the single highest-μ survivor.
    4. selection — with prob explore_rate (and a COLD/high-σ survivor exists)
       pick the highest-σ survivor (exploration); else cost-min, with a bounded
       quality-premium: a pricier SETTLED survivor whose μ exceeds the cost-min
       pick's μ by >15 is preferred.

Determinism for tests: a local `random.Random(rng_seed)` drives the exploration
coin flip and tie-breaks. Default deps are constructed lazily against the
host-local ledger/breaker under ~/.business-agent/rightmodel/.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import rightmodel_harness as rm_harness
from .action_receipt import TaintClass
from .circuit_breaker import CircuitBreaker
from .model_prices import CANDIDATES, blended_cost
from .rightmodel_ledger import RightModelLedger
from .rightmodel_matrix import CapabilityMatrix
from .rightmodel_types import CellState, Outcome
from .router_policy import check as taint_check
from .router_policy import model_to_provider
from .stakes_score import StakesDecision, append_stakes_decision

_QUALITY_PREMIUM_MU = 15.0
_HIGH_SIGMA = 12.0

FUSION_MODEL = "council:budget-trio"


def _harness_value(model: str, execution_mode: str) -> str | None:
    """The harness string for the chosen (model, execution_mode), or None."""
    harness = rm_harness.harness_for(model, execution_mode)
    return harness.value if harness is not None else None


@dataclass
class RouteDecision:
    """Outcome of one cascade run — the chosen model plus a full audit trace."""

    model: str | None
    task_class: str
    reason: str
    exploration: bool = False
    rejected: dict[str, str] = field(default_factory=dict)
    candidates_considered: list[str] = field(default_factory=list)
    execution_mode: str = "completion"
    harness: str | None = None  # the (model × harness) pairing chosen to run the task


class RightModelRouter:
    """Lexicographic-cascade model router over the capability matrix + breaker."""

    def __init__(
        self,
        ledger: RightModelLedger | None = None,
        matrix: CapabilityMatrix | None = None,
        breaker: CircuitBreaker | None = None,
        explore_rate: float = 0.10,
        rng_seed: int | None = None,
    ) -> None:
        self.ledger = ledger if ledger is not None else RightModelLedger()
        self.matrix = matrix if matrix is not None else CapabilityMatrix(self.ledger)
        self.breaker = breaker if breaker is not None else CircuitBreaker()
        self.explore_rate = explore_rate
        self._rng = random.Random(rng_seed)

    def _emit_stakes_decision(
        self, run_id: str | None, stakes_in: str | None, action: str, resolved_stakes: str
    ) -> None:
        append_stakes_decision(
            StakesDecision(
                ts=datetime.now(timezone.utc).isoformat(),
                kind="route",
                card_id=None,
                run_id=run_id,
                stakes_in=stakes_in,
                score=None,
                signals={},
                action=action,
                near_threshold=False,
                resolved_stakes=resolved_stakes,
            )
        )

    def choose(
        self,
        *,
        task_class: str,
        taint_set: set[TaintClass],
        candidates: list[str] | None = None,
        run_id: str | None = None,
        lang: str = "en",
        execution_mode: str = "completion",
        stakes: str | None = None,
    ) -> RouteDecision:
        """Run the cascade and return a RouteDecision (model=None if none survive).

        ``execution_mode`` (``completion`` | ``agentic`` | ``skill:<name>`` |
        ``script:<path>``) adds an eligibility stage: a candidate survives only if
        a wired (model × harness) pairing provides the required capability. For the
        default ``completion`` mode every candidate is eligible, so the stage is a
        no-op and routing is unchanged. ``script:`` needs no LLM — the caller should
        short-circuit before routing (the consumer does).
        """
        cands = list(candidates) if candidates is not None else list(CANDIDATES.get(task_class, []))
        rejected: dict[str, str] = {}
        resolved_stakes = stakes if stakes is not None else "low"

        # Stage 0 — stakes short-circuit. High-stakes callers escalate to the
        # fusion panel, BUT taint still governs external fan-out: if the taint set
        # forbids the external panel (e.g. AUTH_SECRETS, host-local only), fall
        # through to the normal in-house cost-min cascade.
        if resolved_stakes == "high" and not taint_check(FUSION_MODEL, taint_set, run_id).blocked:
            self._emit_stakes_decision(run_id, stakes, "escalated", resolved_stakes)
            return RouteDecision(
                model=FUSION_MODEL,
                task_class=task_class,
                reason="stakes-escalation",
                candidates_considered=cands,
                execution_mode=execution_mode,
            )

        # Stage 1 — taint.
        survivors = []
        for model in cands:
            decision = taint_check(model, taint_set, run_id)
            if decision.blocked:
                rejected[model] = f"taint: {decision.block_reason}"
            else:
                survivors.append(model)
        if not survivors:
            self._emit_stakes_decision(run_id, stakes, "cost-min", resolved_stakes)
            return RouteDecision(
                model=None,
                task_class=task_class,
                reason="no candidate survived the taint stage",
                rejected=rejected,
                candidates_considered=cands,
            )

        # Stage 2 — breaker availability.
        avail = []
        for model in survivors:
            provider = model_to_provider(model).value
            if self.breaker.is_available(provider):
                avail.append(model)
            else:
                rejected[model] = f"breaker: provider {provider} unavailable"
        if not avail:
            self._emit_stakes_decision(run_id, stakes, "cost-min", resolved_stakes)
            return RouteDecision(
                model=None,
                task_class=task_class,
                reason="all taint-clean candidates have an open circuit breaker",
                rejected=rejected,
                candidates_considered=cands,
                execution_mode=execution_mode,
            )

        # Stage 3 — execution-mode eligibility. Drop candidates with no wired
        # (model × harness) pairing for the requested mode. No-op for completion.
        eligible = []
        for model in avail:
            if rm_harness.eligible(model, execution_mode):
                eligible.append(model)
            else:
                rejected[model] = f"execution-mode: no harness for {execution_mode}"
        if not eligible:
            self._emit_stakes_decision(run_id, stakes, "cost-min", resolved_stakes)
            return RouteDecision(
                model=None,
                task_class=task_class,
                reason=f"no candidate has a harness providing execution_mode={execution_mode}",
                rejected=rejected,
                candidates_considered=cands,
                execution_mode=execution_mode,
            )
        avail = eligible

        # Stage 4 — capability bar (with safety floor).
        bar = self.matrix.min_bar(task_class)
        cells = {m: self.matrix.cell(task_class, m) for m in avail}
        above = [m for m in avail if cells[m].mu >= bar]
        if above:
            qualified = above
        else:
            best = max(avail, key=lambda m: cells[m].mu)
            for m in avail:
                if m != best:
                    rejected[m] = f"below-bar: mu={cells[m].mu:.1f} < {bar:.1f}"
            qualified = [best]
            bar_note = (
                f" (all below bar {bar:.1f}; kept highest-mu survivor "
                f"{best} at mu={cells[best].mu:.1f})"
            )
        if above:
            for m in avail:
                if m not in above:
                    rejected[m] = f"below-bar: mu={cells[m].mu:.1f} < {bar:.1f}"
            bar_note = ""

        # Stage 5 — selection.
        explorable = [
            m
            for m in qualified
            if cells[m].state == CellState.COLD.value or cells[m].sigma >= _HIGH_SIGMA
        ]
        if explorable and self._rng.random() < self.explore_rate:
            chosen = max(explorable, key=lambda m: (cells[m].sigma, -blended_cost(m), m))
            reason = (
                f"exploration: highest-sigma survivor {chosen} "
                f"(sigma={cells[chosen].sigma:.1f}, state={cells[chosen].state}){bar_note}"
            )
            self._emit_stakes_decision(run_id, stakes, "cost-min", resolved_stakes)
            return RouteDecision(
                model=chosen,
                task_class=task_class,
                reason=reason,
                exploration=True,
                rejected=rejected,
                candidates_considered=cands,
                execution_mode=execution_mode,
                harness=_harness_value(chosen, execution_mode),
            )

        cost_min = min(qualified, key=lambda m: (blended_cost(m), -cells[m].mu, m))
        chosen = cost_min
        premium_note = ""
        for m in qualified:
            if (
                cells[m].state == CellState.SETTLED.value
                and cells[m].mu - cells[cost_min].mu > _QUALITY_PREMIUM_MU
            ):
                if chosen == cost_min or cells[m].mu > cells[chosen].mu:
                    chosen = m
        if chosen != cost_min:
            premium_note = (
                f" (bounded quality-premium: {chosen} mu={cells[chosen].mu:.1f} "
                f"beats cost-min {cost_min} mu={cells[cost_min].mu:.1f} by >{_QUALITY_PREMIUM_MU:.0f})"
            )
            reason = f"quality-premium pick {chosen}{premium_note}{bar_note}"
        else:
            reason = (
                f"cost-min pick {chosen} "
                f"(blended_cost={blended_cost(chosen):.3f}, mu={cells[chosen].mu:.1f}){bar_note}"
            )
        self._emit_stakes_decision(run_id, stakes, "cost-min", resolved_stakes)
        return RouteDecision(
            model=chosen,
            task_class=task_class,
            reason=reason,
            exploration=False,
            rejected=rejected,
            candidates_considered=cands,
            execution_mode=execution_mode,
            harness=_harness_value(chosen, execution_mode),
        )

    def choose_panel(
        self,
        *,
        task_class: str,
        taint_set: set[TaintClass],
        n: int = 3,
        candidates: list[str] | None = None,
        run_id: str | None = None,
        execution_mode: str = "completion",
    ) -> list[str]:
        """Select up to ``n`` models for a fusion panel — RightModel composes the
        budget-trio instead of a hard-coded roster.

        Survivors of the same constraint cascade as :meth:`choose` (taint → breaker →
        capability bar) are ranked cheapest-first and taken **one per provider**, so the
        panel genuinely cross-checks rather than sampling near-duplicates. Stays
        per-task-class and cost-minimal, and evolves as the ledger does. Returns ``[]``
        if nothing survives; may return fewer than ``n`` when fewer distinct providers
        qualify — the caller decides whether quorum (≥2) is met.
        """
        cands = list(candidates) if candidates is not None else list(CANDIDATES.get(task_class, []))
        survivors = [
            m
            for m in cands
            if not taint_check(m, taint_set, run_id).blocked
            and self.breaker.is_available(model_to_provider(m).value)
            and rm_harness.eligible(m, execution_mode)
        ]
        if not survivors:
            return []
        bar = self.matrix.min_bar(task_class)
        cells = {m: self.matrix.cell(task_class, m) for m in survivors}
        above = [m for m in survivors if cells[m].mu >= bar]
        # Safety floor mirrors choose(): if the bar empties the set, keep the best survivor.
        pool = above if above else [max(survivors, key=lambda m: cells[m].mu)]
        pool.sort(key=lambda m: (blended_cost(m), -cells[m].mu, m))
        picked: list[str] = []
        seen_providers: set[str] = set()
        for m in pool:
            provider = model_to_provider(m).value
            if provider in seen_providers:
                continue
            picked.append(m)
            seen_providers.add(provider)
            if len(picked) >= n:
                break
        return picked

    def record_result(
        self,
        decision: RouteDecision,
        *,
        outcome: str,
        verified: bool,
        cost_usd: float,
        difficulty: float = 0.5,
        lang: str = "en",
        collision_group: str | None = None,
        reviewer: str = "",
        review_kind: str = "none",
        origin: str = "agent",
        basis: str = "",
    ) -> None:
        """Append the outcome to the ledger and update the provider breaker.

        A PASS records breaker success; FAIL/PARTIAL records a breaker failure.
        No-op on the breaker if the decision chose no model. Provenance is recorded
        for attribution: ``author`` is the chosen model, plus ``reviewer``/
        ``review_kind``/``origin``/``basis``. ``verified`` is recorded as the caller
        passes it — the caller decides whether an independent check happened.
        """
        if decision.model is None:
            return
        self.ledger.append(
            task_class=decision.task_class,
            model=decision.model,
            outcome=outcome,
            verified=verified,
            cost_usd=cost_usd,
            difficulty=difficulty,
            lang=lang,
            source="production",
            collision_group=collision_group,
            author=decision.model,
            reviewer=reviewer,
            review_kind=review_kind,
            origin=origin,
            basis=basis,
        )
        provider = model_to_provider(decision.model).value
        if outcome == Outcome.PASS.value:
            self.breaker.record_success(provider)
        else:
            self.breaker.record_failure(provider, kind="error")
