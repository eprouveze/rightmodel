"""RightModel Arena — blind pairwise HITL for the subjective classes (Module 9).

The second capability stream Emmanuel specified: content_writing / design /
marketing have no auto-grader, so capability is measured by blind human pairwise
votes. Each vote lands in the outcome ledger as TWO ``source="arena"`` collision
rows (winner PASS, loser FAIL; tie → both PARTIAL) sharing a ``collision_group``,
so the existing Bradley–Terry matrix consumes Arena duels with zero matrix
changes.

Provenance (Proof v2 spine): every Arena row carries ``origin="arena"``,
``author=<model>`` (the producing model), ``reviewer=<voter>``, and
``review_kind="human"`` — a genuine independent human review.

Triggering uses uncertainty sampling: only pair models whose capability
intervals (μ±σ) overlap, prioritising COLD / high-σ cells, so votes go where they
move the matrix most. Stopping uses a Wilson confidence interval over the duels
between a pair (settled when its width < 0.15).

Safe on an empty ledger (queue still proposes from priors; is_settled → False).
No live LLM calls.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .model_prices import CANDIDATES
from .rightmodel_types import CellState, Outcome

_SETTLE_WIDTH: float = 0.15
_DIVERGE_PTS: float = 15.0


@dataclass
class ArenaMatchup:
    """One blind pairwise task to be voted on.

    ``item_id`` is the shared ``collision_group`` for the two ledger rows a vote
    produces, and is a stable hash of (task_class, model_a, model_b, index) so
    the same logical matchup always maps to the same id.
    """

    task_class: str
    item_id: str
    model_a: str
    model_b: str
    prompt: str


def _item_id(task_class: str, model_a: str, model_b: str, index: int) -> str:
    """Stable hash id for a matchup (order-independent over the model pair)."""
    lo, hi = sorted((model_a, model_b))
    raw = f"{task_class}|{lo}|{hi}|{index}".encode("utf-8")
    return "arena-" + hashlib.sha1(raw).hexdigest()[:12]


class Arena:
    """Blind pairwise HITL stream feeding the ledger as arena collision rows."""

    def __init__(self, ledger, matrix, candidates: dict | None = None) -> None:
        """Default candidate pool is ``model_prices.CANDIDATES``."""
        self._ledger = ledger
        self._matrix = matrix
        self._candidates = candidates if candidates is not None else CANDIDATES

    def _pool(self, task_class: str) -> list[str]:
        return list(self._candidates.get(task_class, []))

    def should_trigger(self, task_class: str) -> tuple[bool, str]:
        """Whether the Arena should run more duels for this class.

        Fires on either: a cold candidate cell (cold-start — and since identity is
        ``model_id``, a brand-new model is simply a new cold cell, so a new model
        re-arms the Arena automatically); or production μ diverging >15pts (or >1σ)
        from the arena/prior estimate for some candidate. There is no version-based
        trigger — a different model_id is a different model (handled by cold-start);
        a "version revision" of a stable id is not a re-trigger (see the matrix /
        freshness IDENTITY PRINCIPLE).
        """
        self._matrix.refresh()
        pool = self._pool(task_class)
        if not pool:
            return False, "no candidates configured for class"

        for model in pool:
            if self._matrix.cell(task_class, model).state == CellState.COLD.value:
                return True, f"cold-start: {model} cell is COLD"

        for model in pool:
            prod = [
                r for r in self._ledger.rows(task_class=task_class, model=model)
                if r.source == "production"
            ]
            if not prod:
                continue
            passes = sum(1 for r in prod if r.outcome == Outcome.PASS.value)
            prod_mu = 100.0 * passes / len(prod)
            cell = self._matrix.cell(task_class, model)
            gap = abs(prod_mu - cell.mu)
            if gap > _DIVERGE_PTS or gap > cell.sigma:
                return True, (
                    f"production diverges for {model}: "
                    f"prod={prod_mu:.0f} vs est={cell.mu:.0f}"
                )

        return False, "all candidate cells warm and consistent"

    def queue(self, task_class: str, n_pairs: int = 3, prompt: str = "") -> list[ArenaMatchup]:
        """Propose up to ``n_pairs`` matchups via uncertainty sampling.

        Only pairs models whose capability CIs (μ±σ) overlap, prioritising the
        pairs with the most combined uncertainty (COLD/high-σ cells). Cold cells
        with no overlap are still pairable because their wide σ overlaps almost
        everything. Cold-start safe: cells fall back to priors with maximal σ.
        """
        self._matrix.refresh()
        pool = self._pool(task_class)
        cells = {m: self._matrix.cell(task_class, m) for m in pool}

        scored: list[tuple[float, str, str]] = []
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                a, b = pool[i], pool[j]
                ca, cb = cells[a], cells[b]
                lo_a, hi_a = ca.mu - ca.sigma, ca.mu + ca.sigma
                lo_b, hi_b = cb.mu - cb.sigma, cb.mu + cb.sigma
                if hi_a < lo_b or hi_b < lo_a:
                    continue  # CIs disjoint → an uninformative duel
                cold_bonus = sum(
                    1.0 for c in (ca, cb) if c.state == CellState.COLD.value
                )
                score = ca.sigma + cb.sigma + cold_bonus
                scored.append((score, a, b))

        scored.sort(key=lambda t: t[0], reverse=True)
        out: list[ArenaMatchup] = []
        for index, (_, a, b) in enumerate(scored[:n_pairs]):
            out.append(
                ArenaMatchup(
                    task_class=task_class,
                    item_id=_item_id(task_class, a, b, index),
                    model_a=a,
                    model_b=b,
                    prompt=prompt,
                )
            )
        return out

    def record_vote(
        self,
        matchup: ArenaMatchup,
        winner: str,
        *,
        voter: str = "human",
        difficulty: float = 0.5,
        lang: str = "en",
    ) -> None:
        """Record one blind vote as TWO arena collision rows.

        Winner → ``outcome="pass"``, loser → ``"fail"``. A tie (``winner=""``)
        records both as ``"partial"``. Both rows: ``source="arena"``,
        ``collision_group=item_id``, ``origin="arena"``, ``author=<model>``,
        ``reviewer=voter``, ``review_kind="human"`` — a genuine human review, so
        they are recorded ``verified=True``.
        """
        models = (matchup.model_a, matchup.model_b)
        if winner and winner not in models:
            raise ValueError(f"winner {winner!r} is not in matchup {models}")
        tie = not winner
        for model in models:
            if tie:
                outcome = Outcome.PARTIAL.value
            elif model == winner:
                outcome = Outcome.PASS.value
            else:
                outcome = Outcome.FAIL.value
            self._ledger.append(
                task_class=matchup.task_class,
                model=model,
                outcome=outcome,
                verified=True,
                cost_usd=0.0,
                difficulty=difficulty,
                lang=lang,
                source="arena",
                collision_group=matchup.item_id,
                origin="arena",
                author=model,
                reviewer=voter,
                review_kind="human",
            )

    @staticmethod
    def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
        """Wilson score interval for a binomial proportion (lo, hi) in [0, 1].

        Returns (0.0, 1.0) for n == 0 (maximal uncertainty).
        """
        if n <= 0:
            return 0.0, 1.0
        phat = wins / n
        z2 = z * z
        denom = 1.0 + z2 / n
        center = (phat + z2 / (2 * n)) / denom
        margin = (z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))) / denom
        lo = max(0.0, center - margin)
        hi = min(1.0, center + margin)
        return lo, hi

    def is_settled(self, task_class: str, model_a: str, model_b: str) -> bool:
        """True when the Arena duels between this pair have a tight win-rate CI.

        Counts model_a's wins (PASS while the opponent did not pass; ties count
        as half a win) over duels sharing a collision_group. Settled when the
        Wilson CI width over those duels is < 0.15. Empty/young → False.
        """
        groups = self._ledger.collisions(task_class)
        wins = 0.0
        n = 0
        for group in groups:
            by_model = {r.model: r for r in group if r.source == "arena"}
            ra, rb = by_model.get(model_a), by_model.get(model_b)
            if ra is None or rb is None:
                continue
            n += 1
            sa = 1.0 if ra.outcome == Outcome.PASS.value else (
                0.5 if ra.outcome == Outcome.PARTIAL.value else 0.0
            )
            sb = 1.0 if rb.outcome == Outcome.PASS.value else (
                0.5 if rb.outcome == Outcome.PARTIAL.value else 0.0
            )
            if sa > sb:
                wins += 1.0
            elif sa == sb:
                wins += 0.5
        if n == 0:
            return False
        lo, hi = self.wilson_interval(int(round(wins)), n)
        return (hi - lo) < _SETTLE_WIDTH
