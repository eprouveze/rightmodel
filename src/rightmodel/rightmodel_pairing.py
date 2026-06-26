"""Deliberate duels to fill sparse cells — RightModel Module 11 (v2 addendum).

The ~5% complement to natural collisions (round 2 §3). Natural collisions only
form when two models happen to run the same task; sparse cells (cold/high-σ
models in a class) may never collide on their own. This module proposes
deliberate model-vs-model duels, ranked by expected information gain, capped at a
per-call slice of a small Thompson-flavoured budget.

The info-gain heuristic rewards duels that are *uncertain* (both cells carry
σ mass) and *overlapping* (their μ±σ confidence intervals overlap, so the duel
genuinely discriminates). A SETTLED-confident cell — low σ — contributes ~0 gain;
re-probing it wastes an inference. A duel is only proposed when its expected σ
reduction, priced at `gain_per_cost`, clears the second inference's cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model_prices import CANDIDATES, blended_cost
from .rightmodel_matrix import CapabilityMatrix, Cell


@dataclass
class PairProposal:
    """One proposed deliberate duel between two models on a task class."""

    task_class: str
    model_a: str
    model_b: str
    expected_info_gain: float
    reason: str


def _ci_overlap(cell_a: Cell, cell_b: Cell) -> float:
    """Overlap width of the two μ±σ confidence intervals (0 if disjoint)."""
    lo = max(cell_a.mu - cell_a.sigma, cell_b.mu - cell_b.sigma)
    hi = min(cell_a.mu + cell_a.sigma, cell_b.mu + cell_b.sigma)
    return max(0.0, hi - lo)


def expected_sigma_reduction(cell_a: Cell, cell_b: Cell) -> float:
    """Heuristic info-gain for a duel between two cells.

    Grows with the combined σ (uncertain cells are informative) and with the
    fraction of their CIs that overlap (overlapping cells genuinely discriminate
    — a duel between a clearly-strong and a clearly-weak model teaches little).
    Returns ~0 when either cell is SETTLED-confident (its σ has collapsed), so
    re-probing a well-known cell scores no gain.
    """
    combined_sigma = cell_a.sigma + cell_b.sigma
    if combined_sigma <= 0.0:
        return 0.0
    overlap = _ci_overlap(cell_a, cell_b)
    overlap_frac = overlap / combined_sigma
    return combined_sigma * overlap_frac


def should_pair(
    cell_a: Cell,
    cell_b: Cell,
    *,
    second_inference_cost: float,
    gain_per_cost: float = 50.0,
) -> bool:
    """Pair only when the priced expected σ reduction beats the duel's cost.

    `gain_per_cost` converts info-gain (σ points) into a comparable USD scale.
    An infinite/unknown second-inference cost never pairs.
    """
    gain = expected_sigma_reduction(cell_a, cell_b)
    return gain * gain_per_cost > second_inference_cost


def propose_pairs(
    matrix: CapabilityMatrix,
    task_class: str,
    *,
    candidates: list[str] | None = None,
    budget_frac: float = 0.05,
    max_pairs: int = 3,
    second_inference_cost_fn=None,
) -> list[PairProposal]:
    """Rank candidate pairs by info-gain, keep those passing `should_pair`, cap.

    Thompson-flavoured: every distinct candidate pair is scored by
    `expected_sigma_reduction`, sorted high-to-low, filtered by `should_pair`
    (the second model's `blended_cost` is the per-pair inference cost unless a
    `second_inference_cost_fn` overrides it), then truncated to `max_pairs` —
    the per-call slice of the `budget_frac` (default 5%) deliberate-duel budget.

    Cold-start safe: with an empty ledger every cell sits on its prior at max σ,
    so overlapping priors still yield proposals.
    """
    if candidates is None:
        candidates = CANDIDATES.get(task_class, [])
    if second_inference_cost_fn is None:
        second_inference_cost_fn = blended_cost

    proposals: list[PairProposal] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            model_a, model_b = candidates[i], candidates[j]
            cell_a = matrix.cell(task_class, model_a)
            cell_b = matrix.cell(task_class, model_b)
            gain = expected_sigma_reduction(cell_a, cell_b)
            cost = second_inference_cost_fn(model_b)
            if not should_pair(cell_a, cell_b, second_inference_cost=cost):
                continue
            proposals.append(
                PairProposal(
                    task_class=task_class,
                    model_a=model_a,
                    model_b=model_b,
                    expected_info_gain=gain,
                    reason=(
                        f"σ_a={cell_a.sigma:.1f} ({cell_a.state}) "
                        f"σ_b={cell_b.sigma:.1f} ({cell_b.state}) "
                        f"gain={gain:.2f} cost={cost:.3f}"
                    ),
                )
            )

    proposals.sort(key=lambda p: p.expected_info_gain, reverse=True)
    return proposals[:max_pairs]
