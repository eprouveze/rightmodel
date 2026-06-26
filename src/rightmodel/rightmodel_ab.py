"""RightModel A/B sampler — the counterfactual dual-run (Workstream E2).

``rightmodel_shadow.py`` is observe-only: it asks the router what it WOULD pick and
records the ONE model that actually ran. It never runs two models, so it can never
answer "does fusion beat cost-min ON THE SAME TASK". This module does.

On a sampled fraction (``FUSION_AB_RATE``, default 0.1) of escalation-eligible cards,
``run_ab`` runs BOTH routes on the same task — the cost-min route AND the fusion route
— and tags both ledger rows with a shared ``collision_group`` (the duel mechanism
``rightmodel-delegate-card.py`` already exposes via ``--collision-group``), so the
capability matrix sees a real head-to-head pair instead of two unrelated solo rows.

The actual model calls are injected (``costmin_runner`` / ``fusion_runner``) so tests
need no network — each is a callable ``(card, *, collision_group) -> DelegationResult``
that itself records its own ledger row under that ``collision_group``.

``report()`` compares fusion vs cost-min STRATIFIED within each stakes bucket and never
across — fusion runs only on the hardest cards, so cross-bucket aggregation has a
selection bias that makes fusion look worse (architecture §1.5).
"""

from __future__ import annotations

import os
import random
import uuid
from dataclasses import dataclass

from .rightmodel_ledger import RightModelLedger


_DEFAULT_AB_RATE = 0.1


def _ab_rate() -> float:
    """Sampling fraction from FUSION_AB_RATE (default 0.1); clamped to [0, 1]."""
    try:
        rate = float(os.environ.get("FUSION_AB_RATE", _DEFAULT_AB_RATE))
    except (TypeError, ValueError):
        return _DEFAULT_AB_RATE
    return min(1.0, max(0.0, rate))


@dataclass
class AbRun:
    """One A/B dual-run: the shared collision_group + both route outcomes."""

    collision_group: str
    costmin: object
    fusion: object


class AbSampler:
    """Samples escalation-eligible cards and runs cost-min + fusion as a duel."""

    def __init__(
        self,
        *,
        costmin_runner=None,
        fusion_runner=None,
        ledger: RightModelLedger | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.costmin_runner = costmin_runner
        self.fusion_runner = fusion_runner
        self.ledger = ledger if ledger is not None else RightModelLedger()
        self.rng = rng if rng is not None else random.Random()

    def should_sample(self, *, sample_index: int | None = None) -> bool:
        """True on a ``FUSION_AB_RATE`` fraction of cards (deterministic when seeded).

        With an injected ``rng`` (or a fixed ``rng_seed``) the draw is reproducible; a
        ``sample_index`` instead gives a fully deterministic 1-in-N pattern for tests.
        """
        rate = _ab_rate()
        if rate <= 0.0:
            return False
        if rate >= 1.0:
            return True
        if sample_index is not None:
            period = max(1, round(1.0 / rate))
            return (sample_index % period) == 0
        return self.rng.random() < rate

    def run_ab(self, card, *, sample_index: int | None = None) -> AbRun | None:
        """If sampled, run BOTH routes on the same card under a shared collision_group.

        Returns the ``AbRun`` (both DelegationResults + the group) when the card was
        sampled, else ``None`` (no extra runs fired). The injected runners each record
        their own ledger row tagged with the shared group.
        """
        if not self.should_sample(sample_index=sample_index):
            return None
        if self.costmin_runner is None or self.fusion_runner is None:
            raise ValueError("run_ab requires both costmin_runner and fusion_runner")
        group = "ab-" + uuid.uuid4().hex[:12]
        costmin = self.costmin_runner(card, collision_group=group)
        fusion = self.fusion_runner(card, collision_group=group)
        return AbRun(collision_group=group, costmin=costmin, fusion=fusion)

    def report(self) -> dict:
        """Per-stakes-bucket fusion vs cost-min comparison from the A/B ledger rows.

        Only rows under an ``ab-`` collision_group are considered. Within each group a
        fusion row (``council:*`` model) is paired against the cost-min row(s); the
        bucket is the row's stakes — taken from the ``basis`` tag's ``stakes=<b>`` field,
        falling back to ``"unknown"``. Comparison is STRICTLY within a bucket, never
        across, because fusion runs only on the hardest cards (selection bias).
        """
        buckets: dict[str, dict[str, list[bool]]] = {}
        for group in self._ab_groups():
            for row in group:
                bucket = _bucket_of(row)
                side = "fusion" if _is_fusion(row.model) else "costmin"
                b = buckets.setdefault(bucket, {"fusion": [], "costmin": []})
                b[side].append(row.outcome == "pass")
        out: dict[str, dict] = {}
        for bucket, sides in buckets.items():
            f, c = sides["fusion"], sides["costmin"]
            fusion_wr = (sum(f) / len(f)) if f else 0.0
            costmin_wr = (sum(c) / len(c)) if c else 0.0
            out[bucket] = {
                "n": len(f) + len(c),
                "n_fusion": len(f),
                "n_costmin": len(c),
                "fusion_winrate": fusion_wr,
                "costmin_winrate": costmin_wr,
                "marginal_lift": fusion_wr - costmin_wr,
            }
        return out

    def _ab_groups(self) -> list[list]:
        """All rows grouped by an ``ab-`` collision_group (any size)."""
        groups: dict[str, list] = {}
        for row in self.ledger.rows():
            cg = row.collision_group
            if cg is None or not cg.startswith("ab-"):
                continue
            groups.setdefault(cg, []).append(row)
        return list(groups.values())


def _is_fusion(model: str | None) -> bool:
    return bool(model) and model.startswith("council:")


def _bucket_of(row) -> str:
    """Read the stakes bucket off a row's ``basis`` (``...stakes=<bucket>...``)."""
    basis = row.basis or ""
    for tok in basis.replace(";", " ").replace(",", " ").split():
        if tok.startswith("stakes="):
            return tok.split("=", 1)[1] or "unknown"
    return "unknown"
