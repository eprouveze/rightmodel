"""RightModel capability matrix — per-class skill estimates from the ledger.

Module 4 of the RightModel routing substrate. Reads the outcome ledger
(`rightmodel_ledger`), fits a per-task-class strength model over *natural
collisions* (same task run by >1 model, linked by `collision_group`), blends the
fitted strength with the bootstrapped `SEED_PRIOR`, and exposes **distribution
cells** (μ ± σ) the router consults for the capability-bar and quality-premium
cascade stages.

MODEL IDENTITY (Emmanuel, 2026-06-16): a cell is keyed by ``(task_class, model_id)``.
``model_id`` IS the model's identity — claude-opus-4-7 and claude-opus-4-8 (or kimi-k2.6
vs k2.7) are *different models* with *different cells*, each cold-started and evaluated
from scratch. Capabilities are never blended across versions: a new version can regress,
so a blended estimate would describe neither. A new model is just a new model_id → a new
cold cell (automatic). ``version_hash`` is informational provenance only; it is never an
identity or re-eval key (see `rightmodel_freshness` IDENTITY PRINCIPLE).

Elo vs MM (Bradley–Terry minorization–maximization) — design choice
------------------------------------------------------------------
The contract permits either. We use **online Elo** (single sequential pass over
each collision's pairwise duels) rather than iterative MM. Rationale:

- **Stdlib-only / cheap.** Elo is one closed-form update per duel; MM needs many
  iterations and a normalization that's fiddly without numpy. At v1 ledger
  volumes the two converge to the same ordering, so the extra MM cost buys
  nothing measurable.
- **Difficulty weighting is natural.** Elo's K-factor absorbs the contract's
  `w = 0.5 + difficulty` weight directly (harder duels move strength more) and
  the synthetic half-weight just halves K for synthetic rows.
- **Incremental-friendly.** Elo composes with the append-only ledger without a
  full refit, which matches how the heartbeat will call `refresh()` cheaply.

The Elo strength is mapped onto a 0–100 scale and blended with the prior:

    mu = (PRIOR_WEIGHT * prior + n * bt_mu) / (PRIOR_WEIGHT + n)
    sigma = SIGMA_SCALE / sqrt(n + 1)

so a cold cell (n=0) sits exactly on its prior with maximum σ, and a settled cell
(n≫PRIOR_WEIGHT) is dominated by the data. Synthetic observations count toward n
at half weight versus production.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .model_prices import SEED_PRIOR
from .rightmodel_ledger import LedgerRow, RightModelLedger
from .rightmodel_types import TASK_CLASSES, CellState, Outcome

PRIOR_WEIGHT: float = 10.0
SIGMA_SCALE: float = 30.0
DEFAULT_MIN_BAR: float = 60.0
DEFAULT_PRIOR: float = 60.0

_ELO_BASE: float = 1500.0
_ELO_K: float = 24.0
_ELO_SPAN: float = 400.0
_SYNTHETIC_WEIGHT: float = 0.5
_SOLO_WEIGHT: float = 0.5


@dataclass
class Cell:
    """A single (task_class, model) capability distribution cell."""

    task_class: str
    model: str
    mu: float
    sigma: float
    n: int
    state: str
    version_hash: str
    lang_coverage: dict[str, int] = field(default_factory=dict)


def _cell_state(n: int) -> str:
    """Map effective observation count to a CellState value."""
    if n < 5:
        return CellState.COLD.value
    if n < 30:
        return CellState.WARMING.value
    return CellState.SETTLED.value


def _elo_to_mu(rating: float) -> float:
    """Map a latent Elo rating onto the 0-100 capability scale (logistic, centered 50)."""
    return 100.0 / (1.0 + 10.0 ** ((_ELO_BASE - rating) / _ELO_SPAN))


def _outcome_score(outcome: str) -> float:
    """Duel score for an outcome: PASS=1.0, PARTIAL=0.5, FAIL=0.0."""
    if outcome == Outcome.PASS.value:
        return 1.0
    if outcome == Outcome.PARTIAL.value:
        return 0.5
    return 0.0


class CapabilityMatrix:
    """Per-class capability estimates over the ledger, blended with SEED_PRIOR.

    Lazily computed and cached. Call `refresh()` after the ledger grows to
    recompute; `cell`/`rank` trigger a refresh on first use.
    """

    def __init__(self, ledger: RightModelLedger) -> None:
        self._ledger = ledger
        self._cells: dict[tuple[str, str], Cell] | None = None

    def refresh(self) -> None:
        """Recompute every cell from the current ledger contents."""
        cells: dict[tuple[str, str], Cell] = {}
        for task_class in TASK_CLASSES:
            cells.update(self._fit_class(task_class))
        self._cells = cells

    def _ensure(self) -> dict[tuple[str, str], Cell]:
        if self._cells is None:
            self.refresh()
        assert self._cells is not None
        return self._cells

    def _fit_class(self, task_class: str) -> dict[tuple[str, str], Cell]:
        rows = self._ledger.rows(task_class=task_class)
        ratings: dict[str, float] = {}
        eff_n: dict[str, float] = {}
        version_hash: dict[str, str] = {}
        lang_cov: dict[str, dict[str, int]] = {}
        collision_n: dict[str, float] = {}
        solo_weight: dict[str, float] = {}
        solo_score_sum: dict[str, float] = {}

        for row in rows:
            eff_n[row.model] = eff_n.get(row.model, 0.0) + (
                _SYNTHETIC_WEIGHT if row.source == "synthetic" else 1.0
            )
            if row.version_hash:
                version_hash[row.model] = row.version_hash
            cov = lang_cov.setdefault(row.model, {})
            cov[row.lang] = cov.get(row.lang, 0) + 1

            if row.collision_group is None:
                w = _SOLO_WEIGHT * (
                    _SYNTHETIC_WEIGHT if row.source == "synthetic" else 1.0
                )
                solo_weight[row.model] = solo_weight.get(row.model, 0.0) + w
                solo_score_sum[row.model] = solo_score_sum.get(
                    row.model, 0.0
                ) + w * _outcome_score(row.outcome)
            else:
                collision_n[row.model] = collision_n.get(row.model, 0.0) + (
                    _SYNTHETIC_WEIGHT if row.source == "synthetic" else 1.0
                )

        for group in self._ledger.collisions(task_class):
            self._update_ratings_from_group(group, ratings)

        priors = SEED_PRIOR.get(task_class, {})
        models = set(eff_n) | set(priors)
        out: dict[tuple[str, str], Cell] = {}
        for model in models:
            out[(task_class, model)] = self._build_cell(
                task_class,
                model,
                ratings.get(model),
                eff_n.get(model, 0.0),
                priors.get(model, DEFAULT_PRIOR),
                version_hash.get(model, ""),
                lang_cov.get(model, {}),
                collision_n.get(model, 0.0),
                solo_weight.get(model, 0.0),
                solo_score_sum.get(model, 0.0),
            )
        return out

    def _update_ratings_from_group(
        self, group: list[LedgerRow], ratings: dict[str, float]
    ) -> None:
        """Online-Elo update over every pair of distinct models in one collision."""
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a.model == b.model:
                    continue
                ra = ratings.setdefault(a.model, _ELO_BASE)
                rb = ratings.setdefault(b.model, _ELO_BASE)
                sa = _outcome_score(a.outcome)
                sb = _outcome_score(b.outcome)
                if sa == sb:
                    continue
                actual_a = 1.0 if sa > sb else 0.0
                expected_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / _ELO_SPAN))
                w = 0.5 + max(a.difficulty, b.difficulty)
                if a.source == "synthetic" or b.source == "synthetic":
                    w *= _SYNTHETIC_WEIGHT
                delta = _ELO_K * w * (actual_a - expected_a)
                ratings[a.model] = ra + delta
                ratings[b.model] = rb - delta

    def _build_cell(
        self,
        task_class: str,
        model: str,
        rating: float | None,
        eff_n: float,
        prior: float,
        version_hash: str,
        lang_coverage: dict[str, int],
        collision_n: float = 0.0,
        solo_weight: float = 0.0,
        solo_score_sum: float = 0.0,
    ) -> Cell:
        n = int(round(eff_n))
        bt_mu = _elo_to_mu(rating) if rating is not None else prior
        solo_mu = (
            100.0 * solo_score_sum / solo_weight if solo_weight > 0.0 else 0.0
        )
        mu = (
            PRIOR_WEIGHT * prior
            + collision_n * bt_mu
            + solo_weight * solo_mu
        ) / (PRIOR_WEIGHT + collision_n + solo_weight)
        sigma = SIGMA_SCALE / math.sqrt(eff_n + 1.0)
        return Cell(
            task_class=task_class,
            model=model,
            mu=mu,
            sigma=sigma,
            n=n,
            state=_cell_state(n),
            version_hash=version_hash,
            lang_coverage=dict(lang_coverage),
        )

    def cell(self, task_class: str, model: str) -> Cell:
        """Return the cell for (task_class, model). Cold cells fall back to prior."""
        cells = self._ensure()
        existing = cells.get((task_class, model))
        if existing is not None:
            return existing
        prior = SEED_PRIOR.get(task_class, {}).get(model, DEFAULT_PRIOR)
        return self._build_cell(task_class, model, None, 0.0, prior, "", {})

    def rank(self, task_class: str, candidates: list[str]) -> list[Cell]:
        """Return cells for the candidates, highest μ first (σ breaks ties, higher first)."""
        cells = [self.cell(task_class, m) for m in candidates]
        return sorted(cells, key=lambda c: (c.mu, c.sigma), reverse=True)

    def min_bar(self, task_class: str) -> float:
        """Capability bar for the class. Default 60.0; constant for v1."""
        return DEFAULT_MIN_BAR
