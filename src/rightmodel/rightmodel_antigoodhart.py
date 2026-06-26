"""RightModel anti-Goodhart analyzers — flag, don't mutate routing.

The anti-Goodhart trio + bias check (round 2 §"biggest failure mode"). Each
analyzer reads the ledger (Module 2) and capability matrix (Module 4) and emits
HealthFlags. None of them change routing — they surface conditions that mean the
matrix may be optimizing a proxy instead of the real objective:

  - starvation: a model is winning volume while its cell stays under-observed
    (the matrix never gathers enough evidence to challenge it).
  - gold_regression: a settled-high model fails hand-crafted gold items.
  - escrow_divergence: a matrix-preferred model correlates with a degraded
    lagged business metric.
  - ja_bias: Japanese tasks systematically routed away from a non-Western model
    that scores higher on ja.

All methods are safe on an empty/young ledger — they return [] rather than raise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .rightmodel_types import CellState, TASK_CLASSES
from .router_policy import model_to_provider

_STARVATION_N = 20
_STARVATION_VOLUME_SHARE = 0.25
_STARVATION_WINDOW_DAYS = 90
_JA_MIN_N = 30
_JA_BIAS_MU_DELTA = 5.0
_WESTERN_PROVIDERS = frozenset({"claude", "openai", "gemini", "perplexity"})


@dataclass
class HealthFlag:
    """One anti-Goodhart finding over the ledger/matrix.

    kind is one of "starvation" | "escrow_divergence" | "gold_regression" |
    "ja_bias". severity gates downstream attention: "info" < "warn" < "critical".
    """

    kind: str
    task_class: str
    model: str | None
    detail: str
    severity: str


class AntiGoodhart:
    """Health analyzers over the RightModel ledger + capability matrix.

    Flags only; never mutates routing state. Construct with a RightModelLedger
    and a CapabilityMatrix and call the individual analyzers or report() to run
    them all.
    """

    def __init__(self, ledger, matrix) -> None:
        self.ledger = ledger
        self.matrix = matrix

    def starvation_flags(self) -> list[HealthFlag]:
        """Flag models taking >25% of a class's recent volume while cell n<20.

        Such a model is steering routing without enough observations to verify
        it deserves to — the matrix is starved of the duels that would test it.
        Recommendation embedded in detail: raise the Arena budget for the cell.
        """
        flags: list[HealthFlag] = []
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=_STARVATION_WINDOW_DAYS)
        ).isoformat()
        for task_class in TASK_CLASSES:
            recent = self.ledger.rows(task_class=task_class, since_iso=cutoff)
            total = len(recent)
            if total == 0:
                continue
            by_model: dict[str, int] = {}
            for row in recent:
                by_model[row.model] = by_model.get(row.model, 0) + 1
            for model, count in by_model.items():
                share = count / total
                if share <= _STARVATION_VOLUME_SHARE:
                    continue
                cell = self.matrix.cell(task_class, model)
                if cell.n >= _STARVATION_N:
                    continue
                flags.append(
                    HealthFlag(
                        kind="starvation",
                        task_class=task_class,
                        model=model,
                        detail=(
                            f"{model} took {share:.0%} of {task_class} volume "
                            f"(last {_STARVATION_WINDOW_DAYS}d) but cell n={cell.n}"
                            f" (<{_STARVATION_N}) — raise Arena budget for this cell"
                        ),
                        severity="warn",
                    )
                )
        return flags

    def gold_spotcheck(self, gold_results: list[dict]) -> list[HealthFlag]:
        """Flag settled-high models that fail hand-crafted gold items.

        gold_results = recent gold outcomes, each a dict with at least
        {"task_class", "model", "passed": bool}. A SETTLED model whose matrix μ
        is high yet fails a gold item is a regression worth a critical flag —
        gold items are weighted heavily and resist Goodharting the auto-scorer.
        """
        if not gold_results:
            return []
        flags: list[HealthFlag] = []
        for result in gold_results:
            task_class = result.get("task_class")
            model = result.get("model")
            passed = result.get("passed")
            if not task_class or not model or passed:
                continue
            cell = self.matrix.cell(task_class, model)
            if cell.state == CellState.SETTLED.value and cell.mu >= self.matrix.min_bar(
                task_class
            ):
                flags.append(
                    HealthFlag(
                        kind="gold_regression",
                        task_class=task_class,
                        model=model,
                        detail=(
                            f"{model} is SETTLED at μ={cell.mu:.0f} on {task_class}"
                            f" but failed gold item {result.get('id', '?')!r} — "
                            "possible auto-scorer Goodhart or model regression"
                        ),
                        severity="critical",
                    )
                )
        return flags

    def escrow_flags(self, downstream: dict[str, float] | None = None) -> list[HealthFlag]:
        """Flag matrix-preferred models tied to a degraded lagged business metric.

        Reserves attention for the 10% of the objective that only shows up in
        lagged downstream metrics. downstream maps a label to a delta where a
        negative value means the metric degraded. Stub-safe: None → []. Each
        degraded metric whose label names a candidate model yields a flag.
        """
        if not downstream:
            return []
        flags: list[HealthFlag] = []
        for label, delta in downstream.items():
            if delta >= 0:
                continue
            task_class = None
            model = label
            if "::" in label:
                task_class, model = label.split("::", 1)
            flags.append(
                HealthFlag(
                    kind="escrow_divergence",
                    task_class=task_class or "*",
                    model=model,
                    detail=(
                        f"{model} preferred but downstream metric {label!r} moved "
                        f"{delta:+.2f} — escrow check: verify the proxy still tracks value"
                    ),
                    severity="warn",
                )
            )
        return flags

    def ja_bias_check(self) -> list[HealthFlag]:
        """Flag Japanese tasks routed away from a higher-μ non-Western model.

        Grok's standing metric: at n>30 ja observations for a class, if a
        non-Western model has a higher ja μ yet ~0 ja volume, the router is
        culturally biased — flag it. Safe on young ledgers (returns [] until a
        class has enough ja evidence).
        """
        flags: list[HealthFlag] = []
        for task_class in TASK_CLASSES:
            ja_rows = [
                r for r in self.ledger.rows(task_class=task_class) if r.lang == "ja"
            ]
            if len(ja_rows) <= _JA_MIN_N:
                continue
            ja_volume: dict[str, int] = {}
            for row in ja_rows:
                ja_volume[row.model] = ja_volume.get(row.model, 0) + 1
            candidates = sorted(set(r.model for r in self.ledger.rows(task_class=task_class)))
            if not candidates:
                continue
            ranked = self.matrix.rank(task_class, candidates)
            picked = {m for m, v in ja_volume.items() if v > 0}
            best_picked_mu = max(
                (c.mu for c in ranked if c.model in picked), default=0.0
            )
            for cell in ranked:
                if model_to_provider(cell.model).value in _WESTERN_PROVIDERS:
                    continue
                if ja_volume.get(cell.model, 0) > 0:
                    continue
                if cell.mu - best_picked_mu > _JA_BIAS_MU_DELTA:
                    flags.append(
                        HealthFlag(
                            kind="ja_bias",
                            task_class=task_class,
                            model=cell.model,
                            detail=(
                                f"non-Western {cell.model} has higher ja μ="
                                f"{cell.mu:.0f} (vs picked {best_picked_mu:.0f}) on "
                                f"{task_class} yet ~0 ja volume — routing bias"
                            ),
                            severity="warn",
                        )
                    )
        return flags

    def report(
        self,
        gold_results: list[dict] | None = None,
        downstream: dict[str, float] | None = None,
    ) -> list[HealthFlag]:
        """Run every analyzer and return the concatenated flag list (may be [])."""
        flags: list[HealthFlag] = []
        flags.extend(self.starvation_flags())
        flags.extend(self.gold_spotcheck(gold_results or []))
        flags.extend(self.escrow_flags(downstream))
        flags.extend(self.ja_bias_check())
        return flags
