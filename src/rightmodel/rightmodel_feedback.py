"""RightModel feedback ingestion — HITL ratings + /battle results into the ledger.

Two human-grounded quality signals that close the self-improvement loop:

  rate()         — Emmanuel (or any human) rates a delegated output. Lands as a
                   verified=True / review_kind="human" production row, which the matrix
                   weights ABOVE the LLM-judge — the gold signal the council wanted.
  ingest_battle()— a /battle run is a real head-to-head coding benchmark; its per-model
                   results land as one duel (shared collision_group), source="battle", so
                   the matrix learns coding ability from genuine competition, not just the
                   synthetic item pack.

Both append to the same ledger every other consumer reads, so they self-improve routing
the moment they land.
"""

from __future__ import annotations

from .rightmodel_ledger import LedgerRow, RightModelLedger
from .rightmodel_types import Outcome

_VERDICTS = {"pass": Outcome.PASS.value, "partial": Outcome.PARTIAL.value, "fail": Outcome.FAIL.value}


def _verdict_from(score) -> str:
    """Map a 'pass'/'partial'/'fail' string or a 0-100 number to an Outcome value."""
    if isinstance(score, str) and score.lower() in _VERDICTS:
        return _VERDICTS[score.lower()]
    try:
        n = float(score)
    except (TypeError, ValueError):
        return Outcome.FAIL.value
    if n >= 70:
        return Outcome.PASS.value
    if n >= 40:
        return Outcome.PARTIAL.value
    return Outcome.FAIL.value


def rate(*, task_class: str, model: str, score, ledger: RightModelLedger | None = None,
         lang: str = "en", reviewer: str = "emmanuel", difficulty: float = 0.5,
         cost_usd: float = 0.0, note: str = "") -> LedgerRow:
    """Record a HUMAN quality rating for a (task_class, model) output.

    Lands verified=True / review_kind="human" so it outranks any LLM-judge row in the
    matrix. ``score`` accepts 'pass'|'partial'|'fail' or a 0-100 number.
    """
    led = ledger if ledger is not None else RightModelLedger()
    return led.append(
        task_class=task_class, model=model, outcome=_verdict_from(score),
        verified=True, cost_usd=cost_usd, difficulty=difficulty, lang=lang,
        source="production", origin="hitl-rating", author=model,
        reviewer=reviewer, review_kind="human", basis="human-rating", note=note or None,
    )


def ingest_battle(task_id: str, results: list[dict], *, task_class: str = "coding",
                  ledger: RightModelLedger | None = None, difficulty: float = 0.7,
                  lang: str = "en", human_judged: bool = True,
                  reviewer: str = "emmanuel") -> list[LedgerRow]:
    """Ingest one /battle round as a duel — all models share a collision_group.

    ``results`` = list of dicts each with ``model`` and either ``passed`` (bool) or
    ``score`` (0-100). source="battle". When human_judged, rows are verified=True/human
    (a /battle leaderboard scored by Emmanuel); else verified=False (auto-scored battle).
    """
    led = ledger if ledger is not None else RightModelLedger()
    rows: list[LedgerRow] = []
    for r in results:
        model = r.get("model")
        if not model:
            continue
        if "passed" in r:
            outcome = Outcome.PASS.value if r["passed"] else Outcome.FAIL.value
        else:
            outcome = _verdict_from(r.get("score", 0))
        rows.append(led.append(
            task_class=task_class, model=model, outcome=outcome,
            verified=bool(human_judged), cost_usd=float(r.get("cost_usd", 0.0) or 0.0),
            difficulty=difficulty, lang=lang, source="battle",
            collision_group=task_id, origin=f"battle:{task_id}", author=model,
            reviewer=reviewer if human_judged else "",
            review_kind="human" if human_judged else "none", basis="battle-leaderboard",
        ))
    return rows
