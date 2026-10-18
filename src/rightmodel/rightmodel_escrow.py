"""RightModel escrow join — feed lagged world-verdicts into anti-Goodhart escrow.

The **outcome ledger** (`docs/outcomes/ledger.jsonl`) records falsifiable predictions about
the world and their verdicts; the **RightModel ledger** records which model produced each
delegated action. This module is the READ-SIDE JOIN between the two — **not a store merge**
(see `docs/solutions/architecture/system-integration.md` §"Two ledgers"). It attributes a
settled world-verdict back to the `(task_class, model)` that produced the action and returns the
``{"task_class::model": net_delta}`` dict ``AntiGoodhart.escrow_flags`` already consumes, where
``net_delta < 0`` means "the actions this model produced have been failing in the world" — the
escrow signal that the matrix's quality proxy may have drifted from real value.

**Link key.** The join key is ``rightmodel_run_id`` on the outcome prediction — the ``run_id`` of
the RightModel delegation the prediction is the world-effect of (stamped by the acting session that
both delegated and predicted; an additive, nullable field). A card-scoped prediction may instead
carry ``card_id`` and join on the RightModel row's ``origin == "heartbeat-card:<id>"``.

The join is **sparse by design**: most predictions have no delegation behind them and most
delegations have no world-prediction (synthetic eval, arena votes, low-stakes runs). Escrow is the
~10% of the objective that only shows up in lagged metrics. This module NEVER mutates routing or the
matrix — its output is advisory, exactly like the rest of the anti-Goodhart layer.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

# A terminal world-verdict scores +1 (confirmed) or -1 (failed). UNVERIFIED is NOT yet a fact
# (the first law: a tool failure / unresolved horizon is never a negative fact) → skipped.
_TERMINAL_SCORE = {"CONFIRMED": 1.0, "FAILED": -1.0}

_CARD_ORIGIN_PREFIX = "heartbeat-card:"


def _repo_root() -> Path:
    # src/rightmodel/rightmodel_escrow.py -> src/rightmodel -> repo root is 2 parents up.
    return Path(__file__).resolve().parents[2]


def _outcome_ledger_path() -> Path:
    return _repo_root() / "docs" / "outcomes" / "ledger.jsonl"


def read_outcome_rows(path: Path | None = None) -> list[dict]:
    """Parse the outcome ledger JSONL into dicts. Tolerant: blank/unparseable lines skipped,
    a missing file → ``[]`` (cold-start safe)."""
    p = path if path is not None else _outcome_ledger_path()
    rows: list[dict] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _status_of(row: dict) -> str | None:
    """Terminal verdict of an outcome row, or None. Tolerates both the flat schema
    (``status`` at root) and the nested one (``verification.status``)."""
    status = row.get("status")
    if status is None:
        v = row.get("verification")
        if isinstance(v, dict):
            status = v.get("status")
    return status.upper() if isinstance(status, str) else None


def _verdict_ts(row: dict) -> str:
    """When the verdict landed (for the recency window): verified_at, else ts, else ''."""
    return str(row.get("verified_at") or row.get("ts") or "")


def build_downstream(
    outcome_rows: list[dict],
    rm_ledger,
    *,
    window_days: int = 30,
    min_n: int = 3,
) -> dict[str, float]:
    """Join settled world-verdicts to the model that produced them.

    Returns ``{"task_class::model": net_delta}`` over outcome verdicts within
    ``window_days``, where ``net_delta = mean(+1 confirmed / -1 failed)`` and only
    ``(task_class, model)`` pairs with at least ``min_n`` joined verdicts are emitted.
    A negative value is the escrow signal. Sparse by design; cold-start safe (``{}``).

    The result is exactly the shape ``AntiGoodhart.escrow_flags(downstream=...)`` consumes,
    so no change to the consumer is required. A fusion action is attributed to its pseudo-model
    (``council:budget-trio``) — the tier is flag-able as a unit.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()

    # Index the RightModel ledger by the two possible join keys. Later rows win (a retry under
    # the same run_id attributes to its most recent attempt).
    by_run: dict[str, object] = {}
    by_card: dict[str, object] = {}
    for r in rm_ledger.rows():
        if r.run_id:
            by_run[str(r.run_id)] = r
        if r.origin and r.origin.startswith(_CARD_ORIGIN_PREFIX):
            card_id = r.origin[len(_CARD_ORIGIN_PREFIX):]
            if card_id:
                by_card[card_id] = r

    buckets: dict[str, list[float]] = {}
    for o in outcome_rows:
        ts = _verdict_ts(o)
        if ts and ts < cutoff:
            continue
        score = _TERMINAL_SCORE.get(_status_of(o) or "")
        if score is None:  # UNVERIFIED / unknown — not a terminal world-fact
            continue
        rm = None
        rid = o.get("rightmodel_run_id")
        if rid is not None and str(rid) in by_run:
            rm = by_run[str(rid)]
        else:
            cid = o.get("card_id")
            if cid is not None and str(cid) in by_card:
                rm = by_card[str(cid)]
        if rm is None:  # a prediction with no delegation behind it is not an escrow signal
            continue
        buckets.setdefault(f"{rm.task_class}::{rm.model}", []).append(score)

    return {
        label: sum(scores) / len(scores)
        for label, scores in buckets.items()
        if len(scores) >= min_n
    }
