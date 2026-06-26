"""RightModel shadow mode — observe what the router WOULD do, without acting.

The zero-risk gate before live routing (Tier 4). A consumer that still picks its
own model calls ``observe(...)`` with the task and the model it actually used; the
shadow recorder:

  1. asks the router what it WOULD have picked (``router.choose`` — no LLM call),
  2. records the ACTUAL (model, outcome) as a normal production ledger row, so the
     matrix organically learns from whatever the live system is already doing, and
  3. appends a ShadowRecord (would-route vs actual + agreement) to a separate shadow
     log, so agreement can be measured over time.

No model is ever invoked here — shadow mode adds zero spend. When agreement is high
and divergences look sane, that is the signal to graduate one task class to live
``delegate()`` (Tier 5).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .rightmodel_router import RightModelRouter
from .rightmodel_types import rightmodel_home


@dataclass
class ShadowRecord:
    """One would-route vs actual-route comparison (no model was called)."""

    ts: str
    task_class: str
    actual_model: str
    would_model: str | None
    agree: bool
    actual_outcome: str
    would_reason: str


class ShadowRecorder:
    """Records would-route vs actual decisions to a shadow log + the main ledger."""

    def __init__(self, router: RightModelRouter | None = None, path: Path | None = None) -> None:
        self.router = router if router is not None else RightModelRouter()
        if path is None:
            path = rightmodel_home() / "shadow.jsonl"
        self.path = Path(path)

    def observe(
        self,
        *,
        task_class: str,
        taint_set: set,
        actual_model: str,
        actual_outcome: str,
        cost_usd: float = 0.0,
        verified: bool = False,
        lang: str = "en",
        run_id: str | None = None,
        origin: str = "shadow",
        reviewer: str = "",
        review_kind: str = "none",
    ) -> ShadowRecord:
        """Compare the router's would-pick to what actually ran; record both.

        Records the ACTUAL outcome as a real production ledger row (the live system's
        own data) and appends a ShadowRecord. Makes NO model call.
        """
        decision = self.router.choose(
            task_class=task_class, taint_set=taint_set, run_id=run_id, lang=lang
        )
        # The actual outcome is real production data for the model that ran.
        self.router.ledger.append(
            task_class=task_class,
            model=actual_model,
            outcome=actual_outcome,
            verified=verified,
            cost_usd=cost_usd,
            lang=lang,
            source="production",
            run_id=run_id,
            origin=origin,
            reviewer=reviewer,
            review_kind=review_kind,
        )
        record = ShadowRecord(
            ts=datetime.now(timezone.utc).isoformat(),
            task_class=task_class,
            actual_model=actual_model,
            would_model=decision.model,
            agree=(decision.model == actual_model),
            actual_outcome=actual_outcome,
            would_reason=decision.reason,
        )
        self._append(record)
        return record

    def _append(self, record: ShadowRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def records(self) -> list[ShadowRecord]:
        """Read the shadow log; robust to a missing/partial file."""
        if not self.path.exists():
            return []
        out: list[ShadowRecord] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(ShadowRecord(**json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        return out

    def agreement_report(self) -> dict:
        """Aggregate agreement: overall + per-class rate, and the divergences.

        A high agreement rate with sane divergences is the green light to graduate
        a class to live routing.
        """
        recs = self.records()
        total = len(recs)
        by_class: dict[str, dict[str, int]] = {}
        divergences: list[dict] = []
        agree = 0
        for r in recs:
            agree += int(r.agree)
            b = by_class.setdefault(r.task_class, {"n": 0, "agree": 0})
            b["n"] += 1
            b["agree"] += int(r.agree)
            if not r.agree:
                divergences.append(
                    {
                        "task_class": r.task_class,
                        "actual": r.actual_model,
                        "would": r.would_model,
                        "outcome": r.actual_outcome,
                    }
                )
        return {
            "total": total,
            "agreement_rate": (agree / total) if total else 0.0,
            "by_class": {
                k: {**v, "rate": (v["agree"] / v["n"]) if v["n"] else 0.0}
                for k, v in by_class.items()
            },
            "divergences": divergences,
        }
