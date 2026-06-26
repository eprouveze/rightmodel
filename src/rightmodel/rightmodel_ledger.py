"""RightModel ledger — per-delegation outcome ledger (JSONL append + fsync).

One row per delegated (or synthetic/arena) task. Everything downstream in the
RightModel substrate (capability matrix, anti-Goodhart analyzers, the report)
reads this ledger, so it is the canonical record of "what model did what, how
well, at what cost".

Storage: append-only JSONL at ``rightmodel_home()/ledger.jsonl``. Each append is
flushed and fsync'd so a tick that dies mid-write does not lose committed rows.
Reads are robust to a missing or partially-written file (bad lines are skipped).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .rightmodel_types import rightmodel_home
from .router_policy import model_to_provider


@dataclass
class LedgerRow:
    """One outcome observation for a (task_class, model) delegation.

    ``collision_group`` links rows where >1 model ran the SAME task (a natural
    duel) — shared value = same item, consumed by Bradley–Terry fitting.

    Provenance attribution (origin/basis/review + author IDs) — pure metadata the
    matrix/anti-Goodhart layers read to weight trust in an outcome:
      ``author``      — who produced the work (model id for AI, human id for a person)
      ``reviewer``    — who reviewed/verified the outcome ("" when none)
      ``review_kind`` — "none" | "self" | "ai-peer" | "human"
      ``origin``      — provenance tag of the producing context (e.g. "heartbeat",
                        "foreman", "arena", "synthetic")
      ``basis``       — short reference to the evidence the outcome rests on
    These fields ATTRIBUTE; they do not gate. ``verified`` is a factual signal set
    by the caller (independently-checked vs self-reported), recorded as given. The
    "AI cannot approve its own work" auto-land rule is Proof-specific governance and
    lives in Proof — RightModel does not enforce it on the capability ledger. New
    fields are defaulted so pre-provenance JSONL rows still deserialize.
    """

    id: str
    ts: str
    task_class: str
    model: str
    provider: str
    outcome: str
    verified: bool
    cost_usd: float
    difficulty: float
    lang: str
    source: str
    version_hash: str
    run_id: str | None
    collision_group: str | None
    note: str | None
    author: str = ""
    reviewer: str = ""
    review_kind: str = "none"
    origin: str = "agent"
    basis: str = ""


def is_independently_reviewed(*, author: str, reviewer: str, review_kind: str) -> bool:
    """Descriptive helper: did a genuine second party review this outcome?

    True when review_kind is ai-peer/human, a reviewer is recorded, and the
    reviewer differs from the author. This only *describes* the provenance — it
    does NOT gate ``verified`` (that is the caller's factual call) and is not the
    Proof 'AI cannot approve its own work' rule, which lives in Proof. Callers may
    use it to decide what ``verified`` value to record.
    """
    return bool(
        review_kind in ("ai-peer", "human") and reviewer and reviewer != author
    )


class RightModelLedger:
    """Append-only JSONL ledger of delegation outcomes."""

    def __init__(self, path: Path | None = None) -> None:
        """Default path is ``rightmodel_home()/ledger.jsonl`` (dir ensured)."""
        if path is None:
            path = rightmodel_home() / "ledger.jsonl"
        self.path = Path(path)

    def record(self, row: LedgerRow) -> None:
        """Append one row as a JSONL line, flushing and fsync'ing to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(row), ensure_ascii=False, separators=(",", ":"))
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def append(
        self,
        *,
        task_class: str,
        model: str,
        outcome: str,
        verified: bool,
        cost_usd: float,
        difficulty: float = 0.5,
        lang: str = "en",
        source: str = "production",
        version_hash: str = "",
        run_id: str | None = None,
        collision_group: str | None = None,
        note: str | None = None,
        author: str = "",
        reviewer: str = "",
        review_kind: str = "none",
        origin: str = "agent",
        basis: str = "",
    ) -> LedgerRow:
        """Build id/ts/provider, append the row, and return it.

        ``verified`` is recorded as given — a factual signal (independently-checked
        vs self-reported), not a governance gate. The provenance fields attribute
        who produced/reviewed it. ``author`` defaults to ``model`` when not given.
        """
        author = author or model
        row = LedgerRow(
            id="rm-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:6],
            ts=datetime.now(timezone.utc).isoformat(),
            task_class=task_class,
            model=model,
            provider=model_to_provider(model).value,
            outcome=outcome,
            verified=verified,
            cost_usd=cost_usd,
            difficulty=difficulty,
            lang=lang,
            source=source,
            version_hash=version_hash,
            run_id=run_id,
            collision_group=collision_group,
            note=note,
            author=author,
            reviewer=reviewer,
            review_kind=review_kind,
            origin=origin,
            basis=basis,
        )
        self.record(row)
        return row

    def _iter_rows(self) -> list[LedgerRow]:
        """Read every well-formed row; skip blank/corrupt lines; [] if missing."""
        if not self.path.exists():
            return []
        out: list[LedgerRow] = []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    try:
                        out.append(LedgerRow(**data))
                    except TypeError:
                        continue
        except OSError:
            return out
        return out

    def rows(
        self,
        *,
        task_class: str | None = None,
        model: str | None = None,
        source: str | None = None,
        since_iso: str | None = None,
    ) -> list[LedgerRow]:
        """Return rows matching the given filters (all None → everything)."""
        result = self._iter_rows()
        if task_class is not None:
            result = [r for r in result if r.task_class == task_class]
        if model is not None:
            result = [r for r in result if r.model == model]
        if source is not None:
            result = [r for r in result if r.source == source]
        if since_iso is not None:
            result = [r for r in result if r.ts >= since_iso]
        return result

    def collisions(self, task_class: str) -> list[list[LedgerRow]]:
        """Group rows of ``task_class`` by non-null collision_group.

        Returns one list per group that has >=2 rows (a real duel). Order of
        groups follows first-seen order; rows within a group keep ledger order.
        """
        groups: dict[str, list[LedgerRow]] = {}
        for r in self.rows(task_class=task_class):
            if r.collision_group is None:
                continue
            groups.setdefault(r.collision_group, []).append(r)
        return [g for g in groups.values() if len(g) >= 2]

    def stats(self) -> dict:
        """Counts by task_class, model, and source for the substrate report."""
        rows = self._iter_rows()
        by_class: dict[str, int] = {}
        by_model: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for r in rows:
            by_class[r.task_class] = by_class.get(r.task_class, 0) + 1
            by_model[r.model] = by_model.get(r.model, 0) + 1
            by_source[r.source] = by_source.get(r.source, 0) + 1
        return {
            "total": len(rows),
            "by_class": by_class,
            "by_model": by_model,
            "by_source": by_source,
        }
