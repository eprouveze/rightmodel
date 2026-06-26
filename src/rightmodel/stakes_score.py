"""Stakes classifier — the cost-blind heuristic floor that triggers fusion.

Implements architecture §5a: four binary signals over a card's free-text fields
yield a `low|med|high` stakes label. The rater is deliberately separate from the
cost optimizer (§5a fix 1) — it never sees a model price, so a cost-aware loop
can't quietly score borderline cards down to dodge a fusion escalation.

`score()` is a transparent keyword/structure heuristic, not an LLM call — a floor,
not a ceiling; judgment on top still sets `stakes` at card-creation. Every scoring
pass emits a `StakesDecision` to the `stakes-decisions.jsonl` sidecar so the
near-threshold boundary is auditable (the data E3's weekly audit reads).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from .rightmodel_types import rightmodel_home


@dataclass
class StakesDecision:
    """One stakes scoring or routing decision (the audit sidecar row)."""

    ts: str
    kind: str
    card_id: str | None
    run_id: str | None
    stakes_in: str | None
    score: int | None
    signals: dict
    action: str
    near_threshold: bool
    resolved_stakes: str | None


def append_stakes_decision(record: StakesDecision) -> None:
    path = rightmodel_home() / "stakes-decisions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


_REVERSIBILITY_KEYWORDS = (
    "prod",
    "production",
    "customer",
    "client",
    "money",
    "payment",
    "billing",
    "stripe",
    "revenue",
    "security",
    "secret",
    "credential",
    "auth",
    "deploy",
    "publish",
    "send",
    "outward",
    "irreversible",
    "delete",
    "drop",
    "migration",
)

_BLAST_KEYWORDS = (
    "shared",
    "infra",
    "infrastructure",
    "fleet",
    "many files",
    "across",
    "consumers",
    "downstream",
    "global",
    "schema",
    "refactor",
    "rename",
    "breaking",
    "all projects",
    "every",
)

_OPEN_KEYWORDS = (
    "architecture",
    "architect",
    "research",
    "strategy",
    "strategic",
    "design",
    "open-ended",
    "open ended",
    "explore",
    "investigate",
    "approach",
    "tradeoff",
    "trade-off",
    "evaluate",
    "decide",
    "proposal",
)

_MECHANICAL_KEYWORDS = (
    "typo",
    "rename",
    "bump",
    "lint",
    "format",
    "one-line",
    "one line",
    "mechanical",
    "trivial",
    "comment",
    "whitespace",
)

_NOVELTY_KEYWORDS = (
    "novel",
    "new",
    "first time",
    "no precedent",
    "no pattern",
    "from scratch",
    "greenfield",
    "unprecedented",
    "never done",
    "experimental",
    "prototype",
)


def _card_text(card: dict) -> str:
    parts = [
        str(card.get("title", "")),
        str(card.get("note", "")),
        str(card.get("lane", "")),
    ]
    delegate = card.get("delegate")
    if isinstance(delegate, dict):
        parts.append(str(delegate.get("taint", "")))
    return " ".join(parts).lower()


def _any(text: str, keywords) -> bool:
    return any(kw in text for kw in keywords)


def score(card: dict) -> tuple[str, dict]:
    text = _card_text(card)

    irreversible = _any(text, _REVERSIBILITY_KEYWORDS)
    blast = _any(text, _BLAST_KEYWORDS)
    mechanical = _any(text, _MECHANICAL_KEYWORDS)
    open_space = _any(text, _OPEN_KEYWORDS) and not mechanical
    novel = _any(text, _NOVELTY_KEYWORDS)

    signals = {
        "reversibility": irreversible,
        "blast_radius": blast,
        "solution_space": "open" if open_space else "mechanical",
        "novelty": novel,
    }

    total = int(irreversible) + int(blast) + int(open_space) + int(novel)

    if total >= 3:
        label = "high"
    elif total >= 1:
        label = "med"
    else:
        label = "low"

    if irreversible or (novel and open_space):
        label = "high"

    append_stakes_decision(
        StakesDecision(
            ts=datetime.now(timezone.utc).isoformat(),
            kind="score",
            card_id=card.get("id"),
            run_id=card.get("run_id"),
            stakes_in=(card.get("delegate") or {}).get("stakes")
            if isinstance(card.get("delegate"), dict)
            else None,
            score=total,
            signals=signals,
            action="scored",
            near_threshold=(total == 2),
            resolved_stakes=label,
        )
    )

    return label, signals


def resolve_stakes(block: dict | None, card: dict) -> str | None:
    """Stakes for a delegate card: explicit ``delegate.stakes`` wins (judgment on top);
    otherwise derive the cost-blind floor from ``score(card)``.

    This is the card-side classifier pass the architecture's §5a calls for — it gives a
    card stakes at delegation time so a high-stakes card actually reaches the router's
    Stage-0 short-circuit instead of silently routing cost-min. Deriving here (not inside
    ``choose()``) keeps the rater with full card context and the cost optimizer separate.
    Returns None only when there is no card to score, leaving the caller's cost-min
    default intact (a non-card caller never auto-escalates, per §5a)."""
    explicit = (block or {}).get("stakes")
    if explicit:
        return str(explicit)
    if not card:
        return None
    label, _signals = score(card)
    return label
