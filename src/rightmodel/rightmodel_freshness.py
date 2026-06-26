"""RightModel anti-staleness / live-mutation — freshness flags + item rotation (Module 10).

Round 2 §6. Detects when the matrix's predictions have diverged from observed reality
(`prediction_drift`), and keeps the synthetic item-pack fresh by rotating in anonymized
production-derived items while a sealed per-class holdout is preserved untouched.

IDENTITY PRINCIPLE (Emmanuel, 2026-06-16): **a model's identity is its `model_id`.** A
different id (claude-opus-4-7 vs 4-8, kimi-k2.6 vs k2.7) is a *different model* and gets its
own cell, cold-started from scratch — capabilities are NEVER blended across versions, because
a new version can regress and a blend would describe neither model. Cells are keyed by
`model_id`, so this is automatic. We therefore deliberately do **not** have a `version_hash`-
based re-eval trigger (it would re-open and blend a stable id's data — removed 2026-06-16).
The only staleness signal is behavioral: `prediction_drift`, whose correct remedy is to RESET
the model's cell (archive its old rows, cold-start), not to append-and-blend. `version_hash`
remains a purely informational provenance field on rows; nothing routes or merges on it.

This module never mutates routing — it only flags (for the heartbeat / report) and
proposes item-pack changes. All flag functions are safe on empty/young inputs (return []).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .rightmodel_types import TASK_CLASSES

_DRIFT_SEVERITY: str = "warn"
_MIN_DRIFT_ROWS: int = 5


@dataclass
class FreshnessFlag:
    """A single freshness signal raised over the matrix/ledger."""

    kind: str       # "prediction_drift"
    task_class: str
    model: str | None
    detail: str
    severity: str   # "info" | "warn"


def _classes(classes: list[str] | None) -> list[str]:
    return list(classes) if classes is not None else list(TASK_CLASSES)


def prediction_drift(
    ledger,
    matrix,
    *,
    window: int = 30,
    threshold: float = 0.10,
    classes: list[str] | None = None,
) -> list[FreshnessFlag]:
    """Flag classes where predicted success diverges from observed pass-rate.

    Over the last ``window`` production rows per class, compare mean predicted
    success (the routed model's ``cell.mu/100``) against observed pass-rate.
    ``|pred - obs| > threshold`` raises a re-probe flag. Classes with fewer than
    ``_MIN_DRIFT_ROWS`` production rows are too young to judge and are skipped.
    """
    flags: list[FreshnessFlag] = []
    for task_class in _classes(classes):
        rows = [r for r in ledger.rows(task_class=task_class, source="production")]
        rows = rows[-window:]
        if len(rows) < _MIN_DRIFT_ROWS:
            continue
        preds = [matrix.cell(task_class, r.model).mu / 100.0 for r in rows]
        pred = sum(preds) / len(preds)
        obs = sum(1 for r in rows if r.outcome == "pass") / len(rows)
        gap = abs(pred - obs)
        if gap > threshold:
            flags.append(
                FreshnessFlag(
                    kind="prediction_drift",
                    task_class=task_class,
                    model=None,
                    detail=(
                        f"predicted {pred:.2f} vs observed {obs:.2f} pass-rate over "
                        f"{len(rows)} rows (gap {gap:.2f} > {threshold:.2f}); re-probe"
                    ),
                    severity=_DRIFT_SEVERITY,
                )
            )
    return flags


def sealed_holdout(item_pack: list, *, k_per_class: int = 2) -> set[str]:
    """Designate a stable per-class holdout set of item ids rotate_items never replaces.

    Selection is deterministic (lowest ids per class), so the holdout is identical
    across runs and machines. Items must expose ``id`` and ``task_class`` attributes.
    """
    by_class: dict[str, list[str]] = {}
    for item in item_pack:
        by_class.setdefault(item.task_class, []).append(item.id)
    held: set[str] = set()
    for ids in by_class.values():
        held.update(sorted(ids)[:k_per_class])
    return held


def _production_item(sample: dict, task_class: str, index: int) -> dict:
    """Build an EvalItem-shaped dict from one production sample.

    WARNING: the prompt is copied VERBATIM — this does NOT anonymize/scrub PII.
    Only the id is hashed. ``rotate_items`` is library-only (no live caller); a PII
    scrubbing pass MUST be added before any consumer wires production samples in.
    See ``rightmodel-deferred.md`` (production-sample ingestion / anonymization).
    """
    prompt = str(sample.get("prompt", ""))
    digest = hashlib.sha256(f"{task_class}|{prompt}".encode("utf-8")).hexdigest()[:8]
    return {
        "id": f"prod-{task_class}-{digest}-{index}",
        "task_class": task_class,
        "prompt": prompt,
        "difficulty": float(sample.get("difficulty", 0.5)),
        "lang": str(sample.get("lang", "en")),
        "grader": str(sample.get("grader", "contains")),
        "expected": str(sample.get("expected", "")),
    }


def rotate_items(
    item_pack: list,
    production_samples: list[dict],
    *,
    frac: float = 0.20,
    seed: int,
    holdout_ids: set[str] | None = None,
) -> list:
    """Replace ~``frac`` of non-holdout synthetic items with production-derived items.

    Deterministic via ``seed``: the same inputs always yield the same output. Holdout
    items (by id) are never replaced. Returns a new list the same length as
    ``item_pack``, preserving original positions for kept items. Replacements are
    EvalItem-shaped dicts derived from ``production_samples``. With no production
    samples to draw from, the original pack is returned unchanged.

    NOTE: production prompts are copied verbatim (see ``_production_item``) — PII
    scrubbing is a prerequisite before wiring a real production source. No live
    caller today; this is a library primitive.
    """
    held = holdout_ids or set()
    rng = random.Random(seed)
    eligible = [i for i, it in enumerate(item_pack) if it.id not in held]
    n_replace = int(len(eligible) * frac)
    if n_replace <= 0 or not production_samples:
        return list(item_pack)
    chosen = sorted(rng.sample(eligible, min(n_replace, len(eligible))))
    out = list(item_pack)
    for k, pos in enumerate(chosen):
        sample = production_samples[k % len(production_samples)]
        task_class = item_pack[pos].task_class
        out[pos] = _production_item(sample, task_class, pos)
    return out
