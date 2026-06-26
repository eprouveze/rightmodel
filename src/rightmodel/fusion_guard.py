"""Fusion runtime rails — daily budget, per-card leases, nested-fusion guard, seat breakers.

Stage of the orchestrator that gates fusion (council) escalation so the fleet can't
overspend, can't double-fire on one card, and can't recursively escalate from inside a
panel seat. All persistent state lives under `rightmodel_home()` (TCC-safe, host-local).

- **Budget** — a per-day counter at `fusion-<YYYY-MM-DD>.json`. `fusion_budget_ok()` is a
  pure check (does NOT burn budget); the caller commits a run with `record_fusion_run()`
  only once fusion actually fires. A check that decides not to escalate leaves the counter
  untouched. Cap is `FUSION_DAILY_CAP` (env, default 20).

- **Lease** — atomic `O_CREAT|O_EXCL` lock at `leases/fusion-<card_id>.lock`. A second
  concurrent `acquire_fusion_lease` for the same card fails while the first holds it. A
  lease older than `FUSION_LEASE_TTL_S` (env, default 1800) is reclaimable (a session that
  died mid-fusion doesn't wedge the card forever).

- **Nested-fusion guard** — `nested_fusion_forbidden()` returns True when
  `RIGHTMODEL_NO_FUSION=1`, which panel-seat subprocesses set so a seat can never re-escalate.

- **Seat breaker** — per-seat circuit breaker reusing `spine.circuit_breaker.CircuitBreaker`
  (seat name keys the provider slot), persisted at `seat-breaker.json`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .circuit_breaker import CircuitBreaker
from .rightmodel_types import rightmodel_home


_DEFAULT_DAILY_CAP = 20
_DEFAULT_LEASE_TTL_S = 1800.0


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _daily_cap() -> int:
    try:
        return int(os.environ.get("FUSION_DAILY_CAP", _DEFAULT_DAILY_CAP))
    except ValueError:
        return _DEFAULT_DAILY_CAP


def _lease_ttl_s() -> float:
    try:
        return float(os.environ.get("FUSION_LEASE_TTL_S", _DEFAULT_LEASE_TTL_S))
    except ValueError:
        return _DEFAULT_LEASE_TTL_S


def _budget_path() -> Path:
    return rightmodel_home() / f"fusion-budget-{_today()}.json"


def _read_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if isinstance(raw, dict):
        return int(raw.get("count", 0))
    return 0


def fusion_budget_ok() -> bool:
    """True if another fusion run would stay within today's cap.

    Pure check — does not increment the counter. The caller commits an actual run with
    `record_fusion_run()` so a check that doesn't escalate never burns budget.
    """
    return _read_count(_budget_path()) < _daily_cap()


def record_fusion_run() -> int:
    """Increment today's fusion counter and return the new count.

    Call exactly once per fusion run that actually fired.
    """
    path = _budget_path()
    count = _read_count(path) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"date": _today(), "count": count}, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
    return count


def _leases_dir() -> Path:
    return rightmodel_home() / "leases"


def _lease_path(card_id: str) -> Path:
    return _leases_dir() / f"fusion-{card_id}.lock"


def acquire_fusion_lease(card_id: str) -> bool:
    """Atomically claim the fusion lease for one card.

    Returns True if the lease was taken, False if another holder is live. A held lease
    older than `FUSION_LEASE_TTL_S` is reclaimed (stale holder presumed dead).
    """
    path = _lease_path(card_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            age = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        except OSError:
            return False
        if age < _lease_ttl_s():
            return False
        try:
            path.unlink()
        except OSError:
            return False
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"card_id": card_id, "pid": os.getpid(), "acquired_at": datetime.now(timezone.utc).isoformat()}
            )
        )
    return True


def release_fusion_lease(card_id: str) -> None:
    """Drop the fusion lease for a card. Idempotent (a missing lease is fine)."""
    try:
        _lease_path(card_id).unlink()
    except OSError:
        pass


def nested_fusion_forbidden() -> bool:
    """True when RIGHTMODEL_NO_FUSION=1 — a panel seat must never re-escalate."""
    return os.environ.get("RIGHTMODEL_NO_FUSION") == "1"


def _seat_breaker_path() -> Path:
    return rightmodel_home() / "seat-breaker.json"


def seat_breaker_open(seat: str) -> bool:
    """True if the seat is in cooldown and must be dropped from the panel."""
    return not CircuitBreaker(path=_seat_breaker_path()).is_available(seat)


def record_seat_failure(seat: str) -> None:
    """Record a seat timeout/auth-failure toward its breaker threshold."""
    CircuitBreaker(path=_seat_breaker_path()).record_failure(seat)


def record_seat_success(seat: str) -> None:
    """Reset the seat's breaker after a clean panel response."""
    CircuitBreaker(path=_seat_breaker_path()).record_success(seat)
