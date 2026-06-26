"""Circuit breaker — per-provider availability gate for the RightModel cascade.

Stage 2 of the routing cascade (round 1 §2, LeK's pattern). Drops providers whose
breaker is OPEN before capability is even considered. Persisted as JSON at
`rightmodel_home()/breaker.json` so breaker state survives across heartbeat ticks
(a 402 that opened a provider at tick N must still be open at tick N+1).

Failure handling by kind:
    - "error"      — exponential backoff (base 30s, factor 2, cap 3600s) once
                     consecutive failures reach 3; below that the breaker stays
                     CLOSED (transient blips don't trip it).
    - "rate_limit" (429) — cooldown honours the provider's Retry-After when given,
                     else falls back to the current error backoff.
    - "payment"    (402) — OPEN immediately, cooldown 3600s (a billing failure
                     won't self-heal in seconds; stop hammering it).

Providers are keyed by plain `str` (the Provider value) to avoid an import cycle
with router_policy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path


_BACKOFF_BASE_S = 30.0
_BACKOFF_FACTOR = 2.0
_BACKOFF_CAP_S = 3600.0
_FAILURE_THRESHOLD = 3
_PAYMENT_COOLDOWN_S = 3600.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class BreakerState(str, Enum):
    """Lifecycle states of a per-provider breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ProviderBreaker:
    """Durable breaker record for one provider."""

    provider: str
    state: str
    consecutive_failures: int
    opened_at: str | None
    cooldown_until: str | None
    last_reason: str | None


def _default_path() -> Path:
    """Resolve the default breaker.json path under rightmodel_home().

    Imported lazily so this module loads even before Module 0 lands; tests pass
    an explicit path and never hit this.
    """
    from .rightmodel_types import rightmodel_home

    return rightmodel_home() / "breaker.json"


class CircuitBreaker:
    """JSON-persisted per-provider circuit breaker.

    State is loaded from disk on construction and rewritten on every mutation, so
    multiple heartbeat ticks (separate processes) share one breaker view.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else _default_path()
        self._breakers: dict[str, ProviderBreaker] = self._load()

    # ---- persistence ----------------------------------------------------------

    def _load(self) -> dict[str, ProviderBreaker]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        out: dict[str, ProviderBreaker] = {}
        if isinstance(raw, dict):
            for provider, rec in raw.items():
                if not isinstance(rec, dict):
                    continue
                out[provider] = ProviderBreaker(
                    provider=rec.get("provider", provider),
                    state=rec.get("state", BreakerState.CLOSED.value),
                    consecutive_failures=int(rec.get("consecutive_failures", 0)),
                    opened_at=rec.get("opened_at"),
                    cooldown_until=rec.get("cooldown_until"),
                    last_reason=rec.get("last_reason"),
                )
        return out

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {p: asdict(b) for p, b in self._breakers.items()}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def _get(self, provider: str) -> ProviderBreaker:
        b = self._breakers.get(provider)
        if b is None:
            b = ProviderBreaker(
                provider=provider,
                state=BreakerState.CLOSED.value,
                consecutive_failures=0,
                opened_at=None,
                cooldown_until=None,
                last_reason=None,
            )
            self._breakers[provider] = b
        return b

    # ---- queries --------------------------------------------------------------

    def is_available(self, provider: str) -> bool:
        """True if the provider may be routed to right now.

        CLOSED/HALF_OPEN → available. OPEN → available only once cooldown has
        elapsed, in which case the breaker flips to HALF_OPEN (allowing one probe)
        and this returns True. An OPEN breaker with no cooldown timestamp is
        treated as elapsed (fail-open on corrupt state).
        """
        b = self._breakers.get(provider)
        if b is None or b.state != BreakerState.OPEN.value:
            return True
        if b.cooldown_until is None:
            b.state = BreakerState.HALF_OPEN.value
            self._persist()
            return True
        cooldown = datetime.fromisoformat(b.cooldown_until)
        if _now() >= cooldown:
            b.state = BreakerState.HALF_OPEN.value
            self._persist()
            return True
        return False

    # ---- mutations ------------------------------------------------------------

    def record_success(self, provider: str) -> None:
        """Reset the provider's breaker to CLOSED with zero failures."""
        b = self._get(provider)
        b.state = BreakerState.CLOSED.value
        b.consecutive_failures = 0
        b.opened_at = None
        b.cooldown_until = None
        b.last_reason = None
        self._persist()

    def record_failure(
        self,
        provider: str,
        *,
        kind: str = "error",
        retry_after_s: float | None = None,
    ) -> None:
        """Record a failed call and update breaker state per failure kind.

        kind="payment" (402) opens immediately for 1h. kind="rate_limit" (429)
        honours retry_after_s when given. kind="error" trips exponential backoff
        only once consecutive_failures >= 3.
        """
        b = self._get(provider)
        b.consecutive_failures += 1
        now = _now()

        if kind == "payment":
            b.state = BreakerState.OPEN.value
            b.opened_at = _iso(now)
            b.cooldown_until = _iso(now + timedelta(seconds=_PAYMENT_COOLDOWN_S))
            b.last_reason = "payment_required(402)"
            self._persist()
            return

        if kind == "rate_limit":
            if retry_after_s is not None:
                cooldown_s = max(0.0, float(retry_after_s))
            else:
                cooldown_s = self._backoff_seconds(b.consecutive_failures)
            b.state = BreakerState.OPEN.value
            b.opened_at = _iso(now)
            b.cooldown_until = _iso(now + timedelta(seconds=cooldown_s))
            b.last_reason = f"rate_limit(429) cooldown={cooldown_s:.0f}s"
            self._persist()
            return

        if b.consecutive_failures >= _FAILURE_THRESHOLD:
            cooldown_s = self._backoff_seconds(b.consecutive_failures)
            b.state = BreakerState.OPEN.value
            b.opened_at = _iso(now)
            b.cooldown_until = _iso(now + timedelta(seconds=cooldown_s))
            b.last_reason = f"error backoff={cooldown_s:.0f}s (fails={b.consecutive_failures})"
        else:
            b.last_reason = f"error (fails={b.consecutive_failures}, below threshold)"
        self._persist()

    @staticmethod
    def _backoff_seconds(consecutive_failures: int) -> float:
        """Exponential backoff: min(3600, 30 * 2**(fails-3)) once fails>=3."""
        exponent = max(0, consecutive_failures - _FAILURE_THRESHOLD)
        return min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (_BACKOFF_FACTOR ** exponent))

    # ---- reporting ------------------------------------------------------------

    def snapshot(self) -> dict[str, ProviderBreaker]:
        """Return a copy of all known provider breakers for the report."""
        return {
            p: ProviderBreaker(**asdict(b)) for p, b in self._breakers.items()
        }
