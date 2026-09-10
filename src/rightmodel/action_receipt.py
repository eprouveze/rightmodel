"""ActionReceipt — Pydantic v2 model with 5-field metric contract (LexT R4).

Every action the agent takes writes a receipt. The 5 metric fields are mandatory:
without them, weekly cost-per-outcome reallocation has no data to act on.

PRD §6 + §10 mitigation.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger("foreman")


class TaintClass(str, Enum):
    """12-class PII taint enum. See PRD §7 for per-provider allowlists."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CUSTOMER_PII = "customer_pii"
    PROSPECT_PII = "prospect_pii"
    PAYMENT_DATA = "payment_data"
    VOICE_DNA = "voice_dna"
    INFERRED_PII = "inferred_pii"
    AUTH_SECRETS = "auth_secrets"
    PRIVATE_REPO_CODE = "private_repo_code"
    UNPUBLISHED_STRATEGY = "unpublished_strategy"
    STEALTH_IDENTITY_LINKAGE = "stealth_identity_linkage"
    FAMILY_PERSONAL = "family_personal"
    REPLY_CONTENT = "reply_content"


class ActionReceipt(BaseModel):
    """Durable record of one agent action.

    The 5-field metric contract (expected_metric, measurement_source, baseline,
    review_date, abort_condition) is mandatory — LexT R4: prevents the agent
    from producing actions with no verifiable outcome link.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)

    # Identity
    theme_id: str = Field(..., min_length=1, description="Theme that originated this action")
    brand_id: str = Field(..., min_length=1, description="identifier for the brand or product this action belongs to")
    action_id: str = Field(..., min_length=1, description="Unique within this run")
    run_id: str = Field(..., min_length=1, description="Pulse run identifier")

    # Execution
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str = Field(..., description="Model used (e.g. claude-sonnet-5, deepseek-v4-pro)")
    cost_usd: float = Field(..., ge=0.0, description="USD cost of this action")
    data_taint: set[TaintClass] = Field(
        default_factory=lambda: {TaintClass.INTERNAL},
        description="Union of taints carried by this action (propagates run-scoped)",
    )

    # Guardrails
    guardrails_passed: list[str] = Field(
        default_factory=list, description="Names of guardrails that ran and passed"
    )
    guardrails_failed: list[str] = Field(
        default_factory=list, description="Names of guardrails that triggered a block"
    )

    # External refs
    external_ids: dict[str, str] = Field(
        default_factory=dict,
        description="e.g. {'etsy_listing_id': '123', 'slack_thread_ts': '1234.567'}",
    )
    payload_hash: str | None = Field(None, description="SHA256 of the action payload for dedup")

    # 5-field metric contract — MANDATORY (LexT R4)
    expected_metric: str = Field(
        ..., min_length=1, description="What metric this action is expected to move"
    )
    measurement_source: str = Field(
        ..., min_length=1, description="Where the metric is measured (e.g. posthog.events.X)"
    )
    baseline: str = Field(
        ..., min_length=1, description="Current value of the metric pre-action"
    )
    review_date: date = Field(
        ..., description="When to check if expected_metric moved (must be future at write time)"
    )
    abort_condition: str = Field(
        ..., min_length=1, description="When to give up on this action-type (e.g. '3 sends no replies')"
    )

    # Optional decision-impact flag (anamnesis-closure-lite filter)
    decision_impact: bool = Field(
        False, description="True if this action carries a decision worth Anamnesis preservation"
    )

    @field_validator("data_taint", mode="before")
    @classmethod
    def _coerce_taint_set(cls, v: Any) -> set[TaintClass]:
        if isinstance(v, set):
            return v
        if isinstance(v, (list, tuple)):
            return {TaintClass(x) if isinstance(x, str) else x for x in v}
        if isinstance(v, str):
            return {TaintClass(v)}
        if isinstance(v, TaintClass):
            return {v}
        raise ValueError(f"data_taint must be set/list/str/TaintClass, got {type(v).__name__}")

    @field_validator("review_date")
    @classmethod
    def _review_date_must_be_future(cls, v: date) -> date:
        today = datetime.now(timezone.utc).date()
        if v < today:
            raise ValueError(f"review_date must be today or future, got {v} (today={today})")
        return v


_CLAMP_HORIZON_DAYS = 7


def review_in(days: int) -> date:
    """Forward review horizon for a new receipt, relative to today (UTC).

    Receipts must never hardcode calendar dates: a literal becomes invalid the
    moment 'today' passes it — the cause of the 2026-06-04..09 daily crash
    (review_date=date(2026, 6, 3) outliving its own validity in 6 themes).
    """
    return datetime.now(timezone.utc).date() + timedelta(days=days)


def safe_receipt(**kwargs: Any) -> ActionReceipt:
    """Two-mode receipt construction (council 2026-06-10, Fork 5 ruling).

    Mode 1 (default path): strict validation — a NEW forward commitment with a
    past review_date is an authoring bug and must surface.
    Mode 2 (runtime degradation): if validation fails ONLY on review_date, the
    receipt must not kill the theme — clamp to today+_CLAMP_HORIZON_DAYS,
    record the clamp in guardrails_failed so the outcome sweep can see it, and
    warn. Any other validation error still raises: those are real contract
    breaches, not staleness.
    """
    try:
        return ActionReceipt(**kwargs)
    except ValidationError as e:
        locs = {err.get("loc", ()) for err in e.errors()}
        if locs != {("review_date",)}:
            raise
        original = kwargs.get("review_date")
        kwargs["review_date"] = review_in(_CLAMP_HORIZON_DAYS)
        failed = list(kwargs.get("guardrails_failed") or [])
        failed.append(f"receipt.review_date_clamped:{original}")
        kwargs["guardrails_failed"] = failed
        logger.warning(
            f"receipt review_date {original!r} stale for theme "
            f"{kwargs.get('theme_id')!r} — clamped to {kwargs['review_date']} "
            f"(degrade-don't-die; flagged in guardrails_failed)"
        )
        return ActionReceipt(**kwargs)
