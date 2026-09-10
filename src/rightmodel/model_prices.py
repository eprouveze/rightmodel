"""Model prices + bootstrapped capability prior (RightModel Module 1, no deps).

Compact price table (USD/1M tokens, source: model-scan 2026-06-15), a blended-cost
helper for ranking, the weak SEED_PRIOR capability matrix the ledger overrides as
production volume arrives, and the admin-curated CANDIDATES routing pool per task class.

All capability numbers are LOW confidence — qualitative council knowledge on a 0-100
scale, kept inside a 55-90 band so the learned posterior can move freely.
"""

from __future__ import annotations

# USD per 1M tokens, (input, output). Source: model-scan 2026-06-15.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.2, 1.25),
    "gpt-5.3-codex": (1.75, 14.0),
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.3, 2.5),
    "deepseek-v4-pro": (0.435, 0.87),
    "deepseek-v4-flash": (0.09, 0.18),
    "kimi-k2.6": (0.68, 3.41),
    "minimax-m3": (0.3, 1.2),
    "glm-5.2": (0.6, 2.2),  # EST — z.ai (Zhipu) GLM tier, pending /model-scan
    "sonar-pro": (3.0, 15.0),
    # Fusion pseudo-models — NOT real per-token models and NEVER in CANDIDATES (cost-min must
    # never pick them). Registered as (0.0, 0.0) only so existing PRICES/blended_cost/matrix
    # lookups never KeyError; their REAL cost is the A2 summed marginal seat cost at run time.
    # "budget-trio" is a label, not a fixed roster: the actual panel is composed at run time by
    # RightModelRouter.choose_panel (cheapest above-bar model per provider for the task class).
    "council:budget-trio": (0.0, 0.0),
    "council:self-lite": (0.0, 0.0),
}


def blended_cost(model: str) -> float:
    """Blended ranking cost (USD/1M): input*0.25 + output*0.75.

    Cheaper = lower. Unknown model → float("inf") so it sorts last and never wins
    a cost-min tie. Weighting favours output tokens, the dominant cost in practice.
    """
    price = PRICES.get(model)
    if price is None:
        return float("inf")
    input_cost, output_cost = price
    return input_cost * 0.25 + output_cost * 0.75


# --- Subscription-seat awareness ---------------------------------------------
# PRICES is a metered API rate card. Many deployments reach some models through a
# flat-rate subscription CLI/UI instead of the metered API (e.g. a Claude Max /
# ChatGPT Plus / Google AI subscription), where the true marginal cost of a call
# is $0 even though the same model has a nonzero entry in PRICES. Ranking those
# calls on the metered rate card inverts cost-min routing: subscription-backed
# models look like the MOST expensive options and lose to models that are
# actually billed per token but happen to have a low sticker price. If your
# deployment routes some model families through a subscription vehicle, wire
# that mapping into `vehicle_for` below and prefer `SUBSCRIPTION_NOTIONAL` over
# `blended_cost` for those models in your cost comparisons.
#
# `_CODEX_UNSUPPORTED` exists because a subscription CLI may not actually be able
# to serve every model in its family — if a call silently falls back to the
# billed API instead of erroring, it must be costed as metered even though this
# function would otherwise claim it's free. Populate it with any model IDs your
# subscription vehicle is known not to serve; leave it empty if not applicable.
_CODEX_UNSUPPORTED: frozenset[str] = frozenset()

# Notional USD/1M for a subscription seat. The true marginal cash cost is 0.0, so
# these encode relative QUOTA SCARCITY across shared seats, not money: a heavily
# contended pool (e.g. one shared across every interactive session and background
# worker) should rank last among the free seats so its headroom is preserved for
# work that has nowhere else to go. Keep every value well below the cheapest
# metered seat in your PRICES table so a subscription always beats a metered
# model in cost-min comparisons — that ordering is usually the intended policy.
# These are deliberate knobs to tune per deployment, not measurements.
SUBSCRIPTION_NOTIONAL: dict[str, float] = {}


def vehicle_for(model: str) -> str | None:
    """The subscription vehicle that serves this model, or None if it is metered.

    Adapt the prefix matching below to your own provider naming and subscription
    vehicles; returns None (metered/unknown) by default for every model. Mirror
    this mapping in your own runner's backend-selection logic so the two never
    drift apart — a test asserting that is recommended.
    """
    m = (model or "").lower()
    if m.startswith("council:"):
        return None  # fusion pseudo-model: cost is the summed seat cost at run time
    if m in _CODEX_UNSUPPORTED:
        return None
    return None


# Weak bootstrapped capability prior on a 0-100 scale, per (task_class, model).
# All values in the 55-90 band — the ledger posterior overrides this with volume.
SEED_PRIOR: dict[str, dict[str, float]] = {
    "coding": {
        "claude-opus-4-8": 90.0,
        "gpt-5.3-codex": 89.0,
        "claude-sonnet-4-6": 86.0,
        "gpt-5.5": 86.0,
        "gemini-3.1-pro-preview": 83.0,
        "deepseek-v4-pro": 80.0,
        "gemini-2.5-pro": 79.0,
        "claude-haiku-4-5": 68.0,
        "gpt-5.4-mini": 66.0,
        "kimi-k2.6": 70.0,
        "gpt-5.4-nano": 58.0,
    },
    "content_writing": {
        "claude-opus-4-8": 90.0,
        "claude-sonnet-4-6": 87.0,
        "gpt-5.5": 84.0,
        "gemini-3.1-pro-preview": 82.0,
        "gemini-2.5-pro": 78.0,
        "claude-haiku-4-5": 70.0,
        "minimax-m3": 64.0,
        "gpt-5.4-mini": 65.0,
    },
    "design": {
        "claude-opus-4-8": 86.0,
        "gpt-5.5": 84.0,
        "gemini-3.1-pro-preview": 85.0,
        "claude-sonnet-4-6": 82.0,
        "gemini-2.5-pro": 78.0,
        "claude-haiku-4-5": 66.0,
        "gpt-5.4-mini": 63.0,
    },
    "planning_pm": {
        "claude-opus-4-8": 88.0,
        "gpt-5.5": 86.0,
        "claude-sonnet-4-6": 84.0,
        "gemini-3.1-pro-preview": 82.0,
        "gemini-2.5-pro": 78.0,
        "claude-haiku-4-5": 68.0,
        "gpt-5.4-mini": 66.0,
    },
    "marketing": {
        "claude-opus-4-8": 87.0,
        "claude-sonnet-4-6": 85.0,
        "gpt-5.5": 84.0,
        "gemini-3.1-pro-preview": 81.0,
        "gemini-2.5-pro": 77.0,
        "claude-haiku-4-5": 69.0,
        "minimax-m3": 63.0,
        "gpt-5.4-mini": 65.0,
    },
    "data_analysis": {
        "gpt-5.5": 89.0,
        "claude-opus-4-8": 88.0,
        "claude-sonnet-4-6": 84.0,
        "gemini-3.1-pro-preview": 83.0,
        "deepseek-v4-pro": 80.0,
        "gemini-2.5-pro": 79.0,
        "claude-haiku-4-5": 67.0,
        "gpt-5.4-mini": 66.0,
    },
    "debugging": {
        "claude-opus-4-8": 89.0,
        "gpt-5.3-codex": 88.0,
        "gpt-5.5": 86.0,
        "claude-sonnet-4-6": 85.0,
        "gemini-3.1-pro-preview": 82.0,
        "deepseek-v4-pro": 79.0,
        "claude-haiku-4-5": 67.0,
        "gpt-5.4-mini": 64.0,
    },
    "testing": {
        "claude-opus-4-8": 86.0,
        "gpt-5.3-codex": 85.0,
        "claude-sonnet-4-6": 83.0,
        "gpt-5.5": 83.0,
        "deepseek-v4-pro": 78.0,
        "gemini-2.5-pro": 77.0,
        "claude-haiku-4-5": 70.0,
        "gpt-5.4-mini": 68.0,
        "gpt-5.4-nano": 60.0,
    },
    "security_check": {
        "claude-opus-4-8": 88.0,
        "gpt-5.5": 85.0,
        "claude-sonnet-4-6": 84.0,
        "gemini-3.1-pro-preview": 81.0,
        "deepseek-v4-pro": 76.0,
        "claude-haiku-4-5": 66.0,
        "gpt-5.4-mini": 62.0,
    },
    "research": {
        "sonar-pro": 89.0,
        "claude-opus-4-8": 85.0,
        "gpt-5.5": 84.0,
        "gemini-3.1-pro-preview": 83.0,
        "claude-sonnet-4-6": 82.0,
        "gemini-2.5-pro": 78.0,
        "claude-haiku-4-5": 67.0,
        "gpt-5.4-mini": 64.0,
    },
}


# Admin-curated default routing pool per task class (subset of PRICES keys).
CANDIDATES: dict[str, list[str]] = {
    "coding": [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "deepseek-v4-pro",
        "gpt-5.4-mini",
    ],
    "content_writing": [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "gpt-5.5",
        "claude-haiku-4-5",
        "minimax-m3",
    ],
    "design": [
        "claude-opus-4-8",
        "gemini-3.1-pro-preview",
        "gpt-5.5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ],
    "planning_pm": [
        "claude-opus-4-8",
        "gpt-5.5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "gpt-5.4-mini",
    ],
    "marketing": [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "gpt-5.5",
        "claude-haiku-4-5",
        "minimax-m3",
    ],
    "data_analysis": [
        "gpt-5.5",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "deepseek-v4-pro",
        "gemini-2.5-pro",
        "gpt-5.4-mini",
    ],
    "debugging": [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "deepseek-v4-pro",
    ],
    "testing": [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ],
    "security_check": [
        "claude-opus-4-8",
        "gpt-5.5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "deepseek-v4-pro",
    ],
    "research": [
        "sonar-pro",
        "claude-opus-4-8",
        "gpt-5.5",
        "gemini-3.1-pro-preview",
        "claude-sonnet-4-6",
    ],
}
