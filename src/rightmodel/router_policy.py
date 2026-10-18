"""Router policy — 12-class × 4-provider PII taint matrix enforcement.

Every LLM call in the Foreman loop must pass through `router_policy.check()`
before the model is invoked. A block here is a hard guardrail: no model call,
caller emits a Slack alert + writes a CostLedger violation row.

Providers (matrix updated 2026-06-01 — risk-based, not country-based):
    - CLAUDE (claude-*)         — Max plan. Allowed on everything EXCEPT auth_secrets.
    - GEMINI (gemini-*)         — Google AI Pro free-tier has GDPR-grey processing
                                  for PII workloads. Allowed on public, internal,
                                  private_repo_code only until Gemini Business is
                                  wired up with a DPA.
    - DEEPSEEK (deepseek-*)     — Pay-per-use, treated as Claude-tier.
    - KIMI (kimi-*, moonshot-*) — Pay-per-use, treated as Claude-tier.
    - MINIMAX (minimax-*)       — Pay-per-use, treated as Claude-tier. (Adapter
                                  should set zero-retention flag where available.)
    - OPENAI (gpt-*, codex-*)   — Business tier. Allowed on most taints; the old
                                  carve-out for unpublished_strategy / stealth /
                                  family_personal is conservative and can be lifted
                                  if Emmanuel removes it.
    - PERPLEXITY (sonar-*)      — Pro sub. Treated as Claude-tier.
    - UNKNOWN                   — Unrecognised model. Allowed only on public + internal.

What the matrix actually enforces (the parts that ARE load-bearing):
    1. AUTH_SECRETS — never sent to any model (host-local only).
    2. CUSTOMER_PII / PAYMENT_DATA — provider must have a DPA-equivalent.
    3. Unknown models — default-restricted until classified.
Provider-of-origin is NOT a factor.

12 taint classes (see spine.action_receipt.TaintClass):
    public, internal, customer_pii, prospect_pii, payment_data, voice_dna,
    inferred_pii, auth_secrets, private_repo_code, unpublished_strategy,
    stealth_identity_linkage, family_personal.

auth_secrets is NEVER sent to any model (host-local only). A check that contains
AUTH_SECRETS in its taint set is unconditionally blocked.

Taint propagation (Lexi R4 information-flow control):
    Within one run_id, taints UNION across calls. Once a tool returns a tainted
    value into history, every subsequent model call in that run inherits the
    union. No laundering — propagate() only adds, never removes. clear_run()
    resets at end of run.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .action_receipt import TaintClass


class Provider(str, Enum):
    CLAUDE = "claude"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    KIMI = "kimi"
    MINIMAX = "minimax"
    OPENAI = "openai"
    PERPLEXITY = "perplexity"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Decision:
    allowed: bool
    blocked: bool
    block_reason: str | None
    requires_zero_retention: bool


# ---- Per-provider allowlist matrix --------------------------------------------------

_T = TaintClass

_CLAUDE_TIER_ALLOWED: frozenset[TaintClass] = frozenset(
    {
        _T.PUBLIC,
        _T.INTERNAL,
        _T.CUSTOMER_PII,
        _T.PROSPECT_PII,
        _T.PAYMENT_DATA,
        _T.VOICE_DNA,
        _T.INFERRED_PII,
        _T.PRIVATE_REPO_CODE,
        _T.UNPUBLISHED_STRATEGY,
        _T.STEALTH_IDENTITY_LINKAGE,
        _T.FAMILY_PERSONAL,
    }
)

_GEMINI_ALLOWED: frozenset[TaintClass] = frozenset(
    {_T.PUBLIC, _T.INTERNAL, _T.PRIVATE_REPO_CODE}
)

# DeepSeek / Kimi / MiniMax: promoted to Claude-tier 2026-06-01 (Emmanuel call).
# The previous "no documented no-training tier → restrict" rule treated
# provider-of-origin as a proxy for training-data risk. For this business, that
# distinction isn't load-bearing — external is external regardless of whether
# the provider contractually promises no-training. The real differentiators
# the matrix enforces (auth_secrets host-local + customer_pii DPA + GDPR-ish
# obligations) are provider-neutral. See lex/docs/sessions/log.md entry
# 2026-06-01 21:00 JST and discussion of why the old default was lazy.
_PAY_PER_USE_ALLOWED: frozenset[TaintClass] = _CLAUDE_TIER_ALLOWED

# Legacy alias (kept for any downstream import; safe to drop after grep confirms).
_DEEPSEEK_KIMI_ALLOWED: frozenset[TaintClass] = _PAY_PER_USE_ALLOWED
_MINIMAX_ALLOWED: frozenset[TaintClass] = _PAY_PER_USE_ALLOWED

_OPENAI_ALLOWED: frozenset[TaintClass] = frozenset(
    {
        _T.PUBLIC,
        _T.INTERNAL,
        _T.CUSTOMER_PII,
        _T.PROSPECT_PII,
        _T.PAYMENT_DATA,
        _T.VOICE_DNA,
        _T.INFERRED_PII,
        _T.PRIVATE_REPO_CODE,
    }
)

_UNKNOWN_ALLOWED: frozenset[TaintClass] = frozenset({_T.PUBLIC, _T.INTERNAL})

_PROVIDER_ALLOWLIST: dict[Provider, frozenset[TaintClass]] = {
    Provider.CLAUDE: _CLAUDE_TIER_ALLOWED,
    Provider.PERPLEXITY: _CLAUDE_TIER_ALLOWED,
    Provider.GEMINI: _GEMINI_ALLOWED,
    Provider.DEEPSEEK: _DEEPSEEK_KIMI_ALLOWED,
    Provider.KIMI: _DEEPSEEK_KIMI_ALLOWED,
    Provider.MINIMAX: _MINIMAX_ALLOWED,
    Provider.OPENAI: _OPENAI_ALLOWED,
    Provider.UNKNOWN: _UNKNOWN_ALLOWED,
}


# ---- Process-level run-scoped taint propagation -------------------------------------

_RUN_TAINTS: dict[str, set[TaintClass]] = {}
_RUN_TAINTS_LOCK = threading.Lock()


def propagate(run_id: str, new_taints: set[TaintClass]) -> None:
    """Union new_taints into the run-scoped taint set. Never removes."""
    if not new_taints:
        return
    with _RUN_TAINTS_LOCK:
        existing = _RUN_TAINTS.get(run_id)
        if existing is None:
            _RUN_TAINTS[run_id] = set(new_taints)
        else:
            existing |= new_taints


def get_run_taints(run_id: str) -> set[TaintClass]:
    """Return a copy of the current run-scoped taint set."""
    with _RUN_TAINTS_LOCK:
        return set(_RUN_TAINTS.get(run_id, set()))


def clear_run(run_id: str) -> None:
    """Reset run-scoped taints. Called at end of run."""
    with _RUN_TAINTS_LOCK:
        _RUN_TAINTS.pop(run_id, None)


# ---- Provider mapping ---------------------------------------------------------------


def model_to_provider(model: str) -> Provider:
    """Map a model ID to its Provider. Checks active catalog first, then prefix mapping."""
    if not model:
        return Provider.UNKNOWN
    # Check active catalog for user-defined provider mapping
    try:
        from .model_catalog import get_active_catalog
        catalog = get_active_catalog()
        prov_name = catalog.providers.get(model)
        if prov_name:
            for p in Provider:
                if p.value == prov_name:
                    return p
    except ImportError:
        pass

    m = model.lower()
    if m.startswith("claude-"):
        return Provider.CLAUDE
    if m.startswith("gemini-"):
        return Provider.GEMINI
    if m.startswith("deepseek-"):
        return Provider.DEEPSEEK
    if m.startswith("kimi-") or m.startswith("moonshot-"):
        return Provider.KIMI
    if m.startswith("minimax-"):
        return Provider.MINIMAX
    if m.startswith("gpt-") or m.startswith("codex-") or m == "o1" or m.startswith("o1-") or m == "o3" or m.startswith("o3-"):
        return Provider.OPENAI
    if m.startswith("sonar-") or m.startswith("perplexity-"):
        return Provider.PERPLEXITY
    return Provider.UNKNOWN


# ---- Core check ---------------------------------------------------------------------


def check(
    model: str,
    taint_set: set[TaintClass],
    run_id: str | None = None,
) -> Decision:
    """Enforce per-provider allowlist against the (call taints ∪ run taints).

    AUTH_SECRETS → blocked unconditionally regardless of provider.
    If the active catalog specifies explicit allowed_taints for the model, those are honored.
    """
    effective: set[TaintClass] = set(taint_set) if taint_set else set()
    if run_id is not None:
        effective |= get_run_taints(run_id)

    if TaintClass.AUTH_SECRETS in effective:
        return Decision(
            allowed=False,
            blocked=True,
            block_reason=(
                "auth_secrets in taint set — never sent to any model "
                "(host-local only)"
            ),
            requires_zero_retention=False,
        )

    # If the user configured custom allowed taints for this model, honor their config
    try:
        from .model_catalog import get_active_catalog
        catalog = get_active_catalog()
        if model in catalog.taints:
            user_allowed = {
                TaintClass(t)
                for t in catalog.taints[model]
                if t in TaintClass._value2member_map_
            }
            disallowed = effective - user_allowed
            if disallowed:
                names = sorted(t.value for t in disallowed)
                return Decision(
                    allowed=False,
                    blocked=True,
                    block_reason=(
                        f"model={model} not cleared for taint(s): {', '.join(names)}"
                    ),
                    requires_zero_retention=False,
                )
            return Decision(
                allowed=True,
                blocked=False,
                block_reason=None,
                requires_zero_retention=False,
            )
    except ImportError:
        pass

    provider = model_to_provider(model)
    allowed_set = _PROVIDER_ALLOWLIST[provider]
    disallowed = effective - allowed_set
    if disallowed:
        names = sorted(t.value for t in disallowed)
        return Decision(
            allowed=False,
            blocked=True,
            block_reason=(
                f"provider={provider.value} not allowed on taint(s): "
                f"{', '.join(names)} (model={model})"
            ),
            requires_zero_retention=False,
        )

    return Decision(
        allowed=True,
        blocked=False,
        block_reason=None,
        requires_zero_retention=False,
    )


# ---- Secrets blocklist post-hook ----------------------------------------------------


def _blocklist_path() -> Path:
    return Path(os.path.expanduser("~/.business-agent/secrets-blocklist.txt"))


def secrets_blocklist_check(payload_text: str) -> bool:
    """Return True if any literal secret from the blocklist appears in payload_text.

    Blocklist format: one literal secret per line; lines starting with `#` are
    comments; blank lines ignored. Missing file → returns False (nothing to check,
    no error). Used as a post-hook before sending payloads to any model.
    """
    if not payload_text:
        return False
    path = _blocklist_path()
    if not path.exists():
        return False
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return False
    for line in raw.splitlines():
        secret = line.strip()
        if not secret or secret.startswith("#"):
            continue
        if secret in payload_text:
            return True
    return False
