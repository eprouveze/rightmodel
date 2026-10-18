"""RightModel harness registry — the (model × harness) capability layer (Module 13).

RightModel v1 assumed pure completion (prompt → text). Real tasks may need an
*agentic* harness (drive tools/files/bash across steps to a goal), a SKILL, or a
deterministic SCRIPT. The corrected v2 spec (rightmodel-deferred.md §"Execution
mode", 2026-06-16) makes the key correction:

    Agentic capability is a property of the (model × HARNESS) pairing, NOT the
    model. Gemini, Kimi, GPT, Claude are ALL agentic when driven through an
    agentic harness — Gemini via Antigravity (`agy`), GPT via codex, Claude via
    the claude CLI, Kimi via tool-calling. The limitation today is an *adapter
    gap*, not a model limit: `llm_runner._call_api` drives gemini/deepseek/
    minimax/sonar through plain /chat/completions (no tools), so those models are
    completion-only *in that harness*. Wiring an agentic harness (e.g. `agy` for
    Gemini) makes them agentic.

So the registry is ``(model, harness) → {completion, agentic, skills}`` and
eligibility is "do we have a harness wired for this model that provides the
required capability." It IS an eligibility filter (a tool-requiring task routed
to a completion-only *pairing* fails — it does not merely score lower), but the
filter is over *pairings*, not models.

This module is pure stdlib (no deps) so the router/runner can both import it.
Capability is keyed by provider family × harness (a model_id resolves to its
family); that mirrors how `llm_runner.backend_for` and `router_policy.
model_to_provider` already resolve models by prefix, and stays maintainable as
the candidate pool grows. A genuinely per-model override can be added to
`_MODEL_OVERRIDES` when one model in a family diverges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Harness(str, Enum):
    """A concrete adapter that drives a model. Capability is (model × harness)."""

    CLAUDE_CLI = "claude_cli"   # `claude -p` subscription CLI — agentic (tools, turns)
    CODEX = "codex"             # `codex exec` subscription CLI — agentic
    AGY = "agy"                 # Antigravity CLI — the agentic harness for Gemini
    API = "api"                 # plain OpenAI-compatible /chat/completions — completion-only


class ExecutionMode(str, Enum):
    """How a task must be executed. ``skill:`` / ``script:`` carry an argument."""

    COMPLETION = "completion"   # prompt → text (what the v1 eval measures)
    AGENTIC = "agentic"         # drive tools/files/steps to a goal
    SKILL = "skill"             # run a named skill (agentic structure)
    SCRIPT = "script"           # deterministic — NO LLM routing


@dataclass(frozen=True)
class HarnessCapability:
    """What a (model, harness) pairing can do.

    ``skills`` is the set of skill names the pairing exposes; ``("*",)`` means
    "any skill" (the harness is a general agent that can run arbitrary skills,
    e.g. the claude CLI / codex / agy). Empty means no skill support.
    """

    completion: bool
    agentic: bool
    skills: tuple[str, ...] = ()


# Provider family → ordered {harness: capability}. Insertion order is PREFERENCE:
# for a given execution mode the first harness that satisfies it wins. The
# completion default per family therefore stays exactly what `llm_runner.
# backend_for` produces today (claude→CLAUDE_CLI, gpt→CODEX, gemini/others→API).
_PROVIDER_HARNESSES: dict[str, dict[Harness, HarnessCapability]] = {
    # Claude — the claude CLI is both the completion and agentic harness.
    "claude": {
        Harness.CLAUDE_CLI: HarnessCapability(completion=True, agentic=True, skills=("*",)),
    },
    # GPT — codex is the agentic + default completion harness; the OpenAI API is a
    # completion-only fallback (mirrors llm_runner.run()'s codex→api fallback).
    "gpt": {
        Harness.CODEX: HarnessCapability(completion=True, agentic=True, skills=("*",)),
        Harness.API: HarnessCapability(completion=True, agentic=False),
    },
    # Gemini — completion rides the cheap /chat/completions API (today's default);
    # AGENTIC rides Antigravity (`agy`), already used agentically in /research, /plan.
    # THIS pairing is the adapter gap this module + llm_runner close.
    "gemini": {
        Harness.API: HarnessCapability(completion=True, agentic=False),
        Harness.AGY: HarnessCapability(completion=True, agentic=True, skills=("*",)),
    },
    # Kimi is agentic via tool-calling, but that harness is NOT wired in llm_runner
    # yet — so today it is completion-only. (Registered here so the gap is explicit;
    # add a TOOLCALL harness entry when the adapter lands — it is a (model×harness)
    # wiring task, not a model limit.)
    "kimi": {
        Harness.API: HarnessCapability(completion=True, agentic=False),
    },
    "deepseek": {
        Harness.API: HarnessCapability(completion=True, agentic=False),
    },
    "minimax": {
        Harness.API: HarnessCapability(completion=True, agentic=False),
    },
    "sonar": {
        Harness.API: HarnessCapability(completion=True, agentic=False),
    },
    "generic": {
        Harness.API: HarnessCapability(completion=True, agentic=False),
    },
}

# Per-model overrides for when a single model_id diverges from its family. Empty
# today; populate with model_id → {harness: capability} when needed.
_MODEL_OVERRIDES: dict[str, dict[Harness, HarnessCapability]] = {}


def provider_family(model: str) -> str:
    """Resolve a model_id to its provider family. Checks active catalog first."""
    m = (model or "").lower()
    try:
        from .model_catalog import get_active_catalog
        catalog = get_active_catalog()
        prov = catalog.providers.get(model)
        if prov:
            prov_clean = prov.lower()
            if prov_clean in _PROVIDER_HARNESSES:
                return prov_clean
            return "generic"
        elif model in catalog.prices and not any(
            m.startswith(p) for p in ("claude-", "gpt-", "gemini-", "kimi-", "deepseek-", "minimax-", "sonar-")
        ):
            return "generic"
    except ImportError:
        pass

    if m.startswith("claude-") or m in ("opus", "sonnet", "haiku"):
        return "claude"
    if m.startswith(("gpt-", "o1", "o3", "o4", "codex")):
        return "gpt"
    if m.startswith("gemini-"):
        return "gemini"
    if m.startswith(("kimi-", "moonshot-")):
        return "kimi"
    if m.startswith("deepseek-"):
        return "deepseek"
    if m.startswith("minimax-"):
        return "minimax"
    if m.startswith(("sonar-", "perplexity-")):
        return "sonar"
    return "unknown"




def harnesses_for(model: str) -> dict[Harness, HarnessCapability]:
    """The ordered {harness: capability} wired for a model (override > family)."""
    if model in _MODEL_OVERRIDES:
        return _MODEL_OVERRIDES[model]
    return _PROVIDER_HARNESSES.get(provider_family(model), {})


def parse_mode(execution_mode: str) -> tuple[ExecutionMode, str | None]:
    """Split an execution_mode string into (mode, arg).

    ``"completion"`` → (COMPLETION, None); ``"agentic"`` → (AGENTIC, None);
    ``"skill:logo-creator"`` → (SKILL, "logo-creator");
    ``"script:scripts/x.sh"`` → (SCRIPT, "scripts/x.sh").
    Unknown / empty defaults to COMPLETION.
    """
    raw = (execution_mode or "completion").strip()
    head, _, arg = raw.partition(":")
    head = head.lower()
    try:
        mode = ExecutionMode(head)
    except ValueError:
        return ExecutionMode.COMPLETION, None
    return mode, (arg or None)


def _capability_satisfies(cap: HarnessCapability, mode: ExecutionMode, arg: str | None) -> bool:
    if mode is ExecutionMode.COMPLETION:
        return cap.completion
    if mode is ExecutionMode.AGENTIC:
        return cap.agentic
    if mode is ExecutionMode.SKILL:
        # A skill needs an agentic harness that exposes the skill (or any skill, "*").
        if not cap.agentic:
            return False
        return "*" in cap.skills or (arg is not None and arg in cap.skills)
    # SCRIPT — deterministic, no LLM (handled upstream); no harness satisfies it.
    return False


def harness_for(model: str, execution_mode: str = "completion") -> Harness | None:
    """The preferred harness driving ``model`` for ``execution_mode``, or None.

    Iterates the model's harnesses in preference order and returns the first whose
    capability satisfies the requested mode. None means the pairing does not exist
    — i.e. the model is INELIGIBLE for this execution mode (an eligibility miss,
    not a low score). SCRIPT mode always returns None (no LLM routing).
    """
    mode, arg = parse_mode(execution_mode)
    if mode is ExecutionMode.SCRIPT:
        return None
    for harness, cap in harnesses_for(model).items():
        if _capability_satisfies(cap, mode, arg):
            return harness
    return None


def eligible(model: str, execution_mode: str = "completion") -> bool:
    """True iff a wired harness lets ``model`` satisfy ``execution_mode``.

    SCRIPT mode is eligible for ALL models (it needs no LLM — the deterministic
    path runs regardless), so this returns True for script: without consulting the
    registry. Every other mode requires a satisfying harness to exist.
    """
    mode, _ = parse_mode(execution_mode)
    if mode is ExecutionMode.SCRIPT:
        return True
    return harness_for(model, execution_mode) is not None


def is_script(execution_mode: str) -> bool:
    """True for ``script:<path>`` modes — deterministic, no LLM routing."""
    return parse_mode(execution_mode)[0] is ExecutionMode.SCRIPT
