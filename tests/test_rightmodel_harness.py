"""Tests for spine.rightmodel_harness — the (model × harness) capability registry.

Pins the corrected v2 spec contract (rightmodel-deferred.md §"Execution mode"):
agentic capability is a property of the (model × harness) pairing, not the model;
Gemini is agentic via the `agy` harness; completion-only API pairings are an
eligibility miss for agentic/skill modes; script never routes to an LLM.
"""

from __future__ import annotations

import pytest

from rightmodel import rightmodel_harness as h
from rightmodel.rightmodel_harness import ExecutionMode, Harness


# ── provider family resolution ──────────────────────────────────────────────

@pytest.mark.parametrize("model,family", [
    ("claude-opus-4-8", "claude"),
    ("haiku", "claude"),
    ("gpt-5.5", "gpt"),
    ("codex", "gpt"),
    ("gemini-3.1-pro-preview", "gemini"),
    ("kimi-k2.6", "kimi"),
    ("moonshot-v1", "kimi"),
    ("deepseek-v4-pro", "deepseek"),
    ("minimax-m3", "minimax"),
    ("sonar-pro", "sonar"),
    ("perplexity-x", "sonar"),
    ("something-weird", "unknown"),
])
def test_provider_family(model, family):
    assert h.provider_family(model) == family


# ── parse_mode ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,mode,arg", [
    ("completion", ExecutionMode.COMPLETION, None),
    ("agentic", ExecutionMode.AGENTIC, None),
    ("skill:logo-creator", ExecutionMode.SKILL, "logo-creator"),
    ("script:scripts/x.sh", ExecutionMode.SCRIPT, "scripts/x.sh"),
    ("", ExecutionMode.COMPLETION, None),
    ("garbage", ExecutionMode.COMPLETION, None),  # unknown → completion
])
def test_parse_mode(raw, mode, arg):
    assert h.parse_mode(raw) == (mode, arg)


# ── the core correction: agentic is (model × harness), not the model ─────────

def test_gemini_is_agentic_via_agy_not_api():
    # completion default rides the cheap API; agentic rides Antigravity.
    assert h.harness_for("gemini-3.1-pro-preview", "completion") is Harness.API
    assert h.harness_for("gemini-3.1-pro-preview", "agentic") is Harness.AGY
    assert h.eligible("gemini-3.1-pro-preview", "agentic") is True


def test_claude_and_gpt_are_agentic_via_their_clis():
    assert h.harness_for("claude-opus-4-8", "agentic") is Harness.CLAUDE_CLI
    assert h.harness_for("gpt-5.5", "agentic") is Harness.CODEX
    # gpt completion default is codex (mirrors llm_runner backend_for / fallback).
    assert h.harness_for("gpt-5.5", "completion") is Harness.CODEX


def test_completion_only_models_are_ineligible_for_agentic():
    # No agentic harness wired for these yet → eligibility miss (not a low score).
    for m in ("deepseek-v4-pro", "minimax-m3", "sonar-pro", "kimi-k2.6"):
        assert h.harness_for(m, "agentic") is None
        assert h.eligible(m, "agentic") is False
        # ...but they ARE eligible for completion.
        assert h.eligible(m, "completion") is True


def test_every_known_model_is_completion_eligible():
    for m in ("claude-opus-4-8", "gpt-5.5", "gemini-2.5-pro", "deepseek-v4-pro",
              "kimi-k2.6", "minimax-m3", "sonar-pro"):
        assert h.eligible(m, "completion") is True


def test_unknown_model_is_ineligible_everywhere_but_script():
    assert h.eligible("nope-1", "completion") is False
    assert h.eligible("nope-1", "agentic") is False
    assert h.eligible("nope-1", "script:x") is True  # script needs no harness


# ── skills ───────────────────────────────────────────────────────────────────

def test_skill_mode_needs_an_agentic_harness():
    # Claude/GPT/Gemini agentic harnesses expose "*" (any skill).
    assert h.harness_for("claude-opus-4-8", "skill:logo-creator") is Harness.CLAUDE_CLI
    assert h.harness_for("gemini-3.1-pro-preview", "skill:geo") is Harness.AGY
    # A completion-only model can't run a skill.
    assert h.harness_for("deepseek-v4-pro", "skill:geo") is None


# ── script ────────────────────────────────────────────────────────────────────

def test_script_never_routes_to_an_llm():
    assert h.is_script("script:scripts/build.sh") is True
    assert h.is_script("agentic") is False
    # harness_for returns None for script (no LLM route) even for a capable model.
    assert h.harness_for("claude-opus-4-8", "script:x.sh") is None
