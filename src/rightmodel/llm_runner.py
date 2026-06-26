"""llm_runner — the real LLM execution layer Foreman themes call (v1.1).

Until v1.1 every theme that declared a ``MODEL`` used a mock; the mocks
referenced a ``claude_runner`` that did not exist. This module is that runner.
It dispatches a model id to the right backend:

  - ``claude-*``            → ``claude -p`` subscription CLI (free under Max; never API-billed)
  - ``gpt-*`` / ``codex``   → ``codex exec`` subscription CLI
  - ``deepseek-*`` / ``kimi-*`` / ``moonshot-*`` → OpenAI-compatible HTTP API (httpx, metered)

A thin wrapper over provider model CLIs (same flags,
same env-strip, same isolated-cwd trick), with Foreman-specific guardrails so a
bad call can NEVER kill the daily pulse:

  1. MASTER KILL-SWITCH — unless ``FOREMAN_LLM_LIVE=1`` in the environment,
     ``run()`` returns an inert ``LLMResult(ok=False, reason="kill_switch_off")``
     without invoking anything. Themes treat that as "no result" and fall back
     to their mock/skip path. Going live is therefore ONE deliberate env flip,
     not a surprise on the next cron.
  2. NEVER RAISES — every failure (missing CLI, nonzero exit, timeout, missing
     API key, unparseable structured output) is caught and returned as
     ``ok=False``. A theme can always `if not result.ok: <fallback>`.
  3. API-KEY STRIP — ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY /
     GOOGLE_API_KEY / CLAUDECODE are stripped before any CLI call, so a stray
     key can't silently bill the paid API (belt-and-suspenders; the
     claude-cli-guard already does this for claude).
  4. STRUCTURED OUTPUT — pass a Pydantic model as ``schema``; the prompt is
     wrapped with a JSON-only instruction, the output is parsed + validated,
     and a schema mismatch leaves ``result.parsed=None`` (text still returned).
  5. COST — subscription CLI calls are ``cost_usd=0.0``; API calls carry the
     metered estimate. Themes record this to the CostLedger as before.

Pure stdlib + Pydantic + httpx, matching the spine dependency contract.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from . import rightmodel_harness as rm_harness

# Default per-call wall-clock cap for COMPLETION calls (one inference → text). gpt-5.x CLIs
# can be slow on web-search-heavy prompts (120s proved too short in practice);
# 300s is the documented safe ceiling for the slow models. Themes may override per call.
DEFAULT_TIMEOUT_S = 300
# AGENTIC runs (claude -p / codex / agy driving tools, files, skills across many turns to a
# goal — e.g. a content pipeline: keyword-research → draft → translate → image → PR) routinely
# need 10–30 min, not one inference. Applying the 300s completion cap SIGKILLs real agentic work
# mid-run (c-1b70c9e4 sonnet redo died at 300s, 2026-06-16). Agentic mode gets this larger cap by
# default; a caller (or a card's delegate.timeout) may still override per call.
AGENTIC_TIMEOUT_S = 1800

# claude CLI model aliases (mirrors CLAUDE_MODEL_MAP in llm-cli.ts).
_CLAUDE_MODEL_MAP: dict[str, str] = {
    "claude-opus-4-8": "opus",
    "claude-opus-4-6": "opus",
    "claude-sonnet-4-6": "sonnet",
    "claude-haiku-4-5": "haiku",
    "claude-haiku-4-5-20251001": "haiku",
    "opus": "opus",
    "sonnet": "sonnet",
    "haiku": "haiku",
}

# OpenAI-compatible API providers: provider_key → (base_url, env var holding key).
_API_PROVIDERS: dict[str, tuple[str, str]] = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "kimi": ("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    "moonshot": ("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
    "minimax": ("https://api.minimax.io/v1", "MINIMAX_API_KEY"),
    "glm": ("https://api.z.ai/api/paas/v4", "ZAI_API_KEY"),  # z.ai (Zhipu) GLM, OpenAI-compatible
    # Gemini + Perplexity expose OpenAI-compatible /chat/completions endpoints, so they
    # ride the same _call_api path (added 2026-06-16 to make the full RightModel candidate
    # set runnable — the router must be able to execute every model it can route to).
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "sonar": ("https://api.perplexity.ai", "PERPLEXITY_API_KEY"),
    "perplexity": ("https://api.perplexity.ai", "PERPLEXITY_API_KEY"),
    # OpenAI direct API — the fallback for gpt models the Codex CLI rejects (see run()).
    "gpt": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "o1": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "o3": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
}

_ENV_KEYS_TO_STRIP = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "CLAUDECODE",
)

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResult:
    """Outcome of one model call. Never raised — always returned."""

    ok: bool
    backend: str  # "claude" | "codex" | "api" | "none"
    model: str
    text: str | None = None
    parsed: BaseModel | None = None  # set when a schema was given AND validated
    cost_usd: float = 0.0
    tokens_in: int | None = None
    tokens_out: int | None = None
    reason: str | None = None  # failure reason when ok is False
    meta: dict[str, Any] = field(default_factory=dict)


def is_live() -> bool:
    """True only when the master kill-switch is explicitly enabled."""
    return os.environ.get("FOREMAN_LLM_LIVE", "").strip() in ("1", "true", "yes", "on")


def backend_for(model: str) -> str:
    """Map a model id to its execution backend. Mirrors router_policy prefixes."""
    m = (model or "").lower()
    if m.startswith("claude-") or m in ("opus", "sonnet", "haiku"):
        return "claude"
    if m.startswith(("gpt-", "o1", "o3", "o4", "codex")):
        return "codex"
    if m.startswith(("deepseek-", "kimi-", "moonshot-", "minimax-",
                     "gemini-", "sonar-", "perplexity-", "glm-")):
        return "api"
    return "unknown"


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for k in _ENV_KEYS_TO_STRIP:
        env.pop(k, None)
    return env


def _resolve_cli(command: str) -> str | None:
    """Find a CLI on PATH or in the common Homebrew/npm-global locations."""
    # Antigravity ships TWO binaries named `agy`: the CLI (~/.local/bin/agy) and
    # the IDE 2.0 shim (/opt/homebrew/bin/agy). We need the CLI — prefer it
    # explicitly so `shutil.which` can't pick the shim (see /research SKILL.md).
    if command == "agy":
        for cand in (os.environ.get("AGY_BIN", ""), os.path.expanduser("~/.local/bin/agy")):
            if cand and os.path.exists(cand):
                return cand
    found = shutil.which(command)
    if found:
        return found
    for cand in (
        os.environ.get(f"{command.upper()}_BIN", ""),
        f"/usr/local/bin/{command}",
        f"/opt/homebrew/bin/{command}",
        os.path.expanduser(f"~/.npm-global/bin/{command}"),
    ):
        if cand and os.path.exists(cand):
            return cand
    return None


def _run_cli(
    command: str, args: list[str], timeout: int, *, isolated: bool,
    workdir: str | None = None,
) -> tuple[bool, str, str]:
    """Execute a CLI. Returns (ok, stdout, error). Never raises.

    ``workdir`` pins the working directory (used by agentic runs so the agent works
    INSIDE the project repo — git/gh/file ops land in the right place). When unset,
    ``isolated`` runs in a throwaway tempdir (completion default), else the caller's cwd.
    """
    binary = _resolve_cli(command)
    if not binary:
        return False, "", f"{command} CLI not found on PATH"
    cwd = workdir or (tempfile.mkdtemp(prefix="foreman-llm-") if isolated else None)
    try:
        proc = subprocess.run(
            [binary, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_clean_env(),
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return False, "", f"{command} timed out after {timeout}s"
    except OSError as e:  # spawn failure
        return False, "", f"{command} spawn failed: {e}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        return False, "", f"{command} exit {proc.returncode}: {tail}"
    return True, (proc.stdout or "").strip(), ""


def _compose_prompt(prompt: str, system: str | None) -> str:
    return f"{system}\n\n---\n\n{prompt}" if system else prompt


def _git(workdir: str, *args: str) -> str:
    """Run a read-only git command in ``workdir``; return stdout (or "" on any failure)."""
    try:
        p = subprocess.run(
            ["git", *args], cwd=workdir, capture_output=True, text=True,
            timeout=30, env=_clean_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (p.stdout or "").strip() if p.returncode == 0 else ""


def _git_artifact(workdir: str, baseline: str | None) -> str:
    """The textual ARTIFACT an agentic run produced in ``workdir`` — the diff it actually
    wrote, not its narration. Combines commits since ``baseline`` plus any uncommitted
    changes. Returns "" when nothing changed / not a git repo. Capped so the judge prompt
    stays bounded. This is what quality scoring should grade for agentic runs (Bug 1)."""
    parts: list[str] = []
    if baseline:
        stat = _git(workdir, "diff", "--stat", baseline, "HEAD")
        patch = _git(workdir, "diff", baseline, "HEAD")
        if stat:
            parts.append("# Committed since run start:\n" + stat)
        if patch:
            parts.append(patch)
    for label, d in (("Staged", _git(workdir, "diff", "--cached")),
                     ("Unstaged", _git(workdir, "diff"))):
        if d:
            parts.append(f"# {label} changes:\n{d}")
    return "\n\n".join(parts).strip()[:20000]


def _call_claude(
    model: str, prompt: str, timeout: int, isolated: bool,
    *, agentic: bool = False, add_dirs: list[str] | None = None,
) -> LLMResult:
    cli_model = _CLAUDE_MODEL_MAP.get(model, model)
    args = ["-p", prompt, "--model", cli_model, "--output-format", "text"]
    workdir = None
    if not agentic:
        # Completion = single turn, sandboxed (no settings, throwaway cwd).
        args += ["--max-turns", "1"]
        if isolated:
            args += ["--setting-sources", ""]
    else:
        # Agentic = drive tools across turns to a goal. To actually work it MUST be able to
        # act (write files, run git/gh) and reach the PROJECT's skills+MCP — so run with
        # bypassPermissions and pin cwd to the project dir (the first granted add_dir) so
        # git/gh/file ops land in the repo, not a temp sandbox.
        #
        # Scope settings to project+local, NEVER user (Bug 2): loading ~/.claude/ would pull
        # Lex's session-end Stop hooks + persona into the delegated agent — it then ran Lex's
        # session-summary ritual and committed to the lex repo (f73c5a3f, 2026-06-16). The
        # agent should inherit only the TARGET project's settings, not the orchestrator's.
        args += ["--permission-mode", "bypassPermissions",
                 "--setting-sources", "project,local"]
        workdir = next((d for d in (add_dirs or []) if os.path.isdir(d)), None)
    for d in add_dirs or []:
        args += ["--add-dir", d]
    # Capture the pre-run commit so we can diff the artifact the agent actually wrote (Bug 1).
    baseline = _git(workdir, "rev-parse", "HEAD") or None if (agentic and workdir) else None
    ok, out, err = _run_cli("claude", args, timeout, isolated=isolated, workdir=workdir)
    meta = {"harness": "claude_cli", "execution_mode": "agentic" if agentic else "completion"}
    if ok and agentic and workdir:
        artifact = _git_artifact(workdir, baseline)
        if artifact:
            meta["artifact"] = artifact
    return LLMResult(
        ok=ok, backend="claude", model=model, text=out if ok else None,
        cost_usd=0.0, reason=None if ok else err, meta=meta,
    )


def _call_codex(
    model: str, prompt: str, timeout: int, isolated: bool,
    *, agentic: bool = False, add_dirs: list[str] | None = None,
) -> LLMResult:
    args = ["exec", prompt]
    if model:
        args += ["-m", model]
    for d in add_dirs or []:
        args += ["--add-dir", d]
    if isolated:
        args.append("--skip-git-repo-check")
    ok, out, err = _run_cli("codex", args, timeout, isolated=isolated)
    return LLMResult(
        ok=ok, backend="codex", model=model, text=out if ok else None,
        cost_usd=0.0, reason=None if ok else err,
        meta={"harness": "codex", "execution_mode": "agentic" if agentic else "completion"},
    )


def _call_api(model: str, prompt: str, system: str | None, timeout: int) -> LLMResult:
    prefix = model.lower().split("-")[0]
    conf = _API_PROVIDERS.get(prefix)
    if conf is None:
        return LLMResult(ok=False, backend="api", model=model, reason=f"no API config for {prefix}")
    base_url, key_env = conf
    api_key = os.environ.get(key_env)
    if not api_key:
        return LLMResult(ok=False, backend="api", model=model, reason=f"{key_env} not set")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        import httpx

        resp = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "stream": False},
            timeout=float(timeout),
        )
        if resp.status_code != 200:
            return LLMResult(ok=False, backend="api", model=model,
                             reason=f"HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        text = (data["choices"][0]["message"]["content"] or "").strip()
        usage = data.get("usage") or {}
        return LLMResult(
            ok=bool(text), backend="api", model=model, text=text or None,
            tokens_in=usage.get("prompt_tokens"), tokens_out=usage.get("completion_tokens"),
            cost_usd=0.0, reason=None if text else "empty completion",
            meta={"raw_usage": usage},
        )
    except Exception as e:  # network/JSON/shape failures all degrade to ok=False
        return LLMResult(ok=False, backend="api", model=model, reason=f"api call failed: {e}")


def _call_agy(
    model: str, prompt: str, system: str | None, timeout: int, add_dirs: list[str] | None
) -> LLMResult:
    """Drive Gemini (or whatever Antigravity is configured for) AGENTICALLY via the
    Antigravity CLI (`agy`) — the agentic harness already used in /research and /plan.

    `agy` has live web/tool access and runs to a goal across steps. It has no
    per-call `-m` flag (model is set globally in Antigravity settings), so ``model``
    is informational here — the registry routes Gemini ids to this harness. Free
    under the Google AI Pro subscription → cost_usd=0.0. ``add_dirs`` grants the
    agent read/write access to project paths (`--add-dir`, repeatable).
    """
    args = ["-p", _compose_prompt(prompt, system)]
    for d in add_dirs or []:
        args += ["--add-dir", d]
    args += ["--print-timeout", f"{timeout}s"]
    ok, out, err = _run_cli("agy", args, timeout, isolated=False)
    return LLMResult(
        ok=ok, backend="agy", model=model, text=out if ok else None,
        cost_usd=0.0, reason=None if ok else err,
        meta={"harness": "agy", "execution_mode": "agentic"},
    )


def _extract_json(text: str) -> str | None:
    """Pull a JSON object out of a model response (tolerates ```json fences / prose)."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def _apply_schema(result: LLMResult, schema: type[BaseModel] | None) -> LLMResult:
    """Parse+validate ``result.text`` against ``schema`` in place; never raises.

    On a miss, ``result.ok`` stays True (text came back) but ``parsed`` is None and
    ``reason`` is set. Shared by the completion and agentic paths.
    """
    if not (result.ok and schema is not None):
        return result
    raw = _extract_json(result.text or "")
    if raw is None:
        result.parsed, result.reason = None, "no JSON object in output"
    else:
        try:
            result.parsed = schema.model_validate_json(raw)
        except ValidationError as e:
            result.parsed = None
            result.reason = f"schema validation failed: {e.error_count()} error(s)"
    return result


def _run_agentic(
    model: str, prompt: str, system: str | None, timeout: int, isolated: bool,
    execution_mode: str, add_dirs: list[str] | None,
) -> LLMResult:
    """Drive ``model`` through its agentic harness for a non-completion mode.

    Resolves the (model × harness) pairing via the registry; a miss returns
    ok=False (eligibility violation — the router should already have filtered it,
    but the runner defends in depth and never raises). SKILL mode rides the same
    agentic harness; SCRIPT never reaches here (it is not an LLM route).
    """
    harness = rm_harness.harness_for(model, execution_mode)
    if harness is None:
        return LLMResult(
            ok=False, backend="none", model=model,
            reason=f"no agentic harness for {model!r} providing execution_mode={execution_mode!r}",
        )
    composed = _compose_prompt(prompt, system)
    if harness is rm_harness.Harness.CLAUDE_CLI:
        return _call_claude(model, composed, timeout, isolated, agentic=True, add_dirs=add_dirs)
    if harness is rm_harness.Harness.CODEX:
        return _call_codex(model, composed, timeout, isolated, agentic=True, add_dirs=add_dirs)
    if harness is rm_harness.Harness.AGY:
        return _call_agy(model, prompt, system, timeout, add_dirs)
    # API harness is completion-only; reaching here means the registry is misconfigured.
    return LLMResult(
        ok=False, backend="api", model=model,
        reason=f"harness {harness.value} cannot satisfy execution_mode={execution_mode!r}",
    )


def run(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    schema: type[BaseModel] | None = None,
    timeout: int | None = None,
    isolated: bool = True,
    execution_mode: str = "completion",
    add_dirs: list[str] | None = None,
) -> LLMResult:
    """Execute one model call. Never raises.

    Returns an ``LLMResult``. When ``schema`` is given and the output validates,
    ``result.parsed`` holds the model instance; on a schema miss ``result.ok``
    stays True (text came back) but ``result.parsed`` is None and ``reason`` is
    set — callers that require structure should check ``result.parsed``.

    ``execution_mode`` selects the harness: ``"completion"`` (default) uses the
    plain backend (claude CLI / codex / chat-completions API); ``"agentic"`` or
    ``"skill:<name>"`` drive the model's AGENTIC harness (Gemini → Antigravity
    `agy`, GPT → codex, Claude → claude CLI) via the (model × harness) registry,
    granting ``add_dirs`` as tool scope. A model with no agentic harness wired
    returns ok=False (an eligibility miss, never a raise).

    Honors the FOREMAN_LLM_LIVE kill-switch: returns an inert result if off.
    """
    if not is_live():
        return LLMResult(ok=False, backend="none", model=model, reason="kill_switch_off")

    # Resolve the wall-clock cap by mode when the caller didn't pin one: agentic/skill runs do
    # long multi-turn tool work and get AGENTIC_TIMEOUT_S; completion calls get the 300s default.
    is_completion = rm_harness.parse_mode(execution_mode)[0] is rm_harness.ExecutionMode.COMPLETION
    if timeout is None:
        timeout = DEFAULT_TIMEOUT_S if is_completion else AGENTIC_TIMEOUT_S

    if schema is not None:
        prompt = (
            f"{prompt}\n\nRespond with ONLY a single JSON object matching this schema "
            f"(no prose, no markdown fences):\n{json.dumps(schema.model_json_schema())}"
        )

    # Non-completion modes route through the agentic harness registry.
    if rm_harness.parse_mode(execution_mode)[0] is not rm_harness.ExecutionMode.COMPLETION:
        result = _run_agentic(model, prompt, system, timeout, isolated, execution_mode, add_dirs)
        return _apply_schema(result, schema)

    backend = backend_for(model)
    if backend == "claude":
        result = _call_claude(model, _compose_prompt(prompt, system), timeout, isolated)
    elif backend == "codex":
        result = _call_codex(model, _compose_prompt(prompt, system), timeout, isolated)
        # The Codex CLI (ChatGPT-account auth) only supports some gpt models; for the rest
        # (e.g. gpt-5.3-codex, gpt-5.4-nano) it 400s "model is not supported". Fall back to
        # the OpenAI API so every gpt candidate is actually runnable — otherwise the router
        # would route to a model that always fails and the matrix would learn a false zero.
        if not result.ok and "not supported" in (result.reason or "").lower():
            api = _call_api(model, prompt, system, timeout)
            if api.ok or "not set" not in (api.reason or ""):
                result = api
    elif backend == "api":
        result = _call_api(model, prompt, system, timeout)
    else:
        return LLMResult(ok=False, backend="unknown", model=model,
                         reason=f"no backend for model {model!r}")

    return _apply_schema(result, schema)
