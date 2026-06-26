"""Tests for spine.rightmodel_eval — graders + run_item with a fake llm_runner.

No live LLM calls: score() graders are tested directly and run_item runs against a
monkeypatched llm_runner.run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

import rightmodel.rightmodel_eval as ev
from rightmodel.rightmodel_eval import (
    ITEM_PACK,
    EvalItem,
    run_item,
    score,
    seed,
)
from rightmodel.rightmodel_ledger import RightModelLedger
from rightmodel.rightmodel_types import Outcome


# --------------------------------------------------------------------------- item pack


def test_item_pack_covers_each_auto_class() -> None:
    by_class: dict[str, int] = {}
    for item in ITEM_PACK:
        by_class[item.task_class] = by_class.get(item.task_class, 0) + 1
    for cls in ("coding", "data_analysis", "testing"):
        assert by_class.get(cls, 0) >= 3, f"{cls} has <3 items"


def test_item_pack_is_bilingual() -> None:
    langs = {i.lang for i in ITEM_PACK}
    assert "en" in langs and "ja" in langs


# --------------------------------------------------------------------------- graders


def test_score_exact() -> None:
    item = EvalItem("x", "coding", "p", 0.1, "en", "exact", "5")
    assert score(item, "5") is True
    assert score(item, "  5\n") is True
    assert score(item, "the answer is 5") is False


def test_score_contains() -> None:
    item = EvalItem("x", "coding", "p", 0.1, "en", "contains", "cba")
    assert score(item, "result: cba") is True
    assert score(item, "abc") is False


def test_score_python_assert_pass_and_fail() -> None:
    item = EvalItem("x", "coding", "p", 0.4, "en", "python_assert",
                    "assert int(output.strip()) == 55")
    assert score(item, "55") is True
    assert score(item, "54") is False


def test_python_assert_errors_grade_false() -> None:
    item = EvalItem("x", "coding", "p", 0.4, "en", "python_assert",
                    "assert int(output.strip()) == 55")
    assert score(item, "not-an-int") is False  # ValueError -> False


def test_python_assert_blocks_import(tmp_path) -> None:
    item = EvalItem("x", "coding", "p", 0.4, "en", "python_assert",
                    "import os; assert True")
    assert score(item, "anything") is False  # __import__ disabled -> False


def test_python_assert_no_file_access(tmp_path) -> None:
    item = EvalItem("x", "coding", "p", 0.4, "en", "python_assert",
                    "open('/etc/passwd'); assert True")
    assert score(item, "anything") is False  # open not in whitelist -> NameError -> False


def test_score_unknown_grader_is_false() -> None:
    item = EvalItem("x", "coding", "p", 0.1, "en", "bogus", "5")
    assert score(item, "5") is False


def test_score_none_output_is_false() -> None:
    item = EvalItem("x", "coding", "p", 0.1, "en", "exact", "5")
    assert score(item, None) is False  # type: ignore[arg-type]


# --------------------------------------------------------------------------- run_item


@dataclass
class _FakeResult:
    ok: bool
    text: str | None
    cost_usd: float = 0.001
    reason: str | None = None


def _fake_run_factory(text: str, ok: bool = True):
    def _fake_run(model: str, prompt: str, **kwargs):
        return _FakeResult(ok=ok, text=text)
    return _fake_run


def test_run_item_pass_writes_synthetic_pass_row(tmp_path, monkeypatch) -> None:
    item = EvalItem("duel-1", "coding", "p", 0.4, "en", "exact", "5")
    monkeypatch.setattr(ev.llm_runner, "run", _fake_run_factory("5"))
    ledger = RightModelLedger(path=tmp_path / "ledger.jsonl")

    result = run_item(item, "claude-haiku-4-5", ledger=ledger)

    assert result["passed"] is True
    assert result["outcome"] == Outcome.PASS.value
    rows = ledger.rows()
    assert len(rows) == 1
    assert rows[0].source == "synthetic"
    assert rows[0].collision_group == "duel-1"
    assert rows[0].outcome == Outcome.PASS.value
    assert rows[0].lang == "en"


def test_run_item_fail_on_wrong_output(tmp_path, monkeypatch) -> None:
    item = EvalItem("duel-2", "coding", "p", 0.4, "en", "exact", "5")
    monkeypatch.setattr(ev.llm_runner, "run", _fake_run_factory("nope"))
    ledger = RightModelLedger(path=tmp_path / "ledger.jsonl")

    result = run_item(item, "gpt-5.4-mini", ledger=ledger)

    assert result["passed"] is False
    assert ledger.rows()[0].outcome == Outcome.FAIL.value


def test_run_item_kill_switch_off_records_fail(tmp_path, monkeypatch) -> None:
    item = EvalItem("duel-3", "coding", "p", 0.4, "en", "exact", "5")
    monkeypatch.setattr(
        ev.llm_runner, "run",
        lambda model, prompt, **kw: _FakeResult(ok=False, text=None, reason="kill_switch_off"),
    )
    ledger = RightModelLedger(path=tmp_path / "ledger.jsonl")

    result = run_item(item, "claude-haiku-4-5", ledger=ledger)

    assert result["passed"] is False
    assert result["ok"] is False
    row = ledger.rows()[0]
    assert row.outcome == Outcome.FAIL.value
    assert "kill_switch_off" in (row.note or "")


# --------------------------------------------------------------------------- seed


def test_seed_runs_pack_across_models(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ev.llm_runner, "run", _fake_run_factory("5"))
    ledger = RightModelLedger(path=tmp_path / "ledger.jsonl")

    summary = seed(
        ["claude-haiku-4-5", "gpt-5.4-mini"],
        classes=["coding"],
        ledger=ledger,
    )

    coding_items = [i for i in ITEM_PACK if i.task_class == "coding"]
    assert summary["runs"] == len(coding_items) * 2
    assert set(summary["by_model"]) == {"claude-haiku-4-5", "gpt-5.4-mini"}
    assert len(ledger.rows()) == summary["runs"]


# --------------------------------------------------------------------------- test_suite
# grader fixture pack: feed hardcoded model-output strings (every format a model
# emits in the wild) to the public score() API for a test_suite item, asserting the
# right pass/fail. No live LLM calls. Items are constructed locally with
# grader="test_suite" and expected=<hidden asserts> so the tests only couple to the
# public score(item, output) contract (internal grader fns may be renamed).


# Two reference problems with hidden assert suites (mirror ITEM_PACK shape).
_PALINDROME = EvalItem(
    "ts-pal", "coding", "p", 0.5, "en", "test_suite",
    "assert is_palindrome('A man, a plan, a canal: Panama') is True\n"
    "assert is_palindrome('race a car') is False\n"
    "assert is_palindrome('') is True\n",
)
_TWOSUM = EvalItem(
    "ts-two", "coding", "p", 0.6, "en", "test_suite",
    "assert sorted(two_sum([2,7,11,15], 9)) == [0,1]\n"
    "assert sorted(two_sum([3,2,4], 6)) == [1,2]\n"
    "assert sorted(two_sum([3,3], 6)) == [0,1]\n",
)


# A correct is_palindrome body (no imports), reused across fence-format fixtures.
_PAL_BODY = (
    "def is_palindrome(s):\n"
    "    t = [c.lower() for c in s if c.isalnum()]\n"
    "    return t == t[::-1]\n"
)


def test_ts_bare_code_no_fences() -> None:
    # 1. bare code, no fences
    assert score(_PALINDROME, _PAL_BODY) is True


def test_ts_single_python_fence() -> None:
    # 2. single ```python fenced block
    output = f"```python\n{_PAL_BODY}```"
    assert score(_PALINDROME, output) is True


def test_ts_single_bare_fence_no_lang() -> None:
    # 3. single ``` fence with no language tag
    output = f"```\n{_PAL_BODY}```"
    assert score(_PALINDROME, output) is True


def test_ts_explain_then_code() -> None:
    # 4. prose paragraph, then a fenced block
    output = (
        "Here's a clean solution. We normalize the string to alphanumeric, "
        "lowercase it, then compare against its reverse.\n\n"
        f"```python\n{_PAL_BODY}```"
    )
    assert score(_PALINDROME, output) is True


def test_ts_code_then_explanation() -> None:
    # 5. fenced block, then prose
    output = (
        f"```python\n{_PAL_BODY}```\n\n"
        "This runs in O(n) time and O(n) extra space for the filtered list."
    )
    assert score(_PALINDROME, output) is True


def test_ts_multiple_fenced_blocks_both_needed() -> None:
    # 6. a helper-function block + the main-function block — BOTH needed.
    helper = (
        "def _clean(s):\n"
        "    return [c.lower() for c in s if c.isalnum()]\n"
    )
    main = (
        "def is_palindrome(s):\n"
        "    t = _clean(s)\n"
        "    return t == t[::-1]\n"
    )
    output = (
        "First a normalizer helper:\n\n"
        f"```python\n{helper}```\n\n"
        "Then the palindrome check that uses it:\n\n"
        f"```python\n{main}```"
    )
    # Only passes if the grader concatenates ALL code blocks; if it grabs just the
    # first block, is_palindrome is undefined and this fails.
    assert score(_PALINDROME, output) is True


def test_ts_stdlib_imports_must_pass() -> None:
    # 7. uses stdlib imports (import re / from collections import Counter) — the
    # real-interpreter sandbox must allow curated stdlib, so this PASSES.
    output = (
        "```python\n"
        "import re\n"
        "from collections import Counter\n"
        "def is_palindrome(s):\n"
        "    t = re.sub(r'[^a-z0-9]', '', s.lower())\n"
        "    # Counter import is intentional — proves collections is importable\n"
        "    _ = Counter(t)\n"
        "    return t == t[::-1]\n"
        "```"
    )
    assert score(_PALINDROME, output) is True


def test_ts_code_then_usage_demo() -> None:
    # 8. code followed by a usage example / print(...) demo after the function def.
    output = (
        "```python\n"
        f"{_PAL_BODY}"
        "\n"
        "# quick demo\n"
        "print(is_palindrome('Never odd or even'))\n"
        "```"
    )
    assert score(_PALINDROME, output) is True


def test_ts_wrong_solution_fails() -> None:
    # 9. a WRONG solution (ignores non-alphanumeric / case) — must FAIL.
    wrong = (
        "```python\n"
        "def is_palindrome(s):\n"
        "    return s == s[::-1]\n"
        "```"
    )
    assert score(_PALINDROME, wrong) is False


def test_ts_syntax_error_fails_not_crash() -> None:
    # 10a. code with a syntax error — must FAIL, not crash the grader.
    broken = "```python\ndef is_palindrome(s)\n    return True\n```"  # missing colon
    assert score(_PALINDROME, broken) is False


def test_ts_runtime_raises_fails_not_crash() -> None:
    # 10b. code that raises at call time — must FAIL, not crash the grader.
    raising = (
        "```python\n"
        "def is_palindrome(s):\n"
        "    raise RuntimeError('boom')\n"
        "```"
    )
    assert score(_PALINDROME, raising) is False


def test_ts_infinite_loop_times_out_fails() -> None:
    # 11. an infinite loop — must FAIL via the grader's own short timeout, not hang
    # the suite. Relies on the hardened grader killing the subprocess; if this test
    # ever hangs, the timeout regressed.
    looping = (
        "```python\n"
        "def is_palindrome(s):\n"
        "    return True\n"
        "while True:\n"
        "    pass\n"
        "```"
    )
    started = time.monotonic()
    assert score(_PALINDROME, looping, test_suite_timeout_s=2.0) is False
    elapsed = time.monotonic() - started
    # the grader's timeout must fire — not hang the suite at the production default.
    assert elapsed < 10.0, f"timeout did not fire fast: {elapsed:.1f}s"


def test_ts_twosum_correct_passes() -> None:
    # second reference problem, correct solution — PASS.
    output = (
        "```python\n"
        "def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for i, n in enumerate(nums):\n"
        "        if target - n in seen:\n"
        "            return [seen[target - n], i]\n"
        "        seen[n] = i\n"
        "```"
    )
    assert score(_TWOSUM, output) is True


def test_ts_twosum_wrong_fails() -> None:
    # second reference problem, wrong solution (reuses the same element) — FAIL.
    output = (
        "```python\n"
        "def two_sum(nums, target):\n"
        "    return [0, 0]\n"
        "```"
    )
    assert score(_TWOSUM, output) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
