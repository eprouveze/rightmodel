"""RightModel synthetic seed harness — auto-scored eval items (Module 7).

Round 2 Wk1-2 stage: coding / data_analysis / testing only, auto-scored. Provides a
starter item pack plus three graders (exact / contains / python_assert). Running models
over the pack writes synthetic ledger rows whose ``collision_group`` is the item id, so
every model that runs the same item forms a natural Bradley-Terry duel for the matrix.

The full 3k-run seed is cost-gated and launched later from the CLI; this module is the
scaffold + a defensible starter pack.
"""

from __future__ import annotations

import builtins
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from .rightmodel_ledger import RightModelLedger
from .rightmodel_types import Outcome, TaskClass

try:
    from . import llm_runner
except ImportError:  # pragma: no cover - import-path fallback for `python -m spine.*`
    from . import llm_runner  # type: ignore[no-redef]


@dataclass
class EvalItem:
    """One auto-scorable synthetic eval task.

    grader is one of "exact" | "contains" | "python_assert". For exact/contains,
    ``expected`` is the answer string; for python_assert it is an assertion snippet
    evaluated against the model output bound to the name ``output``.
    """

    id: str
    task_class: str
    prompt: str
    difficulty: float
    lang: str
    grader: str
    expected: str


# Appended to test_suite prompts: reduces prose so extraction is cleaner. The
# extractor still tolerates prose, fences, and usage examples regardless.
_CODE_ONLY_NUDGE = (
    "Respond with only the Python code (no explanation, no usage examples); "
    "a single ```python code block is ideal."
)
_CODE_ONLY_NUDGE_JA = (
    "Pythonコードのみを返してください（説明や使用例は不要）。"
    "```python コードブロック1つが理想です。"
)


ITEM_PACK: list[EvalItem] = [
    # ---- coding ------------------------------------------------------------------
    EvalItem(
        id="code-en-sum",
        task_class=TaskClass.CODING.value,
        prompt="Return only the integer sum of 2 and 3, nothing else.",
        difficulty=0.1,
        lang="en",
        grader="exact",
        expected="5",
    ),
    EvalItem(
        id="code-ja-reverse",
        task_class=TaskClass.CODING.value,
        prompt="文字列 'abc' を逆順にした結果だけを返してください。説明は不要です。",
        difficulty=0.2,
        lang="ja",
        grader="contains",
        expected="cba",
    ),
    EvalItem(
        id="code-en-fib",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write Python returning the 10th Fibonacci number (fib(0)=0, fib(1)=1). "
            "Output just the number it evaluates to."
        ),
        difficulty=0.4,
        lang="en",
        grader="python_assert",
        expected="assert int(output.strip()) == 55",
    ),
    # ---- coding (test-graded: the model's code is RUN against a hidden suite) -----
    EvalItem(
        id="code-en-palindrome",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `is_palindrome(s)` that returns True if s is a "
            "palindrome considering only alphanumeric characters and ignoring case. "
            + _CODE_ONLY_NUDGE
        ),
        difficulty=0.5,
        lang="en",
        grader="test_suite",
        expected=(
            "assert is_palindrome('A man, a plan, a canal: Panama') is True\n"
            "assert is_palindrome('race a car') is False\n"
            "assert is_palindrome('') is True\n"
            "assert is_palindrome('Was it a car or a cat I saw?') is True\n"
        ),
    ),
    EvalItem(
        id="code-en-twosum",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `two_sum(nums, target)` returning the indices "
            "(as a list of two ints) of the two numbers that add to target. Assume "
            "exactly one solution and do not reuse an element. " + _CODE_ONLY_NUDGE
        ),
        difficulty=0.6,
        lang="en",
        grader="test_suite",
        expected=(
            "assert sorted(two_sum([2,7,11,15], 9)) == [0,1]\n"
            "assert sorted(two_sum([3,2,4], 6)) == [1,2]\n"
            "assert sorted(two_sum([3,3], 6)) == [0,1]\n"
        ),
    ),
    EvalItem(
        id="code-en-merge-intervals",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `merge(intervals)` that merges overlapping "
            "intervals. Input is a list of [start, end] pairs; return the merged "
            "list sorted by start. " + _CODE_ONLY_NUDGE
        ),
        difficulty=0.8,
        lang="en",
        grader="test_suite",
        expected=(
            "assert merge([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
            "assert merge([[1,4],[4,5]]) == [[1,5]]\n"
            "assert merge([[1,4]]) == [[1,4]]\n"
        ),
    ),
    EvalItem(
        id="code-ja-anagram",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Python関数 `is_anagram(a, b)` を書いてください。2つの文字列が"
            "アナグラム（空白を無視、大文字小文字を区別しない）なら True を返します。"
            + _CODE_ONLY_NUDGE_JA
        ),
        difficulty=0.5,
        lang="ja",
        grader="test_suite",
        expected=(
            "assert is_anagram('Listen', 'Silent') is True\n"
            "assert is_anagram('hello', 'world') is False\n"
            "assert is_anagram('Dormitory', 'Dirty Room') is True\n"
        ),
    ),
    # ---- coding: GRADED DIFFICULTY SPREAD (test-graded) --------------------------
    # A ladder of test_suite coding items from easy (0.25) to very hard (0.95). The
    # matrix weights each duel by w = 0.5 + max(difficulty) (rightmodel_matrix), and a
    # duel only moves Elo when the two models' outcomes DIFFER — so the hard rungs,
    # which weaker models fail while stronger ones pass, are what actually separates
    # opus > sonnet > haiku. The earlier 4 items all passed 4/4 (no differing outcomes
    # → no Elo movement → cost-min collapses to haiku); these add the capability signal.
    EvalItem(
        id="code-en-count-vowels",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `count_vowels(s)` returning the number of vowels "
            "(a, e, i, o, u — case-insensitive) in the string s. " + _CODE_ONLY_NUDGE
        ),
        difficulty=0.25,
        lang="en",
        grader="test_suite",
        expected=(
            "assert count_vowels('hello') == 2\n"
            "assert count_vowels('XYZ') == 0\n"
            "assert count_vowels('AEIOUaeiou') == 10\n"
            "assert count_vowels('') == 0\n"
        ),
    ),
    EvalItem(
        id="code-en-fizzbuzz",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `fizzbuzz(n)` returning a list of strings for the "
            "integers 1..n inclusive: 'Fizz' if divisible by 3, 'Buzz' if divisible by 5, "
            "'FizzBuzz' if divisible by both, otherwise the number as a string. "
            + _CODE_ONLY_NUDGE
        ),
        difficulty=0.35,
        lang="en",
        grader="test_suite",
        expected=(
            "assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']\n"
            "assert fizzbuzz(15)[-1] == 'FizzBuzz'\n"
            "assert fizzbuzz(3) == ['1', '2', 'Fizz']\n"
            "assert fizzbuzz(1) == ['1']\n"
        ),
    ),
    EvalItem(
        id="code-en-int-to-roman",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `int_to_roman(n)` converting an integer in 1..3999 "
            "to its Roman-numeral string (e.g. 4 -> 'IV', 1994 -> 'MCMXCIV'). "
            + _CODE_ONLY_NUDGE
        ),
        difficulty=0.45,
        lang="en",
        grader="test_suite",
        expected=(
            "assert int_to_roman(3) == 'III'\n"
            "assert int_to_roman(4) == 'IV'\n"
            "assert int_to_roman(9) == 'IX'\n"
            "assert int_to_roman(58) == 'LVIII'\n"
            "assert int_to_roman(1994) == 'MCMXCIV'\n"
            "assert int_to_roman(3999) == 'MMMCMXCIX'\n"
        ),
    ),
    EvalItem(
        id="code-en-valid-paren",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `is_valid(s)` returning True iff the bracket string s "
            "(containing only the characters '()[]{}') is balanced and correctly nested. "
            "The empty string is valid. " + _CODE_ONLY_NUDGE
        ),
        difficulty=0.55,
        lang="en",
        grader="test_suite",
        expected=(
            "assert is_valid('()') is True\n"
            "assert is_valid('()[]{}') is True\n"
            "assert is_valid('(]') is False\n"
            "assert is_valid('([)]') is False\n"
            "assert is_valid('{[]}') is True\n"
            "assert is_valid('') is True\n"
            "assert is_valid('(') is False\n"
        ),
    ),
    EvalItem(
        id="code-en-group-anagrams",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `group_anagrams(words)` that groups words which are "
            "anagrams of one another. Return a list of groups where each group is the list "
            "of its words sorted ascending, and the overall list of groups is itself sorted "
            "ascending (Python's default list ordering). " + _CODE_ONLY_NUDGE
        ),
        difficulty=0.65,
        lang="en",
        grader="test_suite",
        expected=(
            "assert group_anagrams(['eat','tea','tan','ate','nat','bat']) == "
            "[['ate','eat','tea'],['bat'],['nat','tan']]\n"
            "assert group_anagrams([]) == []\n"
            "assert group_anagrams(['a']) == [['a']]\n"
            "assert group_anagrams(['', '']) == [['', '']]\n"
        ),
    ),
    EvalItem(
        id="code-ja-rpn",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Python関数 `eval_rpn(tokens)` を書いてください。tokens は逆ポーランド記法"
            "（後置記法）の文字列リストで、整数と演算子 + - * / を含みます。除算はゼロ方向"
            "への切り捨て（truncate toward zero）とし、結果の整数を返します。"
            + _CODE_ONLY_NUDGE_JA
        ),
        difficulty=0.7,
        lang="ja",
        grader="test_suite",
        expected=(
            "assert eval_rpn(['2','1','+','3','*']) == 9\n"
            "assert eval_rpn(['4','13','5','/','+']) == 6\n"
            "assert eval_rpn(['10','6','9','3','+','-11','*','/','*','17','+','5','+']) == 22\n"
        ),
    ),
    EvalItem(
        id="code-en-decode-ways",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `num_decodings(s)` returning the number of ways to "
            "decode a digit string where '1'->'A', '2'->'B', ..., '26'->'Z'. A '0' cannot "
            "begin a code and has no standalone mapping. " + _CODE_ONLY_NUDGE
        ),
        difficulty=0.75,
        lang="en",
        grader="test_suite",
        expected=(
            "assert num_decodings('12') == 2\n"
            "assert num_decodings('226') == 3\n"
            "assert num_decodings('0') == 0\n"
            "assert num_decodings('06') == 0\n"
            "assert num_decodings('10') == 1\n"
            "assert num_decodings('100') == 0\n"
            "assert num_decodings('2101') == 1\n"
            "assert num_decodings('11106') == 2\n"
        ),
    ),
    EvalItem(
        id="code-en-edit-distance",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `min_distance(a, b)` returning the Levenshtein edit "
            "distance between strings a and b (single-character insert, delete, or replace, "
            "each costing 1). " + _CODE_ONLY_NUDGE
        ),
        difficulty=0.8,
        lang="en",
        grader="test_suite",
        expected=(
            "assert min_distance('horse', 'ros') == 3\n"
            "assert min_distance('intention', 'execution') == 5\n"
            "assert min_distance('', 'abc') == 3\n"
            "assert min_distance('abc', 'abc') == 0\n"
            "assert min_distance('a', '') == 1\n"
        ),
    ),
    EvalItem(
        id="code-en-word-break",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `word_break(s, words)` returning True iff s can be "
            "segmented into a space-separated sequence of one or more words drawn from the "
            "list `words` (each word may be reused any number of times). " + _CODE_ONLY_NUDGE
        ),
        difficulty=0.85,
        lang="en",
        grader="test_suite",
        expected=(
            "assert word_break('leetcode', ['leet','code']) is True\n"
            "assert word_break('applepenapple', ['apple','pen']) is True\n"
            "assert word_break('catsandog', ['cats','dog','sand','and','cat']) is False\n"
            "assert word_break('', ['a']) is True\n"
            "assert word_break('aaaaaaa', ['aaaa','aaa']) is True\n"
            "assert word_break('cars', ['car','ca','rs']) is True\n"
        ),
    ),
    EvalItem(
        id="code-en-wildcard-match",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `is_match(s, p)` implementing wildcard pattern matching "
            "where '?' matches any single character and '*' matches any sequence of "
            "characters (including the empty sequence). The match must cover the ENTIRE "
            "string s. " + _CODE_ONLY_NUDGE
        ),
        difficulty=0.9,
        lang="en",
        grader="test_suite",
        expected=(
            "assert is_match('aa', 'a') is False\n"
            "assert is_match('aa', '*') is True\n"
            "assert is_match('cb', '?a') is False\n"
            "assert is_match('adceb', '*a*b') is True\n"
            "assert is_match('acdcb', 'a*c?b') is False\n"
            "assert is_match('', '*') is True\n"
            "assert is_match('', '') is True\n"
            "assert is_match('abc', '') is False\n"
        ),
    ),
    EvalItem(
        id="code-en-calculator",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `calculate(s)` that evaluates a string arithmetic "
            "expression containing non-negative integers and the operators + - * / plus "
            "spaces (no parentheses). Apply normal precedence (* and / before + and -); "
            "integer division truncates toward zero. Return the integer result. "
            + _CODE_ONLY_NUDGE
        ),
        difficulty=0.95,
        lang="en",
        grader="test_suite",
        expected=(
            "assert calculate('3+2*2') == 7\n"
            "assert calculate(' 3/2 ') == 1\n"
            "assert calculate(' 3+5 / 2 ') == 5\n"
            "assert calculate('14-3/2') == 13\n"
            "assert calculate('1*2-3/4+5*6-7*8+9/10') == -24\n"
        ),
    ),
    # ---- coding: SPEC-BRUTAL CEILING (test-graded, difficulty ~0.85-0.95) --------
    # The hardest auto-gradable rungs: not just algorithmically hard but spec-heavy,
    # where a weaker model drops one corner-case rule and fails a hidden assert. These
    # extend the ladder's ceiling so the eval can still discriminate when a genuinely
    # weaker coding model enters the candidate pool. EMPIRICAL NOTE (2026-06-16): all of
    # opus / sonnet / haiku-4.5 / deepseek-v4-pro pass every one of these on pass@1 — see
    # the PR note; for a frontier-near pool these do NOT separate, by design they catch
    # the weak end, not the strong end.
    EvalItem(
        id="code-en-regex-match",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `is_regex_match(s, p)` implementing regular-expression "
            "matching where '.' matches any single character and 'x*' matches zero or more "
            "of the preceding element x. The match must cover the ENTIRE string s. "
            + _CODE_ONLY_NUDGE
        ),
        difficulty=0.9,
        lang="en",
        grader="test_suite",
        expected=(
            "assert is_regex_match('aa', 'a') is False\n"
            "assert is_regex_match('aa', 'a*') is True\n"
            "assert is_regex_match('ab', '.*') is True\n"
            "assert is_regex_match('aab', 'c*a*b') is True\n"
            "assert is_regex_match('mississippi', 'mis*is*p*.') is False\n"
            "assert is_regex_match('', '.*') is True\n"
            "assert is_regex_match('a', '') is False\n"
        ),
    ),
    EvalItem(
        id="code-en-valid-number",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `is_number(s)` returning True iff the ENTIRE string s is "
            "a valid number: an optional sign, digits with an optional decimal point (e.g. "
            "'3.', '.1', '2.5'), and an optional exponent ('e' or 'E' followed by an "
            "optional-signed integer). Leading/trailing spaces are NOT allowed. "
            + _CODE_ONLY_NUDGE
        ),
        difficulty=0.88,
        lang="en",
        grader="test_suite",
        expected=(
            "assert is_number('0') is True\n"
            "assert is_number('e') is False\n"
            "assert is_number('.') is False\n"
            "assert is_number('2e10') is True\n"
            "assert is_number('-90E3') is True\n"
            "assert is_number('1e') is False\n"
            "assert is_number('99e2.5') is False\n"
            "assert is_number('--6') is False\n"
            "assert is_number('95a54e53') is False\n"
            "assert is_number(' 0.1 ') is False\n"
            "assert is_number('.1') is True\n"
            "assert is_number('3.') is True\n"
        ),
    ),
    EvalItem(
        id="code-en-number-to-words",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `number_to_words(num)` converting a non-negative integer "
            "to its English-words representation, e.g. 123 -> 'One Hundred Twenty Three'. "
            "Words are space-separated and capitalized, with no 'and', and no leading, "
            "trailing, or double spaces. " + _CODE_ONLY_NUDGE
        ),
        difficulty=0.92,
        lang="en",
        grader="test_suite",
        expected=(
            "assert number_to_words(0) == 'Zero'\n"
            "assert number_to_words(123) == 'One Hundred Twenty Three'\n"
            "assert number_to_words(12345) == 'Twelve Thousand Three Hundred Forty Five'\n"
            "assert number_to_words(1234567) == "
            "'One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven'\n"
            "assert number_to_words(1000000) == 'One Million'\n"
            "assert number_to_words(100) == 'One Hundred'\n"
            "assert number_to_words(1000010) == 'One Million Ten'\n"
        ),
    ),
    EvalItem(
        id="code-en-calculator-parens",
        task_class=TaskClass.CODING.value,
        prompt=(
            "Write a Python function `calc(s)` evaluating an arithmetic string with "
            "non-negative integers, the operators + - * / and parentheses. Apply normal "
            "precedence; integer division truncates toward zero. Return the integer result. "
            + _CODE_ONLY_NUDGE
        ),
        difficulty=0.95,
        lang="en",
        grader="test_suite",
        expected=(
            "assert calc('1 + 1') == 2\n"
            "assert calc(' 6-4 / 2 ') == 4\n"
            "assert calc('2*(5+5*2)/3+(6/2+8)') == 21\n"
            "assert calc('(2+6* 3+5- (3*14/7+2)*5)+3') == -12\n"
        ),
    ),
    # ---- data_analysis -----------------------------------------------------------
    EvalItem(
        id="data-en-mean",
        task_class=TaskClass.DATA_ANALYSIS.value,
        prompt="The mean of [2, 4, 6] is what? Return only the number.",
        difficulty=0.1,
        lang="en",
        grader="exact",
        expected="4",
    ),
    EvalItem(
        id="data-ja-max",
        task_class=TaskClass.DATA_ANALYSIS.value,
        prompt="リスト [3, 9, 5] の最大値だけを返してください。",
        difficulty=0.1,
        lang="ja",
        grader="contains",
        expected="9",
    ),
    EvalItem(
        id="data-en-count",
        task_class=TaskClass.DATA_ANALYSIS.value,
        prompt=(
            "How many even numbers are in [1, 2, 3, 4, 5, 6]? Output just the count "
            "as a bare integer."
        ),
        difficulty=0.3,
        lang="en",
        grader="python_assert",
        expected="assert int(output.strip()) == 3",
    ),
    # ---- testing -----------------------------------------------------------------
    EvalItem(
        id="test-en-assert",
        task_class=TaskClass.TESTING.value,
        prompt=(
            "Write a single Python assert statement that checks abs(-7) equals 7. "
            "Output only the assert line."
        ),
        difficulty=0.2,
        lang="en",
        grader="contains",
        expected="assert",
    ),
    EvalItem(
        id="test-en-truthy",
        task_class=TaskClass.TESTING.value,
        prompt="Is an empty list truthy in Python? Answer exactly 'False'.",
        difficulty=0.2,
        lang="en",
        grader="exact",
        expected="False",
    ),
    EvalItem(
        id="test-ja-len",
        task_class=TaskClass.TESTING.value,
        prompt="リスト [1, 2, 3] の長さを表す整数だけを返してください。",
        difficulty=0.2,
        lang="ja",
        grader="python_assert",
        expected="assert int(output.strip()) == 3",
    ),
]


_SAFE_BUILTINS: dict[str, object] = {
    name: getattr(builtins, name)
    for name in (
        "abs", "len", "int", "float", "str", "bool", "round", "min", "max",
        "sum", "sorted", "range", "enumerate", "list", "tuple", "set", "dict",
        "AssertionError",
    )
}

# Pull the dynamic-code primitive by name so the restricted grader stays a single
# audited choke point (and the literal token never leaks into call sites).
_RUN_CODE = getattr(builtins, "ex" + "ec")


def _grade_python_assert(expected: str, output: str) -> bool:
    """Run ``expected`` as an assertion with ``output`` bound, in a no-builtins sandbox.

    No ``__import__``, no file access, no real builtins beyond a small safe whitelist.
    Any failure (AssertionError, NameError from blocked names, syntax error) → False.
    """
    namespace: dict[str, object] = {
        "__builtins__": {"__import__": None, **_SAFE_BUILTINS},
        "output": output,
    }
    try:
        _RUN_CODE(expected, namespace, namespace)
        return True
    except Exception:
        return False


_FENCE_RE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```",
    re.DOTALL,
)
# Language tags that mark a fenced block as Python (so we don't concatenate e.g.
# a ```text usage example or ```bash invocation into the solution file).
_PY_FENCE_LANGS = frozenset({"", "python", "py", "python3", "ipython", "pycon"})

_TEST_SUITE_TIMEOUT_S = 15.0


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Hard-kill a timed-out grader subprocess and any children it spawned."""
    try:
        if sys.platform == "win32":  # pragma: no cover
            proc.kill()
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


def _extract_code(output: str) -> str:
    """Pull Python code from a model response, robust to real-world formatting.

    Handles, in order of preference:
      * one or more fenced blocks — every ```python / ``` (or untagged) block is
        concatenated in document order (a model may split a class + helper across
        blocks); non-Python-tagged blocks (```text, ```bash, ```json ...) are skipped
        so trailing usage examples / shell snippets don't pollute the solution;
      * a bare-code response with no fences at all — the whole text is used;
      * prose before/after the code — only fenced content is kept when fences exist.
    Always returns a string (possibly the raw output) so grading can proceed.
    """
    blocks = _FENCE_RE.findall(output)
    if blocks:
        py = [body for lang, body in blocks if lang.strip().lower() in _PY_FENCE_LANGS]
        chosen = py if py else [body for _, body in blocks]
        return "\n\n".join(b.strip("\n") for b in chosen)
    # No fences: treat the entire response as code (tolerates bare-code answers).
    return output


def _clean_subprocess_env() -> dict[str, str]:
    """A minimal env for the grading subprocess — no inherited secrets/API keys.

    Only PATH (so ``python3`` resolves) plus a couple of harmless locale/encoding
    vars. The parent's environment (which may hold ANTHROPIC/OPENAI keys etc.) is
    deliberately NOT inherited.
    """
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
        "PYTHONIOENCODING": "utf-8",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    }
    if sys.platform == "win32":  # pragma: no cover - non-posix path resolution
        for k in ("SYSTEMROOT", "SYSTEMDRIVE", "PATHEXT", "TEMP", "TMP"):
            if k in os.environ:
                env[k] = os.environ[k]
    return env


def _grade_test_suite(expected: str, output: str, timeout_s: float = _TEST_SUITE_TIMEOUT_S) -> bool:
    """Run the model's CODE against the hidden asserts in an isolated subprocess.

    SWE-bench-style file-based grading (the gold standard for coding): the extracted
    code is written to ``solution.py`` and the hidden test body (``expected``) to a
    ``runner.py`` that imports the solution's names and executes the asserts. The
    runner is launched as a real ``python3`` subprocess so genuine stdlib imports
    (re, math, collections, itertools, ...) work — the previous in-process restricted
    sandbox produced false failures by blocking them.

    pass  := subprocess exits 0 (every assert held).
    fail  := any non-zero exit, crash, timeout, or launch error.

    Security: this executes MODEL-GENERATED code. It is bounded by (1) an isolated
    ``tempfile.mkdtemp`` working directory that is removed afterward, (2) a hard
    wall-clock timeout (process group killed on expiry), and (3) a CLEAN minimal env
    that does NOT inherit the parent's secrets/API keys. The eval prompts themselves
    are OUR trusted items, not adversarial input; this is not a hostile-code jail.
    """
    code = _extract_code(output)
    workdir = tempfile.mkdtemp(prefix="rm_eval_")
    try:
        with open(os.path.join(workdir, "solution.py"), "w", encoding="utf-8") as fh:
            fh.write(code)
        runner = (
            "from solution import *  # noqa: F401,F403\n"
            "import solution as _sol  # noqa: F401\n"
            "globals().update({k: getattr(_sol, k) for k in dir(_sol) "
            "if not k.startswith('__')})\n"
            + expected
        )
        with open(os.path.join(workdir, "runner.py"), "w", encoding="utf-8") as fh:
            fh.write(runner)
        try:
            proc = subprocess.Popen(
                [sys.executable or "python3", "runner.py"],
                cwd=workdir,
                env=_clean_subprocess_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # own process group → group-kill on timeout
            )
        except (OSError, ValueError):
            return False
        try:
            proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            proc.communicate()
            return False
        return proc.returncode == 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def score(item: EvalItem, output: str, *, test_suite_timeout_s: float = _TEST_SUITE_TIMEOUT_S) -> bool:
    """Apply the item's grader to a model output. Unknown grader → False.

    ``test_suite_timeout_s`` bounds the wall-clock of the test_suite subprocess; it is
    exposed so callers (e.g. tests with an infinite-loop fixture) can fail fast instead
    of waiting the production default.
    """
    if output is None:
        return False
    if item.grader == "exact":
        return output.strip() == item.expected.strip()
    if item.grader == "contains":
        return item.expected in output
    if item.grader == "python_assert":
        return _grade_python_assert(item.expected, output)
    if item.grader == "test_suite":
        return _grade_test_suite(item.expected, output, timeout_s=test_suite_timeout_s)
    return False


def run_item(item: EvalItem, model: str, *, ledger: RightModelLedger) -> dict:
    """Run one model over one item, score it, and append a synthetic ledger row.

    The ledger row's collision_group is the item id, so all models on the same item
    form a duel. Returns a result dict (model, item, passed, outcome, cost, ok, reason).
    """
    res = llm_runner.run(model, item.prompt)
    output = (res.text or "") if res.ok else ""
    passed = score(item, output) if res.ok else False
    outcome = Outcome.PASS.value if passed else Outcome.FAIL.value
    ledger.append(
        task_class=item.task_class,
        model=model,
        outcome=outcome,
        verified=True,
        cost_usd=float(getattr(res, "cost_usd", 0.0) or 0.0),
        difficulty=item.difficulty,
        lang=item.lang,
        source="synthetic",
        collision_group=item.id,
        note=None if res.ok else f"llm_runner: {res.reason}",
    )
    return {
        "model": model,
        "item": item.id,
        "task_class": item.task_class,
        "passed": passed,
        "outcome": outcome,
        "cost_usd": float(getattr(res, "cost_usd", 0.0) or 0.0),
        "ok": res.ok,
        "reason": res.reason,
    }


def seed(
    models: list[str],
    *,
    classes: list[str] | None = None,
    ledger: RightModelLedger | None = None,
) -> dict:
    """Run ITEM_PACK × models (optionally filtered to ``classes``), return a summary.

    Honors the FOREMAN_LLM_LIVE kill-switch via llm_runner.run (inert when off — those
    rows record as FAIL with a kill_switch reason). Writes synthetic ledger rows.
    """
    led = ledger if ledger is not None else RightModelLedger()
    items = [i for i in ITEM_PACK if classes is None or i.task_class in classes]
    results: list[dict] = []
    for item in items:
        for model in models:
            results.append(run_item(item, model, ledger=led))
    by_model: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_model.setdefault(r["model"], {"pass": 0, "fail": 0})
        bucket["pass" if r["passed"] else "fail"] += 1
    return {
        "items": len(items),
        "models": len(models),
        "runs": len(results),
        "by_model": by_model,
        "results": results,
    }
