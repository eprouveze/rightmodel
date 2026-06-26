"""Tests for spine.rightmodel_types — foundation enums/constants.

Other modules build against these exact values; a drift here breaks integration.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rightmodel.rightmodel_types import (
    RIGHTMODEL_HOME,
    SCORING_MODE,
    TASK_CLASSES,
    CellState,
    Outcome,
    TaskClass,
    cell_state_for_n,
    rightmodel_home,
)


# --------------------------------------------------------------------------- task classes


def test_task_classes_frozen_set_of_ten() -> None:
    assert len(TASK_CLASSES) == 10
    assert isinstance(TASK_CLASSES, tuple)
    assert set(TASK_CLASSES) == {
        "coding",
        "content_writing",
        "design",
        "planning_pm",
        "marketing",
        "data_analysis",
        "debugging",
        "testing",
        "security_check",
        "research",
    }


def test_taskclass_enum_matches_tuple() -> None:
    assert {tc.value for tc in TaskClass} == set(TASK_CLASSES)
    assert TaskClass.CODING == "coding"
    assert TaskClass("research") is TaskClass.RESEARCH


# --------------------------------------------------------------------------- scoring mode


def test_scoring_mode_covers_all_classes() -> None:
    assert set(SCORING_MODE) == set(TASK_CLASSES)
    assert set(SCORING_MODE.values()) <= {"auto", "arena", "guarded"}


def test_scoring_mode_known_splits() -> None:
    for cls in ("coding", "debugging", "testing", "data_analysis"):
        assert SCORING_MODE[cls] == "auto"
    for cls in ("content_writing", "design", "marketing"):
        assert SCORING_MODE[cls] == "arena"
    for cls in ("security_check", "research", "planning_pm"):
        assert SCORING_MODE[cls] == "guarded"


# --------------------------------------------------------------------------- enums


def test_outcome_values() -> None:
    assert Outcome.PASS == "pass"
    assert Outcome.FAIL == "fail"
    assert Outcome.PARTIAL == "partial"
    assert {o.value for o in Outcome} == {"pass", "fail", "partial"}


def test_cell_state_values() -> None:
    assert {s.value for s in CellState} == {"cold", "warming", "settled"}


# --------------------------------------------------------------------------- thresholds


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, CellState.COLD),   # cold-start
        (4, CellState.COLD),   # boundary just below WARMING
        (5, CellState.WARMING),
        (29, CellState.WARMING),  # boundary just below SETTLED
        (30, CellState.SETTLED),
        (1000, CellState.SETTLED),
    ],
)
def test_cell_state_thresholds(n: int, expected: CellState) -> None:
    assert cell_state_for_n(n) == expected


# --------------------------------------------------------------------------- home path


def test_rightmodel_home_constant_is_tcc_safe_and_uncreated(tmp_path: Path) -> None:
    expected = Path(os.path.expanduser("~/.business-agent/rightmodel"))
    assert RIGHTMODEL_HOME == expected
    assert ".business-agent" in str(RIGHTMODEL_HOME)
    assert "Documents" not in str(RIGHTMODEL_HOME)
    assert "/Volumes" not in str(RIGHTMODEL_HOME)


def test_rightmodel_home_creates_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "rm-home"
    monkeypatch.setattr("rightmodel.rightmodel_types.RIGHTMODEL_HOME", target)
    assert not target.exists()
    result = rightmodel_home()
    assert result == target
    assert target.is_dir()
    # idempotent
    assert rightmodel_home() == target
