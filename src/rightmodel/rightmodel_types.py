"""RightModel shared types — foundation module (no deps).

Frozen enums/constants for the fleet routing substrate so other modules don't
import each other for these primitives. See
`docs/solutions/architecture/rightmodel-substrate-contract.md` Module 0.

All persistent state lives under `~/.business-agent/rightmodel/` (TCC-safe,
host-local). `RIGHTMODEL_HOME` is the expanded path but is NOT created at import
time; call `rightmodel_home()` to materialize the directory.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


TASK_CLASSES: tuple[str, ...] = (
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
)


class TaskClass(str, Enum):
    """The 10 frozen v1 task classes routed by the substrate."""

    CODING = "coding"
    CONTENT_WRITING = "content_writing"
    DESIGN = "design"
    PLANNING_PM = "planning_pm"
    MARKETING = "marketing"
    DATA_ANALYSIS = "data_analysis"
    DEBUGGING = "debugging"
    TESTING = "testing"
    SECURITY_CHECK = "security_check"
    RESEARCH = "research"


SCORING_MODE: dict[str, str] = {
    "coding": "auto",
    "debugging": "auto",
    "testing": "auto",
    "data_analysis": "auto",
    "content_writing": "arena",
    "design": "arena",
    "marketing": "arena",
    "security_check": "guarded",
    "research": "guarded",
    "planning_pm": "guarded",
}


class Outcome(str, Enum):
    """Independently-resolvable result of one delegated task."""

    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"


class CellState(str, Enum):
    """Confidence tier of a (task_class, model) matrix cell, derived from n.

    COLD: n<5 (≈ prior). WARMING: 5<=n<30. SETTLED: n>=30 (≈ data).
    """

    COLD = "cold"
    WARMING = "warming"
    SETTLED = "settled"


_COLD_MAX = 5
_SETTLED_MIN = 30


def cell_state_for_n(n: int) -> CellState:
    """Map an observation count to its CellState per the frozen thresholds."""
    if n < _COLD_MAX:
        return CellState.COLD
    if n < _SETTLED_MIN:
        return CellState.WARMING
    return CellState.SETTLED


RIGHTMODEL_HOME: Path = Path(os.path.expanduser("~/.business-agent/rightmodel"))


def rightmodel_home() -> Path:
    """Ensure the host-local state directory exists, returning its path."""
    RIGHTMODEL_HOME.mkdir(parents=True, exist_ok=True)
    return RIGHTMODEL_HOME
