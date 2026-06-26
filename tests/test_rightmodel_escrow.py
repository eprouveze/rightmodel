"""Tests for spine.rightmodel_escrow — the read-side outcome↔RightModel join.

A lightweight fake RightModel ledger (only .rows() + the LedgerRow attributes the join
reads) keeps these self-contained. The join is sparse and cold-start safe by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from rightmodel.rightmodel_escrow import build_downstream, read_outcome_rows


@dataclass
class FakeRMRow:
    task_class: str
    model: str
    run_id: str | None = None
    origin: str = "agent"


class FakeRMLedger:
    def __init__(self, rows: list[FakeRMRow]) -> None:
        self._rows = rows

    def rows(self):
        return list(self._rows)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pred(status, *, run_id=None, card_id=None, ts=None, nested=False):
    row: dict = {"id": "ol-x", "ts": ts or _now()}
    if run_id is not None:
        row["rightmodel_run_id"] = run_id
    if card_id is not None:
        row["card_id"] = card_id
    if nested:
        row["verification"] = {"status": status}
    else:
        row["status"] = status
    return row


# --- cold-start / empty -------------------------------------------------------

def test_empty_inputs_yield_empty():
    assert build_downstream([], FakeRMLedger([])) == {}


def test_no_join_key_yields_empty():
    # A terminal verdict with no rightmodel_run_id and no card_id joins to nothing.
    rm = FakeRMLedger([FakeRMRow("coding", "deepseek-v4-pro", run_id="rm-1")])
    assert build_downstream([_pred("FAILED")], rm) == {}


def test_read_outcome_rows_missing_file(tmp_path):
    assert read_outcome_rows(tmp_path / "nope.jsonl") == []


def test_read_outcome_rows_skips_blank_and_bad(tmp_path):
    p = tmp_path / "ledger.jsonl"
    p.write_text('{"id":"a"}\n\nnot json\n{"id":"b"}\n', encoding="utf-8")
    assert [r["id"] for r in read_outcome_rows(p)] == ["a", "b"]


# --- the core join ------------------------------------------------------------

def test_failed_verdict_joins_negative_via_run_id():
    rm = FakeRMLedger([FakeRMRow("coding", "deepseek-v4-pro", run_id="rm-1")])
    rows = [_pred("FAILED", run_id="rm-1") for _ in range(3)]
    out = build_downstream(rows, rm, min_n=3)
    assert out == {"coding::deepseek-v4-pro": -1.0}


def test_confirmed_verdict_joins_positive():
    rm = FakeRMLedger([FakeRMRow("coding", "deepseek-v4-pro", run_id="rm-1")])
    rows = [_pred("CONFIRMED", run_id="rm-1") for _ in range(3)]
    assert build_downstream(rows, rm, min_n=3) == {"coding::deepseek-v4-pro": 1.0}


def test_mixed_verdicts_average():
    rm = FakeRMLedger([FakeRMRow("coding", "m", run_id="rm-1")])
    rows = [
        _pred("FAILED", run_id="rm-1"),
        _pred("FAILED", run_id="rm-1"),
        _pred("CONFIRMED", run_id="rm-1"),
    ]
    # (-1 -1 +1) / 3 = -1/3
    assert build_downstream(rows, rm, min_n=3)["coding::m"] == (-1.0 / 3.0)


def test_unverified_is_skipped():
    rm = FakeRMLedger([FakeRMRow("coding", "m", run_id="rm-1")])
    rows = [_pred("UNVERIFIED", run_id="rm-1") for _ in range(5)]
    assert build_downstream(rows, rm, min_n=1) == {}


def test_min_n_suppresses_thin_evidence():
    rm = FakeRMLedger([FakeRMRow("coding", "m", run_id="rm-1")])
    rows = [_pred("FAILED", run_id="rm-1"), _pred("FAILED", run_id="rm-1")]
    assert build_downstream(rows, rm, min_n=3) == {}          # only 2 < 3
    assert build_downstream(rows, rm, min_n=2) == {"coding::m": -1.0}


def test_card_id_fallback_joins_via_origin():
    rm = FakeRMLedger([FakeRMRow("design", "m", origin="heartbeat-card:c-42")])
    rows = [_pred("FAILED", card_id="c-42") for _ in range(3)]
    assert build_downstream(rows, rm, min_n=3) == {"design::m": -1.0}


def test_run_id_wins_over_card_id():
    rm = FakeRMLedger([
        FakeRMRow("coding", "by-run", run_id="rm-1"),
        FakeRMRow("design", "by-card", origin="heartbeat-card:c-9"),
    ])
    rows = [_pred("FAILED", run_id="rm-1", card_id="c-9") for _ in range(3)]
    assert build_downstream(rows, rm, min_n=3) == {"coding::by-run": -1.0}


def test_fusion_attributed_to_pseudo_model():
    rm = FakeRMLedger([FakeRMRow("coding", "council:budget-trio", run_id="rm-f")])
    rows = [_pred("FAILED", run_id="rm-f") for _ in range(3)]
    assert build_downstream(rows, rm, min_n=3) == {"coding::council:budget-trio": -1.0}


def test_nested_verification_schema_tolerated():
    rm = FakeRMLedger([FakeRMRow("coding", "m", run_id="rm-1")])
    rows = [_pred("FAILED", run_id="rm-1", nested=True) for _ in range(3)]
    assert build_downstream(rows, rm, min_n=3) == {"coding::m": -1.0}


def test_window_excludes_old_verdicts():
    rm = FakeRMLedger([FakeRMRow("coding", "m", run_id="rm-1")])
    old = (datetime.now(timezone.utc) - timedelta(days=99)).isoformat()
    rows = [_pred("FAILED", run_id="rm-1", ts=old) for _ in range(3)]
    assert build_downstream(rows, rm, window_days=30, min_n=3) == {}


def test_retry_attributes_to_latest_run_row():
    # Two RM rows share run_id rm-1; later row (by ledger order) wins the index.
    rm = FakeRMLedger([
        FakeRMRow("coding", "first", run_id="rm-1"),
        FakeRMRow("debugging", "second", run_id="rm-1"),
    ])
    rows = [_pred("FAILED", run_id="rm-1") for _ in range(3)]
    assert build_downstream(rows, rm, min_n=3) == {"debugging::second": -1.0}
