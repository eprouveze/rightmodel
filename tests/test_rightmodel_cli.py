"""Tests for spine.rightmodel_cli — the substrate CLI front-end (Module 8).

Exercises main([...]) directly (no subprocess) against a tmp host-local state
dir, so the real ~/.business-agent/rightmodel/ is never touched. Covers the
happy path (route returns a real model id; record then report reflects it) and
the cold-start edge case (report on an empty ledger must not error).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import rightmodel.rightmodel_types as rmt
from rightmodel.rightmodel_cli import main


@pytest.fixture(autouse=True)
def _tmp_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect all RightModel host-local state into a per-test tmp dir."""
    home = tmp_path / "rightmodel"
    monkeypatch.setattr(rmt, "RIGHTMODEL_HOME", home)
    monkeypatch.setattr(rmt, "rightmodel_home", lambda: (home.mkdir(parents=True, exist_ok=True) or home))
    return home


def test_route_prints_model_to_stdout_and_json_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["route", "--task-class", "coding", "--taint", "public", "--lang", "en"])
    captured = capsys.readouterr()

    model = captured.out.strip()
    assert rc == 0
    assert model and model != "NONE"

    decision = json.loads(captured.err)
    assert decision["task_class"] == "coding"
    assert decision["model"] == model
    assert "candidates_considered" in decision


def test_route_blocked_taint_returns_none_and_rc1(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["route", "--task-class", "coding", "--taint", "auth_secrets"])
    captured = capsys.readouterr()
    assert captured.out.strip() == "NONE"
    assert rc == 1
    decision = json.loads(captured.err)
    assert decision["model"] is None
    assert decision["rejected"]


def test_record_then_report_reflects_row(
    capsys: pytest.CaptureFixture[str], _tmp_home: Path
) -> None:
    rc = main(
        [
            "record",
            "--task-class",
            "coding",
            "--model",
            "claude-haiku-4-5",
            "--outcome",
            "pass",
            "--verified",
            "--cost",
            "0.002",
        ]
    )
    row_id = capsys.readouterr().out.strip()
    assert rc == 0
    assert row_id.startswith("rm-")
    assert (_tmp_home / "ledger.jsonl").exists()

    rc = main(["report"])
    report = capsys.readouterr().out
    assert rc == 0
    assert "Ledger: 1 rows" in report
    assert "claude-haiku-4-5" in report


def test_report_on_empty_ledger_is_cold_start_safe(
    capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["report"])
    report = capsys.readouterr().out
    assert rc == 0
    assert "Ledger: 0 rows" in report
    assert "Anti-Goodhart flags:" in report
    assert "(none)" in report
