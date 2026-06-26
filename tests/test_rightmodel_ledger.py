"""Tests for spine.rightmodel_ledger — JSONL outcome ledger.

Everything downstream reads this ledger, so durability (fsync), filtering,
collision grouping, and robustness to a missing/corrupt file are all covered.
All tests pass an explicit ``tmp_path`` ledger so no host-local state is touched.
"""

from __future__ import annotations

from pathlib import Path

from rightmodel.rightmodel_ledger import LedgerRow, RightModelLedger, is_independently_reviewed


# --------------------------------------------------------------------------- happy path


def test_append_builds_id_ts_provider_and_records(tmp_path: Path) -> None:
    ledger = RightModelLedger(path=tmp_path / "ledger.jsonl")
    row = ledger.append(
        task_class="coding",
        model="claude-haiku-4-5",
        outcome="pass",
        verified=True,
        cost_usd=0.002,
    )
    assert row.id.startswith("rm-") and row.id.endswith(row.id.rsplit("-", 1)[-1])
    assert len(row.id.rsplit("-", 1)[-1]) == 6
    assert row.ts.endswith("+00:00")
    assert row.provider == "claude"
    assert row.task_class == "coding"
    assert row.difficulty == 0.5 and row.lang == "en" and row.source == "production"

    reloaded = RightModelLedger(path=tmp_path / "ledger.jsonl").rows()
    assert len(reloaded) == 1
    assert reloaded[0].id == row.id
    assert reloaded[0].outcome == "pass"
    # verified is recorded as given — no governance coercion (Proof rule lives in Proof).
    assert reloaded[0].verified is True


# ------------------------------------------------------- provenance is attribution, not a gate


def test_provenance_fields_recorded_verbatim(tmp_path: Path) -> None:
    """Provenance attributes who produced/reviewed; it never mutates verified."""
    ledger = RightModelLedger(path=tmp_path / "ledger.jsonl")

    # author defaults to model; verified recorded as passed even with no reviewer.
    bare = ledger.append(
        task_class="coding", model="gpt-5.4-mini", outcome="pass",
        verified=True, cost_usd=0.001,
    )
    assert bare.verified is True
    assert bare.author == "gpt-5.4-mini" and bare.reviewer == "" and bare.review_kind == "none"

    # full provenance round-trips through JSONL.
    full = ledger.append(
        task_class="coding", model="gpt-5.4-mini", outcome="pass",
        verified=True, cost_usd=0.001,
        author="gpt-5.4-mini", reviewer="claude-opus-4-8", review_kind="ai-peer",
        origin="heartbeat", basis="pytest-green",
    )
    reloaded = RightModelLedger(path=tmp_path / "ledger.jsonl").rows()[-1]
    assert reloaded.id == full.id
    assert reloaded.reviewer == "claude-opus-4-8" and reloaded.review_kind == "ai-peer"
    assert reloaded.origin == "heartbeat" and reloaded.basis == "pytest-green"


def test_is_independently_reviewed_helper() -> None:
    # Descriptive only — used by callers to decide what verified to record.
    assert is_independently_reviewed(author="m", reviewer="m", review_kind="ai-peer") is False
    assert is_independently_reviewed(author="m", reviewer="", review_kind="human") is False
    assert is_independently_reviewed(author="m", reviewer="other", review_kind="none") is False
    assert is_independently_reviewed(author="m", reviewer="other", review_kind="human") is True


def test_record_roundtrip_preserves_all_fields(tmp_path: Path) -> None:
    ledger = RightModelLedger(path=tmp_path / "ledger.jsonl")
    original = LedgerRow(
        id="rm-fixed-abc123",
        ts="2026-06-16T00:00:00+00:00",
        task_class="research",
        model="sonar-pro",
        provider="perplexity",
        outcome="partial",
        verified=False,
        cost_usd=0.01,
        difficulty=0.8,
        lang="ja",
        source="arena",
        version_hash="2026-06-15",
        run_id="run-9",
        collision_group="dup-1",
        note="hello",
    )
    ledger.record(original)
    back = ledger.rows()[0]
    assert back == original


# --------------------------------------------------------------------------- filtering


def test_rows_filters(tmp_path: Path) -> None:
    ledger = RightModelLedger(path=tmp_path / "ledger.jsonl")
    ledger.append(task_class="coding", model="gpt-5.4-mini", outcome="pass",
                  verified=True, cost_usd=0.001, source="synthetic")
    ledger.append(task_class="testing", model="claude-haiku-4-5", outcome="fail",
                  verified=True, cost_usd=0.001, source="production")
    ledger.append(task_class="coding", model="claude-haiku-4-5", outcome="pass",
                  verified=False, cost_usd=0.001, source="production")

    assert len(ledger.rows(task_class="coding")) == 2
    assert len(ledger.rows(model="claude-haiku-4-5")) == 2
    assert len(ledger.rows(source="synthetic")) == 1
    assert len(ledger.rows(task_class="coding", source="production")) == 1


def test_rows_since_iso(tmp_path: Path) -> None:
    ledger = RightModelLedger(path=tmp_path / "ledger.jsonl")
    ledger.record(LedgerRow(
        id="a", ts="2026-01-01T00:00:00+00:00", task_class="coding", model="m",
        provider="unknown", outcome="pass", verified=True, cost_usd=0.0,
        difficulty=0.5, lang="en", source="production", version_hash="",
        run_id=None, collision_group=None, note=None,
    ))
    ledger.record(LedgerRow(
        id="b", ts="2026-12-01T00:00:00+00:00", task_class="coding", model="m",
        provider="unknown", outcome="pass", verified=True, cost_usd=0.0,
        difficulty=0.5, lang="en", source="production", version_hash="",
        run_id=None, collision_group=None, note=None,
    ))
    recent = ledger.rows(since_iso="2026-06-01T00:00:00+00:00")
    assert [r.id for r in recent] == ["b"]


# --------------------------------------------------------------------------- collisions


def test_collisions_groups_only_real_duels(tmp_path: Path) -> None:
    ledger = RightModelLedger(path=tmp_path / "ledger.jsonl")
    # Two models on the same item → a duel.
    ledger.append(task_class="coding", model="claude-haiku-4-5", outcome="pass",
                  verified=True, cost_usd=0.0, collision_group="item-1")
    ledger.append(task_class="coding", model="gpt-5.4-mini", outcome="fail",
                  verified=True, cost_usd=0.0, collision_group="item-1")
    # Lone collision group (only one model) → NOT a duel.
    ledger.append(task_class="coding", model="gemini-2.5-flash", outcome="pass",
                  verified=True, cost_usd=0.0, collision_group="item-2")
    # Null collision group → ignored.
    ledger.append(task_class="coding", model="claude-haiku-4-5", outcome="pass",
                  verified=True, cost_usd=0.0)

    groups = ledger.collisions("coding")
    assert len(groups) == 1
    assert {r.model for r in groups[0]} == {"claude-haiku-4-5", "gpt-5.4-mini"}
    # A class with no rows returns [].
    assert ledger.collisions("design") == []


# --------------------------------------------------------------------------- stats


def test_stats(tmp_path: Path) -> None:
    ledger = RightModelLedger(path=tmp_path / "ledger.jsonl")
    ledger.append(task_class="coding", model="m1", outcome="pass",
                  verified=True, cost_usd=0.0, source="production")
    ledger.append(task_class="coding", model="m1", outcome="fail",
                  verified=True, cost_usd=0.0, source="synthetic")
    ledger.append(task_class="testing", model="m2", outcome="pass",
                  verified=True, cost_usd=0.0, source="production")
    s = ledger.stats()
    assert s["total"] == 3
    assert s["by_class"] == {"coding": 2, "testing": 1}
    assert s["by_model"] == {"m1": 2, "m2": 1}
    assert s["by_source"] == {"production": 2, "synthetic": 1}


# --------------------------------------------------------------------------- robustness / cold start


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    ledger = RightModelLedger(path=tmp_path / "nope.jsonl")
    assert ledger.rows() == []
    assert ledger.collisions("coding") == []
    assert ledger.stats() == {"total": 0, "by_class": {}, "by_model": {}, "by_source": {}}


def test_partial_and_corrupt_lines_skipped(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = RightModelLedger(path=path)
    ledger.append(task_class="coding", model="m", outcome="pass",
                  verified=True, cost_usd=0.0)
    # Simulate a torn write / garbage line + a JSON object missing required fields.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
        fh.write("\n")
        fh.write('{"id": "x", "unexpected": true}\n')
    rows = ledger.rows()
    assert len(rows) == 1
    assert rows[0].task_class == "coding"
