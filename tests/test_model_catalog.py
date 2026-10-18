"""Tests for user-configurable ModelCatalog and provider-agnostic routing."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from rightmodel.model_catalog import ModelCatalog, load_catalog, set_active_catalog
from rightmodel.action_receipt import TaintClass
from rightmodel.router_policy import check as taint_check, model_to_provider, Provider
from rightmodel.rightmodel_router import RightModelRouter


@pytest.fixture(autouse=True)
def reset_catalog():
    """Reset the active catalog before and after each test."""
    set_active_catalog(None)
    yield
    set_active_catalog(None)


def test_custom_catalog_from_dict():
    data = {
        "models": {
            "my-custom-llm": {
                "provider": "local",
                "input_price": 0.05,
                "output_price": 0.20,
                "allowed_taints": ["public", "internal", "customer_pii"],
            },
            "my-expensive-reasoner": {
                "provider": "openai",
                "input_price": 5.0,
                "output_price": 20.0,
                "allowed_taints": ["public", "internal"],
            },
        },
        "candidates": {
            "coding": ["my-custom-llm", "my-expensive-reasoner"],
        },
        "priors": {
            "coding": {
                "my-custom-llm": 75.0,
                "my-expensive-reasoner": 90.0,
            }
        },
    }

    catalog = load_catalog(config_dict=data)

    assert "my-custom-llm" in catalog.prices
    assert catalog.prices["my-custom-llm"] == (0.05, 0.20)
    assert catalog.blended_cost("my-custom-llm") == pytest.approx(0.05 * 0.25 + 0.20 * 0.75)
    assert catalog.candidates["coding"] == ["my-custom-llm", "my-expensive-reasoner"]
    assert catalog.priors["coding"]["my-custom-llm"] == 75.0

    # Verify provider mapping
    assert model_to_provider("my-custom-llm") == Provider.UNKNOWN  # "local" maps safely to Provider
    assert model_to_provider("my-expensive-reasoner") == Provider.OPENAI

    # Verify user-configured taints
    # "my-custom-llm" is cleared for customer_pii:
    dec_pii = taint_check("my-custom-llm", {TaintClass.CUSTOMER_PII})
    assert dec_pii.allowed is True

    # "my-expensive-reasoner" is NOT cleared for customer_pii:
    dec_blocked = taint_check("my-expensive-reasoner", {TaintClass.CUSTOMER_PII})
    assert dec_blocked.allowed is False
    assert "not cleared for taint(s): customer_pii" in dec_blocked.block_reason

    # AUTH_SECRETS is always blocked
    dec_secret = taint_check("my-custom-llm", {TaintClass.AUTH_SECRETS})
    assert dec_secret.allowed is False


def test_router_with_custom_catalog(tmp_path: Path):
    cfg_file = tmp_path / "models.json"
    cfg_file.write_text(
        json.dumps({
            "models": {
                "cheap-agent": {
                    "provider": "local",
                    "input_price": 0.0,
                    "output_price": 0.0,
                    "allowed_taints": ["public"],
                },
                "pro-agent": {
                    "provider": "openai",
                    "input_price": 2.0,
                    "output_price": 10.0,
                    "allowed_taints": ["public"],
                },
            },
            "candidates": {
                "coding": ["cheap-agent", "pro-agent"],
            },
            "priors": {
                "coding": {
                    "cheap-agent": 70.0,
                    "pro-agent": 88.0,
                },
            },
        }),
        encoding="utf-8",
    )

    router = RightModelRouter(config=cfg_file)

    # Route public coding task — both clear capability bar (default 60), cheap-agent is cheaper
    # (cost-min with quality premium check: delta is 88 - 70 = 18 > 15, but cheap-agent is cold)
    decision = router.choose(task_class="coding", taint_set={TaintClass.PUBLIC})
    assert decision.model in {"cheap-agent", "pro-agent"}
    assert decision.candidates_considered == ["cheap-agent", "pro-agent"]
