"""User-configurable model catalog for RightModel.

RightModel is provider- and model-agnostic. Rather than hardcoding fixed vendors
or point-in-time model IDs into library code, users define their own model catalog
via a JSON file (e.g., `models.json`), an environment variable (`RIGHTMODEL_CONFIG`),
or programmatically via `ModelCatalog`.

If no configuration file is provided, a neutral baseline fixture catalog is loaded
by default for immediate zero-config testing and evaluation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


# Default neutral fixture models used for out-of-the-box evaluation and testing
_DEFAULT_FIXTURE_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.2, 1.25),
    "gpt-5.3-codex": (1.75, 14.0),
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.3, 2.5),
    "deepseek-v4-pro": (0.435, 0.87),
    "deepseek-v4-flash": (0.09, 0.18),
    "kimi-k2.6": (0.68, 3.41),
    "minimax-m3": (0.3, 1.2),
    "sonar-pro": (3.0, 15.0),
    "council:budget-trio": (0.0, 0.0),
    "council:self-lite": (0.0, 0.0),
}

_DEFAULT_FIXTURE_PRIORS: dict[str, dict[str, float]] = {
    "coding": {
        "claude-opus-4-8": 90.0,
        "gpt-5.3-codex": 89.0,
        "claude-sonnet-4-6": 86.0,
        "gpt-5.5": 86.0,
        "gemini-3.1-pro-preview": 83.0,
        "deepseek-v4-pro": 80.0,
        "gemini-2.5-pro": 79.0,
        "claude-haiku-4-5": 68.0,
        "gpt-5.4-mini": 66.0,
        "kimi-k2.6": 70.0,
        "gpt-5.4-nano": 58.0,
    },
    "content_writing": {
        "claude-opus-4-8": 90.0,
        "claude-sonnet-4-6": 87.0,
        "gpt-5.5": 84.0,
        "gemini-3.1-pro-preview": 82.0,
        "gemini-2.5-pro": 78.0,
        "claude-haiku-4-5": 70.0,
        "minimax-m3": 64.0,
        "gpt-5.4-mini": 65.0,
    },
    "design": {
        "claude-opus-4-8": 86.0,
        "gpt-5.5": 84.0,
        "gemini-3.1-pro-preview": 85.0,
        "claude-sonnet-4-6": 82.0,
        "gemini-2.5-pro": 78.0,
        "claude-haiku-4-5": 66.0,
        "gpt-5.4-mini": 63.0,
    },
    "planning_pm": {
        "claude-opus-4-8": 88.0,
        "gpt-5.5": 86.0,
        "claude-sonnet-4-6": 84.0,
        "gemini-3.1-pro-preview": 82.0,
        "gemini-2.5-pro": 78.0,
        "claude-haiku-4-5": 68.0,
        "gpt-5.4-mini": 66.0,
    },
    "marketing": {
        "claude-opus-4-8": 87.0,
        "claude-sonnet-4-6": 85.0,
        "gpt-5.5": 84.0,
        "gemini-3.1-pro-preview": 81.0,
        "gemini-2.5-pro": 77.0,
        "claude-haiku-4-5": 69.0,
        "minimax-m3": 63.0,
        "gpt-5.4-mini": 65.0,
    },
    "data_analysis": {
        "gpt-5.5": 89.0,
        "claude-opus-4-8": 88.0,
        "claude-sonnet-4-6": 84.0,
        "gemini-3.1-pro-preview": 83.0,
        "deepseek-v4-pro": 80.0,
        "gemini-2.5-pro": 79.0,
        "claude-haiku-4-5": 67.0,
        "gpt-5.4-mini": 66.0,
    },
    "debugging": {
        "claude-opus-4-8": 89.0,
        "gpt-5.3-codex": 88.0,
        "gpt-5.5": 86.0,
        "claude-sonnet-4-6": 85.0,
        "gemini-3.1-pro-preview": 82.0,
        "deepseek-v4-pro": 79.0,
        "claude-haiku-4-5": 67.0,
        "gpt-5.4-mini": 64.0,
    },
    "testing": {
        "claude-opus-4-8": 86.0,
        "gpt-5.3-codex": 85.0,
        "claude-sonnet-4-6": 83.0,
        "gpt-5.5": 83.0,
        "deepseek-v4-pro": 78.0,
        "gemini-2.5-pro": 77.0,
        "claude-haiku-4-5": 70.0,
        "gpt-5.4-mini": 68.0,
        "gpt-5.4-nano": 60.0,
    },
    "security_check": {
        "claude-opus-4-8": 88.0,
        "gpt-5.5": 85.0,
        "claude-sonnet-4-6": 84.0,
        "gemini-3.1-pro-preview": 81.0,
        "deepseek-v4-pro": 76.0,
        "claude-haiku-4-5": 66.0,
        "gpt-5.4-mini": 62.0,
    },
    "research": {
        "sonar-pro": 89.0,
        "claude-opus-4-8": 85.0,
        "gpt-5.5": 84.0,
        "gemini-3.1-pro-preview": 83.0,
        "claude-sonnet-4-6": 82.0,
        "gemini-2.5-pro": 78.0,
        "claude-haiku-4-5": 67.0,
        "gpt-5.4-mini": 64.0,
    },
}

_DEFAULT_FIXTURE_CANDIDATES: dict[str, list[str]] = {
    "coding": [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "deepseek-v4-pro",
        "gpt-5.4-mini",
    ],
    "content_writing": [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "gpt-5.5",
        "claude-haiku-4-5",
        "minimax-m3",
    ],
    "design": [
        "claude-opus-4-8",
        "gemini-3.1-pro-preview",
        "gpt-5.5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ],
    "planning_pm": [
        "claude-opus-4-8",
        "gpt-5.5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "gpt-5.4-mini",
    ],
    "marketing": [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "gpt-5.5",
        "claude-haiku-4-5",
        "minimax-m3",
    ],
    "data_analysis": [
        "gpt-5.5",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "deepseek-v4-pro",
        "gemini-2.5-pro",
        "gpt-5.4-mini",
    ],
    "debugging": [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "deepseek-v4-pro",
    ],
    "testing": [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ],
    "security_check": [
        "claude-opus-4-8",
        "gpt-5.5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "deepseek-v4-pro",
    ],
    "research": [
        "sonar-pro",
        "claude-opus-4-8",
        "gpt-5.5",
        "gemini-3.1-pro-preview",
        "claude-sonnet-4-6",
    ],
}


@dataclass
class ModelCatalog:
    """Catalog of models, token prices, candidate pools, and capability priors."""

    prices: dict[str, tuple[float, float]] = field(default_factory=dict)
    candidates: dict[str, list[str]] = field(default_factory=dict)
    priors: dict[str, dict[str, float]] = field(default_factory=dict)
    taints: dict[str, set[str]] = field(default_factory=dict)
    providers: dict[str, str] = field(default_factory=dict)

    def blended_cost(self, model: str) -> float:
        """Blended ranking cost (USD/1M): input*0.25 + output*0.75."""
        price = self.prices.get(model)
        if price is None:
            return float("inf")
        input_cost, output_cost = price
        return input_cost * 0.25 + output_cost * 0.75

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelCatalog:
        """Parse catalog from a configuration dictionary."""
        prices: dict[str, tuple[float, float]] = {}
        taints: dict[str, set[str]] = {}
        providers: dict[str, str] = {}

        # Parse "models" section if present
        models_data = data.get("models", {})
        for model_id, info in models_data.items():
            if isinstance(info, (list, tuple)) and len(info) >= 2:
                prices[model_id] = (float(info[0]), float(info[1]))
            elif isinstance(info, dict):
                in_price = float(info.get("input_price_usd_per_m", info.get("input_price", 0.0)))
                out_price = float(info.get("output_price_usd_per_m", info.get("output_price", 0.0)))
                prices[model_id] = (in_price, out_price)
                if "allowed_taints" in info:
                    taints[model_id] = {str(t).lower() for t in info["allowed_taints"]}
                elif "taints" in info:
                    taints[model_id] = {str(t).lower() for t in info["taints"]}
                if "provider" in info:
                    providers[model_id] = str(info["provider"]).lower()

        # Direct "prices" dict override if present
        if "prices" in data:
            for m, p in data["prices"].items():
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    prices[m] = (float(p[0]), float(p[1]))

        candidates = {
            task_class: list(model_list)
            for task_class, model_list in data.get("candidates", {}).items()
        }

        priors = {
            task_class: {m: float(val) for m, val in model_priors.items()}
            for task_class, model_priors in data.get("priors", {}).items()
        }

        return cls(
            prices=prices,
            candidates=candidates,
            priors=priors,
            taints=taints,
            providers=providers,
        )

    @classmethod
    def from_file(cls, path: Path | str) -> ModelCatalog:
        """Load catalog from a JSON file."""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


_ACTIVE_CATALOG: ModelCatalog | None = None


def get_active_catalog() -> ModelCatalog:
    """Return the currently active ModelCatalog, resolving via config file or defaults."""
    global _ACTIVE_CATALOG
    if _ACTIVE_CATALOG is not None:
        return _ACTIVE_CATALOG

    # 1. Environment variable override
    env_path = os.environ.get("RIGHTMODEL_CONFIG")
    if env_path and Path(env_path).is_file():
        _ACTIVE_CATALOG = ModelCatalog.from_file(env_path)
        return _ACTIVE_CATALOG

    # 2. Local models.json in current working directory
    local_cfg = Path("models.json")
    if local_cfg.is_file():
        _ACTIVE_CATALOG = ModelCatalog.from_file(local_cfg)
        return _ACTIVE_CATALOG

    # 3. Default fixture catalog
    _ACTIVE_CATALOG = ModelCatalog(
        prices=dict(_DEFAULT_FIXTURE_PRICES),
        candidates=dict(_DEFAULT_FIXTURE_CANDIDATES),
        priors=dict(_DEFAULT_FIXTURE_PRIORS),
    )
    return _ACTIVE_CATALOG


def set_active_catalog(catalog: ModelCatalog | None) -> None:
    """Set or reset the active ModelCatalog."""
    global _ACTIVE_CATALOG
    _ACTIVE_CATALOG = catalog


def load_catalog(path: Path | str | None = None, config_dict: dict | None = None) -> ModelCatalog:
    """Load and activate a custom ModelCatalog from a file or dictionary."""
    if config_dict is not None:
        cat = ModelCatalog.from_dict(config_dict)
    elif path is not None:
        cat = ModelCatalog.from_file(path)
    else:
        cat = get_active_catalog()
    set_active_catalog(cat)
    return cat
