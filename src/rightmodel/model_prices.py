"""Model prices, candidates, and capability priors (RightModel Module 1).

Exposes the active catalog's price table, candidate pools per task class,
bootstrapped capability priors, and the `blended_cost()` helper.

RightModel is provider- and model-agnostic: by default, these structures are loaded
from the user's `models.json` (or path via `RIGHTMODEL_CONFIG` / `load_catalog()`).
If no user configuration is supplied, a baseline test fixture catalog is used.
"""

from __future__ import annotations

from typing import Mapping
from .model_catalog import (
    ModelCatalog,
    get_active_catalog,
    load_catalog,
    set_active_catalog,
    _DEFAULT_FIXTURE_PRICES,
    _DEFAULT_FIXTURE_CANDIDATES,
    _DEFAULT_FIXTURE_PRIORS,
)


class _DynamicDict(dict):
    """Dynamic dict view that reflects the active catalog unless explicitly modified."""

    def __init__(self, key: str, fallback: dict) -> None:
        super().__init__()
        self._key = key
        self._fallback = fallback

    def _target(self) -> dict:
        catalog = get_active_catalog()
        target = getattr(catalog, self._key, None)
        return target if target is not None else self._fallback

    def __getitem__(self, k):
        return self._target()[k]

    def __setitem__(self, k, v):
        self._target()[k] = v

    def __delitem__(self, k):
        del self._target()[k]

    def __contains__(self, k):
        return k in self._target()

    def __iter__(self):
        return iter(self._target())

    def __len__(self):
        return len(self._target())

    def get(self, k, default=None):
        return self._target().get(k, default)

    def keys(self):
        return self._target().keys()

    def values(self):
        return self._target().values()

    def items(self):
        return self._target().items()

    def copy(self):
        return dict(self._target())


# Dynamic views backed by the active catalog
PRICES: dict[str, tuple[float, float]] = _DynamicDict("prices", _DEFAULT_FIXTURE_PRICES)
CANDIDATES: dict[str, list[str]] = _DynamicDict("candidates", _DEFAULT_FIXTURE_CANDIDATES)
SEED_PRIOR: dict[str, dict[str, float]] = _DynamicDict("priors", _DEFAULT_FIXTURE_PRIORS)


def blended_cost(model: str) -> float:
    """Blended ranking cost (USD/1M): input*0.25 + output*0.75.

    Cheaper = lower. Unknown model → float("inf") so it sorts last and never wins
    a cost-min tie. Weighting favours output tokens, the dominant cost in practice.
    """
    return get_active_catalog().blended_cost(model)


__all__ = [
    "PRICES",
    "CANDIDATES",
    "SEED_PRIOR",
    "blended_cost",
    "ModelCatalog",
    "load_catalog",
    "get_active_catalog",
    "set_active_catalog",
]
