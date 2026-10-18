# RightModel

**Earn-as-you-route model selection for AI agents.**

RightModel decides two things on every task: *which* model should run it, and *how much* that model is allowed to do on its own. It is the routing-and-trust substrate behind [Stewardship Engineering](https://emmanuel.prouveze.fr/writing/stewardship-engineering): the layer that turns "call an LLM" into "call the model that has earned this class of work, then check it afterward."

> Runs daily. **205 tests pass.**

---

## Why

Most agents pick a model once and trust it forever. That fails two ways: the wrong model gets sensitive data or expensive work it does not need, and a model that has never proven itself on a task is trusted as much as one that has. RightModel fixes both with a strict routing cascade and a falsifiable record of what each model actually got right.

## How it works

### 1. The routing cascade

Every task runs through a lexicographic cascade. Earlier stages are non-negotiable; cost only breaks ties among survivors.

```
taint  →  availability  →  capability  →  cost
```

- **taint** — data sensitivity (`public`, `customer_pii`, `payment_data`, `auth_secrets`, …). A model without clearance for the data never sees it, however cheap or capable. `auth_secrets` is never sent to any model.
- **availability** — circuit breaker skips models that are failing or rate-limited.
- **capability** — the capability matrix gates models that are not good enough at this task class.
- **cost** — among everything that survives, the cheapest wins (blended input/output price per task class).

### 2. The council

For load-bearing calls, one model's say-so is not enough. `choose_panel()` convenes a multi-provider panel and synthesizes their answers, because the failure mode of single-model (and same-model-different-persona) setups is *confident agreement on a wrong answer*. Combining several models into one stronger answer is an idea owed to Sakana's [Fugu](https://sakana.ai/fugu/) and [OpenRouter Fusion](https://openrouter.ai/fusion), which both do exactly this.

### 3. The outcome ledger and earned autonomy

Before a significant action, the agent writes a falsifiable prediction to an append-only **outcome ledger**; a deliberately dumb verifier settles it later. Authority is then a live meter computed from that record, kept **per model and per task-class**. A model earns latitude on a class by being right about it, and loses it the moment the record slips. A model that has not earned a class does not run it unattended, however cheap it is.

An `antigoodhart` guard flags predictions that look chosen to be easy to pass, so the meter can't be gamed.

## Install

```bash
pip install -e .
```

The only runtime dependency is `pydantic`.

## Quickstart (Python)

```python
from rightmodel import RightModelRouter, TaintClass

router = RightModelRouter()

decision = router.choose(
    task_class="coding",
    taint_set={TaintClass.PUBLIC},
)

print(decision.model)        # e.g. "claude-haiku-4-5" — cheapest model that cleared the cascade
print(decision.reason)       # human-readable trace of why
print(decision.rejected)     # {model: why it was dropped} for every candidate that lost

# later, once you know how the action turned out, feed the ledger:
router.record_result(decision, ...)
```

`choose()` returns a `RouteDecision`:

| field | meaning |
|---|---|
| `model` | the chosen model id (`None` if nothing survived the cascade) |
| `task_class` | the class routed |
| `reason` | human-readable explanation of the pick |
| `rejected` | `{model: reason}` for each rejected candidate |
| `candidates_considered` | the pool the cascade ran over |
| `exploration` | `True` if this was an exploration pick (see `explore_rate`) |
| `harness` | the `(model × harness)` pairing chosen, for non-`completion` modes |

Task classes: `coding`, `content_writing`, `design`, `planning_pm`, `marketing`, `data_analysis`, `debugging`, `testing`, `security_check`, `research`.

## Quickstart (CLI)

```bash
# pick a model for a public coding task
rightmodel route --task-class coding --taint public

# capability matrix + breaker + anti-Goodhart summary
rightmodel report

# append a real (non-synthetic) outcome to the ledger
rightmodel record --task-class coding --model claude-haiku-4-5 ...
```

## Configuration

The candidate pool and prices live in [`model_prices.py`](src/rightmodel/model_prices.py) (`PRICES`, `CANDIDATES`). Edit them for your own providers and model ids. Capability numbers start as low-confidence priors and move as the ledger accumulates real outcomes.

## Layout

| Module | Role |
|---|---|
| `rightmodel_router` | the taint → availability → capability → cost cascade |
| `rightmodel_matrix` | capability matrix (which models can do which classes) |
| `rightmodel_ledger` | the append-only outcome ledger |
| `rightmodel_arena`, `rightmodel_pairing`, `rightmodel_eval` | the council and its scoring |
| `rightmodel_shadow` | shadow mode: run every decision twice to measure graduation |
| `rightmodel_antigoodhart` | guards against predictions chosen to be easy to pass |
| `circuit_breaker` | isolates and reroutes around a failing model |
| `action_receipt`, `router_policy`, `stakes_score` | taint classes, routing policy, stakes scoring |

## Tests

```bash
pip install pytest
python -m pytest tests/
```

## Related

- [heartbeat](https://github.com/eprouveze/heartbeat) — the autonomous work loop RightModel routes for.
- [anamnesis](https://github.com/eprouveze/anamnesis) — persistent memory with decay, dreaming, and falsifiable precedents.
- [Stewardship Engineering](https://github.com/eprouveze/stewardship-engineering) — the umbrella: discipline, manifesto, and the full stack.

## License

MIT. See [LICENSE](LICENSE).
