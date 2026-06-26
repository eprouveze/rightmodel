# RightModel

Earn-as-you-route model selection for agents.

RightModel picks the right model for each task and decides how much it is allowed to do on its own. It is the routing-and-trust substrate from [Stewardship Engineering](https://emmanuel.prouveze.fr/writing/stewardship-engineering): the layer that turns "call an LLM" into "call the model that has earned this class of work, and check it afterward."

## What it does

**Routing.** Every task is routed through a strict cascade:

```
taint  →  availability  →  capability  →  cost
```

Taint (data sensitivity) comes first and is non-negotiable: a model that lacks the clearance for a class of data never sees it, however cheap or capable it is. Among the models that clear taint and are available and capable enough, the cheapest wins.

**Council.** For load-bearing calls, one model's say-so is not enough. RightModel convenes a multi-provider council and synthesizes the answers, because the failure mode of single-model and same-model-multiple-persona setups is confident agreement on a wrong answer. (Combining several models into one stronger answer is an idea owed to Sakana's Fugu.)

**The outcome ledger, and earned autonomy.** Before a significant action, the agent writes a falsifiable prediction to an append-only ledger; a deliberately dumb verifier settles it later. Authority is then a live meter computed from that record, kept **per model and per task-class**. A model earns latitude on a class by being right about it, and loses it the moment the record slips. A model that has not earned a class does not run it unattended, however cheap it is.

## Install

```bash
pip install -e .
```

Only runtime dependency is `pydantic`.

## Layout

| Module | Role |
|---|---|
| `rightmodel_router` | the taint → availability → capability → cost cascade |
| `rightmodel_matrix` | the capability matrix (which models can do which classes) |
| `rightmodel_ledger` | the append-only outcome ledger |
| `rightmodel_arena` / `rightmodel_pairing` / `rightmodel_eval` | the council and its scoring |
| `rightmodel_shadow` | shadow mode: run every routing decision twice to measure graduation |
| `rightmodel_antigoodhart` | guards against predictions chosen to be easy to pass |
| `circuit_breaker` | isolates and reroutes around a failing model |
| `action_receipt` / `router_policy` / `stakes_score` | taint classes, routing policy, stakes scoring |

The candidate model list lives in `model_prices.CANDIDATES`; edit it for your own providers.

## Tests

```bash
pip install pytest && python -m pytest tests/
```

## License

MIT. See [LICENSE](LICENSE).
