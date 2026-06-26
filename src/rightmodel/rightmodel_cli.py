"""RightModel CLI — thin argparse front-end for the routing substrate (Module 8 + v2).

Lets the heartbeat / Foreman call the substrate without importing Python.
Subcommands:

    route        — run the cascade; print the chosen model id (or "NONE") to STDOUT
                   and the full RouteDecision json to STDERR. The stdout/stderr split
                   keeps a shell `MODEL=$(... route ...)` capture clean.
    report       — human-readable matrix summary + breaker snapshot + anti-Goodhart
                   flags + arena triggers + freshness flags. Cold-start safe.
    seed         — run the synthetic eval harness (Module 7) over models × classes,
                   appending to the ledger (honors the llm_runner kill-switch).
    record       — append one production outcome row to the ledger.
    arena queue  — propose blind pairwise matchups for a subjective class (Module 9).
    arena vote   — record one blind vote as two arena collision rows (Module 9).
    freshness    — print prediction_drift flags (Module 10; identity=model_id).
    delegate     — run the route→run→review→record loop (Module 12); honors the
                   kill-switch (dry, non-recorded result when not live).

Importable both as ``python -m spine.rightmodel_cli`` (from the spine package)
and ``python -m agents.business-agent.spine.rightmodel_cli`` (from repo root):
the try/except below picks relative imports inside the package, absolute ones
otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Sequence

# The shipped spine modules (router_policy, action_receipt, state_ledger, …) use
# absolute ``from spine.X import …`` imports, which only resolve when the
# ``agents/business-agent`` dir is on sys.path. Add it so this CLI runs both as
# ``python -m spine.rightmodel_cli`` and ``python -m agents.business-agent.spine.rightmodel_cli``.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

try:  # package-relative (python -m spine.rightmodel_cli)
    from .action_receipt import TaintClass
    from .circuit_breaker import CircuitBreaker
    from .model_prices import CANDIDATES, blended_cost
    from .rightmodel_antigoodhart import AntiGoodhart
    from .rightmodel_arena import Arena, ArenaMatchup
    from .rightmodel_consumer import delegate as consumer_delegate
    from .rightmodel_freshness import prediction_drift
    from .rightmodel_ledger import RightModelLedger
    from .rightmodel_matrix import CapabilityMatrix
    from .rightmodel_router import RightModelRouter
    from .rightmodel_shadow import ShadowRecorder
    from .rightmodel_types import SCORING_MODE, TASK_CLASSES, rightmodel_home
except ImportError:  # absolute (python -m agents.business-agent.spine.rightmodel_cli)
    from .action_receipt import TaintClass
    from .circuit_breaker import CircuitBreaker
    from .model_prices import CANDIDATES, blended_cost
    from .rightmodel_antigoodhart import AntiGoodhart
    from .rightmodel_arena import Arena, ArenaMatchup
    from .rightmodel_consumer import delegate as consumer_delegate
    from .rightmodel_freshness import prediction_drift
    from .rightmodel_ledger import RightModelLedger
    from .rightmodel_matrix import CapabilityMatrix
    from .rightmodel_router import RightModelRouter
    from .rightmodel_shadow import ShadowRecorder
    from .rightmodel_types import SCORING_MODE, TASK_CLASSES, rightmodel_home


def _build_stack() -> tuple[RightModelLedger, CapabilityMatrix, CircuitBreaker, RightModelRouter]:
    """Construct the shared substrate objects over the host-local state dir."""
    ledger = RightModelLedger()
    matrix = CapabilityMatrix(ledger)
    breaker = CircuitBreaker()
    router = RightModelRouter(ledger=ledger, matrix=matrix, breaker=breaker)
    return ledger, matrix, breaker, router


def _parse_taints(raw: str | None) -> set[TaintClass]:
    """Parse a comma-separated taint string into a set of TaintClass values."""
    if not raw:
        return set()
    out: set[TaintClass] = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            out.add(TaintClass(tok))
    return out


def _cmd_route(args: argparse.Namespace) -> int:
    """route: cascade-pick a model; model id → stdout, RouteDecision json → stderr."""
    _, _, _, router = _build_stack()
    candidates = args.candidates.split(",") if args.candidates else None
    decision = router.choose(
        task_class=args.task_class,
        taint_set=_parse_taints(args.taint),
        candidates=candidates,
        run_id=args.run_id,
        lang=args.lang,
    )
    print(json.dumps(dataclasses.asdict(decision), ensure_ascii=False, indent=2), file=sys.stderr)
    print(decision.model if decision.model is not None else "NONE")
    return 0 if decision.model is not None else 1


def _cmd_report(args: argparse.Namespace) -> int:
    """report: matrix summary + breaker snapshot + anti-Goodhart flags (cold-safe)."""
    ledger, matrix, breaker, _ = _build_stack()
    lines: list[str] = []
    lines.append(f"RightModel report — state: {rightmodel_home()}")

    stats = ledger.stats()
    lines.append("")
    lines.append(f"Ledger: {stats['total']} rows  by_source={stats['by_source']}")

    lines.append("")
    lines.append("Capability matrix (μ ± σ, n, state) over default candidates:")
    for task_class in TASK_CLASSES:
        cands = CANDIDATES.get(task_class, [])
        if not cands:
            continue
        lines.append(f"  [{task_class}]  bar={matrix.min_bar(task_class):.1f}")
        for cell in matrix.rank(task_class, cands):
            lines.append(
                f"    {cell.model:<24} μ={cell.mu:5.1f} ±{cell.sigma:4.1f} "
                f"n={cell.n:<3} {cell.state:<8} cost={blended_cost(cell.model):.3f}"
            )

    lines.append("")
    lines.append("Circuit breakers:")
    snap = breaker.snapshot()
    if not snap:
        lines.append("  (none tripped — all providers CLOSED)")
    for provider, st in sorted(snap.items()):
        lines.append(
            f"  {provider:<12} state={st.state:<10} fails={st.consecutive_failures} "
            f"cooldown_until={st.cooldown_until or '-'}"
        )

    lines.append("")
    lines.append("Anti-Goodhart flags:")
    # Read-side join: attribute lagged world-verdicts (the outcome ledger) back to the model
    # that produced the action, feeding escrow_flags. Cold-start safe ({} when nothing joins).
    try:
        from .rightmodel_escrow import build_downstream, read_outcome_rows
    except ImportError:  # absolute-import context (python -m agents.…)
        from .rightmodel_escrow import build_downstream, read_outcome_rows
    downstream = build_downstream(read_outcome_rows(), ledger)
    flags = AntiGoodhart(ledger, matrix).report(downstream=downstream)
    if not flags:
        lines.append("  (none)")
    for f in flags:
        lines.append(f"  [{f.severity}] {f.kind} {f.task_class}/{f.model or '*'}: {f.detail}")

    lines.append("")
    lines.append("Arena triggers (subjective classes):")
    arena = Arena(ledger, matrix)
    subjective = [tc for tc in TASK_CLASSES if SCORING_MODE.get(tc) == "arena"]
    for task_class in subjective:
        fire, reason = arena.should_trigger(task_class)
        mark = "FIRE" if fire else "ok  "
        lines.append(f"  [{mark}] {task_class}: {reason}")

    lines.append("")
    lines.append("Freshness flags:")
    fresh = prediction_drift(ledger, matrix)
    if not fresh:
        lines.append("  (none)")
    for f in fresh:
        lines.append(f"  [{f.severity}] {f.kind} {f.task_class}/{f.model or '*'}: {f.detail}")

    print("\n".join(lines))
    return 0


def _cmd_seed(args: argparse.Namespace) -> int:
    """seed: run the synthetic eval harness (Module 7) over models × classes."""
    try:
        try:
            from .rightmodel_eval import seed as eval_seed
        except ImportError:
            from .rightmodel_eval import seed as eval_seed
    except ImportError:
        print("rightmodel_eval (Module 7) not available — cannot seed", file=sys.stderr)
        return 2
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    classes = [c.strip() for c in args.classes.split(",") if c.strip()] if args.classes else None
    ledger = RightModelLedger()
    summary = eval_seed(models, classes=classes, ledger=ledger)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    """record: append one production outcome row to the ledger."""
    ledger = RightModelLedger()
    row = ledger.append(
        task_class=args.task_class,
        model=args.model,
        outcome=args.outcome,
        verified=args.verified,
        cost_usd=args.cost,
        difficulty=args.difficulty,
        lang=args.lang,
        source="production",
        run_id=args.run_id,
        collision_group=args.collision_group,
        note=args.note,
    )
    print(row.id)
    return 0


def _cmd_arena_queue(args: argparse.Namespace) -> int:
    """arena queue: propose blind pairwise matchups (json) for a subjective class."""
    ledger, matrix, _, _ = _build_stack()
    arena = Arena(ledger, matrix)
    matchups = arena.queue(args.task_class, n_pairs=args.n, prompt=args.prompt or "")
    print(json.dumps([dataclasses.asdict(m) for m in matchups], ensure_ascii=False, indent=2))
    return 0


def _cmd_arena_vote(args: argparse.Namespace) -> int:
    """arena vote: record one blind vote as two arena collision rows."""
    ledger, matrix, _, _ = _build_stack()
    arena = Arena(ledger, matrix)
    matchup = ArenaMatchup(
        task_class=args.task_class,
        item_id=args.item,
        model_a=args.model_a,
        model_b=args.model_b,
        prompt="",
    )
    try:
        arena.record_vote(
            matchup,
            winner=args.winner or "",
            voter=args.voter,
            difficulty=args.difficulty,
            lang=args.lang,
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(args.item)
    return 0


def _cmd_freshness(args: argparse.Namespace) -> int:
    """freshness: print prediction_drift flags (cold-start safe).

    Behavioral staleness only — identity is model_id, so a new model is simply a new cold
    cell (no version-based re-eval; see rightmodel_freshness IDENTITY PRINCIPLE).
    """
    ledger, matrix, _, _ = _build_stack()
    flags = list(prediction_drift(ledger, matrix, window=args.window, threshold=args.threshold))
    print(
        json.dumps(
            [dataclasses.asdict(f) for f in flags], ensure_ascii=False, indent=2
        )
    )
    return 0


def _cmd_delegate(args: argparse.Namespace) -> int:
    """delegate: route→run→review→record (honors kill-switch; prints DelegationResult json)."""
    result = consumer_delegate(
        args.task_class,
        args.prompt,
        _parse_taints(args.taint),
        run_id=args.run_id,
        lang=args.lang,
        origin=args.origin,
        reviewer_model=args.reviewer,
        difficulty=args.difficulty,
        live=(True if args.live else None),
    )
    payload = dataclasses.asdict(result)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.model is not None else 1


def _cmd_shadow_observe(args: argparse.Namespace) -> int:
    """shadow observe: record what the router WOULD pick vs what actually ran (no call)."""
    _, _, _, router = _build_stack()
    rec = ShadowRecorder(router=router)
    record = rec.observe(
        task_class=args.task_class,
        taint_set=_parse_taints(args.taint),
        actual_model=args.actual_model,
        actual_outcome=args.outcome,
        cost_usd=args.cost,
        lang=args.lang,
        run_id=args.run_id,
    )
    print(json.dumps(dataclasses.asdict(record), ensure_ascii=False, indent=2))
    return 0


def _cmd_shadow_report(args: argparse.Namespace) -> int:
    """shadow report: agreement rate (overall + per-class) and divergences."""
    _, _, _, router = _build_stack()
    print(json.dumps(ShadowRecorder(router=router).agreement_report(), ensure_ascii=False, indent=2))
    return 0


def _cmd_rate(args: argparse.Namespace) -> int:
    """rate: record a HUMAN quality rating (verified=human; outranks the LLM-judge)."""
    from .rightmodel_feedback import rate
    row = rate(task_class=args.task_class, model=args.model, score=args.score,
               reviewer=args.reviewer, note=args.note or "")
    print(f"{row.id} {row.model} {args.task_class} -> {row.outcome} (human, verified={row.verified})")
    return 0


def _cmd_battle_ingest(args: argparse.Namespace) -> int:
    """battle-ingest: record a /battle leaderboard as one coding duel."""
    from .rightmodel_feedback import ingest_battle
    results = json.loads(args.results)
    rows = ingest_battle(args.task_id, results, task_class=args.task_class,
                         human_judged=not args.auto)
    print(f"ingested {len(rows)} battle rows for {args.task_id} ({args.task_class})")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with the route/report/seed/record subcommands."""
    parser = argparse.ArgumentParser(
        prog="rightmodel_cli",
        description="RightModel fleet routing substrate CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_route = sub.add_parser("route", help="cascade-pick a model for a task")
    p_route.add_argument("--task-class", dest="task_class", required=True)
    p_route.add_argument("--taint", default=None, help="comma-separated taint classes")
    p_route.add_argument("--candidates", default=None, help="comma-separated model override")
    p_route.add_argument("--run-id", dest="run_id", default=None)
    p_route.add_argument("--lang", default="en")
    p_route.set_defaults(func=_cmd_route)

    p_report = sub.add_parser("report", help="matrix + breaker + anti-Goodhart summary")
    p_report.set_defaults(func=_cmd_report)

    p_seed = sub.add_parser("seed", help="run synthetic eval over models x classes")
    p_seed.add_argument("--models", required=True, help="comma-separated model ids")
    p_seed.add_argument("--classes", default=None, help="comma-separated task classes")
    p_seed.set_defaults(func=_cmd_seed)

    p_record = sub.add_parser("record", help="append a production outcome row")
    p_record.add_argument("--task-class", dest="task_class", required=True)
    p_record.add_argument("--model", required=True)
    p_record.add_argument("--outcome", required=True, help="pass|fail|partial")
    p_record.add_argument("--verified", action="store_true")
    p_record.add_argument("--cost", type=float, default=0.0)
    p_record.add_argument("--difficulty", type=float, default=0.5)
    p_record.add_argument("--lang", default="en")
    p_record.add_argument("--run-id", dest="run_id", default=None)
    p_record.add_argument("--collision-group", dest="collision_group", default=None)
    p_record.add_argument("--note", default=None)
    p_record.set_defaults(func=_cmd_record)

    p_arena = sub.add_parser("arena", help="blind pairwise HITL for subjective classes")
    arena_sub = p_arena.add_subparsers(dest="arena_command", required=True)

    p_aqueue = arena_sub.add_parser("queue", help="propose matchups (json)")
    p_aqueue.add_argument("--task-class", dest="task_class", required=True)
    p_aqueue.add_argument("--n", type=int, default=3, help="max matchups to propose")
    p_aqueue.add_argument("--prompt", default=None, help="task prompt for both models")
    p_aqueue.set_defaults(func=_cmd_arena_queue)

    p_avote = arena_sub.add_parser("vote", help="record one blind vote (two rows)")
    p_avote.add_argument("--item", required=True, help="matchup item_id (collision_group)")
    p_avote.add_argument("--winner", default=None, help="winning model id (empty = tie)")
    p_avote.add_argument("--task-class", dest="task_class", required=True)
    p_avote.add_argument("--model-a", dest="model_a", required=True)
    p_avote.add_argument("--model-b", dest="model_b", required=True)
    p_avote.add_argument("--voter", default="human")
    p_avote.add_argument("--difficulty", type=float, default=0.5)
    p_avote.add_argument("--lang", default="en")
    p_avote.set_defaults(func=_cmd_arena_vote)

    p_fresh = sub.add_parser("freshness", help="prediction_drift flags (behavioral staleness)")
    p_fresh.add_argument("--window", type=int, default=30)
    p_fresh.add_argument("--threshold", type=float, default=0.10)
    p_fresh.set_defaults(func=_cmd_freshness)

    p_delegate = sub.add_parser("delegate", help="route→run→review→record loop")
    p_delegate.add_argument("--task-class", dest="task_class", required=True)
    p_delegate.add_argument("--prompt", required=True)
    p_delegate.add_argument("--taint", default=None, help="comma-separated taint classes")
    p_delegate.add_argument("--reviewer", default=None, help="distinct peer-reviewer model")
    p_delegate.add_argument("--run-id", dest="run_id", default=None)
    p_delegate.add_argument("--lang", default="en")
    p_delegate.add_argument("--origin", default="agent")
    p_delegate.add_argument("--difficulty", type=float, default=0.5)
    p_delegate.add_argument(
        "--live", action="store_true",
        help="force live; default defers to the FOREMAN_LLM_LIVE kill-switch",
    )
    p_delegate.set_defaults(func=_cmd_delegate)

    p_shadow = sub.add_parser("shadow", help="observe-only would-route vs actual (zero spend)")
    shadow_sub = p_shadow.add_subparsers(dest="shadow_command", required=True)
    p_sobs = shadow_sub.add_parser("observe", help="record would-route vs actual (no model call)")
    p_sobs.add_argument("--task-class", dest="task_class", required=True)
    p_sobs.add_argument("--taint", default=None, help="comma-separated taint classes")
    p_sobs.add_argument("--actual-model", dest="actual_model", required=True)
    p_sobs.add_argument("--outcome", required=True, help="pass|fail|partial")
    p_sobs.add_argument("--cost", type=float, default=0.0)
    p_sobs.add_argument("--lang", default="en")
    p_sobs.add_argument("--run-id", dest="run_id", default=None)
    p_sobs.set_defaults(func=_cmd_shadow_observe)
    p_srep = shadow_sub.add_parser("report", help="agreement rate + divergences")
    p_srep.set_defaults(func=_cmd_shadow_report)

    p_rate = sub.add_parser("rate", help="record a HUMAN quality rating (HITL, outranks the judge)")
    p_rate.add_argument("--task-class", dest="task_class", required=True)
    p_rate.add_argument("--model", required=True)
    p_rate.add_argument("--score", required=True, help="pass|partial|fail or 0-100")
    p_rate.add_argument("--reviewer", default="emmanuel")
    p_rate.add_argument("--note", default=None)
    p_rate.set_defaults(func=_cmd_rate)

    p_battle = sub.add_parser("battle-ingest", help="record a /battle leaderboard as one duel")
    p_battle.add_argument("--task-id", dest="task_id", required=True)
    p_battle.add_argument("--task-class", dest="task_class", default="coding")
    p_battle.add_argument("--results", required=True,
                          help='JSON list e.g. [{"model":"claude-opus-4-8","passed":true}]')
    p_battle.add_argument("--auto", action="store_true",
                          help="auto-scored battle (verified=False) instead of human-judged")
    p_battle.set_defaults(func=_cmd_battle_ingest)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse argv and dispatch to the subcommand handler; returns the exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
