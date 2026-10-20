#!/usr/bin/env python3
"""
auto_eval.py — Main entry point for the Manas auto-evaluation pipeline.

Fixed-set mode (--fixed-set PATH): deterministic, isolated evaluation for
comparing model versions on the same scenarios.  Results go to
eval_runs/<run-tag>/ and never touch formatted/ or the approved index.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.panel import Panel

console = Console()

# ---------------------------------------------------------------------------
# Load adapted_data files
# ---------------------------------------------------------------------------
ADAPTED_DATA_DIR = Path(__file__).parent / "adapted_data"

MANAS_SYSTEM_PROMPT = (
    ADAPTED_DATA_DIR / "manas_system_prompt_english.txt"
).read_text(encoding="utf-8").strip()

MANAS_GUIDELINES = (
    ADAPTED_DATA_DIR / "MANAS_GUIDELINES_english.md"
).read_text(encoding="utf-8").strip()

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from prompts.seeds import SEEDS
from prompts.personas import PERSONAS
from prompts.meta import CATEGORIES, CATEGORY_PERSONAS

from eval.conversation_runner import Manas, UserSimulator, load_manas_model
from eval.scorer import Scorer, EvaluationResult, DIMENSION_NAMES
from eval.router import save_and_merge, ensure_dirs
from eval.report import (
    SummaryTracker,
    print_conversation_result,
    print_batch_header,
    print_batch_footer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONCURRENCY_SENSITIVE = int(os.getenv("EVAL_CONCURRENCY_SENSITIVE", "1"))
CONCURRENCY_DEFAULT   = int(os.getenv("EVAL_CONCURRENCY_DEFAULT",   "1"))

# Fixed-set mode: serialize Manas generation and seed deterministically.
_MANAS_LOCK = threading.Lock()

# Scorer model name (hardcoded in eval/scorer.py — keep in sync).
SCORER_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

EVAL_RUNS_DIR = Path(__file__).parent.resolve() / "eval_runs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_persona_by_id(persona_id: str) -> dict:
    for p in PERSONAS:
        if p["id"] == persona_id:
            return p
    raise KeyError(f"Unknown persona ID: {persona_id}")


def get_all_categories() -> list[str]:
    return list(SEEDS.keys())


def generate_id(
    category: str,
    persona_id: str,
    index: int,
    run_ts: int,
) -> str:
    return f"{category}__{persona_id}__{index:02d}__{run_ts}"


def _scenario_seed(scenario_id: str, turn_index: int) -> int:
    """Deterministic seed derived from (scenario_id, turn_index)."""
    raw = f"{scenario_id}:{turn_index}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest(), 16) % (2 ** 32)


def _system_prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, data: object) -> None:
    """Write JSON atomically so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _verdict_from_total(total: float, possible: int = 10) -> str:
    """Verdict from an averaged total — never routed through the int field."""
    fraction = total / possible
    if fraction >= 0.75:
        return "auto_approve"
    if fraction >= 0.60:
        return "manual_review"
    return "auto_reject"


def _adapter_base_model(repo: str) -> Optional[str]:
    """Read base_model_name_or_path from the adapter's config."""
    try:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(
            repo_id=repo, filename="adapter_config.json",
            token=os.getenv("HF_TOKEN"),
        )
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("base_model_name_or_path")
    except Exception as e:
        print(f"[WARN] Could not read adapter config for {repo}: {e}")
        return None


def _load_fixed_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("fixed-set file must be a JSON list")
    for s in data:
        for key in ("scenario_id", "category", "persona_id", "situation", "num_turns"):
            if key not in s:
                raise ValueError(f"fixed-set entry missing key {key!r}: {s}")
    return data


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-evaluate Manas conversations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=get_all_categories(),
        default=None,
    )
    parser.add_argument("--dry-run",        action="store_true")
    parser.add_argument("--skip-user-sim",  action="store_true")
    parser.add_argument(
        "--conversations-per-category", type=int, default=None
    )
    parser.add_argument("--resume",         action="store_true")
    parser.add_argument(
        "--resume-skip-existing",
        action="store_true",
        default=False,
    )
    parser.add_argument("--export-approved", metavar="PATH")
    parser.add_argument("--report-only",    action="store_true")
    parser.add_argument("--skip-manas-load", action="store_true")
    parser.add_argument(
        "--model-max-tokens",
        type=int,
        default=256,
        help="max_new_tokens for Manas inference (default: 256)",
    )
    parser.add_argument(
        "--model-temperature",
        type=float,
        default=0.85,
    )
    # ── Fixed-set mode ────────────────────────────────────────────────────
    parser.add_argument(
        "--fixed-set", metavar="PATH", default=None,
        help="Run exactly the scenarios in this JSON file (deterministic).",
    )
    parser.add_argument(
        "--run-tag", default=None,
        help="Label for the run (e.g. v4, v6, v7). Defaults to fixed_<ts>.",
    )
    parser.add_argument(
        "--model-repo", default=None,
        help="Override HF_MODEL_REPO for this run only. Use 'none' for the base model.",
    )
    parser.add_argument(
        "--eval-temperature", type=float, default=None,
        help="Manas temperature in fixed-set mode (default: 0.6).",
    )
    parser.add_argument(
        "--score-repeats", type=int, default=3,
        help="Score each conversation N times and average (default: 3).",
    )
    parser.add_argument(
        "--novel-seeds", metavar="PATH", default=None,
        help="JSON {category: [situations]} of fresh openings (builds fixed_eval_v2).",
    )
    parser.add_argument(
        "--system-prompt-file", metavar="PATH", default=None,
        help="Replace MANAS_SYSTEM_PROMPT for this run only.",
    )
    parser.add_argument(
        "--rescore-only", metavar="TAG", default=None,
        help="Re-score saved transcripts from run TAG; never regenerate.",
    )
    return parser.parse_args()


# ===========================================================================
# Fixed-set mode helpers
# ===========================================================================

def _fixed_run_dir(tag: str) -> Path:
    return EVAL_RUNS_DIR / tag


def _fixed_results_path(tag: str) -> Path:
    return _fixed_run_dir(tag) / "results.json"


def _fixed_summary_path(tag: str) -> Path:
    return _fixed_run_dir(tag) / "summary.json"


def _load_fixed_results(tag: str) -> dict:
    p = _fixed_results_path(tag)
    if not p.exists():
        return {"run_tag": tag, "config": {}, "scenarios": {}}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _fixed_config_dict(args, system_prompt_text: str) -> dict:
    return {
        "fixed_set": args.fixed_set,
        "run_tag": args.run_tag,
        "model_repo": os.environ.get("HF_MODEL_REPO"),
        "base_model": os.getenv("HF_BASE_MODEL"),
        "eval_temperature": args.eval_temperature,
        "score_repeats": args.score_repeats,
        "system_prompt_sha256": _system_prompt_hash(system_prompt_text),
        "system_prompt_tokens": len(system_prompt_text.split()),
        "scorer_model": SCORER_MODEL,
        "user_sim_model": os.getenv("OLLAMA_USER_MODEL"),
        "user_sim_temperature": 0.5,
        "model_max_tokens": args.model_max_tokens,
        "concurrency": 1,
    }


# ===========================================================================
# Core evaluation logic
# ===========================================================================

def build_messages(
    history: list[dict],
    system_prompt: Optional[str] = None,
) -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    return messages


async def run_single_conversation(
    *,
    manas: Manas,
    user_sim: UserSimulator,
    category: str,
    persona_id: str,
    situation: str,
    num_turns: int,
    scenario_id: str,
    system_prompt: str,
    dry_run: bool = False,
    skip_user_sim: bool = False,
    fixed_set: bool = False,
) -> tuple[list[dict], Optional[str]]:
    """Run one multi-turn conversation.

    Returns (messages, ended_early_reason_or_None).
    """
    persona   = get_persona_by_id(persona_id)
    messages: list[dict] = []
    ended_early: Optional[str] = None

    for turn_i in range(num_turns):

        # ── User turn ────────────────────────────────────────────────────
        if turn_i == 0:
            user_message = situation
        else:
            if skip_user_sim:
                ended_early = "skip_user_sim"
                break
            try:
                seed = _scenario_seed(scenario_id, turn_i)
                user_message = await user_sim.generate_next_turn(
                    conversation=messages,
                    persona=persona,
                    situation=situation,
                    seed=seed,
                    temperature=0.5,
                )
                preview = user_message[:80] + (
                    "..." if len(user_message) > 80 else ""
                )
                console.print(f"    [dim]User: {preview}[/dim]")
            except Exception as e:
                console.print(f"    [red]User simulation error: {e}[/red]")
                ended_early = f"user_sim_error: {e}"
                break

        messages.append({"role": "user", "content": user_message})

        # ── Detect UserSimulator loop ─────────────────────────────────────
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        if len(user_msgs) >= 3:
            recent = user_msgs[-3:]
            if len(set(s.strip() for s in recent)) == 1:
                console.print(
                    "    [yellow]UserSimulator triple loop — ending[/yellow]"
                )
                ended_early = "user_sim_triple_loop"
                break
        if len(user_msgs) >= 2:
            if user_msgs[-1].strip() == user_msgs[-2].strip():
                console.print(
                    "    [yellow]UserSimulator consecutive loop — ending[/yellow]"
                )
                ended_early = "user_sim_consecutive_loop"
                break

        # ── Manas turn ───────────────────────────────────────────────────
        try:
            full_messages = build_messages(
                history=messages,
                system_prompt=system_prompt,
            )
            seed = _scenario_seed(scenario_id, turn_i)
            assistant_reply = await asyncio.to_thread(
                _run_manas_locked,
                manas,
                user_id=persona_id,
                messages=full_messages,
                seed=seed,
            )
            messages.append({"role": "assistant", "content": assistant_reply})
            preview = assistant_reply[:80] + (
                "..." if len(assistant_reply) > 80 else ""
            )
            console.print(f"    [dim]Manas: {preview}[/dim]")

        except Exception as e:
            console.print(f"    [red]Manas inference error: {e}[/red]")
            ended_early = f"manas_error: {e}"
            break

        await asyncio.sleep(0.01)

    return messages, ended_early


def _run_manas_locked(manas, *, user_id, messages, seed):
    """Serialize Manas generation and seed deterministically."""
    with _MANAS_LOCK:
        torch.manual_seed(seed)
        return manas.run(user_id=user_id, messages=messages)


async def _score_conversation_fixed(
    scorer: Scorer,
    messages: list[dict],
    *,
    category: str,
    persona_id: str,
    situation: str,
    conversation_id: str,
    repeats: int,
) -> dict:
    """Score a conversation `repeats` times; average dimension values.

    Returns a dict with averaged scores, raw repeats, and a verdict computed
    from the averaged total.  Never pushes the average back through the int
    field of EvaluationResult.
    """
    raw_totals: list[int] = []
    raw_dims: list[dict] = []
    failures = 0

    for _ in range(repeats):
        try:
            result = await scorer.evaluate(
                conversation=messages,
                category=category,
                persona_id=persona_id,
                situation=situation,
                conversation_id=conversation_id,
            )
        except Exception as e:
            console.print(f"    [red]Scoring repeat failed: {e}[/red]")
            failures += 1
            continue

        raw_totals.append(result.total_score)
        raw_dims.append(
            {k: v.value for k, v in result.dimensions.items()}
        )

    if not raw_totals:
        return {
            "scoring_failed": True,
            "verdict": "scoring_failed",
            "total": None,
            "fraction": None,
            "dimensions": {d: None for d in DIMENSION_NAMES},
            "raw_repeats": [],
            "repeats_attempted": repeats,
            "repeats_succeeded": 0,
        }

    averaged: dict[str, float] = {}
    for dim in DIMENSION_NAMES:
        vals = [r[dim] for r in raw_dims if dim in r]
        averaged[dim] = round(sum(vals) / len(vals), 4) if vals else 0.0

    total = round(sum(averaged.values()), 4)
    fraction = total / 10.0
    verdict = _verdict_from_total(total)

    return {
        "scoring_failed": False,
        "verdict": verdict,
        "total": total,
        "fraction": round(fraction, 4),
        "dimensions": averaged,
        "raw_repeats": raw_dims,
        "raw_totals": raw_totals,
        "repeats_attempted": repeats,
        "repeats_succeeded": len(raw_totals),
        "repeats_failed": failures,
    }


# ===========================================================================
# Fixed-set run
# ===========================================================================

async def run_fixed_set(
    *,
    args,
    scenarios: list[dict],
    system_prompt: str,
    run_tag: str,
    resume: bool,
) -> dict:
    """Run a fixed-set evaluation.  Writes results.json incrementally."""
    run_dir = _fixed_run_dir(run_tag)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = _fixed_config_dict(args, system_prompt)

    # ── Resume: load existing results, skip finished scenario_ids ────────
    state = _load_fixed_results(run_tag) if resume else {
        "run_tag": run_tag, "config": config, "scenarios": {}
    }
    if resume and state.get("config") != config:
        print(
            "[ERR] --resume refused: stored config differs from current flags.\n"
            f"  stored: {json.dumps(state.get('config'), indent=2)}\n"
            f"  current: {json.dumps(config, indent=2)}"
        )
        sys.exit(1)
    state["config"] = config

    nvidia_key = os.getenv("NVIDIA_API_KEY")

    manas = None
    if not args.skip_manas_load:
        manas = load_manas_model(
            max_new_tokens=args.model_max_tokens,
            temperature=args.eval_temperature,
        )

    # Fixed-set mode: Ollama only, fail loudly.
    user_sim = UserSimulator(
        api_key=nvidia_key,
        ollama_only=True,
    )

    scorer = Scorer(api_key=nvidia_key)

    for scenario in scenarios:
        sid = scenario["scenario_id"]
        if sid in state["scenarios"]:
            continue  # already finished

        persona_id = scenario["persona_id"]
        category = scenario["category"]
        situation = scenario["situation"]
        num_turns = scenario["num_turns"]

        print_batch_header(category, 1)

        try:
            messages, ended_early = await run_single_conversation(
                manas=manas,
                user_sim=user_sim,
                category=category,
                persona_id=persona_id,
                situation=situation,
                num_turns=num_turns,
                scenario_id=sid,
                system_prompt=system_prompt,
                dry_run=args.dry_run,
                skip_user_sim=args.skip_user_sim,
                fixed_set=True,
            )
        except Exception as e:
            console.print(f"    [red]Scenario {sid} errored: {e}[/red]")
            state["scenarios"][sid] = {
                "scenario_id": sid,
                "category": category,
                "persona_id": persona_id,
                "situation": situation,
                "num_turns": num_turns,
                "errored": True,
                "error": str(e),
                "ended_early": True,
                "ended_early_reason": f"scenario_error: {e}",
                "messages": [],
                "scoring_failed": True,
                "verdict": "errored",
                "total": None,
                "fraction": None,
                "dimensions": {},
            }
            _atomic_write_json(_fixed_results_path(run_tag), state)
            continue

        if args.dry_run:
            state["scenarios"][sid] = {
                "scenario_id": sid,
                "category": category,
                "persona_id": persona_id,
                "situation": situation,
                "num_turns": num_turns,
                "dry_run": True,
                "messages": [],
            }
            _atomic_write_json(_fixed_results_path(run_tag), state)
            continue

        # ── Score ─────────────────────────────────────────────────────────
        scored = await _score_conversation_fixed(
            scorer,
            messages,
            category=category,
            persona_id=persona_id,
            situation=situation,
            conversation_id=sid,
            repeats=args.score_repeats,
        )

        record = {
            "scenario_id": sid,
            "category": category,
            "persona_id": persona_id,
            "situation": situation,
            "num_turns": num_turns,
            "ended_early": bool(ended_early),
            "ended_early_reason": ended_early,
            "messages": messages,
            **scored,
        }
        state["scenarios"][sid] = record
        _atomic_write_json(_fixed_results_path(run_tag), state)

        print_conversation_result(
            _score_view(record),
            {"action": "written", "path": str(_fixed_results_path(run_tag))},
            index=len(state["scenarios"]),
            total=len(scenarios),
        )

    _write_fixed_summary(run_tag, state)
    return state


def _score_view(record: dict):
    """Build a lightweight EvaluationResult-like view for console printing."""
    return type("View", (), {
        "conversation_id": record.get("scenario_id"),
        "category": record.get("category"),
        "total_score": record.get("total"),
        "total_possible": 10,
        "fraction": record.get("fraction") or 0.0,
        "verdict": record.get("verdict", "unknown"),
        "dimensions": {},
        "rejected": False,
        "strong_flag": False,
    })()


def _write_fixed_summary(run_tag: str, state: dict) -> None:
    scenarios = list(state["scenarios"].values())
    scored = [s for s in scenarios if not s.get("scoring_failed") and not s.get("errored")]

    total_sum = sum(s["total"] for s in scored if s.get("total") is not None)
    n = max(len(scored), 1)

    dim_means: dict[str, Optional[float]] = {d: None for d in DIMENSION_NAMES}
    for dim in DIMENSION_NAMES:
        vals = [s["dimensions"][dim] for s in scored
                if s.get("dimensions") and s["dimensions"].get(dim) is not None]
        if vals:
            dim_means[dim] = round(sum(vals) / len(vals), 4)

    approved = sum(1 for s in scored if s.get("verdict") == "auto_approve")
    review = sum(1 for s in scored if s.get("verdict") == "manual_review")
    rejected = sum(1 for s in scored if s.get("verdict") == "auto_reject")
    errored = sum(1 for s in scenarios if s.get("errored"))
    failed = sum(1 for s in scenarios if s.get("scoring_failed"))

    summary = {
        "run_tag": run_tag,
        "config": state.get("config", {}),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_scenarios": len(scenarios),
        "scored": len(scored),
        "errored": errored,
        "scoring_failed": failed,
        "approval_rate": round(approved / n, 4) if n else None,
        "approval_counts": {
            "auto_approve": approved,
            "manual_review": review,
            "auto_reject": rejected,
        },
        "mean_total_score": round(total_sum / n, 4) if n else None,
        "dimension_means": dim_means,
        "by_category": _category_means(state),
    }
    _atomic_write_json(_fixed_summary_path(run_tag), summary)
    console.print(f"[green][OK] Summary → {_fixed_summary_path(run_tag)}[/green]")


def _category_means(state: dict) -> dict:
    out: dict[str, dict] = {}
    for sid, s in state["scenarios"].items():
        cat = s.get("category")
        if not cat or s.get("errored") or s.get("scoring_failed"):
            continue
        out.setdefault(cat, {"n": 0, "total_sum": 0.0, "dims": {d: [] for d in DIMENSION_NAMES}})
        out[cat]["n"] += 1
        if s.get("total") is not None:
            out[cat]["total_sum"] += s["total"]
        dims = s.get("dimensions") or {}
        for d in DIMENSION_NAMES:
            if not dims.get(d):
                continue
            out[cat]["dims"][d].append(dims[d])
    result = {}
    for cat, data in out.items():
        result[cat] = {
            "n": data["n"],
            "mean_total": round(data["total_sum"] / data["n"], 4) if data["n"] else None,
            "mean_dimensions": {
                d: round(sum(v) / len(v), 4) if v else None
                for d, v in data["dims"].items()
            },
        }
    return result


# ===========================================================================
# Rescore-only mode
# ===========================================================================

async def rescore_only(
    *,
    args,
    source_tag: str,
    system_prompt: str,
) -> dict:
    """Re-score saved transcripts from run TAG; never regenerate."""
    src = _load_fixed_results(source_tag)
    scenarios = src.get("scenarios", {})
    if not scenarios:
        print(f"[ERR] No saved scenarios found for run {source_tag!r}")
        sys.exit(1)

    ts = int(time.time())
    run_tag = f"{source_tag}__rescored_{ts}"
    run_dir = _fixed_run_dir(run_tag)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = _fixed_config_dict(args, system_prompt)
    config["source_run"] = source_tag
    config["rescore"] = True

    state = {"run_tag": run_tag, "config": config, "scenarios": {}}

    nvidia_key = os.getenv("NVIDIA_API_KEY")
    scorer = Scorer(api_key=nvidia_key)

    for sid, record in scenarios.items():
        messages = record.get("messages", [])
        scored = await _score_conversation_fixed(
            scorer,
            messages,
            category=record.get("category", ""),
            persona_id=record.get("persona_id", ""),
            situation=record.get("situation", ""),
            conversation_id=sid,
            repeats=args.score_repeats,
        )
        new_record = {
            **{k: v for k, v in record.items() if k != "messages"},
            "messages": messages,
            "source_run": source_tag,
            **scored,
        }
        state["scenarios"][sid] = new_record
        _atomic_write_json(_fixed_results_path(run_tag), state)

    _write_fixed_summary(run_tag, state)
    return state


# ===========================================================================
# Main pipeline (non-fixed-set mode — unchanged behavior)
# ===========================================================================

async def run_evaluation(
    categories: Optional[list[str]] = None,
    conversations_per_category: Optional[int] = None,
    dry_run: bool = False,
    skip_user_sim: bool = False,
    skip_manas_load: bool = False,
    resume_skip_existing: bool = False,
    model_max_tokens: int = 256,
    model_temperature: float = 0.85,
    **kwargs,
) -> SummaryTracker:
    """Run the full auto-evaluation pipeline (unchanged when no --fixed-set)."""
    ensure_dirs()
    target_categories = categories or get_all_categories()

    console.print(Panel("[bold]Loading models...[/bold]", expand=False))

    nvidia_key = os.getenv("NVIDIA_API_KEY")

    manas = None
    if not skip_manas_load:
        manas = load_manas_model(
            max_new_tokens=model_max_tokens,
            temperature=model_temperature,
        )

    user_sim = UserSimulator(api_key=nvidia_key)
    scorer = Scorer(api_key=nvidia_key)

    skip_ids: set[str] = set()
    if resume_skip_existing:
        try:
            from eval.router import list_approved_ids
            skip_ids = set(list_approved_ids())
            if skip_ids:
                console.print(
                    f"[dim]Resume: skipping {len(skip_ids)} approved IDs[/dim]"
                )
        except Exception as e:
            console.print(f"[yellow]Could not load skip IDs: {e}[/yellow]")

    total_conversations = sum(
        conversations_per_category or CATEGORIES.get(cat, {}).get("count", 10)
        for cat in target_categories
    )
    tracker         = SummaryTracker()
    processed_total = 0

    for cat in target_categories:
        spec  = CATEGORIES.get(cat, {})
        count = conversations_per_category or spec.get("count", 10)
        print_batch_header(cat, count)

        batch_data = await run_batch(
            manas=manas,
            user_sim=user_sim,
            category=cat,
            spec={**spec, "count": count},
            skip_ids=skip_ids,
            dry_run=dry_run,
            skip_user_sim=skip_user_sim,
        )

        if dry_run:
            console.print(
                f"  [green]{len(batch_data)} conversations generated "
                f"(dry run)[/green]"
            )
            continue

        if not batch_data:
            console.print(
                f"  [yellow]No conversations generated for {cat}[/yellow]"
            )
            continue

        # ── Score ─────────────────────────────────────────────────────────
        scoring_concurrency = int(os.getenv("EVAL_CONCURRENCY", "1"))
        score_sem = asyncio.Semaphore(scoring_concurrency)
        seeds     = SEEDS.get(cat, [""])

        async def _score_one(
            messages: list[dict],
            conv_id: str,
            p_id: str,
            conv_idx: int,
        ) -> Optional[EvaluationResult]:
            async with score_sem:
                await asyncio.sleep(0.2)
                try:
                    return await scorer.evaluate(
                        conversation=messages,
                        category=cat,
                        persona_id=p_id,
                        situation=seeds[conv_idx % len(seeds)],
                        conversation_id=conv_id,
                    )
                except Exception as e:
                    console.print(
                        f"  [red]Scoring error ({conv_id}): {e}[/red]"
                    )
                    return None

        scoring_tasks = [
            _score_one(msgs, c_id, p_id, i)
            for i, (msgs, c_id, p_id) in enumerate(batch_data)
        ]
        scored_results = await asyncio.gather(*scoring_tasks)

        # ── Route ─────────────────────────────────────────────────────────
        approved = flagged = rejected = 0

        for (messages, _, _), evaluation in zip(batch_data, scored_results):
            processed_total += 1

            if not evaluation:
                console.print(
                    "  [yellow]Skipping — scoring failed[/yellow]"
                )
                continue

            decision = save_and_merge(evaluation, messages)
            tracker.record(evaluation, decision)

            print_conversation_result(
                evaluation,
                decision,
                index=processed_total,
                total=total_conservations,
            )

            if evaluation.verdict == "auto_approve":
                approved += 1
            elif evaluation.verdict == "manual_review":
                flagged += 1
            else:
                rejected += 1

        print_batch_footer(cat, approved, flagged, rejected)

    tracker.print_summary()
    tracker.export_to_json()
    return tracker


async def run_batch(
    *,
    manas: Manas,
    user_sim: UserSimulator,
    category: str,
    spec: dict,
    skip_ids: set[str],
    dry_run: bool = False,
    skip_user_sim: bool = False,
) -> list[tuple[list[dict], str, str]]:
    """Run a batch of conversations for one category."""
    concurrency = (
        CONCURRENCY_SENSITIVE if spec.get("sensitive") else CONCURRENCY_DEFAULT
    )
    sem = asyncio.Semaphore(concurrency)

    category_persona_ids = CATEGORY_PERSONAS.get(category)
    if not category_persona_ids:
        category_persona_ids = [
            p["id"] for p in PERSONAS if p.get("language") == "english"
        ]

    count     = spec.get("count", 10)
    seeds     = SEEDS.get(category, ["Tell me about your situation."])
    min_turns = spec.get("turns_min", 4)
    max_turns = spec.get("turns_max", 6)

    shuffled_personas = category_persona_ids.copy()
    shuffled_seeds    = seeds.copy()
    random.shuffle(shuffled_personas)
    random.shuffle(shuffled_seeds)

    run_ts = int(time.time()) % 100000

    console.print(
        f"  [dim]Run ID: {run_ts} | "
        f"Personas: {len(shuffled_personas)} | "
        f"Seeds: {len(shuffled_seeds)} | "
        f"Concurrency: {concurrency}[/dim]"
    )

    tasks = []

    for idx in range(count):
        persona_id = shuffled_personas[idx % len(shuffled_personas)]
        situation  = shuffled_seeds[idx % len(shuffled_seeds)]
        conv_id    = generate_id(category, persona_id, idx, run_ts)

        if conv_id in skip_ids:
            console.print(
                f"  [dim]Skipping {conv_id} (already approved)[/dim]"
            )
            continue

        async def _run_one(p_id=persona_id, sit=situation, c_id=conv_id):
            async with sem:
                msgs = await run_single_conversation(
                    manas=manas,
                    user_sim=user_sim,
                    category=category,
                    persona_id=p_id,
                    situation=sit,
                    min_turns=min_turns,
                    max_turns=max_turns,
                    conversation_id=c_id,
                    system_prompt=MANAS_SYSTEM_PROMPT,
                    dry_run=dry_run,
                    skip_user_sim=skip_user_sim,
                )
                return (msgs, c_id, p_id)

        tasks.append(_run_one())

    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid = []
    for r in results:
        if isinstance(r, Exception):
            console.print(f"  [red]Task failed: {r}[/red]")
            continue
        if r and r[0] is not None:
            valid.append(r)

    return valid


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    args = parse_args()

    # ── --rescore-only ────────────────────────────────────────────────────
    if args.rescore_only:
        system_prompt = MANAS_SYSTEM_PROMPT
        if args.system_prompt_file:
            system_prompt = Path(args.system_prompt_file).read_text(
                encoding="utf-8"
            ).strip()
        asyncio.run(rescore_only(
            args=args,
            source_tag=args.rescore_only,
            system_prompt=system_prompt,
        ))
        return

    # ── Fixed-set mode ────────────────────────────────────────────────────
    if args.fixed_set:
        if args.eval_temperature is None:
            args.eval_temperature = 0.6
        if args.run_tag is None:
            args.run_tag = f"fixed_{int(time.time())}"

        system_prompt = MANAS_SYSTEM_PROMPT
        if args.system_prompt_file:
            system_prompt = Path(args.system_prompt_file).read_text(
                encoding="utf-8"
            ).strip()

        # --model-repo override (applies before Manas loads).
        if args.model_repo is not None:
            if args.model_repo.lower() == "none":
                os.environ["HF_MODEL_REPO"] = ""
            else:
                os.environ["HF_MODEL_REPO"] = args.model_repo

        # Adapter/base-model guard (skipped for --model-repo none).
        if args.model_repo and args.model_repo.lower() != "none":
            base = os.getenv("HF_BASE_MODEL")
            adapter_base = _adapter_base_model(args.model_repo)
            if adapter_base and base and adapter_base != base:
                print(
                    "[ERR] Adapter base model mismatch:\\n"
                    f"  adapter base_model_name_or_path: {adapter_base}\\n"
                    f"  HF_BASE_MODEL:                  {base}\\n"
                    "Refusing to run — base/adapter mismatch."
                )
                sys.exit(1)

        scenarios = _load_fixed_set(args.fixed_set)

        # --conversations-per-category N → first N scenarios per category.
        if args.conversations_per_category:
            per_cat = args.conversations_per_category
            scenarios = [
                s for i, s in enumerate(scenarios)
                if sum(1 for t in scenarios[:i + 1] if t["category"] == s["category"]) <= per_cat
            ]

        asyncio.run(run_fixed_set(
            args=args,
            scenarios=scenarios,
            system_prompt=system_prompt,
            run_tag=args.run_tag,
            resume=args.resume,
        ))
        return

    # ── Existing behavior (no --fixed-set) ────────────────────────────────
    if args.export_approved:
        from eval.router import merge_approved
        approved = merge_approved()
        with open(args.export_approved, "w", encoding="utf-8") as f:
            json.dump(approved, f, ensure_ascii=False, indent=2)
        console.print(
            f"[green]Exported {len(approved)} conversations "
            f"→ {args.export_approved}[/green]"
        )
        return

    if args.report_only:
        SummaryTracker().print_summary()
        return

    if args.resume:
        args.resume_skip_existing = True

    tracker = asyncio.run(run_evaluation(**vars(args)))

    console.rule("[bold]Done[/bold]")
    console.print(f"  Total evaluated : {tracker.total_evaluated}")
    console.print(
        f"  Approved        : [green]{tracker.total_auto_approve}[/green]"
    )
    console.print(
        f"  Flagged         : [yellow]{tracker.total_manual_review}[/yellow]"
    )
    console.print(
        f"  Rejected        : [red]{tracker.total_auto_reject}[/red]"
    )


if __name__ == "__main__":
    main()