#!/usr/bin/env python3
"""
compare_runs.py — Compare fixed-set evaluation runs.

Loads eval_runs/*/results.json (or tags passed on the command line) and prints:

  1. One table per run: approval rate, mean total score (0-10) with a 95%
     bootstrap confidence interval (1000 resamples over scenarios), and
     per-dimension means with CIs.
  2. A paired comparison between two chosen runs (--pair A B): mean score
     difference per scenario, bootstrap CI on that difference, and
     win/tie/loss counts.
  3. A --baseline TAG comparison: every other run against one reference.
  4. Per-category means (n=5 per category — directional only).

Usage:
    python compare_runs.py
    python compare_runs.py --pair v4 v6
    python compare_runs.py --baseline v4
    python compare_runs.py v4 v6 v7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

from eval.scorer import DIMENSION_NAMES

BASE_DIR = Path(__file__).parent.resolve()
EVAL_RUNS_DIR = BASE_DIR / "eval_runs"

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_RNG_SEED = 7


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(tag: str) -> Optional[dict]:
    """Load results.json for a run tag."""
    path = EVAL_RUNS_DIR / tag / "results.json"
    if not path.exists():
        print(f"[ERR] No results.json for run {tag!r} at {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_runs() -> dict[str, dict]:
    """Load every run found under eval_runs/."""
    runs: dict[str, dict] = {}
    if not EVAL_RUNS_DIR.exists():
        return runs
    for entry in sorted(EVAL_RUNS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        results = entry / "results.json"
        if results.exists():
            with open(results, "r", encoding="utf-8") as f:
                runs[entry.name] = json.load(f)
    return runs


# ---------------------------------------------------------------------------
# Scored scenarios
# ---------------------------------------------------------------------------

def scored_scenarios(run: dict) -> list[dict]:
    """Scenarios with a real score — excludes errored and scoring_failed."""
    out = []
    for sid, s in run.get("scenarios", {}).items():
        if s.get("errored") or s.get("scoring_failed"):
            continue
        if s.get("total") is None:
            continue
        out.append(s)
    return out


def common_scored(a: dict, b: dict) -> tuple[list[dict], list[dict]]:
    """Scenarios present and scored in both runs, keyed by scenario_id."""
    sa = {s["scenario_id"]: s for s in scored_scenarios(a)}
    sb = {s["scenario_id"]: s for s in scored_scenarios(b)}
    common = sorted(set(sa) & set(sb))
    return [sa[k] for k in common], [sb[k] for k in common]


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _bootstrap_mean(values: list[float], n_resamples: int = BOOTSTRAP_RESAMPLES,
                    rng: Optional[random.Random] = None) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via bootstrap over the values."""
    if not values:
        return 0.0, 0.0, 0.0
    if rng is None:
        rng = random.Random(BOOTSTRAP_RNG_SEED)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples)]
    return sum(values) / n, lo, hi


def _bootstrap_diff(a: list[float], b: list[float],
                    n_resamples: int = BOOTSTRAP_RESAMPLES,
                    rng: Optional[random.Random] = None) -> tuple[float, float, float]:
    """Bootstrap CI on the paired difference (a - b), resampling scenarios."""
    if not a or len(a) != len(b):
        return 0.0, 0.0, 0.0
    if rng is None:
        rng = random.Random(BOOTSTRAP_RNG_SEED)
    n = len(a)
    diffs = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        d = sum(a[i] - b[i] for i in idxs) / n
        diffs.append(d)
    diffs.sort()
    lo = diffs[int(0.025 * n_resamples)]
    hi = diffs[int(0.975 * n_resamples)]
    return sum(a[i] - b[i] for i in range(n)) / n, lo, hi


# ---------------------------------------------------------------------------
# Run summaries
# ---------------------------------------------------------------------------

def run_summary(run: dict) -> dict:
    """Compute the per-run summary stats."""
    sc = scored_scenarios(run)
    totals = [s["total"] for s in sc]
    mean_total, lo, hi = _bootstrap_mean(totals)

    dim_vals: dict[str, list[float]] = {d: [] for d in DIMENSION_NAMES}
    for s in sc:
        dims = s.get("dimensions") or {}
        for d in DIMENSION_NAMES:
            if dims.get(d) is not None:
                dim_vals[d].append(dims[d])

    dim_means: dict[str, tuple[float, float, float]] = {}
    for d in DIMENSION_NAMES:
        dim_means[d] = _bootstrap_mean(dim_vals[d])

    n = len(sc)
    approved = sum(1 for s in sc if s.get("verdict") == "auto_approve")
    review = sum(1 for s in sc if s.get("verdict") == "manual_review")
    rejected = sum(1 for s in sc if s.get("verdict") == "auto_reject")
    errored = sum(1 for s in run.get("scenarios", {}).values() if s.get("errored"))
    failed = sum(1 for s in run.get("scenarios", {}).values() if s.get("scoring_failed"))

    cfg = run.get("config", {})
    return {
        "tag": run.get("run_tag", "?"),
        "n_scored": n,
        "n_errored": errored,
        "n_scoring_failed": failed,
        "approval_rate": approved / n if n else 0.0,
        "approval_counts": (approved, review, rejected),
        "mean_total": mean_total,
        "total_ci": (lo, hi),
        "dim_means": dim_means,
        "scorer_model": cfg.get("scorer_model"),
        "model_repo": cfg.get("model_repo"),
        "eval_temperature": cfg.get("eval_temperature"),
        "system_prompt_sha256": cfg.get("system_prompt_sha256"),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_run_table(sums: dict[str, dict]) -> None:
    """One table per run."""
    print("=" * 100)
    print(f"{'RUN':<24} {'N':>4} {'Appr%':>7} {'Mean':>6} {'CI 95%':>16}")
    print("-" * 100)
    for tag, s in sums.items():
        lo, hi = s["total_ci"]
        print(
            f"{tag:<24} {s['n_scored']:>4} {s['approval_rate']:>6.1%} "
            f"{s['mean_total']:>6.2f} [{lo:.2f}, {hi:.2f}]"
        )
    print()

    for tag, s in sums.items():
        print(f"- {tag} - per-dimension means (0-2) with 95% CI -")
        print(f"{'Dimension':<24} {'Mean':>6} {'CI 95%':>16}")
        print("-" * 50)
        for d in DIMENSION_NAMES:
            m, lo, hi = s["dim_means"][d]
            print(f"{d:<24} {m:>6.3f} [{lo:.3f}, {hi:.3f}]")
        print()


def print_paired(a_tag: str, b_tag: str, a: dict, b: dict) -> None:
    """Paired comparison between two runs."""
    sa, sb = common_scored(a, b)
    if not sa:
        print(f"[ERR] No common scored scenarios between {a_tag} and {b_tag}.")
        return

    ta = [s["total"] for s in sa]
    tb = [s["total"] for s in sb]
    mean_diff, lo, hi = _bootstrap_diff(ta, tb)

    wins = sum(1 for x, y in zip(ta, tb) if x > y)
    ties = sum(1 for x, y in zip(ta, tb) if x == y)
    losses = len(ta) - wins - ties

    print("=" * 100)
    print(f"PAIRED: {a_tag} vs {b_tag}  (n = {len(ta)} common scenarios)")
    print("-" * 100)
    print(f"  Mean total {a_tag}:     {sum(ta)/len(ta):.3f}")
    print(f"  Mean total {b_tag}:     {sum(tb)/len(tb):.3f}")
    print(f"  Mean difference:        {mean_diff:+.3f}   (95% CI [{lo:+.3f}, {hi:+.3f}])")
    print(f"  Wins / Ties / Losses:   {wins} / {ties} / {losses}")
    print()

    # Per-dimension paired analysis
    print(f"  Per-dimension paired differences (a - b) with 95% CI:")
    print(f"  {'Dimension':<24} {'MeanDiff':>10} {'CI 95%':>18} {'W/T/L':>10}")
    print("  " + "-" * 70)
    for d in DIMENSION_NAMES:
        vals_a = [s["dimensions"].get(d, 0) for s in sa]
        vals_b = [s["dimensions"].get(d, 0) for s in sb]
        md, lo_d, hi_d = _bootstrap_diff(vals_a, vals_b)
        w = sum(1 for x, y in zip(vals_a, vals_b) if x > y)
        t = sum(1 for x, y in zip(vals_a, vals_b) if x == y)
        l = len(vals_a) - w - t
        print(f"  {d:<24} {md:>+10.3f} [{lo_d:+.3f}, {hi_d:+.3f}]   {w}/{t}/{l}")
    print()

    print(f"  {'scenario_id':<32} {a_tag:>8} {b_tag:>8} {'diff':>8}")
    print("  " + "-" * 60)
    for sa_, sb_ in zip(sa, sb):
        diff = sa_["total"] - sb_["total"]
        print(
            f"  {sa_['scenario_id']:<32} {sa_['total']:>8.2f} "
            f"{sb_['total']:>8.2f} {diff:>+8.2f}"
        )
    print()


def print_baseline(baseline_tag: str, baseline: dict, others: dict[str, dict]) -> None:
    """Compare every other run against one reference."""
    print("=" * 100)
    print(f"BASELINE: {baseline_tag} vs all others")
    print("  (each extra comparison raises the odds of a false positive)")
    print("-" * 100)
    print(f"{'vs':<24} {'n':>4} {'MeanDiff':>9} {'CI 95%':>16} {'Wins/Ties/Losses':>18}")
    print("-" * 100)
    for tag, run in others.items():
        if tag == baseline_tag:
            continue
        sa, sb = common_scored(baseline, run)
        if not sa:
            print(f"{tag:<24} {'--':>4}  (no common scored scenarios)")
            continue
        ta = [s["total"] for s in sa]
        tb = [s["total"] for s in sb]
        mean_diff, lo, hi = _bootstrap_diff(ta, tb)
        wins = sum(1 for x, y in zip(ta, tb) if x > y)
        ties = sum(1 for x, y in zip(ta, tb) if x == y)
        losses = len(ta) - wins - ties
        print(
            f"{tag:<24} {len(ta):>4} {mean_diff:>+8.2f} "
            f"[{lo:+.2f}, {hi:+.2f}]      {wins}/{ties}/{losses}"
        )
    print()


def print_lowest(runs: dict[str, dict], dimension: str, n: int, tag: str) -> None:
    """Show N lowest-scoring scenarios for a dimension in a specific run."""
    run = runs.get(tag)
    if not run:
        print(f"[ERR] Tag {tag!r} not found.")
        return

    sc = []
    for sid, s in run.get("scenarios", {}).items():
        if s.get("errored") or s.get("scoring_failed"):
            continue
        dims = s.get("dimensions") or {}
        if dims.get(dimension) is not None:
            sc.append((s, dims[dimension]))

    if not sc:
        print(f"[ERR] No scored scenarios with dimension {dimension!r} in {tag!r}.")
        return

    sc.sort(key=lambda x: x[1])
    lowest = sc[:n]

    print(f"\n  Lowest {n} scenarios for {dimension} in {tag}:")
    print(f"  {'scenario_id':<32} {dimension:>6}  verdict")
    print("  " + "-" * 55)
    for s, val in lowest:
        verdict = s.get("verdict", "?")
        print(f"  {s['scenario_id']:<32} {val:>6.2f}  {verdict}")
    print()


def print_category_means(runs: dict[str, dict]) -> None:
    """Per-category means. n=5 per category — directional only."""
    categories: list[str] = []
    seen: set[str] = set()
    for run in runs.values():
        for s in run.get("scenarios", {}).values():
            cat = s.get("category")
            if cat and cat not in seen:
                seen.add(cat)
                categories.append(cat)
    if not categories:
        return

    print("=" * 100)
    print("PER-CATEGORY MEAN TOTAL SCORES  (n=5 per category — directional only)")
    print("-" * 100)
    header = f"{'Category':<24}" + "".join(f"{tag:<16}" for tag in runs)
    print(header)
    print("-" * 100)
    for cat in categories:
        row = f"{cat:<24}"
        for tag, run in runs.items():
            vals = [
                s["total"] for s in run.get("scenarios", {}).values()
                if s.get("category") == cat
                and not s.get("errored") and not s.get("scoring_failed")
                and s.get("total") is not None
            ]
            if vals:
                row += f"{sum(vals)/len(vals):<16.2f}"
            else:
                row += f"{'--':<16}"
        print(row)
    print()


def print_scorer_warning(runs: dict[str, dict]) -> None:
    """Warn when runs differ in scorer model."""
    models = {tag: run.get("config", {}).get("scorer_model") for tag, run in runs.items()}
    unique = set(m for m in models.values() if m)
    if len(unique) > 1:
        print(
            "[WARN] Runs differ in scorer_model — cross-run comparisons are "
            "against different judges:\n"
            + "\n".join(f"    {tag}: {m}" for tag, m in models.items())
            + "\n"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fixed-set evaluation runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("tags", nargs="*", help="Run tags to compare (default: all).")
    parser.add_argument("--pair", nargs=2, metavar=("A", "B"),
                        help="Paired comparison between two runs.")
    parser.add_argument("--baseline", metavar="TAG",
                        help="Compare every other run against this reference.")
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES,
                        help=f"Bootstrap resamples (default: {BOOTSTRAP_RESAMPLES}).")
    parser.add_argument("--show-lowest", nargs=3, metavar=("DIMENSION", "N", "TAG"),
                        help="Show N lowest-scoring scenarios for DIMENSION in TAG (e.g. --show-lowest question_discipline 5 base_qwen7b)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_runs = load_all_runs()
    if args.tags:
        runs = {}
        for tag in args.tags:
            run = load_run(tag)
            if run:
                runs[tag] = run
    else:
        runs = all_runs

    if not runs:
        print("[ERR] No runs found under eval_runs/. Run auto_eval.py --fixed-set first.")
        sys.exit(1)

    print_scorer_warning(runs)

    summaries = {tag: run_summary(run) for tag, run in runs.items()}
    print_run_table(summaries)

    if args.pair:
        a_tag, b_tag = args.pair
        if a_tag not in runs or b_tag not in runs:
            print(f"[ERR] --pair tags must be among: {', '.join(sorted(runs))}")
            sys.exit(1)
        print_paired(a_tag, b_tag, runs[a_tag], runs[b_tag])

    if args.baseline:
        if args.baseline not in runs:
            print(f"[ERR] --baseline tag not found: {args.baseline}")
            sys.exit(1)
        print_baseline(args.baseline, runs[args.baseline], runs)

    if args.show_lowest:
        dim, n_str, tag = args.show_lowest
        try:
            n = int(n_str)
        except ValueError:
            print(f"[ERR] N must be an integer, got {n_str!r}")
            sys.exit(1)
        print_lowest(runs, dim, n, tag)

    print_category_means(runs)


if __name__ == "__main__":
    main()