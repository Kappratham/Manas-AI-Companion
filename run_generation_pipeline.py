"""
run_generation_pipeline.py — End-to-end pipeline runner for Manas v2 data.

What it does, in order:
  1. Runs data_auditor.py to see current state of output/
  2. Calculates how many more conversations each category needs
     to reach the per-category target (default: 40 clean each,
     440 total across 11 categories)
  3. Runs generate_v2.py for each category that is short of the
     target, requesting enough to compensate for an expected
     ~70% v2 validator pass rate
  4. SKIPS categories that already have >= target clean conversations
  5. After all generation finishes, runs data_cleaner_v2.py to
     write the final Unsloth-format dataset
  6. Prints a final summary including whether the 440 target
     was reached and "Ready for fine-tuning: YES/NO"

Usage:
    python run_generation_pipeline.py
    python run_generation_pipeline.py --target 40
    python run_generation_pipeline.py --target 40 --expected-pass-rate 0.7
    python run_generation_pipeline.py --target 40 --no-generate
    python run_generation_pipeline.py --target 40 --no-clean
"""

import argparse
import asyncio
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from config import CATEGORIES
from prompts.seeds import SEEDS
from validators.checks_v2 import validate_conversation_v2

console = Console()

# Hard-coded target categories come from prompts/seeds.py — 11 in total.
# CATEGORIES in config.py only knows about 5, so the pipeline should
# iterate over SEEDS (the source of truth) and look up the rest from
# CATEGORIES when available, falling back to sane defaults.
DEFAULT_CATEGORY_CONFIG = {
    "turns_min": 6,
    "turns_max": 8,
    "sensitive": False,
    "count": 40,
}


def get_category_config(category: str) -> dict:
    """Return the per-category config from CATEGORIES or a safe default."""
    return CATEGORIES.get(category, DEFAULT_CATEGORY_CONFIG.copy())


def count_clean_per_category() -> dict:
    """
    Run v2 validation against the existing output/<cat>.jsonl files
    and return {category: passed_count} (i.e. count of conversations
    that would survive data_cleaner_v2.py).
    """
    output_dir = Path("output")
    counts = {}

    for category in SEEDS.keys():
        path = output_dir / f"{category}.jsonl"
        if not path.exists():
            counts[category] = 0
            continue

        config = get_category_config(category)
        passed = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                result = validate_conversation_v2(record, config)
                if result["passed"]:
                    passed += 1
        counts[category] = passed

    return counts


def run_auditor() -> None:
    """Run data_auditor.py to display the current state of output/."""
    console.print("\n[bold cyan]=== Step 1: Audit current output/ ===[/bold cyan]")
    auditor_path = Path("data_auditor.py")
    if not auditor_path.exists():
        console.print(
            "[yellow]data_auditor.py not found — skipping audit step[/yellow]"
        )
        return

    try:
        subprocess.run(
            [sys.executable, "data_auditor.py"],
            check=False,
        )
    except Exception as e:
        console.print(f"[red]Auditor failed: {e}[/red]")


def calculate_needs(
    clean_counts: dict, target: int, expected_pass_rate: float,
) -> list:
    """
    For each category, return a list of dicts:
        {category, current_clean, need_clean, generate_count, skip}

    `generate_count` is the number of NEW conversations to request
    from generate_v2.py, sized so that we expect to clear the
    `need_clean` gap given `expected_pass_rate`.
    """
    rows = []
    for category in SEEDS.keys():
        current = clean_counts.get(category, 0)
        need_clean = max(0, target - current)
        if need_clean == 0:
            rows.append({
                "category": category,
                "current_clean": current,
                "need_clean": 0,
                "generate_count": 0,
                "skip": True,
            })
            continue

        # If pass rate is 0 (no historical data), assume 1.0 so we
        # at least ask for exactly what we need. Otherwise, scale.
        if expected_pass_rate <= 0:
            rate = 1.0
        else:
            rate = min(expected_pass_rate, 1.0)

        generate_count = int(round(need_clean / rate))
        # Always round up to leave headroom for retries.
        if generate_count * rate < need_clean:
            generate_count += 1
        # Don't go below the actual deficit.
        generate_count = max(generate_count, need_clean)

        rows.append({
            "category": category,
            "current_clean": current,
            "need_clean": need_clean,
            "generate_count": generate_count,
            "skip": False,
        })
    return rows


def print_needs_table(rows: list, target: int) -> None:
    table = Table(
        title=f"Step 2: Generation Plan (target = {target} per category)"
    )
    table.add_column("Category", style="cyan")
    table.add_column("Clean", justify="right", style="green")
    table.add_column("Need", justify="right", style="yellow")
    table.add_column("Generate", justify="right")
    table.add_column("Action", style="magenta")

    for row in rows:
        action = "SKIP" if row["skip"] else "GENERATE"
        style = "[dim]" if row["skip"] else ""
        table.add_row(
            f"{style}{row['category']}",
            f"{style}{row['current_clean']}",
            f"{style}{row['need_clean']}",
            f"{style}{row['generate_count']}",
            f"{style}{action}",
        )
    console.print(table)


async def run_generation(rows: list, no_generate: bool) -> dict:
    """
    Run generate_v2.py for each non-skipped category. Returns a dict
    keyed by category with attempted/passed/failed.
    """
    if no_generate:
        console.print(
            "\n[yellow]=== Step 3: Generation skipped (--no-generate) ===[/yellow]"
        )
        return {}

    # Import lazily so --no-generate does not require the full
    # generation dependency stack at import time.
    from generate_v2 import generate_category_v2

    console.print(
        "\n[bold cyan]=== Step 3: Generate missing conversations ===[/bold cyan]"
    )
    results = {}
    for row in rows:
        category = row["category"]
        if row["skip"]:
            console.print(
                f"[green]SKIP[/green] {category} — already at "
                f"{row['current_clean']} clean"
            )
            continue

        count = row["generate_count"]
        console.print(
            f"\n[bold]→ {category}[/bold] generating {count} "
            f"(current clean = {row['current_clean']}, "
            f"need {row['need_clean']} more)"
        )
        stats = await generate_category_v2(
            category, count, resume=True,
        )
        results[category] = stats

    return results


def run_cleaner(no_clean: bool) -> None:
    """Run data_cleaner_v2.py to produce formatted/train_v2.jsonl."""
    if no_clean:
        console.print(
            "\n[yellow]=== Step 4: Cleaning skipped (--no-clean) ===[/yellow]"
        )
        return

    cleaner_path = Path("data_cleaner_v2.py")
    if not cleaner_path.exists():
        console.print(
            "[red]data_cleaner_v2.py not found — cannot clean[/red]"
        )
        return

    console.print(
        "\n[bold cyan]=== Step 4: Clean and convert to Unsloth format ===[/bold cyan]"
    )
    subprocess.run(
        [sys.executable, "data_cleaner_v2.py"],
        check=False,
    )


def final_summary(target: int) -> dict:
    """
    Recount clean conversations from output/ and report a final summary.
    Returns the per-category breakdown dict.
    """
    console.print(
        "\n[bold cyan]=== Step 5: Final Summary ===[/bold cyan]"
    )
    counts = count_clean_per_category()
    total = sum(counts.values())

    table = Table(title="Final Clean Count per Category")
    table.add_column("Category", style="cyan")
    table.add_column("Clean", justify="right", style="green")
    table.add_column("Target", justify="right")
    table.add_column("Status")

    for category, clean in counts.items():
        status = "OK" if clean >= target else f"short by {target - clean}"
        table.add_row(category, str(clean), str(target), status)

    console.print(table)

    ready = total >= target * len(SEEDS)
    console.print(
        f"\n[bold]Total clean conversations:[/bold] "
        f"[green]{total}[/green] / "
        f"{target * len(SEEDS)} target"
    )
    console.print(
        f"[bold]Ready for fine-tuning:[/bold] "
        f"{'[green]YES[/green]' if ready else '[red]NO[/red]'}"
    )
    return {"per_category": counts, "total": total, "ready": ready}


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Manas v2 generation pipeline."
    )
    parser.add_argument(
        "--target", type=int, default=40,
        help="Clean conversations per category (default: 40)",
    )
    parser.add_argument(
        "--expected-pass-rate", type=float, default=0.7,
        help="Assumed v2 validator pass rate used to size generation "
             "(default: 0.7)",
    )
    parser.add_argument(
        "--no-generate", action="store_true",
        help="Skip generation, only audit and clean",
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="Skip cleaning step",
    )
    parser.add_argument(
        "--no-audit", action="store_true",
        help="Skip the audit step",
    )
    args = parser.parse_args()

    # Step 1 — audit
    if not args.no_audit:
        run_auditor()

    # Step 2 — calculate needs
    console.print(
        "\n[bold cyan]=== Step 2: Calculate per-category needs ===[/bold cyan]"
    )
    clean_counts = count_clean_per_category()
    rows = calculate_needs(
        clean_counts, args.target, args.expected_pass_rate,
    )
    print_needs_table(rows, args.target)

    # Step 3 — generate
    asyncio.run(run_generation(rows, args.no_generate))

    # Step 4 — clean
    run_cleaner(args.no_clean)

    # Step 5 — final summary
    summary = final_summary(args.target)

    if not summary["ready"]:
        console.print(
            f"\n[yellow]Tip: re-run the pipeline to top up short "
            f"categories. Target total = "
            f"{args.target * len(SEEDS)}.[/yellow]"
        )


if __name__ == "__main__":
    main()
