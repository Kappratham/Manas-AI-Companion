"""
data_auditor.py — Audit your output/ folder using v2 validation.

Shows per-category breakdown of what passes and what fails.
Tells you exactly which conversations need to be dropped.

Usage:
    python data_auditor.py
    python data_auditor.py --data-dir output
    python data_auditor.py --output audit_report.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from config import CATEGORIES
from validators.checks_v2 import validate_conversation_v2

console = Console()


def audit_file(filepath: Path, category: str) -> dict:
    """Audit one JSONL file. Returns stats dict."""
    config = CATEGORIES.get(category, {"turns_min": 6, "turns_max": 8})

    total = 0
    passed = 0
    failed = 0
    failure_reasons = defaultdict(int)
    failed_indices = []

    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                failed += 1
                failure_reasons["json_parse_error"] += 1
                failed_indices.append(line_num)
                total += 1
                continue

            total += 1
            result = validate_conversation_v2(record, config)

            if result["passed"]:
                passed += 1
            else:
                failed += 1
                failed_indices.append(line_num)
                for reason in result["checks_failed"]:
                    failure_reasons[reason] += 1

    return {
        "category": category,
        "file": str(filepath),
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total > 0 else 0.0,
        "failure_reasons": dict(failure_reasons),
        "failed_indices": failed_indices,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="output",
        help="Directory containing per-category JSONL files",
    )
    parser.add_argument(
        "--output",
        default="audit_report.json",
        help="Save full audit report here",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    jsonl_files = sorted(data_dir.glob("*.jsonl"))

    if not jsonl_files:
        console.print(f"[red]No JSONL files found in {data_dir}[/red]")
        return

    all_stats = []
    total_conversations = 0
    total_passed = 0
    all_failure_reasons = defaultdict(int)

    for filepath in jsonl_files:
        category = filepath.stem
        stats = audit_file(filepath, category)
        all_stats.append(stats)
        total_conversations += stats["total"]
        total_passed += stats["passed"]
        for reason, count in stats["failure_reasons"].items():
            all_failure_reasons[reason] += count

    # Print per-category summary table
    table = Table(title="V2 Audit — Per Category")
    table.add_column("Category", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Pass", justify="right", style="green")
    table.add_column("Fail", justify="right", style="red")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Top Failure", style="yellow")

    for stats in all_stats:
        top_failure = ""
        if stats["failure_reasons"]:
            top_failure = max(
                stats["failure_reasons"],
                key=stats["failure_reasons"].get,
            )
        table.add_row(
            stats["category"],
            str(stats["total"]),
            str(stats["passed"]),
            str(stats["failed"]),
            f"{stats['pass_rate']:.0%}",
            top_failure,
        )

    console.print(table)

    # Print overall failure reason breakdown
    console.print("\n[bold]Overall failure reasons (v2 checks):[/bold]")
    for reason, count in sorted(
        all_failure_reasons.items(), key=lambda x: -x[1]
    ):
        bar = "█" * min(count, 30)
        console.print(f"  {reason:35s} {count:4d}  {bar}")

    console.print(
        f"\n[bold]Total:[/bold] {total_conversations} conversations | "
        f"[green]{total_passed} pass[/green] | "
        f"[red]{total_conversations - total_passed} fail[/red] | "
        f"[bold]{total_passed / total_conversations:.0%} pass rate[/bold]"
        if total_conversations > 0
        else "\n[red]No conversations found[/red]"
    )

    # Save report
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": total_conversations,
                "passed": total_passed,
                "failed": total_conversations - total_passed,
                "pass_rate": total_passed / total_conversations
                if total_conversations > 0
                else 0,
                "failure_reasons": dict(all_failure_reasons),
                "per_category": all_stats,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    console.print(f"\n[green]Full report saved to {args.output}[/green]")


if __name__ == "__main__":
    main()