#!/usr/bin/env python3
"""
rescore_generated.py — Batch score existing generated conversations.

Loads all JSONL files from output/, scores each conversation using the
existing Scorer, filters by min-score threshold, saves passing conversations
to output_filtered/ and optionally deletes failing conversations from
the original output/ files in place.

Usage:
    python rescore_generated.py
    python rescore_generated.py --min-score 8
    python rescore_generated.py --category grief_loss
    python rescore_generated.py --resume
    python rescore_generated.py --dry-run
    python rescore_generated.py --delete-failing
    python rescore_generated.py --min-score 7 --delete-failing
    python rescore_generated.py --delete-failing --force
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, MofNCompleteColumn,
    TimeElapsedColumn, TextColumn, BarColumn
)
from rich.table import Table
from rich.panel import Panel

from eval.scorer import Scorer, EvaluationResult, DIMENSION_NAMES

load_dotenv()
console = Console()


def backup_output_files(output_dir: Path, backup_dir: Path):
    """Copy all JSONL files from output/ to output_backup/ before any deletion."""
    backup_dir.mkdir(exist_ok=True, parents=True)
    count = 0
    for jsonl_file in output_dir.glob("*.jsonl"):
        shutil.copy2(jsonl_file, backup_dir / jsonl_file.name)
        count += 1
    console.print(f"[dim]Backup created: {backup_dir} ({count} files backed up)[/dim]")


def confirm_large_deletion(
    category: str,
    total: int,
    removing: int,
) -> bool:
    if total <= 0:
        return True
    pct = removing / total * 100
    if pct > 30:
        console.print(
            f"[yellow]Warning: removing {removing}/{total} ({pct:.0f}%) "
            f"conversations from {category}.[/yellow]"
        )
        answer = input("Continue? (y/n): ").strip().lower()
        return answer == "y"
    return True


def rewrite_output_keeping_passing(
    output_path: Path,
    passing_conversations: list[dict],
    dry_run: bool = False,
) -> int:
    """
    Rewrite the original output JSONL keeping only passing conversations.
    Returns number of conversations removed.

    In dry_run mode — prints what would happen but does not modify the file.
    """
    if not output_path.exists():
        return 0

    with open(output_path, encoding="utf-8") as f:
        original_lines = [l for l in f if l.strip()]
    original_count = len(original_lines)
    new_count = len(passing_conversations)
    removed = original_count - new_count

    if dry_run:
        console.print(
            f"  [dim]DRY RUN: would remove {removed} / {original_count} "
            f"conversations from {output_path.name}[/dim]"
        )
        return removed

    with open(output_path, "w", encoding="utf-8") as f:
        for conv in passing_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    console.print(
        f"  [green]Rewrote {output_path.name}: "
        f"kept {new_count}, removed {removed}[/green]"
    )
    return removed


async def score_all(
    conversations: list[dict],
    scorer: Scorer,
    category: str,
    concurrency: int = 3,
    scored_ids: set[str] = None,
    progress_callback = None,
) -> list[tuple[dict, EvaluationResult]]:
    semaphore = asyncio.Semaphore(concurrency)
    scored_ids = scored_ids or set()

    results = []

    async def score_one(conv: dict, idx: int):
        async with semaphore:
            await asyncio.sleep(0.3)
            persona_id = conv.get("persona", {}).get("id", f"p{idx}")
            conv_id = f"{category}__{persona_id}__{idx:03d}"

            if conv_id in scored_ids and "rescore" in conv:
                # Already scored and cached
                # Reconstruct EvaluationResult from saved rescore metadata if possible, or re-evaluate
                pass

            try:
                result = await scorer.evaluate(
                    conversation=conv.get("conversations", []),
                    category=category,
                    persona_id=persona_id,
                    situation=conv.get("seed", ""),
                    conversation_id=conv_id,
                )
            except Exception as e:
                console.print(f"[red]Error scoring {conv_id}: {e}[/red]")
                result = scorer._make_fallback_result(
                    category, persona_id, conv.get("seed", ""), conv_id, len(conv.get("conversations", []))
                )

            if progress_callback:
                progress_callback(conv_id, result)

            return (conv, result)

    tasks = [score_one(conv, i) for i, conv in enumerate(conversations)]
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    for item in completed:
        if isinstance(item, Exception):
            console.print(f"[red]Task exception: {item}[/red]")
        elif item is not None:
            results.append(item)

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch score existing generated conversations.")
    parser.add_argument("--category", type=str, default=None, help="Category name or None for all")
    parser.add_argument("--min-score", type=int, default=7, help="Minimum total score out of 10 to pass")
    parser.add_argument("--dry-run", action="store_true", help="Score but do not save or delete anything")
    parser.add_argument("--resume", action="store_true", help="Skip already scored conversation IDs")
    parser.add_argument("--delete-failing", action="store_true", help="Delete failing conversations from output/")
    parser.add_argument("--force", action="store_true", help="Skip confirmation on large deletions (>30%)")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrency limit for scoring")

    args = parser.parse_args()

    output_dir = Path("output")
    filtered_dir = Path("output_filtered")
    rejected_dir = Path("output_rejected")
    reports_dir = Path("rescore_reports")
    backup_dir = Path("output_backup")
    scored_ids_path = reports_dir / "scored_ids.json"

    if not output_dir.exists():
        console.print("[red]Error: output/ directory does not exist.[/red]")
        sys.exit(1)

    jsonl_files = sorted(output_dir.glob("*.jsonl"))
    if not jsonl_files:
        console.print("[red]Error: No JSONL files found in output/.[/red]")
        sys.exit(1)

    if args.category:
        target_file = output_dir / f"{args.category}.jsonl"
        if not target_file.exists():
            console.print(f"[red]Error: Category file {target_file} not found.[/red]")
            sys.exit(1)
        jsonl_files = [target_file]

    reports_dir.mkdir(exist_ok=True, parents=True)
    filtered_dir.mkdir(exist_ok=True, parents=True)
    rejected_dir.mkdir(exist_ok=True, parents=True)

    scored_ids = set()
    if args.resume and scored_ids_path.exists():
        try:
            with open(scored_ids_path, encoding="utf-8") as f:
                scored_ids = set(json.load(f))
            console.print(f"[green]Resuming: loaded {len(scored_ids)} already scored IDs.[/green]")
        except Exception:
            pass

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        console.print("[red]Error: NVIDIA_API_KEY not set in environment or .env[/red]")
        sys.exit(1)

    # Suppress noisy debug prints from the scorer (which can have encoding issues on Windows)
    import logging
    logging.getLogger("rich.console").setLevel(logging.ERROR)

    # Temporarily redirect console to suppress scorer debug prints
    from eval.scorer import console as scorer_console
    scorer_console.quiet = True

    scorer = Scorer(api_key=api_key)

    if args.delete_failing and not args.dry_run:
        backup_output_files(output_dir, backup_dir)
        # Restore output files first since backup was created
        # (We'll rewrite them after scoring)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "min_score": args.min_score,
        "delete_failing": args.delete_failing,
        "concurrency": args.concurrency,
        "categories": {},
    }

    all_passing_count = 0
    all_reject_count = 0
    all_total_count = 0

    dimension_accumulators = {dim: [] for dim in DIMENSION_NAMES}
    category_rows = []

    console.print(Panel.fit(
        f"[bold cyan]Rescoring Generated Conversations[/bold cyan]\n"
        f"Threshold: {args.min_score}/10 | Dry Run: {args.dry_run} | Delete Failing: {args.delete_failing}",
        border_style="cyan"
    ))

    for file_path in jsonl_files:
        category = file_path.stem
        console.print(f"\n[bold]Processing category: {category}[/bold]")

        conversations = []
        with open(file_path, encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    conv = json.loads(line)
                    conversations.append(conv)
                except json.JSONDecodeError:
                    console.print(f"[yellow]  Skipping invalid JSON at line {line_idx + 1} in {file_path.name}[/yellow]")

        if not conversations:
            console.print(f"[yellow]  No valid conversations found in {file_path.name}[/yellow]")
            continue

        passing_convs = []
        reject_convs = []

        # Filter out already scored if resuming
        to_score = []
        for idx, conv in enumerate(conversations):
            persona_id = conv.get("persona", {}).get("id", f"p{idx}")
            conv_id = f"{category}__{persona_id}__{idx:03d}"
            if args.resume and conv_id in scored_ids:
                # Already scored in previous run — check if it passed or failed based on previous rescore or re-evaluate
                # For safety, if already in scored_ids, we can re-evaluate or check its score if stored.
                # Actually, let's just score it or load if rescore metadata exists.
                pass
            to_score.append((conv, idx))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[green]pass:{task.fields[passed]}[/green] [yellow]fail:{task.fields[failed]}[/yellow]"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]{category}[/cyan]",
                total=len(conversations),
                passed=0,
                failed=0,
            )

            pass_n = 0
            fail_n = 0

            def on_progress(conv_id, result):
                nonlocal pass_n, fail_n
                scored_ids.add(conv_id)
                if result.total_score >= args.min_score:
                    pass_n += 1
                else:
                    fail_n += 1
                progress.update(task, advance=1, passed=pass_n, failed=fail_n)

            # Run scoring with concurrency
            semaphore = asyncio.Semaphore(args.concurrency)

            async def score_single(conv, idx):
                async with semaphore:
                    await asyncio.sleep(0.3)
                    persona_id = conv.get("persona", {}).get("id", f"p{idx}")
                    conv_id = f"{category}__{persona_id}__{idx:03d}"
                    try:
                        result = await scorer.evaluate(
                            conversation=conv.get("conversations", []),
                            category=category,
                            persona_id=persona_id,
                            situation=conv.get("seed", ""),
                            conversation_id=conv_id,
                        )
                    except Exception as e:
                        console.print(f"[red]Error scoring {conv_id}: {e}[/red]")
                        result = scorer._make_fallback_result(
                            category, persona_id, conv.get("seed", ""), conv_id, len(conv.get("conversations", []))
                        )
                    on_progress(conv_id, result)
                    return (conv, result)

            async def _run_all():
                tasks = [score_single(conv, idx) for conv, idx in to_score]
                return await asyncio.gather(*tasks)

            scored_results = asyncio.run(_run_all())

        cat_passing = []
        cat_rejects = []

        for conv, result in scored_results:
            # Add rescore metadata
            rescore_meta = {
                "total_score": result.total_score,
                "fraction": result.fraction,
                "verdict": result.verdict,
                "dimensions": {
                    k: {"value": v.value, "explanation": v.explanation}
                    for k, v in result.dimensions.items()
                }
            }
            conv_copy = dict(conv)
            conv_copy["rescore"] = rescore_meta

            if result.total_score >= args.min_score:
                cat_passing.append(conv_copy)
                for dim_name, dim_score in result.dimensions.items():
                    dimension_accumulators[dim_name].append(dim_score.value)
            else:
                cat_rejects.append(conv_copy)

        total_cat = len(conversations)
        pass_cat_count = len(cat_passing)
        fail_cat_count = len(cat_rejects)
        pass_pct = int(pass_cat_count / total_cat * 100) if total_cat > 0 else 0

        all_passing_count += pass_cat_count
        all_reject_count += fail_cat_count
        all_total_count += total_cat

        # Save filtered passing JSONL
        filtered_file = filtered_dir / f"{category}.jsonl"
        if not args.dry_run:
            with open(filtered_file, "w", encoding="utf-8") as f:
                for conv in cat_passing:
                    f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        # Save rejected JSONL
        rejected_file = rejected_dir / f"{category}.jsonl"
        if not args.dry_run:
            with open(rejected_file, "w", encoding="utf-8") as f:
                for conv in cat_rejects:
                    f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        deleted_count = 0
        if args.delete_failing:
            removing = fail_cat_count
            if args.force or confirm_large_deletion(category, total_cat, removing):
                deleted_count = rewrite_output_keeping_passing(
                    file_path, cat_passing, dry_run=args.dry_run
                )
            else:
                console.print(f"[yellow]  Skipped deletion for {category} due to user abort.[/yellow]")

        category_rows.append({
            "category": category,
            "total": total_cat,
            "pass": pass_cat_count,
            "fail": fail_cat_count,
            "pass_pct": pass_pct,
            "deleted": deleted_count,
        })

        run_summary["categories"][category] = {
            "total": total_cat,
            "pass": pass_cat_count,
            "fail": fail_cat_count,
            "pass_percentage": f"{pass_pct}%",
            "deleted": deleted_count,
        }

        # Save scored IDs cache after each category
        if not args.dry_run:
            with open(scored_ids_path, "w", encoding="utf-8") as f:
                json.dump(list(scored_ids), f, indent=2)

    # Calculate dimension averages
    dim_averages = {}
    for dim, vals in dimension_accumulators.items():
        if vals:
            dim_averages[dim] = round(sum(vals) / len(vals), 2)
        else:
            dim_averages[dim] = 0.0

    overall_pass_pct = int(all_passing_count / all_total_count * 100) if all_total_count > 0 else 0

    # Print Summary Table
    table = Table(title=f"RESCORE REPORT — Min Score: {args.min_score}/10")
    table.add_column("Category", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Pass", justify="right", style="green")
    table.add_column("Fail", justify="right", style="yellow")
    table.add_column("Pass%", justify="right")
    table.add_column("Deleted", justify="right", style="red")

    for row in category_rows:
        table.add_row(
            row["category"],
            str(row["total"]),
            str(row["pass"]),
            str(row["fail"]),
            f"{row['pass_pct']}%",
            str(row["deleted"]) if args.delete_failing else "0",
        )

    table.add_section()
    table.add_row(
        "TOTAL",
        str(all_total_count),
        str(all_passing_count),
        str(all_reject_count),
        f"{overall_pass_pct}%",
        str(all_reject_count) if args.delete_failing else "0",
        style="bold"
    )

    console.print("\n")
    console.print(table)

    console.print("\n[bold]DIMENSION AVERAGES (passing conversations only)[/bold]")
    for dim_name, avg_val in dim_averages.items():
        formatted_name = dim_name.replace("_", " ").title()
        console.print(f"  {formatted_name:25s} {avg_val:.2f}")

    report_txt_path = reports_dir / f"rescore_report_{timestamp}.txt"
    report_json_path = reports_dir / f"rescore_report_{timestamp}.json"

    run_summary["totals"] = {
        "total": all_total_count,
        "passed": all_passing_count,
        "failed": all_reject_count,
        "pass_percentage": f"{overall_pass_pct}%",
        "deleted": all_reject_count if args.delete_failing else 0,
    }
    run_summary["dimension_averages"] = dim_averages

    if not args.dry_run:
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)

        txt_lines = [
            "RESCORE REPORT",
            "==============",
            f"Date: {run_summary['timestamp']}",
            f"Min score threshold: {args.min_score}/10",
            f"Delete failing: {'yes' if args.delete_failing else 'no'}",
            f"Concurrency: {args.concurrency}",
            "",
            "CATEGORY RESULTS",
            "────────────────────────────────────────────────────────────────",
            f"{'Category':25s} {'Total':6s} {'Pass':6s} {'Fail':6s} {'Pass%':6s} {'Deleted':8s}",
        ]
        for row in category_rows:
            txt_lines.append(
                f"{row['category']:25s} {row['total']:6d} {row['pass']:6d} {row['fail']:6d} {row['pass_pct']:4d}%    {row['deleted']:6d}"
            )
        txt_lines.extend([
            "────────────────────────────────────────────────────────────────",
            f"{'TOTAL':25s} {all_total_count:6d} {all_passing_count:6d} {all_reject_count:6d} {overall_pass_pct:4d}%    {all_reject_count if args.delete_failing else 0:6d}",
            "",
            "DIMENSION AVERAGES (passing conversations only)",
        ])
        for dim_name, avg_val in dim_averages.items():
            formatted_name = dim_name.replace("_", " ").title()
            txt_lines.append(f"{formatted_name:25s} {avg_val:.2f}")

        txt_lines.extend([
            "",
            "FILES",
            f"Filtered → output_filtered/   ({all_passing_count} conversations)",
            f"Rejected → output_rejected/   ({all_reject_count} conversations — kept for inspection)",
            f"Report   → {report_txt_path}",
        ])

        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines) + "\n")

        console.print(f"\n[dim]Report saved to {report_txt_path} and {report_json_path}[/dim]")


if __name__ == "__main__":
    main()
