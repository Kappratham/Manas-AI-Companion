"""
rescue_flagged.py — Attempt to rescue fixable flagged conversations.

Reads all review/*_v2_flagged.jsonl files.
Applies cleaning fixes to conversations with only fixable failures.
Re-validates with v2 validator.
Appends rescued conversations to output/category.jsonl.
Discards conversations with non-fixable failures.

Run this AFTER run_generation_pipeline.py completes.

Usage:
    python rescue_flagged.py
    python rescue_flagged.py --review-dir review --output-dir output
    python rescue_flagged.py --dry-run
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from config import CATEGORIES
from validators.checks_v2 import validate_conversation_v2

console = Console()

# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

FIXABLE_FAILURES = {
    "response_length",
    "no_markdown_formatting",
    "no_clinical_openings",
    "no_hallucinated_user_turns",
    "no_empty_assistant_turns",
    "no_lecture_mode",
}

NOT_FIXABLE = {
    "turns_range",
    "no_diagnosis",
    "no_big_steps",
    "no_hollow_reassurance",
    "crisis_referral_present",
    "json_valid",
    "single_question_per_turn",
    "no_early_framework",
}


# ---------------------------------------------------------------------------
# Cleaning functions
# (mirrors data_cleaner_v2.py clean_assistant_content)
# ---------------------------------------------------------------------------

def clean_assistant_content(content: str) -> str:
    """
    Apply fixes to a single assistant response.

    Fixes:
      - Remove special tokens
      - Strip markdown bold
      - Strip numbered and bullet lists
      - Cut at hallucinated user turns
      - Remove clinical opening phrases
      - Remove lecture mode triggers
      - Keep only first question
      - Remove incomplete sentence at end
      - Collapse whitespace
    """
    if not content:
        return content

    content = content.strip()

    # ── 1. Remove special tokens ────────────────────────────────────────
    for token in ["<|im_end|>", "<|endoftext|>", "</s>", "<|im_start|>"]:
        if token in content:
            content = content.split(token)[0].strip()

    # ── 2. Strip markdown bold (**text**) ───────────────────────────────
    content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)

    # ── 3. Strip numbered lists ─────────────────────────────────────────
    content = re.sub(r"(?im)^\s*\d+\.\s+", "", content)

    # ── 4. Strip bullet lists ───────────────────────────────────────────
    content = re.sub(r"(?im)^\s*[-*•]\s+", "", content)

    # ── 5. Remove clinical opening phrases ──────────────────────────────
    clinical_pattern = re.compile(
        r"^(?:"
        r"Cynically speaking[,\s]"
        r"|Cynical Acceptance[,\s:]"
        r"|Structured Cynicism[,\s:]"
        r"|To be honest[,\s]"
        r"|Logically speaking[,\s]"
        r"|Objectively[,\s]"
        r"|In reality[,\s]"
        r"|Statistically[,\s]"
        r"|Technically[,\s]"
        r"|Scientifically[,\s]"
        r"|As an AI[,\s]"
        r"|I should mention[,\s]"
        r"|It is important to note[,\s]"
        r"|It is worth noting[,\s]"
        r")",
        re.IGNORECASE,
    )
    content = clinical_pattern.sub("", content).strip()

    # ── 6. Remove lecture mode triggers ─────────────────────────────────
    # Inline lecture triggers (no newline before them)
    inline_lecture = re.compile(
        r"\.?\s*(?:Here are|Here is|These are|Below are)"
        r"\s+(?:some|a few|the|several)?\s*"
        r"(?:steps|ways|things|strategies|options|approaches|suggestions)",
        re.IGNORECASE,
    )
    match = inline_lecture.search(content)
    if match and match.start() > 50:
        content = content[: match.start()].strip()
        if content and content[-1] not in ".!?":
            content = content + "."

    # Newline lecture triggers
    lecture_pattern = re.compile(
        r"\n\s*(?:"
        r"For example"
        r"|Here are"
        r"|Here is"
        r"|You could also consider"
        r"|Additionally"
        r"|Furthermore"
        r"|Moreover"
        r"|Some options include"
        r"|There are resources"
        r"|Let me suggest"
        r"|I would suggest"
        r"|One thing you could"
        r"|Ask yourself these"
        r"|To know whether"
        r"|Below are"
        r"|These are"
        r")",
        re.IGNORECASE,
    )
    match = lecture_pattern.search(content)
    if match and match.start() > 50:
        content = content[: match.start()].strip()

    # ── 7. Cut at hallucinated user turns ───────────────────────────────
    roleplay_patterns = [
        r"(?:^|\n)\s*user\s*\n",
        r"(?:^|\n)\s*user\s*:",
        r"(?:^|\n)\s*human\s*:",
        r"\?user\s*:",
        r"\?human\s*:",
    ]
    cut_positions = []
    for pattern in roleplay_patterns:
        for m in re.finditer(pattern, content, re.IGNORECASE):
            if m.start() > 0:
                cut_positions.append(m.start())
    if cut_positions:
        content = content[: min(cut_positions)].strip()

    # ── 8. Keep only the first question ─────────────────────────────────
    sentences = re.split(r"(?<=[.!?])\s+", content)
    kept = []
    q_count = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if sentence.endswith("?"):
            q_count += 1
            if q_count > 1:
                break
        kept.append(sentence)
    content = " ".join(kept).strip()

    # ── 9. Truncate to max 5 sentences ──────────────────────────────────
    sentences = re.split(r"(?<=[.!?])\s+", content)
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) > 5:
        content = " ".join(sentences[:5]).strip()

    # ── 10. Remove incomplete sentence at end ───────────────────────────
    if content and content[-1] not in ".!?\"'":
        last_punct = max(
            content.rfind("."),
            content.rfind("!"),
            content.rfind("?"),
        )
        if last_punct > len(content) // 3:
            content = content[: last_punct + 1].strip()

    # ── 11. Collapse whitespace ──────────────────────────────────────────
    content = re.sub(r"\n{2,}", "\n", content)
    content = re.sub(r"[ \t]{2,}", " ", content)

    return content.strip()


def fix_conversation(record: dict) -> dict:
    """
    Apply all fixes to a conversation record.

    Fixes applied:
      - Clean all assistant turn content
      - Remove empty assistant turns
      - Remove empty user turns
    """
    conv = record.get("conversations", [])
    fixed = []

    for msg in conv:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            content = clean_assistant_content(content)
            # Skip empty assistant turns
            if not content.strip():
                continue

        if role == "user" and not content.strip():
            # Skip empty user turns
            continue

        fixed.append({**msg, "content": content})

    return {**record, "conversations": fixed}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_record(record: dict) -> str:
    """
    Classify a flagged record as fixable, not_fixable, or unknown.

    Returns:
      "fixable"     — all failures are in FIXABLE_FAILURES set
      "not_fixable" — at least one failure is in NOT_FIXABLE set
      "unknown"     — no validation info found
    """
    validation = record.get("validation", {})
    failed = set(validation.get("checks_failed", []))

    if not failed:
        # No failure info — check failure_reasons field
        failed = set(record.get("failure_reasons", []))

    if not failed:
        return "unknown"

    if failed & NOT_FIXABLE:
        return "not_fixable"

    if failed & FIXABLE_FAILURES:
        return "fixable"

    return "unknown"


# ---------------------------------------------------------------------------
# Main rescue logic
# ---------------------------------------------------------------------------

def rescue_file(
    flagged_file: Path,
    output_file: Path,
    category: str,
    dry_run: bool = False,
) -> dict:
    """
    Attempt to rescue conversations from one flagged file.

    Returns stats dict.
    """
    cat_config = CATEGORIES.get(category, {"turns_min": 6, "turns_max": 8})

    total = 0
    rescued = 0
    discarded_not_fixable = 0
    discarded_still_fails = 0
    unknown = 0
    rescue_records = []

    with open(flagged_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            classification = classify_record(record)

            if classification == "not_fixable":
                discarded_not_fixable += 1
                continue

            if classification == "unknown":
                unknown += 1
                continue

            # classification == "fixable" — attempt to fix
            fixed_record = fix_conversation(record)

            # Re-validate after fixes
            result = validate_conversation_v2(fixed_record, cat_config)

            if result["passed"]:
                fixed_record["validation"] = result
                fixed_record["rescued"] = True
                fixed_record.pop("failure_reasons", None)
                rescue_records.append(fixed_record)
                rescued += 1
            else:
                discarded_still_fails += 1

    # Append rescued records to output file
    if rescue_records and not dry_run:
        with open(output_file, "a", encoding="utf-8") as f:
            for record in rescue_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "category": category,
        "total_flagged": total,
        "rescued": rescued,
        "discarded_not_fixable": discarded_not_fixable,
        "discarded_still_fails": discarded_still_fails,
        "unknown": unknown,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rescue fixable flagged conversations after generation pipeline."
    )
    parser.add_argument(
        "--review-dir",
        default="review",
        help="Directory containing *_v2_flagged.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory containing per-category output JSONL files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be rescued without writing anything",
    )
    args = parser.parse_args()

    review_dir = Path(args.review_dir)
    output_dir = Path(args.output_dir)

    flagged_files = sorted(review_dir.glob("*_v2_flagged.jsonl"))

    if not flagged_files:
        console.print(f"[yellow]No flagged files found in {review_dir}[/yellow]")
        console.print(
            "[dim]Run run_generation_pipeline.py first to generate conversations.[/dim]"
        )
        return

    console.print(
        f"[bold]Rescue Pipeline[/bold] "
        f"{'[yellow](DRY RUN)[/yellow]' if args.dry_run else ''}"
    )
    console.print(f"[dim]Found {len(flagged_files)} flagged files[/dim]\n")

    all_stats = []
    total_rescued = 0
    total_flagged = 0

    for flagged_file in flagged_files:
        # Extract category from filename
        # e.g. "grief_loss_v2_flagged.jsonl" → "grief_loss"
        stem = flagged_file.stem
        category = stem.replace("_v2_flagged", "").replace("_flagged", "")

        output_file = output_dir / f"{category}.jsonl"

        if not output_file.exists() and not args.dry_run:
            output_file.touch()

        console.print(f"[cyan]Processing {category}...[/cyan]")

        stats = rescue_file(
            flagged_file=flagged_file,
            output_file=output_file,
            category=category,
            dry_run=args.dry_run,
        )

        all_stats.append(stats)
        total_rescued += stats["rescued"]
        total_flagged += stats["total_flagged"]

    # Print summary table
    table = Table(title="Rescue Results")
    table.add_column("Category", style="cyan")
    table.add_column("Flagged", justify="right")
    table.add_column("Rescued", justify="right", style="green")
    table.add_column("Not Fixable", justify="right", style="red")
    table.add_column("Still Fails", justify="right", style="yellow")
    table.add_column("Unknown", justify="right", style="dim")

    for stats in all_stats:
        table.add_row(
            stats["category"],
            str(stats["total_flagged"]),
            str(stats["rescued"]),
            str(stats["discarded_not_fixable"]),
            str(stats["discarded_still_fails"]),
            str(stats["unknown"]),
        )

    console.print(table)

    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"  Total flagged    : {total_flagged}")
    console.print(f"  Total rescued    : [green]{total_rescued}[/green]")
    console.print(
        f"  Rescue rate      : "
        f"{total_rescued / total_flagged:.0%}"
        if total_flagged > 0 else ""
    )

    if args.dry_run:
        console.print(
            "\n[yellow]Dry run — nothing written. "
            "Remove --dry-run to apply fixes.[/yellow]"
        )
    else:
        console.print(
            f"\n[green]Done. {total_rescued} conversations rescued "
            f"and appended to output/*.jsonl[/green]"
        )
        console.print(
            "[dim]Run data_cleaner_v2.py to rebuild "
            "formatted/train_v2.jsonl with rescued conversations.[/dim]"
        )
        console.print(
            "\n[bold]Next steps:[/bold]"
        )
        console.print("  1. python data_cleaner_v2.py")
        console.print("  2. Upload formatted/train_v2.jsonl to Colab")
        console.print("  3. Run notebooks/manas_finetune_colab.ipynb")


if __name__ == "__main__":
    main()