"""
data_cleaner_v2.py — Clean output/ folder using v2 validation.

What it does:
  1. Reads all JSONL files from output/
  2. Runs v2 validation on each conversation
  3. Keeps conversations that pass all v2 checks
  4. Writes clean conversations to formatted/train_v2.jsonl
     in the Unsloth ShareGPT format your notebook expects

Usage:
    python data_cleaner_v2.py
    python data_cleaner_v2.py --data-dir output --output formatted/train_v2.jsonl
"""

import json
import re
import argparse
import random
from pathlib import Path
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from config import CATEGORIES
from validators.checks_v2 import validate_conversation_v2

console = Console()

# ---------------------------------------------------------------------------
# Compact system prompt for training
# Full prompt is loaded at runtime from adapted_data/
# but we use a compact version during training to save context window
# ---------------------------------------------------------------------------

COMPACT_SYSTEM_PROMPT = """You are Manas — a mental health companion, not a therapist.

Warm but not soft. Honest but not harsh. You notice what is underneath without weaponizing it.

Arc: validate first (turns 1-4, silent assessment) → explore → surface → one small suggestion → close gently.
One question per turn. Never two.
No lists. No bullet points. No bold headers.
No hollow reassurance. No diagnosis. No medical advice.
Never shame a coping mechanism — understand its function first.
Mirror the user's language exactly.
Crisis: iCall 9152987821 | Vandrevala 1860-2662-345. Stay present. Never abandon.
Goal: make their internal voice louder, not replace it."""


def to_unsloth_format(record: dict) -> dict:
    """
    Convert internal format to Unsloth ShareGPT format.

    Input format (your current format):
        {"conversations": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
        ]}

    Output format (Unsloth ShareGPT):
        {"conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."},
        ]}
    """
    role_map = {
        "system": "system",
        "user": "human",
        "assistant": "gpt",
    }

    conv = record.get("conversations", [])
    converted = []

    for msg in conv:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Replace system prompt with compact version
        if role == "system":
            content = COMPACT_SYSTEM_PROMPT

        mapped_role = role_map.get(role)
        if mapped_role is None:
            continue

        converted.append({
            "from": mapped_role,
            "value": content,
        })

    return {
        "conversations": converted,
        "category": record.get("category", ""),
        "persona_id": record.get("persona", {}).get("id", ""),
        "source": "v1_cleaned",
    }


def clean_assistant_content(content: str) -> str:
    """
    Apply the same cleaning logic as conversation_runner._clean_generated_reply
    but as a preprocessing step for training data.

    We want training data to already be clean — not rely on runtime cleaning.
    """
    if not content:
        return content

    content = content.strip()

    # Remove special tokens
    for token in ["<|im_end|>", "<|endoftext|>", "</s>", "<|im_start|>"]:
        content = content.split(token)[0].strip()

    # Remove markdown bold
    content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)

    # Remove numbered lists
    content = re.sub(r"(?im)^\s*\d+\.\s+", "", content)

    # Remove bullet lists
    content = re.sub(r"(?im)^\s*[-*•]\s+", "", content)

    # Cut at hallucinated user turns
    roleplay_patterns = [
        r"(?:^|\n)\s*user\s*\n",
        r"(?:^|\n)\s*user\s*:",
        r"(?:^|\n)\s*human\s*:",
    ]
    cut_positions = []
    for pattern in roleplay_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            if match.start() > 0:
                cut_positions.append(match.start())
    if cut_positions:
        content = content[: min(cut_positions)].strip()

    # Keep only first question
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

    # Remove incomplete sentence at end
    if content and content[-1] not in ".!?\"'":
        last_punct = max(
            content.rfind("."),
            content.rfind("!"),
            content.rfind("?"),
        )
        if last_punct > len(content) // 3:
            content = content[: last_punct + 1].strip()

    # Collapse whitespace
    content = re.sub(r"\n{2,}", "\n", content)
    content = re.sub(r"[ \t]{2,}", " ", content)

    return content.strip()


def clean_record(record: dict) -> dict:
    """
    Clean all assistant turns in a record.
    Applies runtime cleaning as a preprocessing step.
    """
    conv = record.get("conversations", [])
    cleaned_conv = []

    for msg in conv:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            content = clean_assistant_content(content)

        cleaned_conv.append({**msg, "content": content})

    return {**record, "conversations": cleaned_conv}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="output",
        help="Directory containing per-category JSONL files",
    )
    parser.add_argument(
        "--output",
        default="formatted/train_v2.jsonl",
        help="Output JSONL file in Unsloth format",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Shuffle output dataset",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="Skip categories with pass rate below this threshold",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        console.print(f"[red]No JSONL files found in {data_dir}[/red]")
        return

    kept_records = []
    dropped_total = 0
    stats_per_category = []
    all_failure_reasons = defaultdict(int)

    for filepath in jsonl_files:
        category = filepath.stem
        config = CATEGORIES.get(category, {"turns_min": 6, "turns_max": 8})

        kept = 0
        dropped = 0
        failure_reasons = defaultdict(int)

        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    dropped += 1
                    failure_reasons["json_parse_error"] += 1
                    continue

                # Clean the record first
                record = clean_record(record)

                # Then validate
                result = validate_conversation_v2(record, config)

                if result["passed"]:
                    kept_records.append(record)
                    kept += 1
                else:
                    dropped += 1
                    for reason in result["checks_failed"]:
                        failure_reasons[reason] += 1
                        all_failure_reasons[reason] += 1

        stats_per_category.append({
            "category": category,
            "kept": kept,
            "dropped": dropped,
            "top_failure": max(
                failure_reasons, key=failure_reasons.get
            ) if failure_reasons else "none",
        })
        dropped_total += dropped

    # Print per-category stats
    table = Table(title="V2 Cleaning Results")
    table.add_column("Category", style="cyan")
    table.add_column("Kept", justify="right", style="green")
    table.add_column("Dropped", justify="right", style="red")
    table.add_column("Top Failure Reason", style="yellow")

    for stat in stats_per_category:
        table.add_row(
            stat["category"],
            str(stat["kept"]),
            str(stat["dropped"]),
            stat["top_failure"],
        )

    console.print(table)

    # Shuffle if requested
    if args.shuffle:
        random.shuffle(kept_records)

    # Convert to Unsloth format and write
    with open(output_path, "w", encoding="utf-8") as f:
        for record in kept_records:
            unsloth_record = to_unsloth_format(record)
            f.write(json.dumps(unsloth_record, ensure_ascii=False) + "\n")

    # Print summary
    total = len(kept_records) + dropped_total
    console.print(f"\n[bold]Cleaning complete[/bold]")
    console.print(f"  Total input  : {total}")
    console.print(f"  Kept         : [green]{len(kept_records)}[/green]")
    console.print(f"  Dropped      : [red]{dropped_total}[/red]")
    console.print(
        f"  Pass rate    : {len(kept_records) / total:.0%}"
        if total > 0 else ""
    )
    console.print(f"\n[bold]Top failure reasons:[/bold]")
    for reason, count in sorted(
        all_failure_reasons.items(), key=lambda x: -x[1]
    ):
        console.print(f"  {reason:35s} {count}")
    console.print(f"\n[green]Clean data saved to {output_path}[/green]")
    console.print(
        f"[dim]Format: Unsloth ShareGPT — ready for manas_finetune_colab.ipynb[/dim]"
    )


if __name__ == "__main__":
    main()