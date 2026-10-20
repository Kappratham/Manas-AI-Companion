#!/usr/bin/env python3
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
"""
Clean and promote flagged conversations.

Applies all known fixes to each flagged conversation, re-validates,
and appends passing ones to the corresponding output file.
"""

import json
import re
from pathlib import Path

from config import CATEGORIES
from validators.checks import validate_conversation, has_crisis_signal
from rich.console import Console
from rich.table import Table

console = Console()


# ──────────────────────────────────────────────────────────
# Fixes (same logic as generate.py, but extended)
# ──────────────────────────────────────────────────────────

def fix_role(role: str) -> str:
    """Strip leading non-alpha chars from role (e.g. ',assistant' → 'assistant')."""
    if role and not role[0].isalpha():
        role = re.sub(r'^[^a-zA-Z]+', '', role)
    return role.strip()


def fix_empty_key_role(msg: dict) -> dict:
    """Handle messages where the role key itself is missing or empty-string key."""
    # Model sometimes outputs {"role": "", "content": ...} or {"": "", "content": ...}
    if "" in msg:
        del msg[""]
    if not msg.get("role"):
        # Try to infer from context — if content is non-empty and no role, skip it
        msg["role"] = "__drop__"
    return msg


def fix_conversation(conversation: dict) -> dict:
    """Apply all cleaning passes to a conversation object."""
    messages = conversation.get("conversations", [])
    if not messages:
        return conversation

    fixed = []
    for msg in messages:
        # Fix 0: empty-key artifacts ({"": "", "role": ...})
        msg = fix_empty_key_role(msg)

        role = fix_role(msg.get("role", ""))
        msg["role"] = role
        content = (msg.get("content") or "").strip()
        msg["content"] = content

        # Drop sentinel messages
        if role == "__drop__":
            continue

        # Fix 1: drop empty-content assistant messages (stubs)
        if role == "assistant" and not content:
            continue

        # Fix 2: drop empty-content user messages ONLY if they break alternation
        if role == "user" and not content:
            # Keep if it's the only signal of presence (grief silences are valid)
            # but drop if it would break turn counting and is truly blank
            if len(content) == 0:
                continue

        # Fix 3: merge duplicate consecutive assistant messages
        if fixed and fixed[-1].get("role") == "assistant" and role == "assistant":
            fixed[-1]["content"] = fixed[-1]["content"].rstrip() + " " + content
            continue

        # Fix 4: single question mark per assistant turn
        if role == "assistant":
            q_count = content.count("?")
            if q_count > 1:
                # Only keep last question mark
                content = content.replace("?", "")
                content = content.rstrip().rstrip(".") + "?"
                msg["content"] = content

        fixed.append(msg)

    conversation["conversations"] = fixed

    # Recompute crisis signal after cleaning
    conversation["has_crisis_signal"] = has_crisis_signal(conversation)

    return conversation


def ensure_valid_alternation(conversation: dict) -> dict:
    """
    Enforce strict system → user → assistant → user → ... alternation.
    Drops or re-labels messages that break the pattern.
    """
    messages = conversation.get("conversations", [])
    if not messages:
        return conversation

    result = []
    for msg in messages:
        role = msg.get("role", "")
        if role == "system":
            result.append(msg)
            continue

        if not result:
            # First non-system message must be user
            if role != "user":
                continue
            result.append(msg)
            continue

        last_role = result[-1].get("role", "")
        if last_role == "system":
            if role == "user":
                result.append(msg)
        elif last_role == "user":
            if role == "assistant":
                result.append(msg)
        elif last_role == "assistant":
            if role == "user":
                result.append(msg)

    conversation["conversations"] = result
    return conversation


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    review_dir = Path("review")
    output_dir = Path("output")

    flagged_files = sorted(review_dir.glob("*_flagged.jsonl"))
    if not flagged_files:
        console.print("[yellow]No flagged files found in review/[/yellow]")
        return

    grand_promoted = 0
    grand_still_failing = 0

    for flagged_path in flagged_files:
        category = flagged_path.stem.replace("_flagged", "")
        output_path = output_dir / f"{category}.jsonl"
        config = CATEGORIES.get(category, {"turns_min": 5, "turns_max": 10, "sensitive": False})

        console.rule(f"[bold cyan]{category}[/bold cyan]")

        lines = [l.strip() for l in flagged_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        promoted = 0
        still_failing = []

        for i, line in enumerate(lines):
            try:
                conv = json.loads(line)
            except json.JSONDecodeError as e:
                console.print(f"  [red]Line {i+1}: JSON parse error — {e}[/red]")
                still_failing.append({"line": i+1, "reason": f"JSON parse error: {e}"})
                continue

            original_failures = conv.get("validation", {}).get("checks_failed", [])

            # Remove old validation/failure metadata before re-checking
            conv.pop("validation", None)
            conv.pop("failure_reasons", None)
            conv.pop("flagged_reason", None)

            # Apply fixes
            conv = fix_conversation(conv)
            conv = ensure_valid_alternation(conv)

            # Re-validate
            validation = validate_conversation(conv, config)
            conv["validation"] = validation

            if validation["passed"]:
                # Append to output
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(conv, ensure_ascii=False) + "\n")
                promoted += 1
                console.print(
                    f"  [green][PASS] Line {i+1}[/green] promoted  "
                    f"(was failing: {original_failures})"
                )
            else:
                still_failing.append({
                    "line": i + 1,
                    "original_failures": original_failures,
                    "remaining_failures": validation["checks_failed"],
                    "pass_count": validation["pass_count"]
                })
                console.print(
                    f"  [red][FAIL] Line {i+1}[/red] still failing: "
                    f"{validation['checks_failed']}  (pass_count={validation['pass_count']}/7)"
                )

        grand_promoted += promoted
        grand_still_failing += len(still_failing)

        table = Table(title=f"{category} result")
        table.add_column("Status")
        table.add_column("Count")
        table.add_row("[green]Promoted to output[/green]", str(promoted))
        table.add_row("[red]Still failing[/red]", str(len(still_failing)))
        console.print(table)

        # Write still-failing back to review file (overwrite)
        if still_failing:
            # Re-read and keep only still-failing lines
            with open(flagged_path, "w", encoding="utf-8") as f:
                for line_info in still_failing:
                    ln = line_info.get("line")
                    if ln:
                        raw = lines[ln - 1]
                        f.write(raw + "\n")
            console.print(f"  [yellow]{len(still_failing)} still-failing written back to {flagged_path.name}[/yellow]")
        else:
            flagged_path.unlink()
            console.print(f"  [green]{flagged_path.name} deleted (all promoted)[/green]")

    console.rule("[bold]Summary[/bold]")
    console.print(f"[green]Total promoted: {grand_promoted}[/green]")
    console.print(f"[red]Total still failing: {grand_still_failing}[/red]")


if __name__ == "__main__":
    main()
