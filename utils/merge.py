"""
Step 1 — Merge all raw category JSONL files into one.
"""

import json
import os
from pathlib import Path

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class _Dummy:
        def print(self, *a, **k): print(*a)
    console = _Dummy()

CATEGORY_FILES = [
    "career_anxiety",
    "parental_expectations",
    "relationship_stress",
    "relocation_loneliness",
    "financial_stress",
    "academic_pressure",
    "identity_confusion",
    "burnout_motivation",
    "grief_loss",
    "coping_mechanisms",
    "trauma_patterns",
]


def merge_all(output_dir: str = "output", dest: str = "formatted/merged_raw.jsonl") -> tuple:
    """
    Merge all category JSONL files into a single merged file.
    Returns (all_conversations, category_counts_dict).
    """
    all_conversations = []
    category_counts = {}

    for category in CATEGORY_FILES:
        path = Path(output_dir) / f"{category}.jsonl"
        if not path.exists():
            console.print(f"[yellow]Warning: {path} not found — skipping[/yellow]")
            continue

        with open(path, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        category_counts[category] = len(lines)
        all_conversations.extend(lines)

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        for conv in all_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    return all_conversations, category_counts


if __name__ == "__main__":
    convs, counts = merge_all()
    print(f"Merged {len(convs)} conversations.")