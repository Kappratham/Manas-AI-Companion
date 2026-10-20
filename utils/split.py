"""
Step 3 — Stratified split 90/10 by category.
"""

import json
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
except ImportError:
    class _Dummy:
        def print(self, *a): print(a)
        def print(self, *a, **k): print(a)
    console = _Dummy()

from sklearn.model_selection import train_test_split as tts


def stratified_split(
    conversations: list,
    test_size: float = 0.10,
    random_state: int = 42,
) -> tuple:
    """
    Perform a stratified train/eval split by category.
    Returns (train_convs, eval_convs).
    """
    labels = [c.get("category", "unknown") for c in conversations]

    # Count per category — stratify only if min count ≥ 2
    cat_counts = {}
    for lbl in labels:
        cat_counts[lbl] = cat_counts.get(lbl, 0) + 1

    stratify = labels if min(cat_counts.values()) >= 2 else None
    if stratify is None:
        console.print("[yellow]Warning: Some categories have <2 items — using unstratified split[/yellow]")

    train, eval_set = tts(
        conversations,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # ── Distribution table
    table = Table(title="Train/Eval Split by Category", header_style="bold cyan")
    table.add_column("Category",       justify="left")
    table.add_column("Total",          justify="right")
    table.add_column("Train",          justify="right")
    table.add_column("Eval",           justify="right")
    table.add_column("% Eval",         justify="right")

    for cat in sorted(cat_counts):
        total = cat_counts[cat]
        trn   = sum(1 for c in train      if c.get("category") == cat)
        evl   = sum(1 for c in eval_set   if c.get("category") == cat)
        pct   = f"{evl/total*100:.0f}%" if total else "N/A"
        table.add_row(cat, str(total), str(trn), str(evl), pct)

    # Totals row
    table.add_row(
        "[bold]TOTAL[/bold]",
        "[bold]" + str(len(conversations)) + "[/bold]",
        "[bold]" + str(len(train)) + "[/bold]",
        "[bold]" + str(len(eval_set)) + "[/bold]",
        "[bold]" + f"{len(eval_set)/len(conversations)*100:.0f}%" + "[/bold]",
    )
    console.print(table)
    console.print()

    return train, eval_set