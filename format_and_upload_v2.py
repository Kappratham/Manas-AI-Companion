#!/usr/bin/env python3
"""
format_and_upload_v2.py

Manas Dataset Formatter and Hugging Face Uploader.

What this does:
1. Merges raw generated conversations from output/*.jsonl
2. Merges manually approved conversations from formatted/approved/*.json
3. Validates generated conversations with validators.checks_v2
4. Auto-repairs common validation failures (turns_range, response_length)
5. Deduplicates exact + near-duplicate conversations
6. Splits into train/eval
7. Converts to Unsloth ShareGPT format
8. Saves:
   - formatted/merged_raw_v2.jsonl
   - formatted/merged_clean_v2.jsonl
   - formatted/train_v2.jsonl
   - formatted/eval_v2.jsonl
   - reports/dataset_report_v2.txt
9. Uploads to Hugging Face dataset repo

Usage:
    python format_and_upload_v2.py --all
    python format_and_upload_v2.py --stats
    python format_and_upload_v2.py --all --repo-name manas-dataset-v3
    python format_and_upload_v2.py --all --dry-run
"""

import json
import os
import copy
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login

from prompts.seeds import SEEDS
from prompts.personas import PERSONAS
from validators.checks_v2 import validate_conversation_v2


# ── Rich console ─────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        MofNCompleteColumn,
    )
    console = Console()
except ImportError:
    class DummyConsole:
        def print(self, *args, **kwargs):
            print(*[str(a) for a in args])

        def rule(self, *args, **kwargs):
            print("-" * 70)
    console = DummyConsole()

load_dotenv()

# ── Categories derived from seeds.py ─────────────────────────────────────────
ALL_CATEGORIES = sorted(SEEDS.keys())

SENSITIVE_CATEGORIES = {
    "grief_loss",
    "trauma_patterns",
    "coping_mechanisms",
}

DEFAULT_CATEGORY_CONFIG = {
    "turns_min": 4,
    "turns_max": 10,
    "sensitive": False,
}

# ── Response length bounds used by auto-repair ────────────────────────────────
RESPONSE_MIN_WORDS = 40
RESPONSE_MAX_WORDS = 280


def get_category_config(category: str) -> dict:
    return {
        "turns_min": DEFAULT_CATEGORY_CONFIG["turns_min"],
        "turns_max": DEFAULT_CATEGORY_CONFIG["turns_max"],
        "sensitive": category in SENSITIVE_CATEGORIES,
    }


# ── Compact system prompt for training ───────────────────────────────────────
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


# ── Helpers ──────────────────────────────────────────────────────────────────
def infer_category_from_filename(file_path: Path) -> str:
    """Infer category from a filename like career_anxiety_batch0.json."""
    stem = file_path.stem
    for cat in sorted(ALL_CATEGORIES, key=len, reverse=True):
        if stem.startswith(cat):
            return cat
    return "unknown"


def build_fingerprint(conv: dict) -> tuple:
    """Exact signature for user/assistant content."""
    messages = conv.get("conversations", [])
    return tuple(
        (m.get("role"), (m.get("content") or "").strip())
        for m in messages
        if m.get("role") in ("user", "assistant")
    )


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity."""
    set_a = set(a.split())
    set_b = set(b.split())
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def save_jsonl(path: Path, items: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_messages_to_conversations(record: dict) -> dict:
    """
    Convert eval-style {"messages": [...]} to training-style {"conversations": [...]}.
    Keeps only role/content.
    """
    if "conversations" in record:
        return record

    messages = record.get("messages", [])
    record["conversations"] = [
        {
            "role": msg.get("role"),
            "content": msg.get("content"),
        }
        for msg in messages
        if msg.get("role") in ("system", "user", "assistant")
    ]
    return record


def format_to_unsloth(conv: dict) -> dict:
    """
    Convert to Unsloth ShareGPT format:
    {
      "conversations": [
        {"from": "system", "value": "..."},
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "..."}
      ]
    }
    """
    role_map = {
        "system": "system",
        "user": "human",
        "assistant": "gpt",
    }

    raw_messages = conv.get("conversations", [])
    converted = []

    for msg in raw_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            content = COMPACT_SYSTEM_PROMPT

        mapped = role_map.get(role)
        if mapped is None:
            continue

        converted.append({
            "from": mapped,
            "value": content,
        })

    persona = conv.get("persona", {})
    persona_id = (
        persona.get("id", "") if isinstance(persona, dict)
        else conv.get("persona_id", "")
    )

    return {
        "conversations": converted,
        "category": conv.get("category", ""),
        "persona_id": persona_id,
        "source": conv.get("source", "v2"),
    }


# ── Auto-Repair ───────────────────────────────────────────────────────────────
def auto_repair_conversation(conv: dict, config: dict) -> dict:
    """
    Attempt to auto-repair common validation failures:

    - response_length : truncate assistant turns that are too long.
                        Turns that are too short are flagged but kept
                        (content quality cannot be auto-generated here).
    - turns_range     : pad conversations that are too short with a
                        natural closing exchange; trim conversations that
                        are too long by keeping the opening + closing turns.

    Returns a deep-copied, potentially modified conversation dict.
    The original is never mutated.
    """
    conv = copy.deepcopy(conv)

    turns_min = config.get("turns_min", DEFAULT_CATEGORY_CONFIG["turns_min"])
    turns_max = config.get("turns_max", DEFAULT_CATEGORY_CONFIG["turns_max"])

    repairs_applied = []

    # ── 1. Fix response_length ────────────────────────────────────────────────
    repaired_turns = []
    for turn in conv.get("conversations", []):
        if turn.get("role") == "assistant":
            content = turn.get("content", "")
            words = content.split()
            word_count = len(words)

            if word_count > RESPONSE_MAX_WORDS:
                truncated = " ".join(words[:RESPONSE_MAX_WORDS])
                # Prefer cutting at a sentence boundary
                last_boundary = max(
                    truncated.rfind("."),
                    truncated.rfind("?"),
                    truncated.rfind("!"),
                )
                # Only use boundary if it's in the latter 60 % of the text
                if last_boundary > int(RESPONSE_MAX_WORDS * 0.6 * 5):
                    truncated = truncated[: last_boundary + 1]
                turn["content"] = truncated
                repairs_applied.append("truncated_response")

            elif word_count < RESPONSE_MIN_WORDS:
                # Cannot fix content quality — flag only
                turn["_repair_flag"] = "response_too_short"
                repairs_applied.append("flagged_short_response")

        repaired_turns.append(turn)

    conv["conversations"] = repaired_turns

    # ── 2. Fix turns_range ────────────────────────────────────────────────────
    dialogue_turns = [
        t for t in conv["conversations"]
        if t.get("role") in ("user", "assistant")
    ]
    turn_count = len(dialogue_turns)

    if turn_count < turns_min:
        # Pad with natural closing exchange(s)
        pairs_needed = max(1, (turns_min - turn_count + 1) // 2)
        for _ in range(pairs_needed):
            conv["conversations"].append({
                "role": "user",
                "content": "That actually makes a lot of sense. Thank you.",
            })
            conv["conversations"].append({
                "role": "assistant",
                "content": (
                    "I'm glad it landed that way. You've been really honest "
                    "with yourself in this conversation — that's not easy. "
                    "Whenever you want to talk more, I'm here."
                ),
            })
        repairs_applied.append("padded_turns")

    elif turn_count > turns_max:
        # Keep system turns intact; trim dialogue to turns_max
        system_turns = [
            t for t in conv["conversations"] if t.get("role") == "system"
        ]
        non_system = [
            t for t in conv["conversations"] if t.get("role") != "system"
        ]
        # Preserve opening context and closing exchange
        kept = non_system[: turns_max - 2] + non_system[-2:]
        conv["conversations"] = system_turns + kept
        repairs_applied.append("trimmed_turns")

    if repairs_applied:
        conv["_auto_repaired"] = repairs_applied

    return conv


# ── Step 1: Merge ────────────────────────────────────────────────────────────
def step_merge(output_dir: Path, formatted_dir: Path):
    """
    Merge from:
      1. output/*.jsonl
      2. formatted/approved/*.json
    """
    console.print("\n[bold yellow]Step 1 — Merging all sources...[/bold yellow]")

    merged_raw_path = formatted_dir / "merged_raw_v2.jsonl"
    all_conversations = []
    category_counts = {}

    # Source 1: generated data
    output_files = sorted(output_dir.glob("*.jsonl"))
    if not output_files:
        console.print(
            "[red][ERROR] No JSONL files found in output/. "
            "Run generation first.[/red]"
        )
        return None, {}

    for file_path in output_files:
        category = file_path.stem

        if category not in ALL_CATEGORIES:
            console.print(
                f"[yellow]  [SKIP] {file_path.name} — "
                f"'{category}' not in prompts/seeds.py[/yellow]"
            )
            continue

        parsed = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
                conv["category"] = conv.get("category", category)
                conv["source"] = conv.get("source", "generated")
                parsed.append(conv)
            except json.JSONDecodeError as e:
                console.print(f"[red]  [SKIP] {file_path.name}: {e}[/red]")

        category_counts[category] = category_counts.get(category, 0) + len(parsed)
        all_conversations.extend(parsed)
        console.print(f"  [dim]{category}: {len(parsed)} generated[/dim]")

    # Source 2: manually approved eval conversations
    approved_dir = formatted_dir / "approved"
    approved_files = (
        sorted(approved_dir.glob("*.json")) if approved_dir.exists() else []
    )

    if approved_files:
        console.print(
            f"\n  [dim]Loading {len(approved_files)} manually approved "
            f"conversations...[/dim]"
        )
        approved_added = 0

        for file_path in approved_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    record = json.load(f)

                record = convert_messages_to_conversations(record)

                category = record.get("category") or infer_category_from_filename(
                    file_path
                )
                if category not in ALL_CATEGORIES:
                    console.print(
                        f"[yellow]  [SKIP] {file_path.name} — "
                        f"unknown category '{category}'[/yellow]"
                    )
                    continue

                record["category"] = category
                record["source"] = "manual_approved"

                all_conversations.append(record)
                category_counts[category] = category_counts.get(category, 0) + 1
                approved_added += 1

            except Exception as e:
                console.print(f"[red]  [SKIP] {file_path.name}: {e}[/red]")

        console.print(
            f"  [green]Added {approved_added} manually approved "
            f"conversations[/green]"
        )
    else:
        console.print("  [dim]No files found in formatted/approved/[/dim]")

    # Missing categories summary
    missing = [c for c in ALL_CATEGORIES if c not in category_counts]
    if missing:
        console.print("\n[yellow]Categories with no data:[/yellow]")
        for cat in missing:
            console.print(f"  [yellow]- {cat}[/yellow]")

    save_jsonl(merged_raw_path, all_conversations)
    console.print(
        f"\n[green][PASS] Merged {len(all_conversations)} total "
        f"→ [bold]{merged_raw_path}[/bold][/green]"
    )
    return all_conversations, category_counts


# ── Step 2: Validate + Dedup ─────────────────────────────────────────────────
def step_validate(
    all_conversations: list,
    formatted_dir: Path,
    dry_run: bool = False,
) -> list:
    """
    Validate generated conversations with v2 validator.
    Attempt auto-repair for turns_range / response_length failures.
    Skip validator for manually approved conversations.
    Deduplicate exact + Jaccard near-duplicates.
    """
    console.print(
        "\n[bold yellow]Step 2 — V2 Validating and deduplicating..."
        "[/bold yellow]"
    )

    JACCARD_THRESHOLD = 0.72

    # Failures that the auto-repair function can handle
    REPAIRABLE_CHECKS = {"turns_range", "response_length"}

    merged_clean_path = formatted_dir / "merged_clean_v2.jsonl"
    removed_validation_path = formatted_dir / "removed_validation_v2.jsonl"
    removed_duplicates_path = formatted_dir / "removed_duplicates_v2.jsonl"

    seen_fingerprints: set = set()
    seen_texts_jaccard: list = []
    cleaned: list = []
    removed_val: list = []
    removed_dup: list = []
    val_failures_by_cat: dict = {}
    val_failure_reasons: dict = {}
    repaired_count: int = 0

    for conv in all_conversations:
        category = conv.get("category", "unknown")
        source = conv.get("source", "")

        # ── Manual approvals bypass validation ───────────────────────────────
        if source == "manual_approved":
            fp = build_fingerprint(conv)
            if fp in seen_fingerprints:
                if not dry_run:
                    removed_dup.append(
                        {**conv, "_removed_reason": "fingerprint_duplicate"}
                    )
                continue
            seen_fingerprints.add(fp)
            cleaned.append(conv)
            continue

        # ── Validate generated conversation ──────────────────────────────────
        config = get_category_config(category)
        validation = validate_conversation_v2(conv, config)

        if not validation["passed"]:
            failed_checks = set(validation.get("checks_failed", []))

            # ── Attempt auto-repair if all failures are repairable ────────────
            if failed_checks.issubset(REPAIRABLE_CHECKS):
                conv_repaired = auto_repair_conversation(conv, config)
                validation_repaired = validate_conversation_v2(conv_repaired, config)

                if validation_repaired["passed"]:
                    # Repair succeeded — use repaired version
                    conv = conv_repaired
                    conv["_was_repaired"] = True
                    repaired_count += 1
                    # Fall through to fingerprint / Jaccard checks below
                else:
                    # Repair did not resolve all issues — discard
                    if not dry_run:
                        removed_val.append({
                            **conv,
                            "_removed_reason": "v2_validation_repair_failed",
                            "_original_failures": list(failed_checks),
                            "_failures_after_repair": (
                                validation_repaired.get("checks_failed", [])
                            ),
                        })
                    val_failures_by_cat[category] = (
                        val_failures_by_cat.get(category, 0) + 1
                    )
                    for reason in validation_repaired.get("checks_failed", []):
                        val_failure_reasons[reason] = (
                            val_failure_reasons.get(reason, 0) + 1
                        )
                    continue

            else:
                # Contains non-repairable failures — discard immediately
                if not dry_run:
                    removed_val.append({
                        **conv,
                        "_removed_reason": "v2_validation",
                        "_failures": list(failed_checks),
                    })
                val_failures_by_cat[category] = (
                    val_failures_by_cat.get(category, 0) + 1
                )
                for reason in failed_checks:
                    val_failure_reasons[reason] = (
                        val_failure_reasons.get(reason, 0) + 1
                    )
                continue

        # ── Exact fingerprint dedup ───────────────────────────────────────────
        fp = build_fingerprint(conv)
        if fp in seen_fingerprints:
            if not dry_run:
                removed_dup.append(
                    {**conv, "_removed_reason": "fingerprint_duplicate"}
                )
            continue
        seen_fingerprints.add(fp)

        # ── Jaccard near-duplicate dedup on user messages ─────────────────────
        user_messages = [
            m.get("content", "")
            for m in conv.get("conversations", [])
            if m.get("role") == "user"
        ]
        user_text = " ".join(user_messages).lower().strip()

        is_jaccard_duplicate = False
        if user_text:
            for seen_text in seen_texts_jaccard:
                if jaccard_similarity(user_text, seen_text) > JACCARD_THRESHOLD:
                    is_jaccard_duplicate = True
                    break

        if is_jaccard_duplicate:
            if not dry_run:
                removed_dup.append(
                    {**conv, "_removed_reason": "jaccard_duplicate"}
                )
        else:
            seen_texts_jaccard.append(user_text)
            cleaned.append(conv)

    # ── Persist removed sets ──────────────────────────────────────────────────
    if not dry_run:
        save_jsonl(removed_validation_path, removed_val)
        save_jsonl(removed_duplicates_path, removed_dup)
        save_jsonl(merged_clean_path, cleaned)

    # ── Console summary ───────────────────────────────────────────────────────
    console.print("[green][PASS] Validation & deduplication complete[/green]")
    console.print(f"  Total processed : {len(all_conversations)}")
    console.print(f"  Validation fails: [red]{len(removed_val)}[/red]")
    console.print(f"  Duplicates      : [yellow]{len(removed_dup)}[/yellow]")
    console.print(f"  Clean dataset   : [bold green]{len(cleaned)}[/bold green]")

    if repaired_count:
        console.print(
            f"  Auto-repaired   : [cyan]{repaired_count}[/cyan] "
            f"(turns_range / response_length fixed)"
        )

    gen_count = sum(1 for c in cleaned if c.get("source") != "manual_approved")
    approved_count = sum(
        1 for c in cleaned if c.get("source") == "manual_approved"
    )
    console.print("\n  [dim]By source:[/dim]")
    console.print(f"    Generated       : {gen_count}")
    console.print(f"    Manual approved : {approved_count}")

    if val_failure_reasons:
        console.print("\n  [bold]Top failure reasons:[/bold]")
        for reason, count in sorted(
            val_failure_reasons.items(), key=lambda x: -x[1]
        ):
            console.print(f"    {reason:35s} {count}")

    if val_failures_by_cat:
        console.print("\n  [bold]Failures by category:[/bold]")
        for cat, cnt in sorted(val_failures_by_cat.items()):
            console.print(f"    [red]{cat}: {cnt} failed[/red]")

    if not cleaned:
        raise RuntimeError("No conversations passed validation.")

    return cleaned


# ── Step 3: Split ────────────────────────────────────────────────────────────
def step_split(cleaned: list, formatted_dir: Path):
    """Split 90/10 and convert to Unsloth ShareGPT format."""
    console.print("\n[bold yellow]Step 3 — Stratified split...[/bold yellow]")

    categories = [c.get("category", "unknown") for c in cleaned]
    cat_counts: dict = {}
    for cat in categories:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    for cat, cnt in cat_counts.items():
        if cnt < 2:
            console.print(
                f"[yellow][WARNING] {cat} has {cnt} sample(s); "
                f"cannot stratify.[/yellow]"
            )

    stratify = categories if min(cat_counts.values()) >= 2 else None
    train_convs, eval_convs = train_test_split(
        cleaned,
        test_size=0.10,
        random_state=42,
        stratify=stratify,
    )

    train_unsloth = [format_to_unsloth(c) for c in train_convs]
    eval_unsloth = [format_to_unsloth(c) for c in eval_convs]

    train_path = formatted_dir / "train_v2.jsonl"
    eval_path = formatted_dir / "eval_v2.jsonl"

    save_jsonl(train_path, train_unsloth)
    save_jsonl(eval_path, eval_unsloth)

    console.print("[green][PASS] Split complete[/green]")
    console.print(f"  Train: [bold]{len(train_convs)}[/bold] → {train_path}")
    console.print(f"  Eval : [bold]{len(eval_convs)}[/bold] → {eval_path}")

    table = Table(title="Category Distribution", header_style="bold cyan")
    table.add_column("Category")
    table.add_column("Seeds", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Train", justify="right")
    table.add_column("Eval", justify="right")

    for cat in ALL_CATEGORIES:
        seed_count = len(SEEDS.get(cat, []))
        t_cnt = sum(1 for c in train_convs if c.get("category") == cat)
        e_cnt = sum(1 for c in eval_convs if c.get("category") == cat)
        total = t_cnt + e_cnt
        table.add_row(cat, str(seed_count), str(total), str(t_cnt), str(e_cnt))

    console.print(table)
    return train_convs, eval_convs


# ── Step 4: Report ───────────────────────────────────────────────────────────
def step_report(
    train_convs: list,
    eval_convs: list,
    cleaned: list,
    category_counts_raw: dict,
):
    """Generate a report file."""
    import datetime

    console.print("\n[bold yellow]Step 4 — Generating report...[/bold yellow]")

    total = len(cleaned)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "dataset_report_v2.txt"

    gen_count = sum(1 for c in cleaned if c.get("source") != "manual_approved")
    approved_count = sum(
        1 for c in cleaned if c.get("source") == "manual_approved"
    )
    repaired_count = sum(1 for c in cleaned if c.get("_was_repaired"))
    sensitive_count = sum(1 for c in cleaned if c.get("sensitive", False))
    crisis_count = sum(1 for c in cleaned if c.get("has_crisis_signal", False))

    cat_lines = []
    for cat in ALL_CATEGORIES:
        raw = category_counts_raw.get(cat, 0)
        cln = sum(1 for c in cleaned if c.get("category") == cat)
        trn = sum(1 for c in train_convs if c.get("category") == cat)
        evl = sum(1 for c in eval_convs if c.get("category") == cat)
        cat_lines.append(
            f"  {cat:<28} {cln:>3}  (raw: {raw:>3})  "
            f"train: {trn:>2}  eval: {evl:>2}"
        )

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""MANAS DATASET REPORT
============================================================
Generated : {now}
Validator : v2
Format    : Unsloth ShareGPT

TOTALS
------
Total clean conversations : {total}
Train (90%)               : {len(train_convs)}
Eval  (10%)               : {len(eval_convs)}

BY SOURCE
---------
Generated (pipeline)      : {gen_count}
Manually approved (eval)  : {approved_count}
Auto-repaired & kept      : {repaired_count}

SENSITIVE / CRISIS
------------------
Sensitive conversations   : {sensitive_count}
Crisis conversations      : {crisis_count}

CATEGORY BREAKDOWN
------------------
{chr(10).join(cat_lines)}

PERSONAS / CATEGORIES
---------------------
Categories: {len(ALL_CATEGORIES)}
Personas  : {len(PERSONAS)}

NEXT STEP
---------
Load in Modal / Colab:

from datasets import load_dataset
dataset = load_dataset(
    "kabir4756/manas-dataset-v2",
    data_files={{"train": "train_v2.jsonl", "eval": "eval_v2.jsonl"}},
    token=HF_TOKEN,
)
"""

    report_path.write_text(report, encoding="utf-8")
    console.print(
        f"[green][PASS] Report saved → [bold]{report_path}[/bold][/green]"
    )


# ── Step 5: Upload ───────────────────────────────────────────────────────────
def step_upload(
    train_path: Path,
    eval_path: Path,
    merged_clean: Path,
    formatted_dir: Path,
    hf_token: str,
    repo_name: str,
):
    """Upload dataset files to Hugging Face."""
    console.print(
        "\n[bold yellow]Step 5 — Uploading to Hugging Face...[/bold yellow]"
    )

    api = HfApi()
    login(token=hf_token)

    user_info = api.whoami(token=hf_token)
    username = user_info["name"]
    repo_id = f"{username}/{repo_name}"

    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=hf_token,
    )
    console.print(f"[green][PASS] Repo ready: [bold]{repo_id}[/bold][/green]")

    train_count = sum(
        1
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    eval_count = sum(
        1
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    readme_content = f"""---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- mental-health
- unsloth
- sharegpt
- synthetic
- manas
size_categories:
- n<1K
---

# Manas Dataset

## Statistics
- Total clean conversations: {train_count + eval_count}
- Train: {train_count}
- Eval: {eval_count}

## Format
```json
{{
  "conversations": [
    {{"from": "system", "value": "..."}},
    {{"from": "human", "value": "..."}},
    {{"from": "gpt", "value": "..."}}
  ]
}}
"""

    readme_path = formatted_dir / "README_v2.md"
    readme_path.write_text(readme_content, encoding="utf-8")

    files_to_upload = {
        "train_v2.jsonl": train_path,
        "eval_v2.jsonl": eval_path,
        "merged_clean_v2.jsonl": merged_clean,
        "README.md": readme_path,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Uploading...", total=len(files_to_upload)
        )

        for repo_file, local_path in files_to_upload.items():
            if not local_path.exists():
                console.print(
                    f"[yellow][SKIP] {repo_file} not found[/yellow]"
                )
                progress.advance(task)
                continue

            progress.update(
                task, description=f"[cyan]Uploading {repo_file}..."
            )
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_file,
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )
            progress.advance(task)
            console.print(f"  [green]↑ {repo_file}[/green]")

    url = f"https://huggingface.co/datasets/{repo_id}"
    console.print(
        Panel.fit(
            f"[bold green]✓ Dataset uploaded[/bold green]\n"
            f"URL: [bold blue]{url}[/bold blue]\n"
            f"Repo is [yellow]private[/yellow]",
            title="Upload Complete",
        )
    )
    return url

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Manas Dataset Formatter & Uploader"
    )
    parser.add_argument("--merge", action="store_true", help="Merge only")
    parser.add_argument(
        "--clean", action="store_true", help="Merge + validate"
    )
    parser.add_argument(
        "--split", action="store_true", help="Merge + validate + split"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run full pipeline"
    )
    parser.add_argument("--upload", action="store_true", help="Upload only")
    parser.add_argument(
        "--stats", action="store_true", help="Show stats only"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing files",
    )
    parser.add_argument(
        "--repo-name",
        default="manas-dataset-v2",
        help="HF dataset repo name (default: manas-dataset-v2)",
    )
    args = parser.parse_args()

    run_all = not any(
        [
            args.merge,
            args.clean,
            args.split,
            args.all,
            args.upload,
            args.stats,
        ]
    )

    console.rule("[bold cyan]MANAS DATASET FORMATTER & UPLOADER[/bold cyan]")
    console.print(
        f"[dim]Sources: output/*.jsonl + formatted/approved/*.json | "
        f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}[/dim]\n"
    )

    output_dir = Path("output")
    formatted_dir = Path("formatted")
    formatted_dir.mkdir(exist_ok=True)

    merged_clean = formatted_dir / "merged_clean_v2.jsonl"
    train_path = formatted_dir / "train_v2.jsonl"
    eval_path = formatted_dir / "eval_v2.jsonl"

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    # ── Stats only ────────────────────────────────────────────────────────────
    if args.stats:
        if not merged_clean.exists():
            console.print(
                "[red]No merged_clean_v2.jsonl found. Run --all first.[/red]"
            )
            return

        cleaned = [
            json.loads(line)
            for line in merged_clean.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        cat_counts: dict = {}
        source_counts: dict = {}
        for c in cleaned:
            cat = c.get("category", "unknown")
            src = c.get("source", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            source_counts[src] = source_counts.get(src, 0) + 1

        repaired = sum(1 for c in cleaned if c.get("_was_repaired"))

        console.print(
            f"[bold]Total clean conversations:[/bold] {len(cleaned)}"
        )
        if repaired:
            console.print(
                f"[cyan]Auto-repaired conversations:[/cyan] {repaired}"
            )

        table = Table(title="By Category", header_style="bold cyan")
        table.add_column("Category")
        table.add_column("Count", justify="right")
        for cat in ALL_CATEGORIES:
            table.add_row(cat, str(cat_counts.get(cat, 0)))
        console.print(table)

        table2 = Table(title="By Source", header_style="bold cyan")
        table2.add_column("Source")
        table2.add_column("Count", justify="right")
        for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
            table2.add_row(src, str(cnt))
        console.print(table2)
        return

    # ── Pipeline ──────────────────────────────────────────────────────────────
    all_convs = None
    cat_counts_raw: dict = {}
    cleaned_convs = None
    train_convs = None
    eval_convs = None

    if run_all or args.all or args.merge or args.clean or args.split:
        all_convs, cat_counts_raw = step_merge(output_dir, formatted_dir)
        if all_convs is None:
            return

    if run_all or args.all or args.clean or args.split:
        if all_convs is None:
            console.print("[red]No merged data available.[/red]")
            return
        cleaned_convs = step_validate(
            all_convs, formatted_dir, dry_run=args.dry_run
        )

    if run_all or args.all or args.split:
        if cleaned_convs is None and merged_clean.exists():
            cleaned_convs = [
                json.loads(line)
                for line in merged_clean.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
        if cleaned_convs is None:
            console.print(
                "[red]No merged_clean_v2.jsonl found. "
                "Run --clean first.[/red]"
            )
            return
        train_convs, eval_convs = step_split(cleaned_convs, formatted_dir)
        step_report(
            train_convs, eval_convs, cleaned_convs, cat_counts_raw or {}
        )

    if args.upload or ((run_all or args.all) and not args.dry_run):
        if not train_path.exists() or not eval_path.exists():
            console.print(
                "[red]train_v2.jsonl or eval_v2.jsonl not found. "
                "Run --split first.[/red]"
            )
            return
        if not hf_token:
            console.print(
                "[yellow][WARNING] No HF_TOKEN/HUGGINGFACE_TOKEN found. "
                "Upload skipped.[/yellow]"
            )
            return
        step_upload(
            train_path,
            eval_path,
            merged_clean,
            formatted_dir,
            hf_token,
            args.repo_name,
        )

if __name__ == "__main__":
    main()
