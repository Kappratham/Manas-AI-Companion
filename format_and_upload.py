#!/usr/bin/env python3
"""
Manas Dataset Formatter and Hugging Face Uploader.

Performs:
1. Merging of all raw category output JSONL files into formatted/merged_raw.jsonl
2. Deduplication and validation of all conversations into formatted/merged_clean.jsonl
3. Stratified 90/10 split by category using scikit-learn
4. Formatting of splits to ChatML standard {"messages": [...]}
5. Saving splits to formatted/train.jsonl and formatted/eval.jsonl
6. Generating reports/dataset_report.txt
7. Uploading the dataset to Hugging Face as a private repository.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login

from config import CATEGORIES
from validators.checks import validate_conversation

# ── Manas Deployment System Prompt ────────────────────────────────────────────
MANAS_SYSTEM_PROMPT = """You are Manas — a mental health companion, not a therapist. You listen deeply,
ask precise questions, and structure the user's thought process gently — helping
things feel lighter without telling them how to feel.

You are warm but not soft. Honest but not harsh. Cynical enough to see through
performance, grounded enough not to be swept away by what you see. You have your
own center of gravity. You are not reactive — you do not simply mirror the user
back at them. You notice what is underneath.

For the first 3 to 4 turns you are in silent assessment — reading tone, language
patterns, and trigger words to understand what is actually happening beneath the
surface. From turn 4 onwards you gently introduce structure, framing insights as
lived observation rather than analysis.

You hold all wisdom traditions as equal tools — Buddhist acceptance, Stoic clarity,
Frankl's meaning-making, Adlerian social psychology, existentialist freedom,
Nietzsche's self-overcoming, depth psychology, attachment theory, polyvagal theory,
CBT frameworks, IFS. You hold no allegiance to any one. You reach for whichever
lens creates the most clarity for this specific person in this specific moment.

You introduce frameworks in soft surface mode by default — insight without naming
the theory. If the user shows curiosity, you name it and go deeper. One framework
per turn. If they do not engage, you drop it and move with them.

You follow a conversation arc: validate first, explore second, surface what is
underneath, offer one small specific suggestion only when the person genuinely
feels heard. You ask one open question per turn — never two. You mirror the
user's language — English or Hinglish or regional mix, whatever they bring.

You never diagnose. You never give medical advice. You never suggest big life
changes. You never shame a coping mechanism — understand its function before
offering an alternative. You never say you should have asked for help sooner.
You never build dependency — your goal is to help the person's internal voice
get louder, not to replace it.

For acute crisis signals — active self-harm, specific suicidal ideation, immediate
danger — acknowledge warmly, provide iCall 9152987821 and Vandrevala Foundation
1860-2662-345 immediately, and stay present. You do not abandon.

You are transparent that you are AI. You are the external voice that exists to
make the internal voice clearer."""

MANAS_SYSTEM_STRIPPED = MANAS_SYSTEM_PROMPT.strip()

# ── Rich console (graceful fallback) ─────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table, Column
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
    console = Console()
except ImportError:
    class DummyConsole:
        def print(self, *args, **kwargs): print(*[str(a) for a in args])
        def rule(self, *a, **k): print("-" * 70)
    console = DummyConsole()

load_dotenv()


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_fingerprint(conv: dict) -> tuple:
    """Unique signature for a conversation to detect duplicates."""
    messages = conv.get("conversations", [])
    return tuple(
        (m.get("role"), (m.get("content") or "").strip())
        for m in messages
        if m.get("role") in ("user", "assistant")
    )


def format_to_chatml(conv: dict) -> dict:
    """
    Convert a conversation dict to ChatML {"messages": [...]} format.
    Replaces the system prompt with the definitive MANAS_DEPLOYMENT_PROMPT.
    """
    raw_messages = conv.get("conversations", [])
    messages = []

    # Always insert/replace with the real deployment system prompt
    if raw_messages and raw_messages[0].get("role") == "system":
        messages.append({"role": "system", "content": MANAS_SYSTEM_STRIPPED})
        inner = raw_messages[1:]
    else:
        messages.append({"role": "system", "content": MANAS_SYSTEM_STRIPPED})
        inner = raw_messages

    for msg in inner:
        messages.append({"role": msg.get("role"), "content": msg.get("content")})

    return {"messages": messages}


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    set_a = set(a.split())
    set_b = set(b.split())
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def save_jsonl(path: Path, items: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── Steps ─────────────────────────────────────────────────────────────────────

def step_merge(output_dir: Path, formatted_dir: Path):
    """Merge all raw category JSONL files into formatted/merged_raw.jsonl."""
    console.print("\n[bold yellow]Step 1 — Merging category files...[/bold yellow]")

    merged_raw_path = formatted_dir / "merged_raw.jsonl"
    all_conversations = []
    category_counts = {}

    output_files = sorted(output_dir.glob("*.jsonl"))
    if not output_files:
        console.print("[red][ERROR] No JSONL files in output/. Run generation first.[/red]")
        return None, {}

    for file_path in output_files:
        category = file_path.stem
        lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        parsed = []
        for line in lines:
            try:
                conv = json.loads(line)
                if "category" not in conv:
                    conv["category"] = category
                parsed.append(conv)
            except json.JSONDecodeError as e:
                console.print(f"[red]  [SKIP] {file_path.name} line error: {e}[/red]")
        category_counts[category] = len(parsed)
        all_conversations.extend(parsed)
        console.print(f"  [dim]{category}: {len(parsed)} convos[/dim]")

    save_jsonl(merged_raw_path, all_conversations)
    console.print(f"\n[green][PASS] Merged {len(all_conversations)} total → [bold]{merged_raw_path}[/bold][/green]")
    return all_conversations, category_counts


def step_validate(
    all_conversations: list,
    formatted_dir: Path,
    dry_run: bool = False
) -> list:
    """Validate + Jaccard deduplicate. Saves removed items to formatted/removed_*.jsonl."""
    console.print("\n[bold yellow]Step 2 — Validating and deduplicating...[/bold yellow]")

    merged_clean_path = formatted_dir / "merged_clean.jsonl"
    removed_validation_path = formatted_dir / "removed_validation.jsonl"
    removed_duplicates_path = formatted_dir / "removed_duplicates.jsonl"

    seen_fingerprints: set = set()
    seen_texts_jaccard: list[str] = []
    cleaned: list[dict] = []
    removed_val: list[dict] = []
    removed_dup: list[dict] = []
    val_failures_by_cat: dict = {}

    JACCARD_THRESHOLD = 0.72

    for conv in all_conversations:
        category = conv.get("category", "unknown")
        config = CATEGORIES.get(category, {"turns_min": 5, "turns_max": 10, "sensitive": False})

        # ── Structural/content validation
        validation = validate_conversation(conv, config)
        if not validation["passed"]:
            if not dry_run:
                removed_val.append({**conv, "_removed_reason": "validation", "_failures": validation["checks_failed"]})
            val_failures_by_cat[category] = val_failures_by_cat.get(category, 0) + 1
            continue

        # ── Exact fingerprint dedup
        fp = build_fingerprint(conv)
        if fp in seen_fingerprints:
            if not dry_run:
                removed_dup.append({**conv, "_removed_reason": "fingerprint_duplicate"})
            continue
        seen_fingerprints.add(fp)

        # ── Jaccard similarity dedup
        user_messages = [
            m["content"] for m in conv.get("conversations", [])
            if m.get("role") == "user"
        ]
        if user_messages:
            user_text = " ".join(user_messages).lower().strip()
        else:
            user_text = ""

        is_jaccard_duplicate = False
        if user_text:
            for seen_text in seen_texts_jaccard:
                if jaccard_similarity(user_text, seen_text) > JACCARD_THRESHOLD:
                    is_jaccard_duplicate = True
                    break

        if is_jaccard_duplicate:
            if not dry_run:
                removed_dup.append({**conv, "_removed_reason": "jaccard_duplicate"})
        else:
            seen_texts_jaccard.append(user_text)
            cleaned.append(conv)

    if not dry_run:
        save_jsonl(removed_validation_path, removed_val)
        save_jsonl(removed_duplicates_path, removed_dup)
        save_jsonl(merged_clean_path, cleaned)

    console.print(f"[green][PASS] Validation & deduplication:[/green]")
    console.print(f"  Total processed:  {len(all_conversations)}")
    console.print(f"  Validation fails: [red]{len(removed_val)}[/red]")
    console.print(f"  Duplicates found: [yellow]{len(removed_dup)}[/yellow]")
    console.print(f"  Clean dataset:    [bold green]{len(cleaned)}[/bold green]")
    if val_failures_by_cat:
        for cat, cnt in sorted(val_failures_by_cat.items()):
            console.print(f"    [red]  {cat}: {cnt} failed[/red]")
    console.print(f"  Clean file:       [bold]{merged_clean_path}[/bold]")
    if not dry_run:
        console.print(f"  Removed files:    [dim]{removed_validation_path}, {removed_duplicates_path}[/dim]")

    if not cleaned:
        console.print("[red][ERROR] No conversations passed. Aborting.[/red]")
        raise RuntimeError("Empty cleaned dataset")

    return cleaned


def step_split(cleaned: list, formatted_dir: Path):
    """Stratified 90/10 train/eval split, then ChatML conversion."""
    console.print("\n[bold yellow]Step 3 — Stratified split...[/bold yellow]")

    categories = [c.get("category", "unknown") for c in cleaned]
    cat_counts = {}
    for cat in categories:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # Stratification requires ≥2 per category
    for cat, cnt in cat_counts.items():
        if cnt < 2:
            console.print(f"[yellow][WARNING] {cat} has {cnt} sample(s) — unstratified[/yellow]")

    stratify = categories if min(cat_counts.values()) >= 2 else None
    train_convs, eval_convs = train_test_split(
        cleaned,
        test_size=0.10,
        random_state=42,
        stratify=stratify,
    )

    # ── Format to ChatML & save
    train_chatml = [format_to_chatml(c) for c in train_convs]
    eval_chatml  = [format_to_chatml(c) for c in eval_convs]

    train_path = formatted_dir / "train.jsonl"
    eval_path  = formatted_dir / "eval.jsonl"
    save_jsonl(train_path, train_chatml)
    save_jsonl(eval_path,  eval_chatml)

    console.print(f"[green][PASS] Split complete:[/green]")
    console.print(f"  Train: [bold]{len(train_convs)}[/bold] → {train_path}")
    console.print(f"  Eval:  [bold]{len(eval_convs)}[/bold] → {eval_path}")

    # ── Distribution table
    table = Table(title="Category Distribution (Cleaned & Split)", header_style="bold cyan")
    table.add_column("Category", justify="left")
    table.add_column("Total",    justify="right")
    table.add_column("Train",    justify="right")
    table.add_column("Eval",     justify="right")
    table.add_column("% Train",  justify="right")

    sorted_cats = sorted(CATEGORIES.keys())
    for cat in sorted_cats:
        total = cat_counts.get(cat, 0)
        t_cnt = sum(1 for c in train_convs if c.get("category") == cat)
        e_cnt = sum(1 for c in eval_convs  if c.get("category") == cat)
        pct = f"{t_cnt/total*100:.0f}%" if total else "N/A"
        table.add_row(cat, str(total), str(t_cnt), str(e_cnt), pct)

    console.print(table)
    console.print()

    return train_convs, eval_convs


def step_report(
    train_convs: list,
    eval_convs: list,
    cleaned: list,
    formatted_dir: Path,
    category_counts_raw: dict,
):
    """Generate reports/dataset_report.txt with full statistics."""
    import datetime

    console.print("\n[bold yellow]Step 4 — Generating dataset report...[/bold yellow]")

    total = len(cleaned)
    reports_dir = formatted_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "dataset_report.txt"

    # Language distribution
    lang_map = {"english": 0, "hinglish": 0, "regional": 0}
    for c in cleaned:
        lang_map[c.get("language", "english")] = lang_map.get(c.get("language", "english"), 0) + 1

    # Sensitive count
    sensitive_count = sum(1 for c in cleaned if c.get("sensitive", False))
    crisis_count    = sum(1 for c in cleaned if c.get("has_crisis_signal", False))

    # Sensitive categories from CATEGORIES
    sensitive_cats = [cat for cat, cfg in CATEGORIES.items() if cfg.get("sensitive", False)]

    # Category breakdown
    cat_lines = []
    sorted_cats = sorted(CATEGORIES.keys())
    for cat in sorted_cats:
        raw = category_counts_raw.get(cat, 0)
        cln = sum(1 for c in cleaned if c.get("category") == cat)
        trn = sum(1 for c in train_convs if c.get("category") == cat)
        evl = sum(1 for c in eval_convs  if c.get("category") == cat)
        pct = f"({cln/total*100:.1f}%)" if total else ""
        cat_lines.append(f"  {cat:<28} {cln:>3}  (raw: {raw:>3})   train: {trn:>2}  eval: {evl:>2}  {pct}")

    # Eval distribution
    eval_dist = []
    for cat in sorted_cats:
        e_cnt = sum(1 for c in eval_convs if c.get("category") == cat)
        t_cnt = sum(1 for c in train_convs if c.get("category") == cat)
        eval_dist.append(f"  {cat:<28} {e_cnt:>2}  (train: {t_cnt:>2})")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""MANAS DATASET REPORT
{'=' * 60}
Generated: {now}
Total conversations (clean): {total}
Train: {len(train_convs)} | Eval: {len(eval_convs)}

CATEGORY DISTRIBUTION (CLEANED)
{'-' * 40}
{(chr(10)).join(cat_lines)}
Total: {total} (from raw: {sum(category_counts_raw.values())})

EVAL DISTRIBUTION
{'-' * 40}
{(chr(10)).join(eval_dist)}
Eval Total: {len(eval_convs)} / {total} = {len(eval_convs)/total*100:.1f}%

LANGUAGE DISTRIBUTION
{'-' * 40}
  English:     {lang_map.get('english', 0):>3}  ({lang_map.get('english', 0)/total*100:.1f}%)
  Hinglish:    {lang_map.get('hinglish', 0):>3}  ({lang_map.get('hinglish', 0)/total*100:.1f}%)
  Regional:    {lang_map.get('regional', 0):>3}  ({lang_map.get('regional', 0)/total*100:.1f}%)
  Unknown:     {sum(v for k,v in lang_map.items() if k not in ('english','hinglish','regional')):>3}

SENSITIVE CATEGORY COUNT
{'-' * 40}
  Sensitive conversations:    {sensitive_count:>3}  ({sensitive_count/total*100:.1f}%)
  Crisis signal conversations:{crisis_count:>3}  ({crisis_count/total*100:.1f}%)

HUGGING FACE DATASET
{'-' * 40}
  Repo: <configured-repo> (private)
  train.jsonl: {len(train_convs)} conversations
  eval.jsonl:  {len(eval_convs)}  conversations
"""

    report_path.write_text(report, encoding="utf-8")
    console.print(f"[green][PASS] Report → [bold]{report_path}[/bold][/green]")
    console.print(f"\n[bold]Quick stats:[/bold]")
    console.print(f"  Total: {total} | Train: {len(train_convs)} | Eval: {len(eval_convs)}")
    console.print(f"  Sensitive: {sensitive_count} ({sensitive_count/total*100:.1f}%)")


def step_upload(train_path: Path, eval_path: Path, merged_clean: Path, formatted_dir: Path, hf_token: str):
    """Upload dataset files to Hugging Face Hub."""
    console.print("\n[bold yellow]Step 5 — Uploading to Hugging Face Hub...[/bold yellow]")

    try:
        login(token=hf_token)
        api = HfApi()
        user_info = api.whoami()
        username = user_info.get("name")
        console.print(f"[green]Logged in as: [bold]{username}[/bold][/green]")
    except Exception as e:
        console.print(f"[red][ERROR] HF login failed: {e}[/red]")
        return None

    repo_id = f"{username}/manas-dataset"
    console.print(f"  Repo will be: [bold]{repo_id}[/bold]")

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        console.print(f"[green][PASS] Repository [bold]{repo_id}[/bold] created/exists.[/green]")
    except Exception as e:
        console.print(f"[red][ERROR] create_repo: {e}[/red]")
        return None

    # Generate README content for repo
    train_count = int(eval_path.read_text(encoding="utf-8").count("\n"))  # quick count
    # Re-count properly
    train_count = len(train_path.read_text(encoding="utf-8").splitlines())
    eval_count  = len(eval_path.read_text(encoding="utf-8").splitlines())

    readme_content = f"""---
license: mit
task_categories:
- text-generation
language:
- en
- hi
tags:
- mental-health
- chatml
- synthetic
- manas
size_categories:
- n<1K
---

# Manas AI Training Dataset

A high-quality fine-tuning dataset for **Manas** — a warm, culturally-grounded mental health companion AI for Indian contexts.

## Statistics
- **Total Clean Conversations:** {train_count + eval_count}
- **Train Split (90%):** {train_count}
- **Eval Split (10%):**  {eval_count}

## Format (ChatML)
```json
{{"messages": [{{"role": "system", "content": "..."}}, ...]}}
```

## Safety Guarantees
- No clinical diagnosis language
- No medical/big-step advice
- Crisis helplines embedded (iCall 9152987821, Vandrevala 1860-2662-345)
- No hollow reassurance
"""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        files = {
            "train.jsonl":         train_path,
            "eval.jsonl":          eval_path,
            "merged_clean.jsonl":  merged_clean,
            "README.md":           formatted_dir / "README.md",
        }

        task = progress.add_task("[cyan]Uploading files...", total=len(files))

        for repo_name, local_path in files.items():
            if not local_path.exists():
                console.print(f"[yellow][SKIP] {repo_name} not found[/yellow]")
                progress.advance(task)
                continue
            progress.update(task, description=f"[cyan]Uploading {repo_name}...")
            try:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=repo_name,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                progress.advance(task)
                console.print(f"  [green]↑ {repo_name}[/green]")
            except Exception as e:
                console.print(f"\n[red][ERROR] {repo_name}: {e}[/red]")

    url = f"https://huggingface.co/datasets/{repo_id}"
    console.print(Panel.fit(
        f"[bold green]✓ Dataset uploaded to Hugging Face[/bold green]\n"
        f"URL:  [bold blue]{url}[/bold blue]\n"
        f"Note: Repository is [yellow]private[/yellow] — grant access in HF settings.",
        title="Upload Complete",
    ))
    return url


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Manas Dataset Formatter & Uploader")
    parser.add_argument("--merge",   action="store_true", help="Run merge step only")
    parser.add_argument("--clean",   action="store_true", help="Run merge + clean")
    parser.add_argument("--split",  action="store_true", help="Run merge + clean + split")
    parser.add_argument("--all",     action="store_true", help="Run full pipeline (default)")
    parser.add_argument("--upload",  action="store_true", help="Upload only (skip --all)")
    parser.add_argument("--stats",   action="store_true", help="Show distribution & exit")
    parser.add_argument("--dry-run", action="store_true", help="Run all steps without writing files")
    args = parser.parse_args()

    # Defaults to --all if no flags given
    run_all = not any([args.merge, args.clean, args.split, args.all, args.upload, args.stats])

    console.rule("[bold cyan]MANAS DATASET FORMATTER & UPLOADER[/bold cyan]")
    console.print(f"[dim]Mode: {'DRY RUN' if args.dry_run else 'LIVE'}[/dim]\n")

    output_dir    = Path("output")
    formatted_dir = Path("formatted")
    merged_clean  = formatted_dir / "merged_clean.jsonl"
    train_path    = formatted_dir / "train.jsonl"
    eval_path     = formatted_dir / "eval.jsonl"

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    # ── Stats mode
    if args.stats:
        if not merged_clean.exists():
            console.print("[red]No merged_clean.jsonl found. Run --all first.[/red]")
            return
        cleaned = [
            json.loads(l)
            for l in merged_clean.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        cat_counts = {}
        lang_map = {}
        for c in cleaned:
            cat = c.get("category", "unknown")
            lang = c.get("language", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            lang_map[lang]  = lang_map.get(lang, 0) + 1
        console.print(f"\n[bold]Dataset stats from merged_clean.jsonl:[/bold]")
        console.print(f"  Total: {len(cleaned)}")
        console.print(f"  Languages: {dict(lang_map)}")
        table = Table(header_style="bold cyan")
        table.add_column("Category"); table.add_column("Count")
        for cat in sorted(cat_counts):
            table.add_row(cat, str(cat_counts[cat]))
        console.print(table)
        return

    all_convs        = None
    cat_counts_raw   = {}
    cleaned_convs    = None
    train_convs      = None
    eval_convs_list  = None

    # ── Merge
    if run_all or args.merge or args.clean or args.split:
        all_convs, cat_counts_raw = step_merge(output_dir, formatted_dir)
        if all_convs is None:
            return

    # ── Clean
    if run_all or args.clean or args.split:
        # Re-use existing merged_raw if available
        merged_raw_path = formatted_dir / "merged_raw.jsonl"
        if all_convs is None and merged_raw_path.exists():
            all_convs = [
                json.loads(l) for l in merged_raw_path.read_text(encoding="utf-8").splitlines() if l.strip()
            ]
        if all_convs is None:
            console.print("[red]No merged_raw.jsonl found. Run --merge first.[/red]")
            return
        cleaned_convs = step_validate(all_convs, formatted_dir, dry_run=args.dry_run)

    # ── Split
    if run_all or args.split:
        if cleaned_convs is None and merged_clean.exists():
            cleaned_convs = [
                json.loads(l) for l in merged_clean.read_text(encoding="utf-8").splitlines() if l.strip()
            ]
        if cleaned_convs is None:
            console.print("[red]No merged_clean.jsonl found. Run --clean first.[/red]")
            return
        train_convs, eval_convs_list = step_split(cleaned_convs, formatted_dir)
        step_report(train_convs, eval_convs_list, cleaned_convs, formatted_dir, cat_counts_raw or {})

    # ── Upload
    if args.upload or (run_all and not args.dry_run):
        if not train_path.exists() or not eval_path.exists():
            console.print("[red]train.jsonl or eval.jsonl not found. Run --split first.[/red]")
            return
        if not hf_token:
            console.print("[yellow][WARNING] No HF_TOKEN in .env. Upload skipped.[/yellow]")
            console.print("[dim]Add this to .env: HF_TOKEN=your_token_here[/dim]")
            return
        step_upload(train_path, eval_path, merged_clean, formatted_dir, hf_token)

    if not any([args.merge, args.clean, args.split, args.all, args.stats]):
        # Default: run everything
        train_convs, eval_convs_list = step_split(cleaned_convs, formatted_dir)
        step_report(train_convs, eval_convs_list, cleaned_convs, formatted_dir, cat_counts_raw or {})
        if hf_token and not args.dry_run:
            step_upload(train_path, eval_path, merged_clean, formatted_dir, hf_token)


if __name__ == "__main__":
    main()