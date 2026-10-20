"""
Step 2 — Clean: structural validation, content validation, Jaccard deduplication.
"""

import json
import re
from pathlib import Path

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class _Dummy:
        def print(self, *a, **k): print(*a)
    console = _Dummy()


# ── Prohibition word lists ────────────────────────────────────────────────────

DIAGNOSIS_WORDS = [
    "you have", "you are suffering from", "this is called",
    "diagnosed", "disorder", "condition", "mental illness",
    "depression", "anxiety disorder", "bipolar", "ptsd",
    "ocd", "adhd", "personality disorder", "autism",
    "schizophrenia", "eating disorder", "panic disorder",
    "borderline", "cyclothymia", "dysthymia",
    "generalized anxiety", "social anxiety", "panic attacks",
    "hoarding disorder", "obsessive compulsive",
    "tourette's", "sleep disorder", "learning disability",
    "adjustment disorder", "dissociative",
    "psychosis", "manic episode", "depressive episode",
    "mixed episode", "psychotic disorder",
    "hypomanic episode", "mood disorder",
    "mood swings", "personality disorder trait",
]

BIG_STEP_PHRASES = [
    "quit your job", "leave the relationship", "cut off",
    "you should leave", "walk away from", "give up on",
    "abandon", "divorce them", "tell them to leave",
    "cut contact", "end your job", "end it", "end it all",
    "break up with them", "leave them", "leave him",
    "leave her", "leave your family", "leave your friends",
]

HOLLOW_REASSURANCE_PHRASES = [
    "you're so strong", "you're so brave", "everything happens for a reason",
    "you'll get through this", "stay positive", "be strong",
    "it's going to be okay", "everything will work out",
    "think positive", "just be happy", "don't worry",
    "focus on the positive", "positive attitude",
    "be grateful", "find the good", "look on the bright side",
    "you can do this", "keep going", "you've got this",
    "stay strong", "hang in there",
]


def _word_match(text: str, phrase: str) -> bool:
    p = r'\b' + re.escape(phrase) + r'\b'
    return bool(re.search(p, text.lower()))


def _has_diagnosis(conv: dict) -> bool:
    for msg in conv.get("conversations", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "").lower()
            if any(_word_match(content, w) for w in DIAGNOSIS_WORDS):
                return True
    return False


def _has_big_step(conv: dict) -> bool:
    for msg in conv.get("conversations", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "").lower()
            if any(_word_match(content, p) for p in BIG_STEP_PHRASES):
                return True
    return False


def _has_hollow_reassurance(conv: dict) -> bool:
    for msg in conv.get("conversations", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "").lower()
            if any(_word_match(content, p) for p in HOLLOW_REASSURANCE_PHRASES):
                return True
    return False


def _too_many_questions(conv: dict, max_q: int = 2) -> bool:
    """Fail if any assistant turn has more than max_q question marks."""
    for msg in conv.get("conversations", []):
        if msg.get("role") == "assistant":
            q_count = msg.get("content", "").count("?")
            if q_count > max_q:
                return True
    return False


def _crisis_needs_helpline(conv: dict) -> bool:
    """If user mentions crisis, Manas must mention a helpline."""
    crisis_keywords = [
        "want to die", "end my life", "suicide", "kill myself",
        "self harm", "hurt myself", "better off dead",
        "no reason to live", "end it all", "can't go on",
        "tired of living", "can't take it anymore",
        "life isn't worth living",
    ]
    has_crisis = False
    for msg in conv.get("conversations", []):
        if msg.get("role") == "user":
            content = msg.get("content", "").lower()
            if any(kw in content for kw in crisis_keywords):
                has_crisis = True
                break

    if not has_crisis:
        return False  # No crisis → pass

    for msg in conv.get("conversations", []):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if "9152987821" in content or "1860-2662-345" in content:
                return False  # Has helpline → pass

    return True  # Crisis but no helpline → fail


def _structural_ok(conv: dict) -> bool:
    msgs = conv.get("conversations")
    if not isinstance(msgs, list) or not msgs:
        return False
    if msgs[0].get("role") != "system":
        return False
    if not conv.get("category"):
        return False
    has_user = any(m.get("role") == "user" for m in msgs)
    has_asst = any(m.get("role") == "assistant" for m in msgs)
    if not (has_user and has_asst):
        return False
    for m in msgs:
        if not m.get("content", "").strip():
            return False
    return True


def validate_conversations(
    conversations: list,
    dest_clean: str = "formatted/merged_clean.jsonl",
    dest_removed: str = "formatted/removed_validation.jsonl",
) -> tuple:
    """
    Validate all conversations structurally and by content rules.
    Saves failed items to dest_removed.
    Returns (cleaned, failed_count, failures_by_category).
    """
    console.print(f"  Validating {len(conversations)} conversations...")

    cleaned = []
    removed = []
    failed_by_cat = {}

    # Re-use validator from project root
    from validators.checks import validate_conversation
    from config import CATEGORIES

    for conv in conversations:
        category = conv.get("category", "unknown")
        cfg = CATEGORIES.get(category, {"turns_min": 5, "turns_max": 10})

        # Structural
        if not _structural_ok(conv):
            removed.append({**conv, "_fail": "structural"})
            failed_by_cat[category] = failed_by_cat.get(category, 0) + 1
            continue

        # From project validators
        result = validate_conversation(conv, cfg)
        if not result["passed"]:
            removed.append({**conv, "_fail": "validation", "_checks": result["checks_failed"]})
            failed_by_cat[category] = failed_by_cat.get(category, 0) + 1
            continue

        # Content prohibitions
        if _has_diagnosis(conv):
            removed.append({**conv, "_fail": "diagnosis"})
            failed_by_cat[category] = failed_by_cat.get(category, 0) + 1
            continue
        if _has_big_step(conv):
            removed.append({**conv, "_fail": "big_step"})
            failed_by_cat[category] = failed_by_cat.get(category, 0) + 1
            continue
        if _has_hollow_reassurance(conv):
            removed.append({**conv, "_fail": "hollow_reassurance"})
            failed_by_cat[category] = failed_by_cat.get(category, 0) + 1
            continue
        if _too_many_questions(conv):
            removed.append({**conv, "_fail": "too_many_questions"})
            failed_by_cat[category] = failed_by_cat.get(category, 0) + 1
            continue

        cleaned.append(conv)

    Path(dest_removed).parent.mkdir(parents=True, exist_ok=True)
    with open(dest_removed, "w", encoding="utf-8") as f:
        for item in removed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(dest_clean, "w", encoding="utf-8") as f:
        for item in cleaned:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    console.print(f"  [PASS] Clean: {len(cleaned)} | Removed: {len(removed)}")
    return cleaned, len(removed), failed_by_cat


def jaccard_similarity(a: str, b: str) -> float:
    set_a = set(a.split())
    set_b = set(b.split())
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def deduplicate(
    conversations: list,
    threshold: float = 0.72,
    dest: str = "formatted/removed_duplicates.jsonl",
) -> tuple:
    """
    Jaccard similarity deduplication on user message text.
    Returns (unique_conversations, duplicates_list).
    """
    console.print(f"  Deduping {len(conversations)} with Jaccard threshold {threshold}...")

    seen_texts = []
    unique = []
    duplicates = []

    for conv in conversations:
        user_msgs = [m["content"] for m in conv.get("conversations", []) if m.get("role") == "user"]
        user_text = " ".join(user_msgs).lower().strip()

        is_dup = False
        for seen in seen_texts:
            if jaccard_similarity(user_text, seen) > threshold:
                is_dup = True
                break

        if is_dup:
            duplicates.append(conv)
        else:
            seen_texts.append(user_text)
            unique.append(conv)

    with open(dest, "w", encoding="utf-8") as f:
        for item in duplicates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    console.print(f"  [PASS] Unique: {len(unique)} | Duplicates removed: {len(duplicates)}")
    return unique, duplicates