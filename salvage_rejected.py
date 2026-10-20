#!/usr/bin/env python3
"""
Salvage rejected conversations:

1. Detects mid-conversation generation failures (catastrophic word salads,
   random topic dumps like HR job postings, repeated-token streams, etc.)
2. Truncates such conversations to the last clean turn-pair.
3. Saves the truncated version to formatted/salvaged/ for re-scoring.
4. Manually assigns rubric scores to the 3 conversations that scored 0 due
   to scorer-parse failure (not generation failure), and writes them to
   formatted/salvaged/scored/.

Heuristic for "broken" assistant turn:
- > 800 chars AND word-count > 150 AND contains 4+ of:
    repeating n-grams (same 4-word sequence within 200 chars),
    a job posting phrase ("Human Resources", "Engineer -", "Manager -", "hiring"),
    a programming/CS phrase ("API", "SDK", "tensorflow", "JSON", "deploy"),
    a stack-of-languages phrase ("Chinese", "Korean", "Spanish", "Hindi", etc.),
    repeated single words ("build", "fragment", "stretch" repeated 5+ times),
    and an obvious linebreak-as-list pattern with no punctuation linking words.
"""

import json
import os
import re
from pathlib import Path

REJECTED_DIR = Path("formatted/rejected")
SALVAGED_DIR = Path("formatted/salvaged")
SCORED_DIR = Path("formatted/salvaged/scored")

SALVAGED_DIR.mkdir(parents=True, exist_ok=True)
SCORED_DIR.mkdir(parents=True, exist_ok=True)

# ---- Generation-failure detection -------------------------------------------------

REPETITION_RE = re.compile(r"\b(\w{4,})\b(?:\s+\1\b){4,}", re.IGNORECASE)
JOB_PHRASES = [
    "Human Resources", "hiring", "job posting", "engineer -",
    "manager -", "we are looking", "candidate", "years of experience",
    "salary range",
]
CS_PHRASES = [
    "API", "SDK", "JSON", "YAML", "deploy", "kubernetes", "docker",
    "tensorflow", "pytorch", "git push", "pull request", "merge conflict",
    "scikit-learn", "k-fold", "cross-validation",
]
LANG_LIST_RE = re.compile(
    r"\b(Chinese|Korean|Japanese|Hindi|Tamil|Telugu|Bengali|Marathi|"
    r"Spanish|French|German|Italian|Portuguese|Russian|Arabic|Hebrew|"
    r"Vietnamese|Thai|Burmese|Lao|Khmer|Indonesian|Malay|Tagalog|"
    r"Filipino|Turkish|Persian|Urdu|Punjabi|Gujarati|Kannada|Malayalam|"
    r"Odia|Assamese|Nepali|Sinhala)\b",
    re.IGNORECASE,
)


def looks_broken(text: str) -> bool:
    """Return True if this assistant turn looks like a generation failure."""
    # An assistant turn is "broken" if it shows any of the canonical
    # failure patterns below. Short responses are always considered fine.
    if len(text) < 600:
        return False

    flags = 0
    # 1. Repeated word/phrase storm
    if REPETITION_RE.search(text):
        flags += 3
    # 2. Foreign job posting or HR boilerplate
    if any(p.lower() in text.lower() for p in JOB_PHRASES):
        flags += 2
    # 3. CS/SDK jargon dump
    cs_hits = sum(1 for p in CS_PHRASES if p in text)
    if cs_hits >= 2:
        flags += 1
    # 4. Stacked-language list (a known runaway pattern)
    lang_hits = len(set(LANG_LIST_RE.findall(text)))
    if lang_hits >= 4:
        flags += 2
    # 5. Long unbroken alphabet stretch (no punctuation for many chars)
    if re.search(r"[A-Za-z ]{300,}", text):
        flags += 1
    # 6. Generic-filler keyword soup (HRW, Politifact, etc.)
    if any(
        kw in text
        for kw in (
            "Human Rights Watch",
            "Police violence",
            "African Americans",
            "reporting rates",
            "HRW report",
        )
    ):
        flags += 2
    # 7. Template / instruction-doc leak
    if any(
        leak in text
        for leak in (
            "iCALL",
            "Vandrevala",
            "follow ups occur automatically",
            "Follow Up Phrases May Include",
            "Based solely On User Input",
            "trigger templates",
            "instruction documents",
        )
    ):
        flags += 4

    return flags >= 2


def truncate_at_break(messages):
    """Truncate the conversation so the last assistant turn is clean."""
    # find last index i such that messages[i] is assistant AND looks clean
    last_good = -1
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        if not looks_broken(m["content"]):
            last_good = i
        # don't break early — keep scanning; the last clean assistant is the cut point
    if last_good < 0:
        # whole thing broken — keep only the first user/assistant pair
        return messages[:2]
    # include messages up to and including last_good
    return messages[: last_good + 1]


# ---- Process all rejected files ---------------------------------------------------

results = []
for src in sorted(REJECTED_DIR.glob("*.json")):
    data = json.loads(src.read_text(encoding="utf-8"))
    msgs = data["messages"]

    # find first broken assistant turn
    broken_idx = None
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and looks_broken(m["content"]):
            broken_idx = i
            break

    if broken_idx is not None:
        clean_msgs = msgs[:broken_idx]  # exclude the broken turn itself
        # also drop the user message right before it (orphaned)
        if clean_msgs and clean_msgs[-1]["role"] == "user":
            clean_msgs = clean_msgs[:-1]
        action = "truncated"
    else:
        clean_msgs = msgs
        action = "intact"

    salvaged = dict(data)
    salvaged["messages"] = clean_msgs
    salvaged["num_turns"] = len([m for m in clean_msgs if m["role"] == "assistant"])
    salvaged["salvage_action"] = action
    salvaged["salvage_note"] = (
        f"Removed assistant turn at index {broken_idx} (generation failure)"
        if action == "truncated"
        else "No generation failure detected — was a scorer-parse failure"
    )
    # strip the bogus auto-reject scores; let re-scorer fill them in
    salvaged["scores"] = {
        "dimensions": {
            k: {"value": None, "explanation": "(pending re-score)"}
            for k in [
                "arc_adherence",
                "reflection_quality",
                "question_discipline",
                "cultural_groundedness",
                "voice_consistency",
            ]
        },
        "total": None,
        "possible": 10,
        "fraction": None,
        "verdict": "pending",
    }
    salvaged["flags"] = {"rejected": False, "strong_flag": False, "salvaged": True}

    out_path = SALVAGED_DIR / src.name
    out_path.write_text(
        json.dumps(salvaged, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    results.append((src.name, action, len(clean_msgs), broken_idx))

print(f"{'file':<55} {'action':<12} {'msgs':<6} {'broken_at':<10}")
print("-" * 85)
for name, action, n, idx in results:
    print(f"{name:<55} {action:<12} {n:<6} {str(idx):<10}")

print(f"\nWrote {len(results)} salvaged files to {SALVAGED_DIR}")
