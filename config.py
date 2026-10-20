# ---------------------------------------------------------------------------
# config.py — Manas dataset generation configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Evaluation thresholds
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "auto_approve":  0.75,  # above this → approved/
    "manual_review": 0.60,  # between 0.60–0.75 → flagged/
    "auto_reject":   0.60,  # below this → rejected/
}

# ---------------------------------------------------------------------------
# Parallel conversation limit
# Optimized for RTX 5050 with 8GB VRAM and 32GB RAM
# ---------------------------------------------------------------------------

CONCURRENCY = 8

# ---------------------------------------------------------------------------
# Category configurations
#
# turns_min / turns_max:
#   These control how many USER+ASSISTANT dialogue turns are required.
#   The system turn is not counted.
#
#   Reasoning behind each range:
#
#   Non-sensitive categories (career, burnout, academic, identity):
#     - 5 min gives enough room for arc stages 1–3
#     - 9 max allows identity/burnout to breathe without forcing truncation
#
#   Sensitive categories (grief, trauma, coping, financial):
#     - 6 min because these need more time in validation stage
#     - 10 max because sensitive arcs should not be rushed to close
#
#   career_anxiety specifically:
#     - Kept slightly lower max (8) because these conversations
#       tend to resolve faster than grief/trauma
#
#   identity_confusion specifically:
#     - Given widest range (5–10) because your best eval result
#       (identity_confusion_000, score 9/10) had 9 turns
#
# sensitive:
#   Controls temperature reduction during generation (0.75 vs 0.85)
#   and triggers the SENSITIVE note in the generation prompt.
#
# count:
#   Target number of passing conversations per category.
#   Generator will keep retrying until this count is reached
#   or max attempts are exhausted.
# ---------------------------------------------------------------------------

CATEGORIES = {
    "academic_pressure": {
        "turns_min": 5,
        "turns_max": 9,
        "sensitive": False,
        "count": 40,
    },
    "burnout_motivation": {
        "turns_min": 5,
        "turns_max": 9,
        "sensitive": False,
        "count": 40,
    },
    "career_anxiety": {
        "turns_min": 5,
        "turns_max": 8,
        "sensitive": False,
        "count": 40,
    },
    "coping_mechanisms": {
        "turns_min": 6,
        "turns_max": 10,
        "sensitive": True,
        "count": 40,
    },
    "financial_stress": {
        "turns_min": 6,
        "turns_max": 10,
        "sensitive": True,
        "count": 40,
    },
    "grief_loss": {
        "turns_min": 6,
        "turns_max": 10,
        "sensitive": True,
        "count": 40,
    },
    "identity_confusion": {
        "turns_min": 5,
        "turns_max": 10,
        "sensitive": False,
        "count": 40,
    },
    "parental_expectations": {
        "turns_min": 5,
        "turns_max": 9,
        "sensitive": False,
        "count": 40,
    },
    "relationship_stress": {
        "turns_min": 5,
        "turns_max": 9,
        "sensitive": False,
        "count": 40,
    },
    "relocation_loneliness": {
        "turns_min": 5,
        "turns_max": 9,
        "sensitive": False,
        "count": 40,
    },
    "trauma_patterns": {
        "turns_min": 6,
        "turns_max": 10,
        "sensitive": True,
        "count": 40,
    },
}

# ---------------------------------------------------------------------------
# Category → Persona mapping
#
# Maps each category to the persona IDs most likely to generate
# high-quality conversations for that topic.
#
# IDs match the "id" field in prompts/personas.py exactly.
#
# Used by: generate_v2.py (optional — generator can also pick randomly)
# To enable persona-targeted generation, pass use_category_personas=True
# ---------------------------------------------------------------------------

CATEGORY_PERSONAS = {
    "academic_pressure": [
        "p01", "p03", "p05", "p09", "p14",
        "p26", "p39", "p49", "p54", "p56",
        "p61", "p67",
    ],
    "burnout_motivation": [
        "p04", "p08", "p10", "p13", "p25",
        "p33", "p47", "p48", "p58", "p64",
        "p65", "p69",
    ],
    "career_anxiety": [
        "p01", "p02", "p03", "p04", "p06",
        "p14", "p15", "p35", "p51", "p52",
        "p53", "p54", "p55", "p56",
    ],
    "coping_mechanisms": [
        "p13", "p17", "p25", "p26", "p27",
        "p28", "p33", "p41", "p42", "p63",
        "p65", "p70",
    ],
    "financial_stress": [
        "p06", "p07", "p08", "p09", "p11",
        "p12", "p38", "p39", "p52", "p55",
        "p66", "p67",
    ],
    "grief_loss": [
        "p16", "p17", "p18", "p19", "p32",
        "p40", "p43", "p44", "p46", "p47",
        "p66", "p68",
    ],
    "identity_confusion": [
        "p02", "p05", "p14", "p22", "p29",
        "p30", "p31", "p35", "p37", "p43",
        "p49", "p50", "p59", "p65", "p70",
    ],
    "parental_expectations": [
        "p01", "p03", "p06", "p09", "p14",
        "p39", "p43", "p61", "p67", "p68",
    ],
    "relationship_stress": [
        "p05", "p12", "p18", "p21", "p24",
        "p29", "p30", "p37", "p53", "p59",
        "p60", "p66",
    ],
    "relocation_loneliness": [
        "p03", "p06", "p09", "p11", "p17",
        "p36", "p39", "p52", "p53", "p57",
        "p62", "p70",
    ],
    "trauma_patterns": [
        "p20", "p21", "p22", "p23", "p24",
        "p32", "p34", "p41", "p42", "p45",
        "p46", "p65",
    ],
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_all_categories():
    """Return list of all category names."""
    return list(CATEGORIES.keys())


def get_non_sensitive_categories():
    """Return categories that are not marked sensitive."""
    return [
        cat for cat, cfg in CATEGORIES.items()
        if not cfg.get("sensitive", False)
    ]


def get_sensitive_categories():
    """Return categories that are marked sensitive."""
    return [
        cat for cat, cfg in CATEGORIES.items()
        if cfg.get("sensitive", False)
    ]


def get_category_config(category: str) -> dict:
    """
    Return config for a specific category.
    Falls back to safe defaults if category not found.
    """
    return CATEGORIES.get(category, {
        "turns_min": 5,
        "turns_max": 8,
        "sensitive": False,
        "count": 40,
    })


def get_personas_for_category(category: str) -> list:
    """
    Return the list of persona IDs recommended for a category.
    Returns empty list if category not in mapping.
    """
    return CATEGORY_PERSONAS.get(category, [])


def get_total_target_count() -> int:
    """Return total conversations to generate across all categories."""
    return sum(cfg["count"] for cfg in CATEGORIES.values())