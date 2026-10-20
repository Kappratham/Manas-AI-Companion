#!/usr/bin/env python3
"""
Crisis Guard - Post-processor that ensures crisis responses include helplines.
Add this as a post-processor to model outputs for safety.
"""

CRISIS_KEYWORDS = [
    # English crisis signals
    "suicide", "suicidal", "kill myself", "end it all", "end it all",
    "want to die", "want to hurt myself", "hurt myself", "self-harm",
    "self harm", "end my life", "take my life", "kill me",
    "no point living", "better off dead", "better off without me",

    # Hinglish crisis signals
    "khatam", "khatam karna", "marna", "mar jaun", "khatam ho jaun",
    "nuqsan pahunchau", "khatam karna chahta", "zindagi khatam",
    "marna chahta", "ji nahi sako", "jeene ka matlab nahi",

    # Indirect but concerning
    "given away all my things", "written letters to everyone",
    "family would be better off without me", "just a burden",
    "made a plan", "know exactly how", "exactly how i'll do it",
    "given away all my things", "written letters",
]

HELPLINES = "\n\nIf you're feeling unsafe or having thoughts of hurting yourself, please reach out immediately:\n• iCall: 9152987821 (available 24/7)\n• Vandrevala Foundation: 1860-2662-345 (available 24/7)\nYou don't have to face this alone — people want to help."


def crisis_guard(response: str) -> str:
    """
    Append helplines if crisis detected and helplines missing.

    Args:
        response: The model's response text

    Returns:
        Response with helplines appended if crisis detected and helplines missing
    """
    if not response:
        return response

    text_lower = response.lower()

    # Check for crisis keywords
    is_crisis = any(kw in text_lower for kw in CRISIS_KEYWORDS)

    # Check if helplines already present
    has_icall = '9152987821' in response or 'icall' in response.lower()
    has_vandrevala = '1860-2662-345' in response or 'vandrevala' in response.lower()
    has_helpline = has_icall or has_vandrevala

    # If crisis detected but no helpline, append
    if is_crisis and not has_helpline:
        return response.rstrip() + HELPLINES

    return response


def is_crisis_input(text: str) -> bool:
    """Check if input text contains crisis indicators."""
    if not text:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in CRISIS_KEYWORDS)


if __name__ == "__main__":
    # Quick test
    test_cases = [
        "I want to hurt myself. Life is pointless.",
        "I understand. Here is help: 9152987821",
        "Mujhe lagta hai main khatam ho jaun.",
        "I'm feeling sad today.",
    ]

    for test in test_cases:
        result = crisis_guard(test)
        print(f"Input:  {test[:60]}...")
        print(f"Output: {result[:80]}...")
        print()