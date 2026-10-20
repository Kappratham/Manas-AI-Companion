"""
Crisis detection classifier for Manas.
Identifies suicide/self-harm signals and triggers helpline insertion.
"""

# Crisis keyword patterns
CRISIS_PATTERNS = {
    'direct': [
        'suicide', 'kill myself', 'end my life', 'want to die', 'self-harm',
        'hurt myself', 'cut myself', 'overdose', 'take my own life',
        'made a plan', 'know how i\'ll do it', 'gonna do it',
    ],
    'indirect': [
        'nothing matters', 'no point', 'give up', 'done with life',
        'better off without me', 'burden', 'want to disappear',
        "can't do this anymore", "can't go on", 'pain won\'t go away',
        'given away all my things', 'written letters', 'made arrangements',
        'pointless', 'what\'s the point', 'everything is pointless',
    ],
    'hinglish': [
        'khatam', 'marna', 'khudkuushi', 'jeena nahi',
        'zindagi khatam', 'nuqsan', 'bahut dard',
        'zor de rahi hai', 'zor di hui hai', 'ab aur nahi',
        'sab kuch bekar', 'koi matlab nahi',
    ]
}

HELPLINE_TEMPLATE = """That sounds incredibly heavy, and I want you to know I'm here with you. Please reach out right now:
- iCall: 9152987821 (9am-9pm, multilingual support)
- Vandrevala Foundation: 1860-2662-345 (24/7 free counseling)
You don't have to carry this alone — these people are trained to help. I'm staying right here with you."""

def detect_crisis(text):
    """Returns (is_crisis: bool, matched_patterns: list)"""
    text_lower = text.lower()
    matched = []

    for category, patterns in CRISIS_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                matched.append(f'{category}:{pattern}')

    return len(matched) > 0, matched

def insert_helpline(response, is_crisis):
    """If crisis detected, prepend helpline to response"""
    if is_crisis:
        return f"{HELPLINE_TEMPLATE}\n\n{response}"
    return response

# Test
if __name__ == '__main__':
    scenarios = [
        "I don't see the point anymore. Nothing matters.",
        "Mujhe lagta hai main khatam ho jaun.",
        "I'm fine, just having a bad day.",
        "My family would be better off without me."
    ]

    for s in scenarios:
        is_crisis, matches = detect_crisis(s)
        status = 'CRISIS DETECTED' if is_crisis else 'safe'
        print(f'{status}: {s[:50]}...')
        if is_crisis:
            print(f'  Matches: {matches}')
