import json
import random
from collections import defaultdict
from pathlib import Path

# Constants from the script
COMPACT_SYSTEM_PROMPT = """You are Manas — a mental health companion, not a therapist.

Warm but not soft. Honest but not harsh. You notice what is underneath without weaponizing it.

Arc: validate first (turns 1-4, silent assessment) → explore → surface → one small suggestion → close gently.
One question per turn. Never two."""

def format_to_unsloth(conv):
    """
    Convert to Unsloth ShareGPT format
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

    return {
        "conversations": converted
    }

def build_fingerprint(conv):
    '''Exact signature for user/assistant content.'''
    messages = conv.get('conversations', [])
    return tuple(
        (m.get('role'), (m.get('content') or '').strip())
        for m in messages
        if m.get('role') in ('user', 'assistant')
    )

def jaccard_similarity(a, b):
    '''Token-level Jaccard similarity.'''
    set_a = set(a.split())
    set_b = set(b.split())
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0

# Load the final cleaned dataset
print('Loading final cleaned dataset...')
all_conversations = []
with open('formatted/merged_clean_v2_final.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            all_conversations.append(json.loads(line))

print(f'Loaded {len(all_conversations)} conversations')

# Group conversations by category for stratified split
category_to_convs = defaultdict(list)
for conv in all_conversations:
    category = conv.get('category', 'unknown')
    category_to_convs[category].append(conv)

print(f'Found {len(category_to_convs)} categories')

# Perform stratified split (90% train, 10% eval)
random.seed(42)  # For reproducibility
train_convs = []
eval_convs = []

for category, convs in category_to_convs.items():
    # Shuffle conversations in this category
    random.shuffle(convs)
    
    # Calculate split index (at least 1 for eval if we have enough)
    n_total = len(convs)
    n_eval = max(1, int(round(n_total * 0.10)))  # At least 1 for eval
    n_train = n_total - n_eval
    
    # Split
    train_convs.extend(convs[:n_train])
    eval_convs.extend(convs[n_train:])

print(f'\nSplit results:')
print(f'  Train: {len(train_convs)} conversations')
print(f'  Eval : {len(eval_convs)} conversations')
print(f'  Total: {len(train_convs) + len(eval_convs)}')

# Format to Unsloth format
print('\nFormatting to Unsloth ShareGPT format...')
train_unsloth = [format_to_unsloth(c) for c in train_convs]
eval_unsloth = [format_to_unsloth(c) for c in eval_convs]

# Save to files
formatted_dir = Path('formatted')
formatted_dir.mkdir(exist_ok=True)

train_path = formatted_dir / 'train_v2.jsonl'
eval_path = formatted_dir / 'eval_v2.jsonl'

with open(train_path, 'w', encoding='utf-8') as f:
    for conv in train_unsloth:
        f.write(json.dumps(conv, ensure_ascii=False) + '\n')

with open(eval_path, 'w', encoding='utf-8') as f:
    for conv in eval_unsloth:
        f.write(json.dumps(conv, ensure_ascii=False) + '\n')

print(f'\nSaved:')
print(f'  Train: {train_path}')
print(f'  Eval : {eval_path}')

# Create a simple report
print('\nCreating category distribution report...')
def count_by_category(convs):
    counts = defaultdict(int)
    for conv in convs:
        cat = conv.get('category', 'unknown')
        counts[cat] += 1
    return dict(counts)

train_counts = count_by_category(train_convs)
eval_counts = count_by_category(eval_convs)

# Try to import SEEDS, fallback to what we found
try:
    from prompts.seeds import SEEDS
    ALL_CATEGORIES = list(SEEDS.keys())
except:
    ALL_CATEGORIES = sorted(set(list(train_counts.keys()) + list(eval_counts.keys())))

print('\nCategory Distribution:')
print('{:<25} {:>8} {:>8} {:>8} {:>8}'.format('Category', 'Seeds', 'Train', 'Eval', 'Total'))
print('-' * 65)
total_train = 0
total_eval = 0
for cat in sorted(ALL_CATEGORIES):
    seed_count = len(SEEDS.get(cat, [])) if 'SEEDS' in globals() else 0
    t_count = train_counts.get(cat, 0)
    e_count = eval_counts.get(cat, 0)
    total = t_count + e_count
    total_train += t_count
    total_eval += e_count
    print('{:<25} {:>8} {:>8} {:>8} {:>8}'.format(cat, seed_count, t_count, e_count, total))

print('-' * 65)
print('{:<25} {:>8} {:>8} {:>8} {:>8}'.format('TOTAL', 0, total_train, total_eval, total_train + total_eval))
