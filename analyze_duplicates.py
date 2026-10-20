import os
import json
from collections import defaultdict

# Path to the approved conversations folder
approved_path = r"C:\Users\yashg\Downloads\manas-project\manas-dataset\formatted\approved"

# Dictionary to store conversations by persona_id and situation
conversations_by_persona_situation = defaultdict(list)

# List to store all conversations for analysis
all_conversations = []

# Read all JSON files in the approved directory
for filename in os.listdir(approved_path):
    if filename.endswith('.json'):
        file_path = os.path.join(approved_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Extract relevant fields
                conversation_id = data.get('conversation_id', '')
                category = data.get('category', '')
                persona_id = data.get('persona_id', '')
                situation = data.get('situation', '')
                scores = data.get('scores', {})

                # Store the conversation data
                conv_data = {
                    'filename': filename,
                    'conversation_id': conversation_id,
                    'category': category,
                    'persona_id': persona_id,
                    'situation': situation,
                    'scores': scores,
                    'file_path': file_path
                }

                all_conversations.append(conv_data)

                # Group by persona_id and situation for duplicate detection
                key = (persona_id, situation)
                conversations_by_persona_situation[key].append(conv_data)

        except Exception as e:
            print(f"Error reading {filename}: {e}")

print(f"Total conversations analyzed: {len(all_conversations)}")
print(f"Unique persona_situation combinations: {len(conversations_by_persona_situation)}")

# Find duplicates (same persona_id and situation but potentially different categories or scores)
duplicates = []
for key, convs in conversations_by_persona_situation.items():
    if len(convs) > 1:
        persona_id, situation = key
        duplicates.append({
            'persona_id': persona_id,
            'situation': situation,
            'conversations': convs
        })

print(f"\nFound {len(duplicates)} duplicate persona_situation combinations:")

# Display details of duplicates
for i, dup in enumerate(duplicates, 1):
    print(f"\n{i}. Persona: {dup['persona_id']}")
    print(f"   Situation: {dup['situation'][:100]}..." if len(dup['situation']) > 100 else f"   Situation: {dup['situation']}")
    print(f"   Found in {len(dup['conversations'])} conversations:")

    for j, conv in enumerate(dup['conversations'], 1):
        print(f"     {j}. File: {conv['filename']}")
        print(f"        Category: {conv['category']}")
        print(f"        Scores: {conv['scores'].get('total', 'N/A')}/{conv['scores'].get('possible', 'N/A')} = {conv['scores'].get('fraction', 'N/A')}")
        print(f"        Verdict: {conv['scores'].get('verdict', 'N/A')}")
        print()

# Also check for duplicates by persona_id only (same persona, different situations)
print("\n" + "="*80)
print("Checking for same persona_id across different situations:")

persona_groups = defaultdict(list)
for conv in all_conversations:
    persona_groups[conv['persona_id']].append(conv)

persona_duplicates = []
for persona_id, convs in persona_groups.items():
    if len(convs) > 1:
        # Check if situations are different
        situations = [c['situation'] for c in convs]
        if len(set(situations)) > 1:  # Different situations
            persona_duplicates.append({
                'persona_id': persona_id,
                'conversations': convs
            })

print(f"Found {len(persona_duplicates)} personae with multiple different situations:")

for i, dup in enumerate(persona_duplicates[:10], 1):  # Show first 10
    print(f"\n{i}. Persona: {dup['persona_id']} ({len(dup['conversations'])} different situations)")
    for j, conv in enumerate(dup['conversations'], 1):
        print(f"     {j}. [{conv['category']}] {conv['situation'][:80]}...")
        print(f"        Score: {conv['scores'].get('total', 'N/A')}/{conv['scores'].get('possible', 'N/A')} = {conv['scores'].get('fraction', 'N/A')}")

if len(persona_duplicates) > 10:
    print(f"\n... and {len(persona_duplicates) - 10} more personae with multiple situations")

# Check for same situation across different personae
print("\n" + "="*80)
print("Checking for same situation across different personae:")

situation_groups = defaultdict(list)
for conv in all_conversations:
    situation_groups[conv['situation']].append(conv)

situation_duplicates = []
for situation, convs in situation_groups.items():
    if len(convs) > 1:
        # Check if personae are different
        personae = [c['persona_id'] for c in convs]
        if len(set(personae)) > 1:  # Different personae
            situation_duplicates.append({
                'situation': situation,
                'conversations': convs
            })

print(f"Found {len(situation_duplicates)} situations shared by multiple personae:")

for i, dup in enumerate(situation_duplicates[:10], 1):  # Show first 10
    print(f"\n{i}. Situation: {dup['situation'][:100]}..." if len(dup['situation']) > 100 else f"\n{i}. Situation: {dup['situation']}")
    print(f"   Found in {len(dup['conversations'])} different personae:")
    for j, conv in enumerate(dup['conversations'], 1):
        print(f"     {j}. [{conv['category']}] Persona: {conv['persona_id']}")
        print(f"        Score: {conv['scores'].get('total', 'N/A')}/{conv['scores'].get('possible', 'N/A')} = {conv['scores'].get('fraction', 'N/A')}")

if len(situation_duplicates) > 10:
    print(f"\n... and {len(situation_duplicates) - 10} more situations shared by multiple personae")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS:")
print("="*80)
print(f"Total conversations: {len(all_conversations)}")
print(f"Unique persona_situation combinations: {len(conversations_by_persona_situation)}")
print(f"Exact duplicates (same persona_id AND situation): {len(duplicates)}")
print(f"Personae with multiple situations: {len(persona_duplicates)}")
print(f"Situations shared by multiple personae: {len(situation_duplicates)}")

# Show some examples of exact duplicates if any exist
if duplicates:
    print(f"\nExamples of exact duplicates (same persona_id AND situation):")
    for i, dup in enumerate(duplicates[:5], 1):
        print(f"\n{i}. Persona: {dup['persona_id']}")
        print(f"   Situation: {dup['situation'][:100]}..." if len(dup['situation']) > 100 else f"   Situation: {dup['situation']}")
        for j, conv in enumerate(dup['conversations'], 1):
            print(f"     {j}. [{conv['category']}] Score: {conv['scores'].get('total', 'N/A')}/{conv['scores'].get('possible', 'N/A')} = {conv['scores'].get('fraction', 'N/A')} ({conv['scores'].get('verdict', 'N/A')})")
else:
    print("\nNo exact duplicates found (same persona_id AND situation).")

# Check score variations for duplicates
if duplicates:
    print("\nScore variations among exact duplicates:")
    for i, dup in enumerate(duplicates[:5], 1):
        scores = [conv['scores'].get('fraction', 0) for conv in dup['conversations']]
        if len(set(scores)) > 1:  # Different scores
            print(f"\n{i}. Persona: {dup['persona_id']}")
            print(f"   Situation: {dup['situation'][:80]}...")
            for j, conv in enumerate(dup['conversations'], 1):
                print(f"     {j}. [{conv['category']}] Score: {conv['scores'].get('total', 'N/A')}/{conv['scores'].get('possible', 'N/A')} = {conv['scores'].get('fraction', 'N/A')} ({conv['scores'].get('verdict', 'N/A')})")