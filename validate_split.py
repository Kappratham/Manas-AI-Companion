import json
import sys
sys.path.append('.')
from config import CATEGORIES
from validators.checks import (
    check_single_question_per_turn,
    check_no_diagnosis,
    check_no_big_steps,
    check_no_hollow_reassurance,
    check_crisis_referral_present,
    check_json_valid,
    check_min_turns,
)
from validators.checks_v2 import (
    check_no_markdown_formatting,
    check_no_hallucinated_user_turns,
    check_no_clinical_openings,
    check_no_lecture_mode,
    check_no_empty_assistant_turns,
    check_no_early_framework,
    check_response_length,
)

def validate_conversation_v2(conversation, category_config):
    '''Run validation on a conversation'''
    min_turns = category_config.get('turns_min', 6)
    max_turns = category_config.get('turns_max', 8)

    checks = [
        # ── V1 checks ────────────────────────────────────────────────────
        (
            'turns_range',
            check_min_turns(conversation, min_turns, max_turns),
        ),
        (
            'single_question_per_turn',
            check_single_question_per_turn(conversation),
        ),
        (
            'crisis_referral_present',
            check_crisis_referral_present(conversation),
        ),
        (
            'json_valid',
            check_json_valid(conversation),
        ),
        (
            'no_diagnosis',
            check_no_diagnosis(conversation),
        ),
        (
            'no_big_steps',
            check_no_big_steps(conversation),
        ),
        (
            'no_hollow_reassurance',
            check_no_hollow_reassurance(conversation),
        ),

        # ── V2 checks ────────────────────────────────────────────────────
        (
            'no_markdown_formatting',
            check_no_markdown_formatting(conversation),
        ),
        (
            'no_hallucinated_user_turns',
            check_no_hallucinated_user_turns(conversation),
        ),
        (
            'no_clinical_openings',
            check_no_clinical_openings(conversation),
        ),
        (
            'no_lecture_mode',
            check_no_lecture_mode(conversation),
        ),
        (
            'no_empty_assistant_turns',
            check_no_empty_assistant_turns(conversation),
        ),
        (
            'no_early_framework',
            check_no_early_framework(conversation),
        ),
        (
            'response_length',
            check_response_length(conversation),
        ),
    ]

    failed = [name for name, passed in checks if not passed]
    return {
        'passed': len(failed) == 0,
        'failed_checks': failed
    }

def get_category_config(category):
    return CATEGORIES.get(category, {'turns_min': 5, 'turns_max': 8, 'sensitive': False})

def validate_file(filepath, description):
    print(f'\nValidating {description}: {filepath}')
    passed = 0
    failed = 0
    fail_details = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                conv = json.loads(line)
                category = conv.get('category', 'unknown')
                config = get_category_config(category)
                
                result = validate_conversation_v2(conv, config)
                if result['passed']:
                    passed += 1
                else:
                    failed += 1
                    fail_details.append({
                        'line': line_num,
                        'category': category,
                        'failed_checks': result['failed_checks']
                    })
            except json.JSONDecodeError as e:
                print(f'  Error parsing line {line_num}: {e}')
                failed += 1
    
    print(f'  Passed: {passed}')
    print(f'  Failed: {failed}')
    
    if fail_details:
        print(f'  Failure details:')
        # Group by failure reason
        from collections import Counter
        failure_reasons = []
        for detail in fail_details:
            failure_reasons.extend(detail['failed_checks'])
        
        reason_counts = Counter(failure_reasons)
        for reason, count in reason_counts.most_common():
            print(f'    {reason}: {count}')
    
    return passed, failed

# Validate both files
train_passed, train_failed = validate_file('formatted/train_v2.jsonl', 'training set')
eval_passed, eval_failed = validate_file('formatted/eval_v2.jsonl', 'evaluation set')

print(f'\n=== SUMMARY ===')
print(f'Training set: {train_passed} passed, {train_failed} failed')
print(f'Evaluation set: {eval_passed} passed, {eval_failed} failed')
print(f'Total: {train_passed + eval_passed} passed, {train_failed + eval_failed} failed')

if train_failed == 0 and eval_failed == 0:
    print('\n🎉 All conversations pass validation!')
else:
    print(f'\n⚠️  {train_failed + eval_failed} conversations still fail validation')
