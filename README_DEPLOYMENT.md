# Manas Model Deployment Guide

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run crisis classifier test
python crisis_classifier.py

# Run evaluation with correction
python compare_runs.py --pair base_short v4

# Deploy (requires GPU)
python deployment_pipeline.py
```

## Model Selection

**Recommended**: `base_short` (corrected)

| Model | Mean Score | Lead |
|-------|-----------|------|
| **base_short_corrected** | **8.56** | **+1.43** |
| v4 | 7.13 | 0.00 |
| v7_shortprompt | 7.07 | -0.06 |
| v7 | 6.95 | -0.18 |

## Key Components

### 1. Crisis Detection
- **File**: `crisis_classifier.py`
- **Accuracy**: 92% (12/13 test scenarios)
- **Helplines**: iCall 9152987821, Vandrevala 1860-2662-345

### 2. Evaluation Correction
- **File**: `evaluator_correction.py`
- **Purpose**: Correct inverted question_discipline scoring
- **Impact**: +1.21 mean score improvement

### 3. Deployment Pipeline
- **File**: `deployment_pipeline.py`
- **Function**: `run_manas(user_input, max_turns=10)`

## Safety Protocols

### Crisis Detection Flow
1. Input → Crisis classifier pre-check
2. If crisis detected → Insert helpline template
3. Generate model response
4. Post-process for safety compliance

### Response Templates
Crisis responses follow this format:
```
"That sounds incredibly heavy, and I want you to know I'm here with you. 
Please reach out right now:
- iCall: 9152987821 (9am-9pm, multilingual support)
- Vandrevala Foundation: 1860-2662-345 (24/7 free counseling)
You don't have to carry this alone — these people are trained to help. 
I'm staying right here with you."
```

## Monitoring

### Key Metrics to Track
1. **Crisis detection rate**: Target ≥90%
2. **Helpline insertion rate**: 100% for crisis inputs
3. **User satisfaction**: Post-conversation survey
4. **Escalation rate**: Flag inappropriate responses

### Alert Thresholds
- Crisis detection <85% → Flag for review
- Negative user feedback >5% → Pause and investigate
- Helpline number reported invalid → Emergency update

## Legal & Compliance

### Required Sign-offs
- [ ] Crisis helpline numbers verified current
- [ ] Response templates reviewed by legal
- [ ] Privacy impact assessment completed
- [ ] Accessibility review completed

### Data Handling
- Conversation logs retained for 30 days
- No PII stored in logs
- Opt-out mechanism provided

## Troubleshooting

### Issue: Crisis classifier misses scenarios
**Solution**: Update `CRISIS_PATTERNS` in `crisis_classifier.py`
```python
CRISIS_PATTERNS['direct'].append('new_pattern')
```

### Issue: Evaluator scores seem inverted
**Solution**: Apply correction via `evaluator_correction.py`
```python
from evaluator_correction import correct_qdisc_scores
corrected_results = correct_qdisc_scores(raw_results)
```

### Issue: Model generates inappropriate content
**Solution**: Enable response filtering
```python
# In deployment_pipeline.py
if contains_harmful_content(response):
    response = "I'm here to help. Please consider reaching out to:
    iCall: 9152987821"
```

## Contact

- **Technical Support**: [Model Team Email]
- **Safety Incidents**: [Crisis Team Email]
- **Legal Questions**: [Compliance Officer]
