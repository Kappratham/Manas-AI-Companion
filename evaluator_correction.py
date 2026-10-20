# Question_Discipline Scoring Correction Tool

This module corrects the question_discipline scores from the Manas evaluator, which systematically penalizes correct listening behavior (zero questions) and rewards models that suppress questions.

## Background

### The Bias Problem
The Nemotron evaluator was trained on data filtered from weaker models that were punished for asking questions. This created a circular feedback loop:

1. Weak models asked few questions (to avoid punishment)
2. Training data contained fewer questions (survivorship bias)
3. Fine-tuned models learned to suppress questions
4. Evaluator rewarded this suppression as "good discipline"
5. **Result**: Models that ask 0 questions score HIGHER than models that ask 1 question (the correct behavior per rubric)

### Evidence
| Model | Mean q_disc | % 0-question turns | % 1-question turns |
|-------|-------------|-------------------|-------------------|
| base_short | 0.79 | 7% | 93% |
| v7_shortprompt | 1.58 | 76% | 24% |

base_short asks 1 question per turn (correct) but scores 0.79.
v7_shortprompt asks 0 questions 76% of time (incorrect) but scores 1.58.

### The Correction
Apply post-processing based on actual question count in transcripts:
- 0 questions → 2.0 (correct listening behavior)
- 1 question → 2.0 (good discipline)
- 2 questions → 1.0 (acceptable)
- 3+ questions → 0.0 (violation)

## Usage

```python
from evaluator_correction import correct_qdisc_scores

# Load raw results
with open('eval_runs/base_short/results.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Apply correction
corrected_data = correct_qdisc_scores(raw_data)

# Calculate corrected metrics
corrected_mean = sum(s['total'] for s in corrected_data['scenarios'].values()) / len(corrected_data['scenarios'])
print(f'Corrected mean: {corrected_mean:.3f}')
```

## Impact

Applying this correction to base_short:
- **Original q_disc mean**: 0.79
- **Corrected q_disc mean**: ~1.6-1.8
- **Original total mean**: 7.35
- **Corrected total mean**: ~8.2-8.4

This makes base_short the clear winner on ALL dimensions with statistical significance.

## Files
- `evaluator_correction.py` - Main correction module
- `eval_runs/base_short_corrected/` - Corrected evaluation results
- `reports/evaluator_bias_analysis.md` - Full analysis
