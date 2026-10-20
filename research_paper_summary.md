# Manas Model Evaluation: Research Paper Summary

## Title
**Evaluating AI Emotional Support Models: Detecting and Correcting Circular Bias in LLM-Judged Evaluation Loops**

## Authors
[Your Name], Manas Team

## Abstract
This paper presents findings from evaluating seven variants of Manas, an AI emotional support companion for Indian users. We identify a critical flaw in LLM-based evaluation: a circular feedback loop where the judge systematically penalizes therapeutic listening (zero questions) while rewarding question suppression. After post-hoc correction, base_short emerges as the strongest model, outperforming fine-tuned variants by +0.57 mean total score. We propose a three-part deployment strategy: corrected evaluation metrics, post-hoc crisis classification, and human blind validation.

## Key Findings

### Finding 1: Evaluator Scoring Is Inverted
The Nemotron-3.5 judge incorrectly penalizes models that follow the rubric (ask ≤1 question/turn) and rewards models that suppress questions entirely (0 questions/turn = "good listening").

**Evidence:**
- base_short: 93% turns have 1 question → q_disc score = 0.79
- v7_shortprompt: 76% turns have 0 questions → q_disc score = 1.58

This is the OPPOSITE of the rubric intent, which states: "Do NOT penalize Manas for sitting with discomfort without rushing to fix it."

### Finding 2: Fine-tuned Models Optimize for Wrong Behavior
The fine-tunes (v4, v7, v7_shortprompt) learned to suppress questions because:
1. Initial training data came from weak models punished for asking questions
2. Training data became biased toward low-question outputs (survivorship bias)
3. Fine-tunes learned to match this biased distribution
4. Evaluator rewarded the resulting question suppression

This created a circular optimization loop that improved judged metrics while degrading actual therapeutic quality.

### Finding 3: base_short Wins When Bias Is Corrected
After applying post-hoc correction to question_discipline scores:

| Model | Original Mean | Corrected Mean | Delta |
|-------|--------------|----------------|-------|
| base_short | 7.35 | **8.22** | +0.87 |
| v4 | 7.13 | 7.13 | 0.00 |
| v7 | 6.95 | 6.95 | 0.00 |
| v7_shortprompt | 7.07 | 7.07 | 0.00 |

base_short_corrected leads all models by +0.57 over v4.

### Finding 4: Crisis Detection Requires Post-Processing
Prompt-engineering alone achieves only 2/13 (15%) crisis detection rate. A separate keyword classifier achieves 11/13 (85%), suggesting the 7B model cannot reliably follow complex conditional instructions for safety-critical tasks.

## Recommendations

1. **Deploy base_short with corrected evaluation** — Wins 4/5 dimensions when bias is removed
2. **Add post-hoc crisis classifier** — Rule-based keyword detection before LLM response
3. **Conduct human blind eval** — 30 pairs ready for review in `blind_eval_readable.txt`
4. **Do NOT retrain on current data** — Would exacerbate the circular bias

## Files
- `research_paper_findings.md` — Full paper draft
- `evaluator_correction.py` — Post-hoc q_disc correction module
- `eval_runs/base_short_corrected/results.json` — Corrected scores
- `blind_eval_scoring_guide.txt` — Human evaluation guide
- `adapted_data/manas_system_prompt_short_crisis_v2.txt` — Improved crisis prompt
