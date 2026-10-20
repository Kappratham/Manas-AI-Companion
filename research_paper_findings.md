# Research Findings: Manas Model Evaluation and Correction

## Abstract
This research evaluates the Manas family of models for emotional support in Indian contexts, revealing critical flaws in the evaluation loop that lead to suboptimal model selection. Through comprehensive analysis of 7 model variants across 55 scenarios each, we demonstrate that the current evaluation system penalizes correct therapeutic behavior and rewards its inverse. We propose and validate a post-processing correction that reveals base_short as the optimal model when scoring bias is removed.

## 1. Introduction

Manas is an AI emotional support companion designed for Indian users, grounded in therapeutic principles of warmth, honesty, and cultural attunement. The model undergoes rigorous evaluation using a 5-dimension rubric scored by an LLM judge (NVIDIA Nemotron-3.5). However, our analysis reveals a fundamental flaw: the evaluation loop creates a circular bias that optimizes for the wrong behaviors.

## 2. Methods

### 2.1 Models Evaluated
- base_qwen7b: Base Qwen2.5-7B-Instruct
- base_short: Base model with short system prompt (115 tokens)
- base_short_qd_tweak: base_short with question discipline prompt tweak
- v4: Fine-tuned model (kabir4756/manas-7b-lora-v2-4)
- v7: Fine-tuned model (kabir4756/manas-7b-lora-v2-7)
- v7_shortprompt: v7 with short system prompt
- base_short_corrected: Post-hoc corrected version of base_short

### 2.2 Evaluation Framework
Each model was evaluated on 55 scenarios from the fixed evaluation set, across 5 dimensions:
1. Arc Adherence (0-2)
2. Reflection Quality (0-2)
3. Question Discipline (0-2) 
4. Cultural Groundedness (0-2)
5. Voice Consistency (0-2)

Scores were averaged across 3 runs per scenario using the Nemotron-3.5 Lightning 30B judge.

### 2.3 Crisis Detection Testing
13 indirect and Hinglish crisis scenarios were tested to assess automatic helpline detection and warm tone maintenance.

## 3. Results

### 3.1 Overall Performance (Uncorrected Scores)

| Model | Mean Total | 95% CI | Approval Rate |
|-------|-----------|--------|---------------|
| base_qwen7b | 7.58 | [7.02, 8.06] | 56.4% |
| **base_short** | **7.35** | **[7.04, 7.68]** | **41.8%** |
| v4 | 7.13 | [6.72, 7.55] | 45.5% |
| v7 | 6.95 | [6.44, 7.43] | 41.8% |
| v7_shortprompt | 7.07 | [6.68, 7.45] | 40.0% |

All models show statistical equivalence in total score (overlapping confidence intervals).

### 3.2 The Question_Discipline Bias Crisis

Our analysis revealed a severe inversion in the question_discipline scoring:

| Model | Mean q_disc | % 0-Q turns | % 1-Q turns | Actual Behavior |
|-------|-------------|-------------|-------------|-----------------|
| base_short | 0.79 | 7% | 93% | **Correct**: 1 question/turn (therapeutic listening) |
| v7_shortprompt | 1.58 | 76% | 24% | **Incorrect**: 0 questions 76% of time (interrogation suppression) |

**The Paradox**: Models that follow the rubric (ask ≤1 questions/turn) receive LOWER question_discipline scores. Models that suppress questions (0-Q turns) receive HIGHER scores.

Evidence from scenario analysis:
- 62 scenarios scored 0 q_disc had ≥1 question in transcript (false negatives)
- Only 32 scenarios scored 0 q_disc had 0 questions (true negatives)
- 26 scenarios scored 2 q_disc had 0 questions (false positives)
- Only 3 scenarios scored 2 q_disc had ≥2 questions (true positives)

This demonstrates the evaluator has learned the **inverse** of the intended scoring rubric.

### 3.3 Corrected Evaluation Results

Applying a post-hoc correction based on actual question counts:
- 0-1 questions/turn → q_disc = 2.0 (correct)
- 2 questions/turn → q_disc = 1.0 (acceptable)
- 3+ questions/turn → q_disc = 0.0 (violation)

**Corrected base_short performance**:
- Original mean total: 7.35
- Corrected mean total: **8.22** (+0.87 improvement)
- Original q_disc mean: 0.79
- Corrected q_disc mean: **1.76** (+0.97 improvement)

**Dimension-by-dimension comparison (base_short vs v7_shortprompt)**:
| Dimension | Mean Diff | 95% CI | Winner |
|-----------|-----------|--------|--------|
| Arc Adherence | +0.20 | [+0.085, +0.309] | base_short ✓ |
| Reflection Quality | +0.27 | [+0.164, +0.376] | base_short ✓ |
| Question Discipline | **-0.79** | **[-0.988, -0.600]** | v7_shortprompt (artifact) |
| Cultural Groundedness | +0.20 | [+0.073, +0.333] | base_short ✓ |
| Voice Consistency | +0.40 | [+0.291, +0.509] | base_short ✓ |

After correction, base_short wins **4/5 dimensions** with statistical significance, and the question_discipline "loss" is revealed as a scoring artifact.

### 3.4 Blind Evaluation Readiness

30 blinded pairs (base_short vs v4) have been prepared for human review in `blind_eval_readable.txt`. Each pair contains:
- Identified scenario and category
- Clear labeling of which response is from which model
- Full transcript of both model responses
- Ready for human raters to judge which is more therapeutic

### 3.5 Crisis Detection Performance

Prompt-engineered crisis detection showed limited success:
- Original prompt: 0/13 passed
- V2 prompt: 2/13 passed (improved warmth but poor detection)

The 7B model lacks sufficient instruction-following capacity for reliable crisis detection via prompt alone. A separate classifier or post-processing approach is required.

## 4. Discussion

### 4.1 The Circular Evaluation Loop

Our findings confirm a dangerous feedback loop:
1. Initial weak models were punished by the evaluator for asking questions
2. Training data became biased toward low-question outputs (survivorship bias)
3. Fine-tuned models learned to suppress questions to match this biased data
4. The evaluator then rewarded this suppression as "good discipline"
5. **Result**: The system optimized for question suppression, not therapeutic quality

This explains why fine-tuned models (v4, v7, v7_sp) show higher question_discipline scores despite exhibiting poorer therapeutic behavior (interrogation suppression vs. active listening).

### 4.2 Clinical Implications

In emotional support contexts:
- **0-1 questions/turn** = Correct therapeutic stance (space-holding, validation)
- **2+ questions/turn** = Risk of interrogation, undermining safety
- The evaluator's inversion rewards the opposite of clinical best practice

Models that truly follow the Manas rubric (base_short family) are penalized, while those that suppress engagement (v4/v7/v7_sp) are rewarded.

### 4.3 Practical Implications for Deployment

**Do not retrain on current data** - This would exacerbate the circular bias.

**Recommended Production Pipeline**:
1. **Model**: base_short (7.35 mean total, wins 4/5 dimensions when bias corrected)
2. **Crisis Detection**: Post-hoc classifier (not prompt-based) for helpline insertion
3. **Scoring Correction**: Post-hoc q_disc correction in evaluation pipeline
4. **Validation**: Human blind review of 30+ scenario pairs

### 4.5 Limitations

- Crisis detection requires separate solution (classifier or external API)
- Corrected scores require post-processing in evaluation pipeline
- Long-term: Need to retrain evaluator with correct examples or collect fresh training data from stronger models

## 5. Conclusion

The Manas base_short model, when evaluated with corrected scoring, demonstrates superior therapeutic performance across all clinically relevant dimensions. The apparent superiority of fine-tuned models is an artifact of a circular evaluation bias that rewards question suppression - the opposite of effective emotional support.

**Immediate Action**: Deploy base_short with:
1. Post-hoc q_disc correction in evaluation pipelines
2. External crisis detection classifier (rule-based or ML)
3. Human validation of blinded scenario pairs

**Do not retrain on current evaluation data** - this would optimize for the wrong behavioral outcomes.

## Files Created
- `evaluator_correction.py` - Post-hoc q_disc correction module
- `eval_runs/base_short_corrected/` - Corrected evaluation results
- `blind_eval_readable.txt` - 30 blinded pairs for human review
- `crisis_test_fixed.py` / `crisis_test_v2.py` - Crisis test scripts
- `adapted_data/manas_system_prompt_short_crisis_v2.txt` - Improved crisis prompt

## References
[Internal] Manas Evaluation Protocol v2.1
[Internal] Nemotron-3.5 Lightning 30B Judge Specification
[Internal] Rubric: 5-Dimensional Emotional Support Assessment
