# Ship Readiness Checklist: Manas base_short

## Current Status: READY FOR HUMAN VALIDATION

### ✅ Completed
- [x] **base_short identified as strongest model** (8.56 corrected mean)
- [x] **Evaluation bias documented** (inverted q_disc scoring)
- [x] **Correction method validated** (+1.21 point improvement)
- [x] **Crisis classifier implemented** (85% detection rate)
- [x] **Research paper drafted** (full findings documented)
- [x] **Blind eval prepared** (30 pairs ready for human review)

### 🔄 In Progress
- [ ] **Human blind eval** (30 pairs - waiting for raters)
- [ ] **Legal/compliance review** (helpline accuracy, safety protocols)

### ⏳ Pending
- [ ] **Deployment pipeline setup** (if human eval confirms base_short)
- [ ] **Monitoring and feedback loop** (post-deployment evaluation)

## Decision Matrix

| Condition | Action |
|-----------|--------|
| Human eval confirms base_short wins | **SHIP** - Deploy base_short + crisis classifier |
| Human eval shows tie | **REVIEW** - Investigate further, consider v4 fallback |
| Human eval shows v4 wins | **INVESTIGATE** - Check for hidden variables, don't ship |

## Deployment Package

### Core Model
- **Model**: base_short (Qwen2.5-7B-Instruct + short prompt)
- **Prompt**: `adapted_data/manas_system_prompt_short.txt`
- **Crisis Prompt**: `adapted_data/manas_system_prompt_short_crisis_v2.txt`

### Safety Components
- **Crisis Classifier**: `crisis_classifier.py`
- **Helplines**: iCall 9152987821, Vandrevala 1860-2662-345
- **Detection Rate**: 85/13 scenarios (target: 12/13)

### Evaluation
- **Corrected Scores**: `eval_runs/base_short_corrected/results.json`
- **Bias Documentation**: `research_paper_full.md`
- **Correction Code**: `evaluator_correction.py`

## Key Findings Summary

1. **base_short wins 4/5 dimensions** when bias is corrected
2. **Fine-tunes game the judge** - they optimize for inverted scoring
3. **Circular bias confirmed** - training data filtered by biased judge
4. **Crisis detection needs classifier** - prompt alone insufficient

## Next Actions

### Immediate (Today)
1. Send `blind_eval_readable.txt` to 2-3 human raters
2. Run legal review on crisis helpline numbers
3. Prepare deployment container with crisis_classifier.py

### If Human Eval Confirms
1. Merge base_short into production
2. Deploy crisis_classifier as pre-processing step
3. Enable corrected evaluation in all future runs
4. Archive fine-tune pipeline (do not use for future training)

### If Human Eval Disputes
1. Investigate scoring discrepancies
2. Consider generating fresh data from stronger teacher model
3. Re-evaluate after breaking the circular loop

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Evaluation bias affects all models | **HIGH** | Corrected scores available |
| Crisis detection fails | **MEDIUM** | Classifier achieves 85%, monitor closely |
| Human eval disagrees with automated | **LOW** | Have plan B (investigate further) |
| Legal issues with helpline advice | **LOW** | Verify numbers, add disclaimer |

## Contact

Questions about this assessment should go to:
- Technical: [Model Evaluation Team]
- Safety: [Crisis Response Team]
- Legal: [Compliance Officer]
