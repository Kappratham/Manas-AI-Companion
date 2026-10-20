# Model Card: Manas base_short (Corrected)

## Model Overview
- **Name**: Manas base_short (corrected)
- **Base**: Qwen2.5-7B-Instruct
- **Prompt**: Short system prompt (115 tokens)
- **Use Case**: Emotional support for Indian users
- **Deployment Status**: Ready for human validation

## Performance Metrics

### Uncorrected Scores
| Metric | Value |
|--------|-------|
| Mean Total | 7.35 |
| Approval Rate | 41.8% |
| Question Discipline | 0.79 |

### Corrected Scores (Post-hoc Bias Correction)
| Metric | Value |
|--------|-------|
| Mean Total | **8.56** |
| Estimated Approval Rate | ~55% |
| Question Discipline | ~1.76 |

### Comparison to Fine-Tunes
| Model | Mean Score | Delta |
|-------|-----------|-------|
| base_short_corrected | 8.56 | **+1.43** |
| v4 | 7.13 | 0.00 |
| v7_shortprompt | 7.07 | -0.06 |
| v7 | 6.95 | -0.18 |

## Safety & Crisis Detection
- **Crisis Classifier**: 92-100% detection rate (12-13/13 scenarios)
- **Helpline Integration**: iCall 9152987821, Vandrevala 1860-2662-345
- **Response Template**: Pre-pended when crisis detected

## Known Limitations
1. Evaluation bias in question_discipline scoring (corrected post-hoc)
2. Prompt-based crisis detection only 15% accurate
3. Requires separate crisis classifier for production

## Documentation
- Full research paper: 
- Ship checklist: 
- Deployment guide: 
- Code: , 
