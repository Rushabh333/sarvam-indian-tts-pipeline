# Emotion Stage Failure Report

## Observed Run

The iteration-2 run produced a usable audio review set, but not trusted emotion/style metadata.

| Item | Value |
| --- | ---: |
| Final clips | 188 |
| Matched raw before-clips | 188 |
| Total duration | 61.11 min |
| English clips | 81 |
| Hindi clips | 107 |
| Pipeline emotion distribution | 188 neutral |
| Pipeline style distribution | 188 conversational |
| Empty normalized transcripts | 110 |
| Emotion rows requiring review | 188 |

## Failure

The pipeline converted ASR/LLM failures into normal-looking labels. When the Sarvam API hit quota/rate failures, Stage 09 defaulted clips to `neutral` and `conversational`. Because the metadata did not preserve the failure status, those placeholder labels were indistinguishable from valid annotation output.

This is a real production issue: a dataset can pass audio quality checks and still carry invalid semantic labels.

## Fix

Stage 09 now records explicit annotation trust fields:

- `emotion_status`
- `emotion_failure_reason`
- `emotion_review_required`

Stage 10 exports these fields to the final metadata/HuggingFace dataset. The before/after manual review sheet also includes them, so reviewers can rank audio quality while seeing whether the label was trusted or fallback.

For the legacy iteration-2 output, `contribution/backfill_legacy_emotion_status.py` marked all 188 existing rows as `legacy_default_needs_review` instead of silently accepting the placeholder labels.

## Calibration Path

The proposed SenseVoice-style neutral collapse contribution is valid only if the emotion model exposes logits or probabilities. A scalar temperature alone calibrates confidence but usually does not change the predicted class. The implemented calibration module therefore uses:

```text
calibrated_logits = logits / T + class_bias
```

This keeps the contribution principled:

- `T` handles confidence calibration.
- `class_bias` can correct over-selection of `neutral`.
- macro-F1, ECE, and prediction distribution can be measured before/after on a manually labeled set.

## Reviewer Artifacts

- `analysis/before_after_manual_review.csv`: all 188 raw/final clip pairs for manual audio ranking.
- `contribution/results/emotion_manual_labels_seed.csv`: seed sheet for manual emotion/style/transcript labels.
- `contribution/results/emotion_failure_audit.json`: machine-readable audit summary.
