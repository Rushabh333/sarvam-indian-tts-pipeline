# Contribution: Emotion Failure Tracking and Calibration

## What We Found

The iteration-2 pipeline produced 188 final clips with matched raw before-clips. The audio extraction and quality stages completed, but the emotion/normalization stage collapsed to placeholder metadata:

- 188/188 clips were labeled `neutral`
- 188/188 clips were labeled `conversational`
- many clips had empty `normalized_transcript`
- the Sarvam API returned quota/rate failures during the run

The important bug is that these failures previously looked like valid model output. A downstream reviewer could not tell whether `neutral/conversational` was a real annotation or just a fallback after ASR/LLM failure.

## Fix Added

Stage 09 now records explicit status fields:

- `emotion_status`: `llm_success`, `llm_partial`, `llm_failed_defaulted`, `asr_failed_defaulted`, or legacy review markers
- `emotion_failure_reason`: the API or parsing failure reason
- `emotion_review_required`: boolean flag for manual review

Stage 10 and the before/after review CSV now preserve those fields, so final exports and manual review sheets expose annotation trust.

## Why This Is Stronger Than a Simple Temperature Story

Scalar temperature scaling is useful for confidence calibration, but by itself it normally cannot change the winning emotion class because it preserves logit order. If the problem is a `NEUTRAL` prior collapse, the technically correct lightweight fix is:

```text
calibrated_logits = logits / T + class_bias
```

The `class_bias` term can move predictions away from an over-selected class, while `T` calibrates confidence. This is implemented in `contribution/emotion_calibration.py` for any emotion model that can provide logits or probabilities.

## How To Use

Audit the current final metadata and create a manual labeling CSV:

```bash
venv/bin/python contribution/emotion_failure_audit.py
```

Backfill legacy runs that were created before failure tracking existed:

```bash
venv/bin/python contribution/backfill_legacy_emotion_status.py
```

After manually labeling a small set and exporting model logits/probabilities, run calibration:

```bash
venv/bin/python contribution/emotion_calibration.py \
  --input contribution/results/labeled_emotion_logits.jsonl
```

Expected JSONL input format:

```json
{"audio_path":"data/final/example.wav","language":"hi-IN","label":"concerned","logits":[0.1,0.2,-0.3,1.8,0.0,-0.1,0.4,-0.2]}
```
