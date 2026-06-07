# Emotion Tag Repair v2

## What Changed

The previous repair assigned many clips from whole-video source hints. v2 retags each clip using:

- transcript keyword evidence
- curated source emotion/register hints
- clip-level speech rate and audio activity/loudness
- confidence and margin thresholds

These are still heuristic labels, not human labels and not trained emotion-model predictions.

## Results

| Item | Value |
| --- | ---: |
| Clips retagged | 188 |
| Rows changed | 57 |
| Manual review required | 68 |
| Mean confidence | 0.448 |

## Before Emotion Distribution

{
  "sarcastic": 11,
  "neutral": 57,
  "concerned": 59,
  "formal": 26,
  "excited": 35
}

## After Emotion Distribution

{
  "sarcastic": 11,
  "neutral": 46,
  "excited": 10,
  "formal": 73,
  "concerned": 48
}

## Style Distribution

{
  "expressive": 33,
  "conversational": 46,
  "authoritative": 70,
  "formal": 39
}

## Files

- `emotion_manual_review_v2.csv`
- `emotion_tag_repair_summary.json`
- `emotion_distribution_before_after_v2.png`
