# Iteration 1 Audit and Pipeline Hardening

## Purpose

This document records the first end-to-end pipeline iteration, what it produced, why the produced dataset was not accepted as final, and what was changed before the next run.

The first iteration was useful as a dry run: it proved that the pipeline could download, transcribe, segment, filter, normalize, package, and export audio. The audit showed that the main failure was not file format or duration. The main failure was content quality: source selection and clip-level semantic validation were too weak.

## Initial Pipeline Flow

The original pipeline followed this structure:

1. Source curation in `sources/sources.jsonl`
2. YouTube audio download into `data/raw`
3. Optional enhancement into `data/enhanced`
4. Sarvam ASR and diarization
5. Macro-chunk extraction by speaker
6. VAD micro-segmentation into 3-30 second clips
7. DPBP boundary padding
8. Speaker verification and SNR filtering
9. Emotion tagging and text normalization
10. Final 24 kHz mono WAV export and HuggingFace dataset packaging

The exported dataset was later extracted from parquet into `extracted_wav` for manual inspection.

## What Iteration 1 Produced

The first exported dataset contained `202` WAV clips:

- `en-IN`: `88` clips, `32.56` minutes
- `hi-IN`: `114` clips, `31.21` minutes
- Train split: `183` clips, `57.37` minutes
- Validation split: `19` clips, `6.40` minutes

The audio container format was correct:

- `24 kHz`
- `mono`
- `16-bit PCM WAV`

So the basic packaging target was met. The rejection decision came from content-level audit, not from WAV formatting.

## Problems Found

The audit script generated `analysis/extracted_wav_assessment/all_clips_assessment.csv`, `high_risk_clips.csv`, and `manual_review_priority.csv`.

The major flags were:

- `73` clips likely came from read, voiceover, monologue, lecture, story, or similar source styles.
- `60` Hindi-labeled clips were mostly Latin-script or English text.
- `36` clips had too much silence or low speech activity.
- `22` clips appeared to start mid-sentence.
- `19` clips had weak speaker match.
- `16` clips were code-mixed enough to require review.
- `10` clips appeared to end mid-sentence.
- `4` English-labeled clips contained substantial Devanagari text.
- `2` clips had low SNR.
- `2` clips had very short transcripts.

Examples of risky source concentration included:

- `Osho Hindi Discourse`: `15` clips
- `Sonu Sharma Motivational Speech`: `15` clips
- `Gaur Gopal Das Hindi Session`: `15` clips
- `Neelesh Misra Story: Mohar`: `15` clips
- `Neelesh Misra Story: Gunahgaar`: `15` clips
- `Pankaj Tripathi Acting Philosophy`: `14` clips
- `Manoj Bajpayee NSD Days Monologue`: `11` clips

These source types may be usable in some speech datasets, but they are risky for a high-quality natural TTS assignment because many sound scripted, read, performative, or source-biased rather than clean natural speech.

## Why We Did Not Select This Dataset

The dataset was not selected as final because it could fail a manual TTS dataset audit on these grounds:

- The language labels were not reliable enough, especially in the Hindi split.
- Some Hindi clips were actually English speech.
- Some clips were sourced from content likely to sound read, narrated, voiceover-like, or scripted.
- Several clips had weak single-speaker confidence.
- Several clips had poor segmentation boundaries, starting or ending mid-sentence.
- The pipeline did not preserve matched raw clip artifacts for exact before/after comparison.
- The first run could prove final export quality, but could not cleanly prove per-clip improvement from raw source to final processed clip.

For a TTS dataset, incorrect language labels and unnatural read/voiceover speech are high-impact problems. They directly affect model training quality and make the dataset harder to defend in a report.

## Root Causes

The first iteration had these pipeline weaknesses:

- Source validation was too permissive.
- Background music, low dominant-speaker percentage, and risky source types were treated mostly as warnings.
- There was no strong transcript-script validation after ASR.
- Hindi and English labels were trusted from source metadata instead of being revalidated after transcription.
- Quality filtering focused mainly on duration, SNR, and speaker verification.
- Read/voiceover/audiobook/news/dubbing/music-style sources were not rejected consistently.
- Clip-level audio checks did not include clipping, peak level, or silence/low-activity ratio.
- Metadata did not fully preserve `raw_audio_path`, exact source timestamps, processed segment path, and final path for before/after review.

## Changes Made

The pipeline was hardened with a shared guardrail module: `pipeline/quality_guardrails.py`.

### Source-Level Guardrails

Stage 00 now blocks risky sources before download.

Hard exclusions include:

- background music
- pre-listen quality below `4`
- dominant speaker below `80%`
- audiobook or read-aloud content
- voiceover or documentary narration
- news or teleprompter-style content
- songs, music videos, karaoke
- dubbed or dubbing-heavy content
- ASMR
- panel discussions, debates, reaction videos

This prevents bad source links from entering the expensive ASR and segmentation stages.

### Clip-Level Language Validation

Stage 08 now checks transcript script ratios:

- Reject `hi-IN` when the transcript is mostly Latin/English.
- Reject `en-IN` when the transcript contains too much Devanagari.
- Flag substantial code-mixing for manual review.

This directly targets the problem where English clips appeared inside the Hindi folder.

### Clip-Level Audio Validation

Stage 08 now adds:

- clipping percentage
- peak dBFS
- RMS dBFS
- speech activity ratio
- low-activity/silence-heavy rejection

The SNR threshold was also raised from `15 dB` to `20 dB`.

### Speaker Verification Tightening

The ECAPA similarity threshold was raised from `0.82` to `0.85`.

This reduces the chance of retaining cross-speaker contamination.

### Prohibited Source-Type Filtering

The same prohibited-source regex is used at both source level and clip level.

This catches cases where metadata title/channel/notes indicate:

- voiceover
- news
- documentary
- audiobook
- dubbing
- music
- panel/debate/reaction content

### Before/After Traceability

The pipeline now preserves trace metadata through macro extraction, VAD segmentation, DPBP, quality filtering, finalization, and dataset export.

New or preserved fields include:

- `raw_audio_path`
- `processed_source_path`
- `enhanced_path`
- `source_start_time_seconds`
- `source_end_time_seconds`
- `processed_segment_path`
- `final_path`
- `raw_segment_path`
- `quality_flags`
- `review_flags`
- `devanagari_ratio`
- `latin_ratio`

Stage 10 now exports matched raw-before clips into `data/raw_segments` when timestamps are available. This enables direct comparison:

```text
data/raw_segments/<clip>_raw.wav
data/final/<clip>.wav
```

That fixes the earlier limitation where only full raw YouTube audio and final exported clips were available, but not exact raw segment pairs.

## Current Expected Behavior

With the stricter configuration, the pipeline is expected to reject more material. This is intentional.

Current key thresholds:

- source dominant speaker minimum: `80%`
- source pre-listen quality minimum: `4/5`
- speaker verification minimum: `0.85`
- SNR minimum: `20 dB`
- Hindi Latin-script hard maximum: `0.70`
- English Devanagari hard maximum: `0.25`
- raw segment export: enabled

When Stage 00 was rerun after hardening, it correctly failed on problematic existing sources:

- `RVkqvpyrGr4`: dominant speaker only `75%`
- `-fpXb-6ND-s`: prohibited source-type match
- `Wvwn90HyYso`: prohibited source-type match

These should be replaced before the next full pipeline run.

## Next Iteration Plan

For the next run, source links should be collected using strict scraping-agent guardrails:

- 50% Indian English and 50% Hindi
- clean natural human speech
- one dominant speaker for at least `80%` of the video
- ideal video length `5-20` minutes
- accept up to `45` minutes only if clean and stable
- reject robotic, synthetic, audiobook, dubbing-heavy, voiceover, songs, music videos, news, panel, debate, reaction, ASMR, and teleprompter-style content
- reject language mismatch based on actual spoken content, not title alone

After the next run, the dataset should be evaluated with:

- automatic quality report
- manual review priority CSV
- before/after raw-vs-final listening using `data/raw_segments` and `data/final`
- language-label audit
- speaker consistency audit
- transcript correctness audit
- boundary quality audit

## Report Framing

The first iteration should be described as a diagnostic run. It validated the mechanics of the pipeline but exposed weaknesses in source curation, language validation, content-type filtering, and before/after traceability.

The revised pipeline is stricter by design. It trades retention rate for higher dataset defensibility, cleaner language splits, stronger single-speaker purity, and measurable before/after comparison.
