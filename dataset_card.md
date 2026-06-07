---
language:
- en
- hi
license: other
task_categories:
- text-to-speech
tags:
- Indian English
- Hindi
- TTS
- speech dataset
- emotion
---

# Sarvam Indian TTS Dataset

Curated Indian English and Hindi/code-mixed speech segments for Text-to-Speech dataset preparation, built for the Sarvam AI assignment. Audio is sourced from manually reviewed YouTube videos; each row includes source metadata for auditability.

## Dataset Stats

| Split | Clips | Duration | Avg proxy SNR (dB) |
|---|---:|---:|---:|
| Indian English (`en-IN`) | 81 | 30.70 min | 39.34 |
| Hindi/code-mixed (`hi-IN`) | 107 | 30.41 min | 39.54 |
| Total | 188 | 61.11 min | 39.45 |

All final clips are mono 24 kHz WAV files. Matched raw before-clips were retained locally for analysis; this public dataset focuses on the final training clips and auditable metadata.

## Pipeline Architecture

1. YouTube download (yt-dlp)
2. Optional enhancement/finalization; this run did not show measurable denoising gain
3. Sarvam Saaras v3 ASR + Diarization
4. Dominant Speaker Extraction
5. Silero VAD Micro-segmentation
6. Dynamic Prosodic Boundary Padding (DPBP)
7. ECAPA-TDNN Speaker Verification
8. SNR & Duration Quality Filtering
9. Transcript repair plus heuristic emotion/style tagging with confidence/evidence fields
10. Finalize (24kHz, -23 LUFS)

## Dataset Schema

Each sample contains:
- `audio`: The 24kHz waveform
- `text`: Normalized transcript
- `language`: `en-IN` or `hi-IN`
- `emotion`: Detected emotion (happy, sad, angry, neutral, etc.)
- `style`: Speaking style (formal, conversational, etc.)
- `style_description`: Rich text description of the style
- `emotion_status`, `emotion_confidence`, `emotion_evidence`: audit fields for heuristic emotion tags
- `snr_db`: Signal-to-Noise Ratio (higher is better)
- `speaker_id` / `speaker_key`: Diarized speaker identity used for verification
- `source_url`: Original YouTube URL

## Quality Notes

The before/after analysis showed strong format/loudness standardization, not denoising improvement. Mean proxy SNR moved from 39.45 dB to 39.20 dB on matched raw/final pairs, while final LUFS moved close to the -23 LUFS target for 96.8% of clips. Emotion labels are heuristic v2 tags and should be treated as review aids rather than ground-truth human emotion labels.

## License / Rights Note

The underlying audio remains governed by the original YouTube/source rights. Do not claim a permissive license unless every selected source has been verified as compatible.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("Rushabh3/sarvam-indian-tts-60min")

# Example: get high SNR english training samples
high_quality = ds["english_train"].filter(lambda x: x["snr_db"] > 20.0)
```
