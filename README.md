# Sarvam Indian TTS Dataset Pipeline

A high-quality Text-to-Speech data processing pipeline designed for the Sarvam AI assignment. This pipeline creates a curated, emotion-tagged, and speaker-verified 60-minute TTS dataset of Indian English and Hindi speech from raw YouTube videos.

## Public Deliverables

- HuggingFace dataset: https://huggingface.co/datasets/Rushabh3/sarvam-indian-tts-60min
- Final report: `report/final_report.pdf`

The published dataset contains 188 final clips: 81 Indian English clips and 107 Hindi/code-mixed clips, totaling 61.11 minutes.

## Features

- **Automated Download**: Extracts 16kHz audio from YouTube using `yt-dlp`.
- **Enhancement / Finalization**: Runs optional vocal enhancement and then standardizes final clips to mono 24 kHz WAV with loudness normalization.
- **ASR & Diarization**: Uses the Sarvam AI API (`saaras:v3`) with smart batch/REST routing and diarization.
- **Smart Segmentation**: 
  - Macro-chunking by dominant speaker
  - Micro-segmentation (3-30s) using Silero VAD
  - Dynamic Prosodic Boundary Padding (DPBP) for click-free phonetic cuts
- **Quality Filtering**: Verifies speaker identity using SpeechBrain ECAPA-TDNN and filters by SNR.
- **Content Guardrails**: Rejects likely language mismatches, prohibited source types, clipped audio, silence-heavy clips, and weak speaker matches.
- **Before/After Traceability**: Carries source timestamps through the pipeline and exports matched raw segments for final clip comparison.
- **Emotion & Text Normalization**: Transcript repair plus heuristic v2 emotion/style tagging with confidence and evidence fields.
- **Delivery Ready**: Upsamples to 24kHz LibriTTS standards, EBU R128 loudness normalizes, and packages for HuggingFace.

## Final Quality Finding

The before/after audit showed that the pipeline improved standardization and reviewability more than denoising. Proxy SNR changed from 39.45 dB to 39.20 dB on average, so the report does not claim enhancement improved noise quality. The useful gains were consistent 24 kHz export, loudness targeting, zero clipping, full transcript coverage, matched raw/final traceability, and explicit audit fields for emotion labels.

## Setup

1. Copy `.env.example` to `.env` and fill in your keys:
   ```bash
   SARVAM_API_KEY=your_key_here
   HF_TOKEN=your_huggingface_write_token
   HF_USERNAME=your_username
   ```

2. Install dependencies (AWS GPU instance recommended):
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Demucs and ECAPA-TDNN will utilize the GPU if available.*

## Usage

1. **Curate Videos**: Manually find high-quality YouTube videos. Add them to `sources/sources.jsonl` following the provided schema.
2. **Validate**:
   ```bash
   python pipeline/00_curate_sources.py
   ```
3. **Run Pipeline**:
   ```bash
   python run_pipeline.py
   ```
   *You can resume from any stage using `--start-stage X`.*

## Robustness Guardrails

The pipeline is intentionally strict. Stage 00 blocks sources with background music, low pre-listen quality, less than 80% dominant speaker, or titles/notes that indicate audiobook, voiceover, dubbing, songs, music videos, news, teleprompter reading, ASMR, panels, debates, or reaction videos.

Stage 08 rejects final candidates with:
- Hindi labels where the transcript is mostly Latin/English
- English labels with substantial Devanagari text
- very short transcripts
- SNR below 20 dB
- ECAPA speaker similarity below 0.85
- clipped or near-clipped audio
- too much silence or low speech activity
- prohibited source-type matches carried from source metadata

Final metadata now preserves `raw_audio_path`, `processed_source_path`, `source_start_time_seconds`, `source_end_time_seconds`, `processed_segment_path`, `final_path`, `quality_flags`, and `review_flags`. Stage 10 also writes matched raw clips to `data/raw_segments` when source timestamps are available, allowing direct before/after listening against `data/final`.

## Analysis

To generate summary statistics for your PDF report:
```bash
python analysis/quality_analysis.py
```

To randomly sample 20% of your data into a CSV for manual grading:
```bash
python pipeline/review_samples.py
```
