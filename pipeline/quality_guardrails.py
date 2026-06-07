#!/usr/bin/env python3
"""Shared source, transcript, and audio guardrails for TTS dataset curation."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")

DEFAULT_PROHIBITED_SOURCE_RE = re.compile(
    r"audiobook|read[- ]?aloud|voiceover|voice[- ]?over|documentary|news|"
    r"teleprompter|song|music video|karaoke|dubbed|dubbing|narration|"
    r"asmr|panel discussion|debate|reaction video",
    re.IGNORECASE,
)


def char_ratios(text: str) -> tuple[float, float]:
    """Return Devanagari and Latin character ratios over script letters."""
    devanagari = len(DEVANAGARI_RE.findall(text or ""))
    latin = len(LATIN_RE.findall(text or ""))
    total = devanagari + latin
    if total == 0:
        return 0.0, 0.0
    return devanagari / total, latin / total


def source_text(entry: dict) -> str:
    fields = (
        "title",
        "channel",
        "speaker_name",
        "genre",
        "content_type",
        "notes",
    )
    return " ".join(str(entry.get(field) or "") for field in fields)


def source_policy_issues(entry: dict, config: dict) -> tuple[list[str], list[str]]:
    """Return hard errors and review warnings for a curated YouTube source."""
    cfg = config.get("source_quality", {})
    hard: list[str] = []
    warnings: list[str] = []

    min_quality = cfg.get("min_pre_listen_quality", 4)
    min_dominant_pct = cfg.get("min_dominant_speaker_pct", 80)
    min_duration = cfg.get("min_video_duration_min", 5)
    max_duration = cfg.get("max_video_duration_min", 45)

    quality = float(entry.get("pre_listen_quality") or 0)
    if quality < min_quality:
        hard.append(f"pre_listen_quality={quality:g} below {min_quality}")

    dominant_pct = float(entry.get("dominant_speaker_pct") or 0)
    if dominant_pct < min_dominant_pct:
        hard.append(f"dominant_speaker_pct={dominant_pct:g} below {min_dominant_pct}")

    if entry.get("background_music", False):
        hard.append("background_music=true")

    duration = entry.get("estimated_duration_min")
    if duration is not None:
        duration = float(duration)
        if duration < min_duration or duration > max_duration:
            warnings.append(
                f"estimated_duration_min={duration:g} outside ideal [{min_duration},{max_duration}]"
            )

    if cfg.get("reject_prohibited_source_types", True):
        pattern = cfg.get("prohibited_source_pattern")
        prohibited_re = re.compile(pattern, re.IGNORECASE) if pattern else DEFAULT_PROHIBITED_SOURCE_RE
        if prohibited_re.search(source_text(entry)):
            hard.append("source text matches prohibited type")

    return hard, warnings


def text_quality_flags(entry: dict, config: dict) -> tuple[list[str], list[str]]:
    """Return hard rejection flags and softer manual-review flags for a clip."""
    cfg = config.get("content_guardrails", {})
    transcript = entry.get("normalized_transcript") or entry.get("approx_transcript") or entry.get("text") or ""
    lang = entry.get("language", "")
    hard: list[str] = []
    review: list[str] = []

    devanagari_ratio, latin_ratio = char_ratios(transcript)
    entry["devanagari_ratio"] = devanagari_ratio
    entry["latin_ratio"] = latin_ratio

    if lang == "hi-IN" and latin_ratio >= cfg.get("hi_latin_hard_max", 0.70):
        hard.append("language_mismatch_hi_label_mostly_latin")
    if lang == "en-IN" and devanagari_ratio >= cfg.get("en_devanagari_hard_max", 0.25):
        hard.append("language_mismatch_en_label_has_devanagari")
    if min(devanagari_ratio, latin_ratio) >= cfg.get("codemix_review_min", 0.20):
        review.append("code_mixed_text")

    stripped = transcript.strip()
    if stripped and stripped[0].islower():
        review.append("possibly_starts_mid_sentence")
    if stripped and stripped[-1] not in ".?!।":
        review.append("possibly_ends_mid_sentence")
    if stripped and len(stripped.split()) < cfg.get("min_transcript_words", 3):
        hard.append("very_short_transcript")

    if cfg.get("reject_prohibited_source_types", True):
        pattern = cfg.get("prohibited_source_pattern")
        prohibited_re = re.compile(pattern, re.IGNORECASE) if pattern else DEFAULT_PROHIBITED_SOURCE_RE
        if prohibited_re.search(source_text(entry)):
            hard.append("prohibited_source_type")

    return hard, review


def audio_quality_metrics(audio_path: str) -> dict:
    """Compute simple audio checks without external model dependencies."""
    audio, sr = sf.read(audio_path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float64)
    if audio.size == 0:
        return {
            "rms_dbfs": -120.0,
            "peak_dbfs": -120.0,
            "clipping_pct": 0.0,
            "active_ratio": 0.0,
            "audio_sample_rate": sr,
        }

    abs_audio = np.abs(audio)
    rms = math.sqrt(float(np.mean(audio**2)))
    peak = float(np.max(abs_audio))
    rms_dbfs = 20 * math.log10(max(rms, 1e-12))
    peak_dbfs = 20 * math.log10(max(peak, 1e-12))
    clipping_pct = float(np.mean(abs_audio >= 0.999) * 100)

    frame = max(int(sr * 0.03), 1)
    usable = (audio.size // frame) * frame
    if usable == 0:
        active_ratio = 0.0
    else:
        framed = audio[:usable].reshape(-1, frame)
        frame_rms = np.sqrt(np.mean(framed**2, axis=1))
        threshold = max(np.percentile(frame_rms, 20) * 3, 10 ** (-45 / 20))
        active_ratio = float(np.mean(frame_rms > threshold))

    return {
        "rms_dbfs": rms_dbfs,
        "peak_dbfs": peak_dbfs,
        "clipping_pct": clipping_pct,
        "active_ratio": active_ratio,
        "audio_sample_rate": sr,
    }


def audio_quality_flags(metrics: dict, config: dict) -> list[str]:
    cfg = config.get("content_guardrails", {})
    flags: list[str] = []
    if metrics.get("clipping_pct", 0.0) > cfg.get("clipping_pct_max", 0.05):
        flags.append("clipped_audio")
    if metrics.get("peak_dbfs", -120.0) > cfg.get("peak_dbfs_max", -0.1):
        flags.append("near_clipping")
    if metrics.get("active_ratio", 1.0) < cfg.get("active_ratio_min", 0.35):
        flags.append("too_much_silence_or_low_activity")
    return flags


def matched_raw_segment_path(raw_segments_dir: Path, seg: dict) -> Path:
    video_id = seg.get("video_id") or Path(seg.get("raw_audio_path", "unknown")).stem
    stem = Path(seg.get("final_path") or seg.get("path") or f"{video_id}_segment").stem
    return raw_segments_dir / f"{stem}_raw.wav"
