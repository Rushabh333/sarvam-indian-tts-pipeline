#!/usr/bin/env python3
"""
Assess the exported WAV dataset and produce CSVs for manual review.

This script intentionally uses cheap, explainable checks. Treat the flags as
triage signals, then listen to the high-risk clips before rejecting them.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
READ_SOURCE_RE = re.compile(
    r"voiceover|documentary|audiobook|story|monologue|news|teleprompter|"
    r"lecture|chapter|motivation|tutorial|sansad|podcast intro",
    re.IGNORECASE,
)


def char_ratios(text: str) -> tuple[float, float]:
    letters = DEVANAGARI_RE.findall(text) + LATIN_RE.findall(text)
    if not letters:
        return 0.0, 0.0
    devanagari = len(DEVANAGARI_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    total = devanagari + latin
    return devanagari / total, latin / total


def wav_header(path: Path) -> dict:
    with wave.open(str(path), "rb") as wav:
        return {
            "actual_duration_s": wav.getnframes() / wav.getframerate(),
            "sample_rate": wav.getframerate(),
            "channels": wav.getnchannels(),
            "bit_depth": wav.getsampwidth() * 8,
        }


def audio_metrics(path: Path) -> dict:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float64)
    if audio.size == 0:
        return {
            "rms_dbfs": -120.0,
            "peak_dbfs": -120.0,
            "clipping_pct": 0.0,
            "active_ratio": 0.0,
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
    }


def flag_row(row: dict) -> list[str]:
    flags: list[str] = []
    lang = row["language"]
    text = row.get("text") or ""
    devanagari_ratio, latin_ratio = char_ratios(text)

    if lang == "hi-IN" and latin_ratio >= 0.70:
        flags.append("language_mismatch_hi_label_mostly_latin")
    if lang == "en-IN" and devanagari_ratio >= 0.25:
        flags.append("language_mismatch_en_label_has_devanagari")
    if min(devanagari_ratio, latin_ratio) >= 0.20:
        flags.append("code_mixed_text")

    stripped = text.strip()
    if stripped and stripped[0].islower():
        flags.append("possibly_starts_mid_sentence")
    if stripped and stripped[-1] not in ".?!।":
        flags.append("possibly_ends_mid_sentence")
    if len(stripped.split()) < 3:
        flags.append("very_short_transcript")

    if float(row.get("snr_db") or 0) < 20:
        flags.append("low_snr")
    if float(row.get("ecapa_similarity") or 0) < 0.85:
        flags.append("weak_speaker_match")
    if not 3 <= float(row.get("duration_seconds") or 0) <= 30:
        flags.append("bad_duration")

    source_text = " ".join(
        str(row.get(k) or "")
        for k in ("source_speaker_name", "source_title", "source_channel")
    )
    if READ_SOURCE_RE.search(source_text):
        flags.append("likely_read_or_voiceover_source")

    if row.get("sample_rate") != 24000:
        flags.append("wrong_sample_rate")
    if row.get("channels") != 1:
        flags.append("not_mono")
    if row.get("bit_depth") != 16:
        flags.append("not_16_bit")
    if abs(float(row["actual_duration_s"]) - float(row["duration_seconds"])) > 0.15:
        flags.append("metadata_duration_mismatch")
    if float(row["peak_dbfs"]) > -0.1:
        flags.append("near_clipping")
    if float(row["clipping_pct"]) > 0.05:
        flags.append("clipped_audio")
    if float(row["active_ratio"]) < 0.45:
        flags.append("too_much_silence_or_low_activity")

    return flags


def load_rows(parquet_dir: Path, wav_dir: Path) -> list[dict]:
    rows = []
    for parquet_path in sorted(parquet_dir.glob("*.parquet")):
        split = parquet_path.name.replace("-00000-of-00001.parquet", "")
        dataset = load_dataset("parquet", data_files=str(parquet_path), split="train")
        for idx, record in enumerate(dataset):
            wav_path = wav_dir / split / f"{split}_{idx:05d}.wav"
            if not wav_path.exists():
                base = {
                    k: v
                    for k, v in record.items()
                    if k != "audio" and not isinstance(v, (bytes, bytearray))
                }
                base["wav_path"] = str(wav_path)
                base["flags"] = "missing_wav"
                rows.append(base)
                continue

            base = {
                k: v
                for k, v in record.items()
                if k != "audio" and not isinstance(v, (bytes, bytearray))
            }
            base["wav_path"] = str(wav_path)
            base.update(wav_header(wav_path))
            base.update(audio_metrics(wav_path))
            devanagari_ratio, latin_ratio = char_ratios(base.get("text") or "")
            base["devanagari_ratio"] = devanagari_ratio
            base["latin_ratio"] = latin_ratio
            flags = flag_row(base)
            base["flags"] = ";".join(flags)
            base["flag_count"] = len(flags)
            rows.append(base)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "wav_path",
        "language",
        "split",
        "duration_seconds",
        "actual_duration_s",
        "sample_rate",
        "channels",
        "bit_depth",
        "snr_db",
        "ecapa_similarity",
        "rms_dbfs",
        "peak_dbfs",
        "clipping_pct",
        "active_ratio",
        "devanagari_ratio",
        "latin_ratio",
        "emotion",
        "style",
        "source_speaker_name",
        "source_title",
        "source_channel",
        "source_url",
        "text",
        "flags",
        "flag_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    by_lang = defaultdict(list)
    by_split = defaultdict(list)
    flag_counts = Counter()
    source_counts = Counter()
    for row in rows:
        by_lang[row.get("language")].append(row)
        by_split[row.get("split")].append(row)
        source_counts[row.get("source_video_id")] += 1
        for flag in filter(None, str(row.get("flags", "")).split(";")):
            flag_counts[flag] += 1

    print("Dataset summary")
    for lang, items in sorted(by_lang.items()):
        total_min = sum(float(r.get("actual_duration_s") or 0) for r in items) / 60
        print(f"  {lang}: {len(items)} clips, {total_min:.2f} min")
    for split, items in sorted(by_split.items()):
        total_min = sum(float(r.get("actual_duration_s") or 0) for r in items) / 60
        print(f"  {split}: {len(items)} clips, {total_min:.2f} min")

    print("\nTop flags")
    for flag, count in flag_counts.most_common(12):
        print(f"  {flag}: {count}")

    print("\nLargest source contributions")
    for video_id, count in source_counts.most_common(10):
        title = next((r.get("source_title") for r in rows if r.get("source_video_id") == video_id), "")
        print(f"  {video_id}: {count} clips - {title}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", type=Path, default=Path.home() / "sarvam-indian-tts-60min/data")
    parser.add_argument("--wav-dir", type=Path, default=Path("extracted_wav"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/extracted_wav_assessment"))
    parser.add_argument("--review-size", type=int, default=60)
    args = parser.parse_args()

    rows = load_rows(args.parquet_dir, args.wav_dir)
    rows.sort(key=lambda r: int(r.get("flag_count") or 0), reverse=True)

    write_csv(args.out_dir / "all_clips_assessment.csv", rows)
    write_csv(args.out_dir / "manual_review_priority.csv", rows[: args.review_size])

    high_risk = [r for r in rows if int(r.get("flag_count") or 0) >= 2]
    write_csv(args.out_dir / "high_risk_clips.csv", high_risk)
    print_summary(rows)
    print(f"\nWrote reports to {args.out_dir}")


if __name__ == "__main__":
    main()
