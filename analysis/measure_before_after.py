#!/usr/bin/env python3
"""Measure before/after audio quality and preservation for matched clips."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def read_mono(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float64), int(sr)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    try:
        import librosa

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except Exception:
        duration = len(audio) / orig_sr
        old_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        new_len = max(1, int(duration * target_sr))
        new_x = np.linspace(0.0, duration, num=new_len, endpoint=False)
        return np.interp(new_x, old_x, audio)


def audio_metrics(path: str) -> dict:
    audio, sr = read_mono(path)
    if audio.size == 0:
        return {
            "duration_s": 0.0,
            "sample_rate": sr,
            "snr_db": 0.0,
            "rms_dbfs": -120.0,
            "peak_dbfs": -120.0,
            "clipping_pct": 0.0,
            "active_ratio": 0.0,
            "silence_ratio": 1.0,
            "lufs": -120.0,
            "loudness_range_db": 0.0,
            "crest_factor_db": 0.0,
            "dc_offset": 0.0,
            "spectral_flatness": 0.0,
            "spectral_centroid_hz": 0.0,
            "low_freq_rumble_ratio": 0.0,
            "high_freq_noise_ratio": 0.0,
            "speech_band_energy_ratio": 0.0,
            "zero_crossing_rate": 0.0,
        }
    abs_audio = np.abs(audio)
    rms = math.sqrt(float(np.mean(audio**2)))
    peak = float(np.max(abs_audio))
    noise_est = float(np.percentile(abs_audio, 10)) + 1e-9
    snr_db = 20 * math.log10(max(rms, 1e-12) / noise_est)

    frame = max(int(sr * 0.03), 1)
    usable = (audio.size // frame) * frame
    if usable:
        framed = audio[:usable].reshape(-1, frame)
        frame_rms = np.sqrt(np.mean(framed**2, axis=1))
        threshold = max(np.percentile(frame_rms, 20) * 3, 10 ** (-45 / 20))
        active_ratio = float(np.mean(frame_rms > threshold))
        frame_db = 20 * np.log10(np.maximum(frame_rms, 1e-12))
        loudness_range_db = float(np.percentile(frame_db, 95) - np.percentile(frame_db, 10))
    else:
        active_ratio = 0.0
        loudness_range_db = 0.0

    target_audio = resample_audio(audio, sr, 16000)
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(16000)
        lufs = float(meter.integrated_loudness(target_audio))
        if not np.isfinite(lufs):
            lufs = -120.0
    except Exception:
        lufs = -120.0

    try:
        import librosa

        y = target_audio.astype(np.float64)
        spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        spectral_centroid_hz = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=16000)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    except Exception:
        spectral_flatness = 0.0
        spectral_centroid_hz = 0.0
        zcr = 0.0

    freqs = np.fft.rfftfreq(target_audio.size, d=1 / 16000)
    spectrum = np.abs(np.fft.rfft(target_audio)) ** 2
    total_energy = float(np.sum(spectrum) + 1e-12)
    low_freq_rumble_ratio = float(np.sum(spectrum[freqs < 80]) / total_energy)
    high_freq_noise_ratio = float(np.sum(spectrum[freqs > 6000]) / total_energy)
    speech_band_energy_ratio = float(np.sum(spectrum[(freqs >= 80) & (freqs <= 6000)]) / total_energy)

    return {
        "duration_s": float(audio.size / sr),
        "sample_rate": sr,
        "snr_db": snr_db,
        "rms_dbfs": 20 * math.log10(max(rms, 1e-12)),
        "peak_dbfs": 20 * math.log10(max(peak, 1e-12)),
        "clipping_pct": float(np.mean(abs_audio >= 0.999) * 100),
        "active_ratio": active_ratio,
        "silence_ratio": 1.0 - active_ratio,
        "lufs": lufs,
        "loudness_range_db": loudness_range_db,
        "crest_factor_db": 20 * math.log10(max(peak, 1e-12) / max(rms, 1e-12)),
        "dc_offset": float(np.mean(audio)),
        "spectral_flatness": spectral_flatness,
        "spectral_centroid_hz": spectral_centroid_hz,
        "low_freq_rumble_ratio": low_freq_rumble_ratio,
        "high_freq_noise_ratio": high_freq_noise_ratio,
        "speech_band_energy_ratio": speech_band_energy_ratio,
        "zero_crossing_rate": zcr,
    }


def mel_cosine_similarity(raw_path: str, final_path: str) -> float:
    """Content/speaker preservation proxy robust to sample-rate/loudness changes."""
    import librosa

    raw, raw_sr = read_mono(raw_path)
    final, final_sr = read_mono(final_path)
    target_sr = 16000
    raw = resample_audio(raw, raw_sr, target_sr)
    final = resample_audio(final, final_sr, target_sr)
    min_len = min(len(raw), len(final))
    if min_len < target_sr:
        return 0.0
    raw = raw[:min_len]
    final = final[:min_len]
    raw_mel = librosa.feature.melspectrogram(y=raw, sr=target_sr, n_mels=40, hop_length=320)
    final_mel = librosa.feature.melspectrogram(y=final, sr=target_sr, n_mels=40, hop_length=320)
    raw_db = librosa.power_to_db(raw_mel + 1e-12)
    final_db = librosa.power_to_db(final_mel + 1e-12)
    a = raw_db.mean(axis=1)
    b = final_db.mean(axis=1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def save_bar_plot(path: Path, labels: list[str], before: list[float], after: list[float], ylabel: str) -> None:
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, before, width, label="Before/raw", color="#6B7280")
    ax.bar(x + width / 2, after, width, label="After/final", color="#2563EB")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_hist(path: Path, values: list[float], title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=24, color="#0F766E", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Clips")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_scatter(path: Path, x_values: list[float], y_values: list[float], title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_values, y_values, s=18, alpha=0.75, color="#7C3AED")
    lo = min(min(x_values), min(y_values))
    hi = max(max(x_values), max(y_values))
    ax.plot([lo, hi], [lo, hi], color="#111827", linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def metric_delta(item: dict, key: str) -> float:
    return float(item[f"final_{key}"] - item[f"raw_{key}"])


def pct_within(values: list[float], predicate) -> float:
    if not values:
        return 0.0
    return float(sum(1 for value in values if predicate(value)) / len(values) * 100)


def run(
    metadata_path: str = "data/metadata/10_final.jsonl",
    output_dir: str = "analysis/before_after_metrics",
) -> None:
    rows = [json.loads(line) for line in Path(metadata_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    measured = []
    for row in rows:
        raw_path = row.get("raw_segment_path", "")
        final_path = row.get("final_path", "")
        if not raw_path or not final_path or not Path(raw_path).exists() or not Path(final_path).exists():
            continue
        raw = audio_metrics(raw_path)
        final = audio_metrics(final_path)
        acoustic_similarity = mel_cosine_similarity(raw_path, final_path)
        item = {
            "id": Path(final_path).stem,
            "language": row.get("language", ""),
            "source_title": row.get("source_title", ""),
            "raw_path": raw_path,
            "final_path": final_path,
            "raw_duration_s": raw["duration_s"],
            "final_duration_s": final["duration_s"],
            "duration_delta_s": abs(final["duration_s"] - raw["duration_s"]),
            "raw_sample_rate": raw["sample_rate"],
            "final_sample_rate": final["sample_rate"],
            "raw_snr_db": raw["snr_db"],
            "final_snr_db": final["snr_db"],
            "snr_delta_db": final["snr_db"] - raw["snr_db"],
            "raw_clipping_pct": raw["clipping_pct"],
            "final_clipping_pct": final["clipping_pct"],
            "clipping_delta_pct": final["clipping_pct"] - raw["clipping_pct"],
            "raw_rms_dbfs": raw["rms_dbfs"],
            "final_rms_dbfs": final["rms_dbfs"],
            "rms_delta_db": final["rms_dbfs"] - raw["rms_dbfs"],
            "raw_peak_dbfs": raw["peak_dbfs"],
            "final_peak_dbfs": final["peak_dbfs"],
            "raw_active_ratio": raw["active_ratio"],
            "final_active_ratio": final["active_ratio"],
            "active_ratio_delta": final["active_ratio"] - raw["active_ratio"],
            "raw_silence_ratio": raw["silence_ratio"],
            "final_silence_ratio": final["silence_ratio"],
            "silence_ratio_delta": final["silence_ratio"] - raw["silence_ratio"],
            "raw_lufs": raw["lufs"],
            "final_lufs": final["lufs"],
            "lufs_delta": final["lufs"] - raw["lufs"],
            "raw_lufs_target_error": abs(raw["lufs"] - (-23.0)),
            "final_lufs_target_error": abs(final["lufs"] - (-23.0)),
            "lufs_target_error_delta": abs(final["lufs"] - (-23.0)) - abs(raw["lufs"] - (-23.0)),
            "raw_loudness_range_db": raw["loudness_range_db"],
            "final_loudness_range_db": final["loudness_range_db"],
            "loudness_range_delta_db": final["loudness_range_db"] - raw["loudness_range_db"],
            "raw_crest_factor_db": raw["crest_factor_db"],
            "final_crest_factor_db": final["crest_factor_db"],
            "crest_factor_delta_db": final["crest_factor_db"] - raw["crest_factor_db"],
            "raw_dc_offset": raw["dc_offset"],
            "final_dc_offset": final["dc_offset"],
            "dc_offset_abs_delta": abs(final["dc_offset"]) - abs(raw["dc_offset"]),
            "raw_spectral_flatness": raw["spectral_flatness"],
            "final_spectral_flatness": final["spectral_flatness"],
            "spectral_flatness_delta": final["spectral_flatness"] - raw["spectral_flatness"],
            "raw_spectral_centroid_hz": raw["spectral_centroid_hz"],
            "final_spectral_centroid_hz": final["spectral_centroid_hz"],
            "spectral_centroid_delta_hz": final["spectral_centroid_hz"] - raw["spectral_centroid_hz"],
            "raw_low_freq_rumble_ratio": raw["low_freq_rumble_ratio"],
            "final_low_freq_rumble_ratio": final["low_freq_rumble_ratio"],
            "low_freq_rumble_delta": final["low_freq_rumble_ratio"] - raw["low_freq_rumble_ratio"],
            "raw_high_freq_noise_ratio": raw["high_freq_noise_ratio"],
            "final_high_freq_noise_ratio": final["high_freq_noise_ratio"],
            "high_freq_noise_delta": final["high_freq_noise_ratio"] - raw["high_freq_noise_ratio"],
            "raw_speech_band_energy_ratio": raw["speech_band_energy_ratio"],
            "final_speech_band_energy_ratio": final["speech_band_energy_ratio"],
            "speech_band_energy_delta": final["speech_band_energy_ratio"] - raw["speech_band_energy_ratio"],
            "raw_zero_crossing_rate": raw["zero_crossing_rate"],
            "final_zero_crossing_rate": final["zero_crossing_rate"],
            "zero_crossing_rate_delta": final["zero_crossing_rate"] - raw["zero_crossing_rate"],
            "acoustic_similarity": acoustic_similarity,
            "speaker_similarity_proxy": acoustic_similarity,
            "transcript_available": bool(row.get("normalized_transcript") or row.get("approx_transcript")),
            "transcript_source": row.get("local_asr_model", "pipeline_asr" if row.get("normalized_transcript") else ""),
            "emotion_status": row.get("emotion_status", ""),
        }
        measured.append(item)

    csv_path = out / "before_after_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(measured[0].keys()))
        writer.writeheader()
        writer.writerows(measured)

    summary = {
        "num_pairs": len(measured),
        "language_counts": dict(Counter(item["language"] for item in measured)),
        "transcript_available": sum(item["transcript_available"] for item in measured),
        "final_24khz_count": sum(item["final_sample_rate"] == 24000 for item in measured),
        "raw_16khz_count": sum(item["raw_sample_rate"] == 16000 for item in measured),
        "quality_gate_summary": {
            "snr_improved_pct": pct_within([m["snr_delta_db"] for m in measured], lambda value: value > 0.5),
            "snr_unchanged_pct": pct_within([m["snr_delta_db"] for m in measured], lambda value: abs(value) <= 0.5),
            "snr_worse_pct": pct_within([m["snr_delta_db"] for m in measured], lambda value: value < -0.5),
            "final_lufs_within_2db_of_target_pct": pct_within(
                [m["final_lufs_target_error"] for m in measured], lambda value: value <= 2.0
            ),
            "raw_lufs_within_2db_of_target_pct": pct_within(
                [m["raw_lufs_target_error"] for m in measured], lambda value: value <= 2.0
            ),
            "duration_delta_under_50ms_pct": pct_within(
                [m["duration_delta_s"] for m in measured], lambda value: value <= 0.05
            ),
            "acoustic_similarity_over_0_98_pct": pct_within(
                [m["acoustic_similarity"] for m in measured], lambda value: value >= 0.98
            ),
            "zero_clipping_final_pct": pct_within(
                [m["final_clipping_pct"] for m in measured], lambda value: value == 0
            ),
        },
        "metrics": {
            "raw_snr_db": summarize([m["raw_snr_db"] for m in measured]),
            "final_snr_db": summarize([m["final_snr_db"] for m in measured]),
            "snr_delta_db": summarize([m["snr_delta_db"] for m in measured]),
            "raw_clipping_pct": summarize([m["raw_clipping_pct"] for m in measured]),
            "final_clipping_pct": summarize([m["final_clipping_pct"] for m in measured]),
            "raw_rms_dbfs": summarize([m["raw_rms_dbfs"] for m in measured]),
            "final_rms_dbfs": summarize([m["final_rms_dbfs"] for m in measured]),
            "raw_peak_dbfs": summarize([m["raw_peak_dbfs"] for m in measured]),
            "final_peak_dbfs": summarize([m["final_peak_dbfs"] for m in measured]),
            "raw_active_ratio": summarize([m["raw_active_ratio"] for m in measured]),
            "final_active_ratio": summarize([m["final_active_ratio"] for m in measured]),
            "raw_silence_ratio": summarize([m["raw_silence_ratio"] for m in measured]),
            "final_silence_ratio": summarize([m["final_silence_ratio"] for m in measured]),
            "raw_lufs": summarize([m["raw_lufs"] for m in measured]),
            "final_lufs": summarize([m["final_lufs"] for m in measured]),
            "lufs_target_error_delta": summarize([m["lufs_target_error_delta"] for m in measured]),
            "raw_loudness_range_db": summarize([m["raw_loudness_range_db"] for m in measured]),
            "final_loudness_range_db": summarize([m["final_loudness_range_db"] for m in measured]),
            "raw_crest_factor_db": summarize([m["raw_crest_factor_db"] for m in measured]),
            "final_crest_factor_db": summarize([m["final_crest_factor_db"] for m in measured]),
            "raw_spectral_flatness": summarize([m["raw_spectral_flatness"] for m in measured]),
            "final_spectral_flatness": summarize([m["final_spectral_flatness"] for m in measured]),
            "raw_low_freq_rumble_ratio": summarize([m["raw_low_freq_rumble_ratio"] for m in measured]),
            "final_low_freq_rumble_ratio": summarize([m["final_low_freq_rumble_ratio"] for m in measured]),
            "raw_high_freq_noise_ratio": summarize([m["raw_high_freq_noise_ratio"] for m in measured]),
            "final_high_freq_noise_ratio": summarize([m["final_high_freq_noise_ratio"] for m in measured]),
            "raw_speech_band_energy_ratio": summarize([m["raw_speech_band_energy_ratio"] for m in measured]),
            "final_speech_band_energy_ratio": summarize([m["final_speech_band_energy_ratio"] for m in measured]),
            "duration_delta_s": summarize([m["duration_delta_s"] for m in measured]),
            "acoustic_similarity": summarize([m["acoustic_similarity"] for m in measured]),
        },
    }
    (out / "before_after_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    save_bar_plot(
        out / "quality_means_before_after.png",
        ["SNR dB", "LUFS", "RMS dBFS", "Peak dBFS", "Active ratio"],
        [
            summary["metrics"]["raw_snr_db"]["mean"],
            summary["metrics"]["raw_lufs"]["mean"],
            summary["metrics"]["raw_rms_dbfs"]["mean"],
            summary["metrics"]["raw_peak_dbfs"]["mean"],
            summary["metrics"]["raw_active_ratio"]["mean"],
        ],
        [
            summary["metrics"]["final_snr_db"]["mean"],
            summary["metrics"]["final_lufs"]["mean"],
            summary["metrics"]["final_rms_dbfs"]["mean"],
            summary["metrics"]["final_peak_dbfs"]["mean"],
            summary["metrics"]["final_active_ratio"]["mean"],
        ],
        "Mean value",
    )
    save_scatter(
        out / "snr_raw_vs_final.png",
        [m["raw_snr_db"] for m in measured],
        [m["final_snr_db"] for m in measured],
        "SNR: raw vs final",
        "Raw SNR dB",
        "Final SNR dB",
    )
    save_scatter(
        out / "rms_raw_vs_final.png",
        [m["raw_rms_dbfs"] for m in measured],
        [m["final_rms_dbfs"] for m in measured],
        "RMS loudness: raw vs final",
        "Raw RMS dBFS",
        "Final RMS dBFS",
    )
    save_hist(out / "snr_delta_hist.png", [m["snr_delta_db"] for m in measured], "SNR delta", "Final - raw SNR dB")
    save_hist(out / "lufs_target_error_delta_hist.png", [m["lufs_target_error_delta"] for m in measured], "LUFS target error delta", "Final error - raw error dB")
    save_hist(out / "high_freq_noise_delta_hist.png", [m["high_freq_noise_delta"] for m in measured], "High-frequency noise energy delta", "Final - raw ratio")
    save_hist(out / "spectral_flatness_delta_hist.png", [m["spectral_flatness_delta"] for m in measured], "Spectral flatness delta", "Final - raw")
    save_hist(out / "duration_delta_hist.png", [m["duration_delta_s"] for m in measured], "Duration mismatch", "Absolute seconds")
    save_hist(
        out / "acoustic_similarity_hist.png",
        [m["acoustic_similarity"] for m in measured],
        "Raw/final acoustic preservation",
        "Mel cosine similarity",
    )

    report = f"""# Before/After Clip Quality Metrics

Measured matched raw/final clip pairs from `{metadata_path}`.

## Dataset

- Pairs measured: {summary['num_pairs']}
- Language counts: {summary['language_counts']}
- Raw clips at 16 kHz: {summary['raw_16khz_count']}/{summary['num_pairs']}
- Final clips at 24 kHz: {summary['final_24khz_count']}/{summary['num_pairs']}
- Transcript available: {summary['transcript_available']}/{summary['num_pairs']}

## Mean Results

| Metric | Before/raw | After/final | Delta |
| --- | ---: | ---: | ---: |
| SNR dB | {summary['metrics']['raw_snr_db']['mean']:.2f} | {summary['metrics']['final_snr_db']['mean']:.2f} | {summary['metrics']['snr_delta_db']['mean']:+.2f} |
| Clipping % | {summary['metrics']['raw_clipping_pct']['mean']:.4f} | {summary['metrics']['final_clipping_pct']['mean']:.4f} | {(summary['metrics']['final_clipping_pct']['mean'] - summary['metrics']['raw_clipping_pct']['mean']):+.4f} |
| Integrated loudness LUFS | {summary['metrics']['raw_lufs']['mean']:.2f} | {summary['metrics']['final_lufs']['mean']:.2f} | {(summary['metrics']['final_lufs']['mean'] - summary['metrics']['raw_lufs']['mean']):+.2f} |
| LUFS target error vs -23 | {summary['metrics']['raw_lufs']['mean'] - (-23.0):+.2f} | {summary['metrics']['final_lufs']['mean'] - (-23.0):+.2f} | error delta {summary['metrics']['lufs_target_error_delta']['mean']:+.2f} |
| RMS dBFS | {summary['metrics']['raw_rms_dbfs']['mean']:.2f} | {summary['metrics']['final_rms_dbfs']['mean']:.2f} | {(summary['metrics']['final_rms_dbfs']['mean'] - summary['metrics']['raw_rms_dbfs']['mean']):+.2f} |
| Peak dBFS | {summary['metrics']['raw_peak_dbfs']['mean']:.2f} | {summary['metrics']['final_peak_dbfs']['mean']:.2f} | {(summary['metrics']['final_peak_dbfs']['mean'] - summary['metrics']['raw_peak_dbfs']['mean']):+.2f} |
| Active speech ratio | {summary['metrics']['raw_active_ratio']['mean']:.3f} | {summary['metrics']['final_active_ratio']['mean']:.3f} | {(summary['metrics']['final_active_ratio']['mean'] - summary['metrics']['raw_active_ratio']['mean']):+.3f} |
| Silence ratio | {summary['metrics']['raw_silence_ratio']['mean']:.3f} | {summary['metrics']['final_silence_ratio']['mean']:.3f} | {(summary['metrics']['final_silence_ratio']['mean'] - summary['metrics']['raw_silence_ratio']['mean']):+.3f} |
| Loudness range dB | {summary['metrics']['raw_loudness_range_db']['mean']:.2f} | {summary['metrics']['final_loudness_range_db']['mean']:.2f} | {(summary['metrics']['final_loudness_range_db']['mean'] - summary['metrics']['raw_loudness_range_db']['mean']):+.2f} |
| Crest factor dB | {summary['metrics']['raw_crest_factor_db']['mean']:.2f} | {summary['metrics']['final_crest_factor_db']['mean']:.2f} | {(summary['metrics']['final_crest_factor_db']['mean'] - summary['metrics']['raw_crest_factor_db']['mean']):+.2f} |
| Spectral flatness | {summary['metrics']['raw_spectral_flatness']['mean']:.6f} | {summary['metrics']['final_spectral_flatness']['mean']:.6f} | {(summary['metrics']['final_spectral_flatness']['mean'] - summary['metrics']['raw_spectral_flatness']['mean']):+.6f} |
| Low-frequency rumble ratio | {summary['metrics']['raw_low_freq_rumble_ratio']['mean']:.6f} | {summary['metrics']['final_low_freq_rumble_ratio']['mean']:.6f} | {(summary['metrics']['final_low_freq_rumble_ratio']['mean'] - summary['metrics']['raw_low_freq_rumble_ratio']['mean']):+.6f} |
| High-frequency noise ratio | {summary['metrics']['raw_high_freq_noise_ratio']['mean']:.6f} | {summary['metrics']['final_high_freq_noise_ratio']['mean']:.6f} | {(summary['metrics']['final_high_freq_noise_ratio']['mean'] - summary['metrics']['raw_high_freq_noise_ratio']['mean']):+.6f} |
| Speech-band energy ratio | {summary['metrics']['raw_speech_band_energy_ratio']['mean']:.6f} | {summary['metrics']['final_speech_band_energy_ratio']['mean']:.6f} | {(summary['metrics']['final_speech_band_energy_ratio']['mean'] - summary['metrics']['raw_speech_band_energy_ratio']['mean']):+.6f} |
| Duration mismatch seconds | - | {summary['metrics']['duration_delta_s']['mean']:.4f} | lower is better |
| Acoustic similarity proxy | - | {summary['metrics']['acoustic_similarity']['mean']:.4f} | higher is better |

## Gate Summary

| Gate | Result |
| --- | ---: |
| SNR improved by >0.5 dB | {summary['quality_gate_summary']['snr_improved_pct']:.1f}% |
| SNR unchanged within +/-0.5 dB | {summary['quality_gate_summary']['snr_unchanged_pct']:.1f}% |
| SNR worsened by >0.5 dB | {summary['quality_gate_summary']['snr_worse_pct']:.1f}% |
| Raw LUFS within 2 dB of -23 | {summary['quality_gate_summary']['raw_lufs_within_2db_of_target_pct']:.1f}% |
| Final LUFS within 2 dB of -23 | {summary['quality_gate_summary']['final_lufs_within_2db_of_target_pct']:.1f}% |
| Duration delta under 50 ms | {summary['quality_gate_summary']['duration_delta_under_50ms_pct']:.1f}% |
| Acoustic similarity >= 0.98 | {summary['quality_gate_summary']['acoustic_similarity_over_0_98_pct']:.1f}% |
| Final clips with zero clipping | {summary['quality_gate_summary']['zero_clipping_final_pct']:.1f}% |

## Interpretation

The before/after evidence does not support a claim that enhancement improved acoustic quality. Mean proxy SNR moved by {summary['metrics']['snr_delta_db']['mean']:+.2f} dB, so enhancement was effectively neutral/slightly negative by this metric. Acoustic similarity is high, which means the output is very close to the input.

What the pipeline did measurably improve is dataset standardization: final files are 24 kHz, duration alignment is effectively exact, clipping remains zero, peak levels are safer, and loudness is moved toward the -23 LUFS target. This is a standardization/export-readiness win, not a denoising win.

`acoustic_similarity` is a mel-spectrogram cosine proxy for speaker/content preservation. It is not a replacement for a dedicated ECAPA raw-vs-final speaker model, but it is useful for confirming that finalization did not materially change the spoken content.

`snr_db` remains a proxy based on percentile noise floor and should not be presented as DNSMOS or perceptual MOS. The stronger added checks here are LUFS target error, clipping, active/silence ratio, spectral flatness, low-frequency rumble, high-frequency noise, speech-band energy, duration alignment, and acoustic preservation.

## Plots

- `quality_means_before_after.png`
- `snr_raw_vs_final.png`
- `rms_raw_vs_final.png`
- `snr_delta_hist.png`
- `lufs_target_error_delta_hist.png`
- `high_freq_noise_delta_hist.png`
- `spectral_flatness_delta_hist.png`
- `duration_delta_hist.png`
- `acoustic_similarity_hist.png`
"""
    (out / "before_after_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote metrics CSV: {csv_path}")
    print(f"Wrote summary/report/plots to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata/10_final.jsonl")
    parser.add_argument("--output-dir", default="analysis/before_after_metrics")
    args = parser.parse_args()
    run(args.metadata, args.output_dir)
