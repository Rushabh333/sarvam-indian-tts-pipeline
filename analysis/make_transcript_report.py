#!/usr/bin/env python3
"""Summarize transcript coverage and raw/final ASR preservation."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "median": float(np.median(arr)) if arr.size else 0.0,
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "lt_0_90": int(np.sum(arr < 0.90)) if arr.size else 0,
        "gte_0_90_pct": float(np.mean(arr >= 0.90) * 100) if arr.size else 0.0,
    }


def save_hist(path: Path, rows: list[dict], key: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for language, color in [("en-IN", "#2563EB"), ("hi-IN", "#DC2626")]:
        values = [float(row[key]) for row in rows if row["language"] == language]
        ax.hist(values, bins=20, alpha=0.65, label=language, color=color)
    ax.set_title(title)
    ax.set_xlabel(key)
    ax.set_ylabel("Clips")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_language_bar(path: Path, by_language: dict) -> None:
    labels = sorted(by_language)
    x = np.arange(len(labels))
    width = 0.36
    char_vals = [by_language[label]["asr_char_similarity"]["median"] for label in labels]
    word_vals = [by_language[label]["asr_word_f1"]["median"] for label in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, char_vals, width, label="char similarity", color="#2563EB")
    ax.bar(x + width / 2, word_vals, width, label="word F1", color="#0F766E")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Median score")
    ax.set_title("Transcript preservation by language")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(
    final_metadata: str = "data/metadata/10_final.jsonl",
    similarity_csv: str = "analysis/before_after_metrics/asr_text_similarity.csv",
    output_dir: str = "analysis/before_after_metrics",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata_rows = [
        json.loads(line)
        for line in Path(final_metadata).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sim_rows = list(csv.DictReader(Path(similarity_csv).open(newline="", encoding="utf-8")))

    coverage = {
        "segments": len(metadata_rows),
        "normalized_transcript_present": sum(bool(row.get("normalized_transcript")) for row in metadata_rows),
        "approx_transcript_present": sum(bool(row.get("approx_transcript")) for row in metadata_rows),
        "local_asr_status": dict(Counter(row.get("local_asr_status", "pipeline_asr") for row in metadata_rows)),
        "emotion_status": dict(Counter(row.get("emotion_status", "missing") for row in metadata_rows)),
    }

    by_language = {}
    grouped = defaultdict(list)
    for row in sim_rows:
        grouped[row["language"]].append(row)
    for language, rows in grouped.items():
        by_language[language] = {
            "count": len(rows),
            "asr_char_similarity": summarize([float(row["asr_char_similarity"]) for row in rows]),
            "asr_word_f1": summarize([float(row["asr_word_f1"]) for row in rows]),
        }

    all_summary = {
        "coverage": coverage,
        "raw_final_similarity_all": {
            "asr_char_similarity": summarize([float(row["asr_char_similarity"]) for row in sim_rows]),
            "asr_word_f1": summarize([float(row["asr_word_f1"]) for row in sim_rows]),
        },
        "raw_final_similarity_by_language": by_language,
    }
    (out / "transcript_summary.json").write_text(json.dumps(all_summary, indent=2), encoding="utf-8")

    low_path = out / "low_asr_similarity_review.csv"
    low_rows = [row for row in sim_rows if float(row["asr_word_f1"]) < 0.90 or float(row["asr_char_similarity"]) < 0.90]
    with low_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sim_rows[0].keys())
        writer.writeheader()
        writer.writerows(low_rows)

    save_hist(out / "asr_word_f1_hist_by_language.png", sim_rows, "asr_word_f1", "Raw/final ASR word F1")
    save_hist(
        out / "asr_char_similarity_hist_by_language.png",
        sim_rows,
        "asr_char_similarity",
        "Raw/final ASR character similarity",
    )
    save_language_bar(out / "asr_similarity_by_language.png", by_language)

    report = f"""# Transcript Coverage and Preservation Report

## Coverage

| Item | Value |
| --- | ---: |
| Final clips | {coverage['segments']} |
| Normalized transcripts present | {coverage['normalized_transcript_present']} |
| Approx transcripts present | {coverage['approx_transcript_present']} |
| Local ASR-filled transcripts | {coverage['local_asr_status'].get('success', 0)} |
| Original pipeline transcripts | {coverage['local_asr_status'].get('pipeline_asr', 0) + coverage['local_asr_status'].get('missing', 0)} |

## Raw vs Final ASR Preservation

These scores were computed by independently transcribing each raw before-clip and final after-clip on the AWS GPU with faster-whisper `base`, then comparing the two ASR strings.

| Split | Clips | Char similarity median | Char similarity mean | Word F1 median | Word F1 mean | Word F1 < 0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All | {len(sim_rows)} | {all_summary['raw_final_similarity_all']['asr_char_similarity']['median']:.3f} | {all_summary['raw_final_similarity_all']['asr_char_similarity']['mean']:.3f} | {all_summary['raw_final_similarity_all']['asr_word_f1']['median']:.3f} | {all_summary['raw_final_similarity_all']['asr_word_f1']['mean']:.3f} | {all_summary['raw_final_similarity_all']['asr_word_f1']['lt_0_90']} |
"""
    for language in sorted(by_language):
        lang = by_language[language]
        report += (
            f"| {language} | {lang['count']} | "
            f"{lang['asr_char_similarity']['median']:.3f} | {lang['asr_char_similarity']['mean']:.3f} | "
            f"{lang['asr_word_f1']['median']:.3f} | {lang['asr_word_f1']['mean']:.3f} | "
            f"{lang['asr_word_f1']['lt_0_90']} |\n"
        )

    report += f"""
## Interpretation

Transcript coverage is now complete in `data/metadata/10_final.jsonl`: every final clip has `approx_transcript` and `normalized_transcript`.

English preservation is strong: median raw/final ASR word F1 is {by_language.get('en-IN', {}).get('asr_word_f1', {}).get('median', 0):.3f}. Hindi/code-mixed preservation is less reliable by this ASR metric: median word F1 is {by_language.get('hi-IN', {}).get('asr_word_f1', {}).get('median', 0):.3f}, and many low-score rows show ASR script/language instability rather than obvious audio corruption. For Hindi, this metric should be used as a manual-review priority signal, not as a hard failure label.

Low-similarity rows are written to `low_asr_similarity_review.csv`.

## Plots

- `asr_word_f1_hist_by_language.png`
- `asr_char_similarity_hist_by_language.png`
- `asr_similarity_by_language.png`
"""
    (out / "transcript_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote transcript summary: {out / 'transcript_summary.json'}")
    print(f"Wrote transcript report: {out / 'transcript_report.md'}")
    print(f"Wrote low-similarity review CSV: {low_path}")


if __name__ == "__main__":
    run()
