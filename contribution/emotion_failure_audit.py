#!/usr/bin/env python3
"""
Audit emotion-stage collapse and prepare manual labels.

This captures the real failure found in iteration 2: API quota/rate failures
were converted into valid-looking neutral/conversational labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(
    metadata_path: str = "data/metadata/10_final.jsonl",
    output_csv: str = "contribution/results/emotion_manual_labels_seed.csv",
    output_summary: str = "contribution/results/emotion_failure_audit.json",
) -> None:
    rows = load_jsonl(Path(metadata_path))
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    emotions = Counter(row.get("emotion", "missing") for row in rows)
    styles = Counter(row.get("style", "missing") for row in rows)
    statuses = Counter(row.get("emotion_status", "missing") for row in rows)
    languages = Counter(row.get("language", "missing") for row in rows)
    empty_norm = sum(not row.get("normalized_transcript") for row in rows)
    review_required = sum(bool(row.get("emotion_review_required")) for row in rows)
    total_minutes = sum(float(row.get("duration_s") or 0) for row in rows) / 60
    raw_pairs = sum(bool(row.get("raw_segment_path")) for row in rows)

    summary = {
        "segments": len(rows),
        "total_minutes": round(total_minutes, 2),
        "languages": dict(languages),
        "emotions": dict(emotions),
        "styles": dict(styles),
        "statuses": dict(statuses),
        "empty_normalized_transcript": empty_norm,
        "emotion_review_required": review_required,
        "before_after_pairs": raw_pairs,
    }

    print("Emotion audit")
    print(f"  segments: {summary['segments']}")
    print(f"  total_minutes: {summary['total_minutes']}")
    print(f"  languages: {summary['languages']}")
    print(f"  emotions: {summary['emotions']}")
    print(f"  styles: {summary['styles']}")
    print(f"  statuses: {summary['statuses']}")
    print(f"  empty_normalized_transcript: {empty_norm}")
    print(f"  emotion_review_required: {review_required}")
    print(f"  before_after_pairs: {raw_pairs}")

    fields = [
        "audio_path",
        "raw_segment_path",
        "language",
        "duration_s",
        "source_title",
        "source_url",
        "source_start_time_seconds",
        "source_end_time_seconds",
        "transcript",
        "pipeline_emotion",
        "pipeline_style",
        "emotion_status",
        "emotion_failure_reason",
        "emotion_review_required",
        "manual_emotion",
        "manual_style",
        "manual_transcript",
        "keep_y_n",
        "notes",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "audio_path": row.get("final_path", ""),
                "raw_segment_path": row.get("raw_segment_path", ""),
                "language": row.get("language", ""),
                "duration_s": round(float(row.get("duration_s") or 0), 2),
                "source_title": row.get("source_title", ""),
                "source_url": row.get("source_url", ""),
                "source_start_time_seconds": row.get("source_start_time_seconds", ""),
                "source_end_time_seconds": row.get("source_end_time_seconds", ""),
                "transcript": row.get("normalized_transcript") or row.get("approx_transcript", ""),
                "pipeline_emotion": row.get("emotion", ""),
                "pipeline_style": row.get("style", ""),
                "emotion_status": row.get("emotion_status", "missing"),
                "emotion_failure_reason": row.get("emotion_failure_reason", ""),
                "emotion_review_required": row.get("emotion_review_required", ""),
                "manual_emotion": "",
                "manual_style": "",
                "manual_transcript": "",
                "keep_y_n": "",
                "notes": "",
            })

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote manual-label seed CSV: {out_path}")
    print(f"Wrote audit summary JSON: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata/10_final.jsonl")
    parser.add_argument("--output", default="contribution/results/emotion_manual_labels_seed.csv")
    parser.add_argument("--summary", default="contribution/results/emotion_failure_audit.json")
    args = parser.parse_args()
    run(args.metadata, args.output, args.summary)
