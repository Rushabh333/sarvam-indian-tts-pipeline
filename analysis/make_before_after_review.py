#!/usr/bin/env python3
"""Create an all-clips before/after manual review CSV from final metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def run(
    input_metadata: str = "data/metadata/10_final.jsonl",
    output_csv: str = "analysis/before_after_manual_review.csv",
) -> list[dict]:
    input_path = Path(input_metadata)
    if not input_path.exists():
        print(f"Final metadata not found: {input_metadata}")
        return []

    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "id",
        "language",
        "duration_s",
        "raw_segment_path",
        "final_path",
        "processed_segment_path",
        "source_title",
        "source_url",
        "source_start_time_seconds",
        "source_end_time_seconds",
        "snr_db",
        "ecapa_similarity",
        "emotion",
        "style",
        "quality_flags",
        "review_flags",
        "emotion_status",
        "emotion_failure_reason",
        "emotion_review_required",
        "transcript",
        "raw_audio_quality_1_5",
        "final_audio_quality_1_5",
        "language_correct_y_n",
        "single_speaker_y_n",
        "natural_human_voice_y_n",
        "good_boundary_y_n",
        "transcript_correct_y_n",
        "keep_y_n",
        "notes",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            final_path = item.get("final_path", "")
            writer.writerow({
                "id": Path(final_path).stem if final_path else item.get("video_id", ""),
                "language": item.get("language", ""),
                "duration_s": round(float(item.get("duration_s") or 0), 2),
                "raw_segment_path": item.get("raw_segment_path", ""),
                "final_path": final_path,
                "processed_segment_path": item.get("path", ""),
                "source_title": item.get("source_title", ""),
                "source_url": item.get("source_url", ""),
                "source_start_time_seconds": item.get("source_start_time_seconds", ""),
                "source_end_time_seconds": item.get("source_end_time_seconds", ""),
                "snr_db": round(float(item.get("snr_db") or 0), 2),
                "ecapa_similarity": round(float(item.get("ecapa_similarity") or 0), 4),
                "emotion": item.get("emotion", ""),
                "style": item.get("style", ""),
                "quality_flags": ";".join(item.get("quality_flags") or []),
                "review_flags": ";".join(item.get("review_flags") or []),
                "emotion_status": item.get("emotion_status", ""),
                "emotion_failure_reason": item.get("emotion_failure_reason", ""),
                "emotion_review_required": item.get("emotion_review_required", ""),
                "transcript": item.get("normalized_transcript") or item.get("approx_transcript", ""),
                "raw_audio_quality_1_5": "",
                "final_audio_quality_1_5": "",
                "language_correct_y_n": "",
                "single_speaker_y_n": "",
                "natural_human_voice_y_n": "",
                "good_boundary_y_n": "",
                "transcript_correct_y_n": "",
                "keep_y_n": "",
                "notes": "",
            })

    print(f"Wrote {len(rows)} rows to {output_csv}")
    return rows


if __name__ == "__main__":
    run()
