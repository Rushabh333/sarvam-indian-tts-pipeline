#!/usr/bin/env python3
"""Fill missing segment transcripts with local faster-whisper ASR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from emotion_fallback import normalize_transcript_text  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def transcribe_one(model: WhisperModel, audio_path: str, language: str) -> tuple[str, float]:
    lang = "hi" if language == "hi-IN" else "en"
    segments, info = model.transcribe(
        audio_path,
        language=lang,
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    return normalize_transcript_text(text), float(getattr(info, "language_probability", 0.0) or 0.0)


def run(
    input_metadata: str,
    output_metadata: str,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> list[dict]:
    rows = load_jsonl(Path(input_metadata))
    missing = [row for row in rows if not row.get("normalized_transcript") and row.get("final_path")]
    print(f"Rows: {len(rows)}. Missing transcripts: {len(missing)}")
    if not missing:
        write_jsonl(Path(output_metadata), rows)
        return rows

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    filled = 0
    failed = 0
    for idx, row in enumerate(rows, start=1):
        if row.get("normalized_transcript") or not row.get("final_path"):
            continue
        audio_path = row["final_path"]
        try:
            transcript, lang_prob = transcribe_one(model, audio_path, row.get("language", "en-IN"))
        except Exception as exc:
            row["local_asr_status"] = "failed"
            row["local_asr_error"] = str(exc)
            failed += 1
            continue

        if transcript:
            row["approx_transcript"] = transcript
            row["normalized_transcript"] = transcript
            row["local_asr_status"] = "success"
            row["local_asr_model"] = f"faster-whisper:{model_size}"
            row["local_asr_language_probability"] = lang_prob
            if row.get("emotion_status") == "heuristic_source_only_no_transcript":
                row["emotion_status"] = "heuristic_fallback_with_local_asr"
                row["emotion_failure_reason"] = (
                    row.get("emotion_failure_reason", "")
                    + "; transcript filled with local faster-whisper ASR"
                ).strip("; ")
            filled += 1
            print(f"[{idx}/{len(rows)}] filled {Path(audio_path).name}: {transcript[:80]}")
        else:
            row["local_asr_status"] = "empty"
            failed += 1

    write_jsonl(Path(output_metadata), rows)
    print(f"Filled transcripts: {filled}. Failed/empty: {failed}. Wrote: {output_metadata}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/metadata/10_final.jsonl")
    parser.add_argument("--output", default="data/metadata/10_final.jsonl")
    parser.add_argument("--model-size", default="base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()
    run(args.input, args.output, args.model_size, args.device, args.compute_type)
