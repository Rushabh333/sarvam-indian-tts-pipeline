#!/usr/bin/env python3
"""Measure ASR text similarity between raw and final matched clips."""

from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from faster_whisper import WhisperModel


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_f1(a: str, b: str) -> float:
    a_words = normalize_text(a).split()
    b_words = normalize_text(b).split()
    if not a_words and not b_words:
        return 1.0
    if not a_words or not b_words:
        return 0.0
    a_counts = {}
    for word in a_words:
        a_counts[word] = a_counts.get(word, 0) + 1
    overlap = 0
    for word in b_words:
        if a_counts.get(word, 0) > 0:
            overlap += 1
            a_counts[word] -= 1
    precision = overlap / len(b_words)
    recall = overlap / len(a_words)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def load_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def transcribe(model: WhisperModel, audio_path: str, language: str, cache: dict[str, str]) -> str:
    key = f"{language}::{audio_path}"
    if key in cache:
        return cache[key]
    lang = "hi" if language == "hi-IN" else "en"
    segments, _ = model.transcribe(
        audio_path,
        language=lang,
        beam_size=1,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    cache[key] = text
    return text


def run(
    metadata_path: str = "data/metadata/10_final.jsonl",
    metrics_csv: str = "analysis/before_after_metrics/before_after_metrics.csv",
    output_csv: str = "analysis/before_after_metrics/asr_text_similarity.csv",
    cache_path: str = "analysis/before_after_metrics/asr_transcript_cache.json",
    model_size: str = "tiny",
    device: str = "cpu",
    compute_type: str = "int8",
) -> None:
    rows = [json.loads(line) for line in Path(metadata_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(Path(cache_path))
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    results = []
    for idx, row in enumerate(rows, start=1):
        raw_path = row.get("raw_segment_path", "")
        final_path = row.get("final_path", "")
        if not raw_path or not final_path or not Path(raw_path).exists() or not Path(final_path).exists():
            continue
        language = row.get("language", "en-IN")
        raw_text = transcribe(model, raw_path, language, cache)
        final_text = transcribe(model, final_path, language, cache)
        result = {
            "id": Path(final_path).stem,
            "language": language,
            "raw_asr_text": raw_text,
            "final_asr_text": final_text,
            "asr_char_similarity": char_similarity(raw_text, final_text),
            "asr_word_f1": word_f1(raw_text, final_text),
        }
        results.append(result)
        if idx % 10 == 0:
            save_cache(Path(cache_path), cache)
            print(f"ASR text similarity: {idx}/{len(rows)}")

    save_cache(Path(cache_path), cache)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    metrics_path = Path(metrics_csv)
    if metrics_path.exists():
        metric_rows = list(csv.DictReader(metrics_path.open(newline="", encoding="utf-8")))
        by_id = {row["id"]: row for row in results}
        fieldnames = list(metric_rows[0].keys())
        for field in ("asr_char_similarity", "asr_word_f1"):
            if field not in fieldnames:
                fieldnames.append(field)
        for row in metric_rows:
            sim = by_id.get(row["id"], {})
            row["asr_char_similarity"] = sim.get("asr_char_similarity", "")
            row["asr_word_f1"] = sim.get("asr_word_f1", "")
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metric_rows)

    print(f"Wrote ASR text similarity CSV: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata/10_final.jsonl")
    parser.add_argument("--metrics-csv", default="analysis/before_after_metrics/before_after_metrics.csv")
    parser.add_argument("--output", default="analysis/before_after_metrics/asr_text_similarity.csv")
    parser.add_argument("--cache", default="analysis/before_after_metrics/asr_transcript_cache.json")
    parser.add_argument("--model-size", default="tiny")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()
    run(args.metadata, args.metrics_csv, args.output, args.cache, args.model_size, args.device, args.compute_type)
