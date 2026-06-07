#!/usr/bin/env python3
"""
Repair current semantic metadata with deterministic fallback labels.

This is for runs where Stage 09 failed before explicit fallback tracking was
implemented. It joins curated source hints back into each segment and rewrites
emotion/style/status fields without changing audio files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))

from emotion_fallback import apply_heuristic_annotation  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_sources(path: Path) -> dict[str, dict]:
    return {row["video_id"]: row for row in load_jsonl(path)}


def repair_rows(rows: list[dict], sources: dict[str, dict]) -> list[dict]:
    repaired = []
    for row in rows:
        item = dict(row)
        source = sources.get(item.get("video_id"), {})
        for key in ("register", "emotions_heard"):
            if key not in item or item.get(key) in (None, "", []):
                item[key] = source.get(key, [] if key == "emotions_heard" else "")
        if "source_notes" not in item or not item.get("source_notes"):
            item["source_notes"] = source.get("notes", "")

        status = item.get("emotion_status", "")
        if status in {"", "missing", "legacy_default_needs_review", "llm_failed_defaulted", "asr_failed_defaulted"}:
            apply_heuristic_annotation(
                item,
                source,
                reason="repaired legacy semantic fallback using curated source hints",
            )
        repaired.append(item)
    return repaired


def run(
    input_metadata: str,
    output_metadata: str,
    sources_path: str = "sources/sources.jsonl",
) -> list[dict]:
    rows = load_jsonl(Path(input_metadata))
    sources = load_sources(Path(sources_path))
    repaired = repair_rows(rows, sources)
    write_jsonl(Path(output_metadata), repaired)
    print(f"Repaired {len(repaired)} rows: {input_metadata} -> {output_metadata}")
    return repaired


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/metadata/10_final.jsonl")
    parser.add_argument("--output", default="data/metadata/10_final.jsonl")
    parser.add_argument("--sources", default="sources/sources.jsonl")
    args = parser.parse_args()
    run(args.input, args.output, args.sources)
