#!/usr/bin/env python3
"""
Backfill explicit review status into legacy metadata.

The iteration-2 run happened before Stage 09 recorded API failures. This script
does not invent labels; it marks the default-looking emotion/style outputs as
legacy fallbacks so manual review can separate trusted labels from placeholders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def backfill_file(path: Path) -> int:
    rows = load_jsonl(path)
    changed = 0
    for row in rows:
        if row.get("emotion_status"):
            continue

        emotion = row.get("emotion")
        style = row.get("style")
        normalized = row.get("normalized_transcript", "")
        if emotion == "neutral" and style == "conversational":
            row["emotion_status"] = "legacy_default_needs_review"
            row["emotion_failure_reason"] = (
                "legacy run lacked failure tracking; neutral/conversational may be an API-failure fallback"
            )
            row["emotion_review_required"] = True
        else:
            row["emotion_status"] = "legacy_untracked"
            row["emotion_failure_reason"] = "legacy run lacked explicit emotion-stage status"
            row["emotion_review_required"] = not bool(normalized)
        changed += 1

    if changed:
        write_jsonl(path, rows)
    return changed


def run(paths: list[str]) -> None:
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"missing: {path}")
            continue
        changed = backfill_file(path)
        print(f"{path}: backfilled {changed} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        default=["data/metadata/09_enriched.jsonl", "data/metadata/10_final.jsonl"],
    )
    args = parser.parse_args()
    run(args.paths)
