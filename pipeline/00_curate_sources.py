#!/usr/bin/env python3
"""
Stage 00 — Source Curation Validator

Validates sources.jsonl schema, checks for missing fields, and prints summary.
Run this AFTER manually curating your YouTube sources.

Usage:
    python pipeline/00_curate_sources.py
"""

import json
import sys
from pathlib import Path
from collections import Counter
import yaml

from quality_guardrails import source_policy_issues

REQUIRED_FIELDS = [
    "video_id", "url", "title", "language", "speaker_name",
    "dominant_speaker_pct", "background_music", "pre_listen_quality"
]

OPTIONAL_FIELDS = [
    "channel", "register", "emotions_heard", "notes",
    "estimated_duration_min", "upload_date", "license"
]

VALID_LANGUAGES = ["en-IN", "hi-IN"]


def load_config(config_path: str = "config/pipeline_config.yaml") -> dict:
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def validate_sources(sources_path: str = "sources/sources.jsonl") -> list:
    """Validate sources.jsonl and return parsed entries."""
    path = Path(sources_path)
    if not path.exists():
        print(f"ERROR: {sources_path} not found!")
        print("Create it with YouTube video entries. See sources/sources.jsonl for schema.")
        sys.exit(1)

    config = load_config()
    strict_sources = config.get("source_quality", {}).get("fail_on_hard_exclusion", True)
    sources = []
    errors = []
    warnings = []

    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON — {e}")
                continue

            # Check required fields
            for field in REQUIRED_FIELDS:
                if field not in entry:
                    errors.append(f"Line {line_num} ({entry.get('video_id', '?')}): Missing required field '{field}'")

            # Validate language
            lang = entry.get("language", "")
            if lang not in VALID_LANGUAGES:
                errors.append(f"Line {line_num}: Invalid language '{lang}'. Must be one of {VALID_LANGUAGES}")

            # Validate quality score
            quality = entry.get("pre_listen_quality", 0)
            if not (1 <= quality <= 5):
                errors.append(f"Line {line_num}: pre_listen_quality must be 1-5, got {quality}")

            hard_issues, review_warnings = source_policy_issues(entry, config)
            for issue in hard_issues:
                msg = f"Line {line_num} ({entry.get('video_id', '?')}): HARD EXCLUSION - {issue}"
                if strict_sources:
                    errors.append(msg)
                else:
                    warnings.append(msg)
            for warning in review_warnings:
                warnings.append(f"Line {line_num} ({entry.get('video_id', '?')}): REVIEW - {warning}")

            sources.append(entry)

    return sources, errors, warnings


def print_summary(sources: list, errors: list, warnings: list):
    """Print a nice summary of the curated sources."""
    print("=" * 60)
    print("SOURCE CURATION SUMMARY")
    print("=" * 60)

    if errors:
        print(f"\n{len(errors)} blocking issues found:")
        for err in errors:
            print(f"  - {err}")
        print()

    if warnings:
        print(f"\n{len(warnings)} review warnings found:")
        for warning in warnings:
            print(f"  - {warning}")
        print()

    if not sources:
        print("No valid sources found. Add entries to sources/sources.jsonl")
        return

    # Language breakdown
    lang_counts = Counter(s["language"] for s in sources)
    print(f"\n📊 Total sources: {len(sources)}")
    for lang, count in lang_counts.items():
        label = "Indian English" if lang == "en-IN" else "Hindi"
        print(f"  {label} ({lang}): {count} videos")

    # Quality distribution
    quality_counts = Counter(s.get("pre_listen_quality", 0) for s in sources)
    print(f"\n⭐ Quality distribution:")
    for q in sorted(quality_counts.keys(), reverse=True):
        print(f"  Quality {q}: {quality_counts[q]} videos")

    # Estimated duration
    total_est = sum(s.get("estimated_duration_min", 15) for s in sources)
    print(f"\n⏱️  Estimated total raw duration: ~{total_est} minutes")
    print(f"   Target after filtering: 60 minutes")
    print(f"   Expected retention: ~50% → need ~120 min raw")
    if total_est < 100:
        print(f"   ⚠️  May need more sources! Aim for {120 - total_est}+ more minutes")
    else:
        print(f"   ✅ Should be sufficient raw material")

    # Emotion coverage
    all_emotions = set()
    for s in sources:
        all_emotions.update(s.get("emotions_heard", []))
    if all_emotions:
        print(f"\n🎭 Emotion coverage: {', '.join(sorted(all_emotions))}")

    # Register coverage
    registers = Counter(s.get("register", "unknown") for s in sources)
    print(f"\n🎤 Register distribution:")
    for reg, count in registers.most_common():
        print(f"  {reg}: {count}")

    print("\n" + "=" * 60)

    # Check for example entries
    example_ids = [s for s in sources if s.get("video_id", "").startswith("EXAMPLE")]
    if example_ids:
        print("\n⚠️  WARNING: Found example entries! Remove them before running the pipeline.")


if __name__ == "__main__":
    sources, errors, warnings = validate_sources()
    print_summary(sources, errors, warnings)
    if errors:
        sys.exit(1)
