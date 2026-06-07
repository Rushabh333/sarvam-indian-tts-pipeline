#!/usr/bin/env python3
"""
Universal Source Adder — Add YouTube videos of ANY genre to the pipeline.

Works with: lectures, talks, interviews, comedy, skits, storytelling,
podcasts, news, debates, monologues, and more.

Usage:
    # Interactive mode (prompts for URLs):
    python pipeline/add_sources.py

    # Single URL:
    python pipeline/add_sources.py --url "https://youtube.com/watch?v=XXXX" --lang hi-IN

    # Batch file (one URL per line, optionally with language tab-separated):
    python pipeline/add_sources.py --batch urls.txt --lang en-IN
"""

import json
import argparse
import re
import sys
from pathlib import Path


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/|/embed/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',  # bare ID
    ]
    for pat in patterns:
        m = re.search(pat, url.strip())
        if m:
            return m.group(1)
    return None


def normalize_url(url: str) -> str:
    """Normalize any YouTube URL to standard watch format."""
    vid = extract_video_id(url)
    if vid:
        return f"https://www.youtube.com/watch?v={vid}"
    return url.strip()


def create_source_entry(url: str, language: str = "en-IN",
                        genre: str = "talk", speaker_name: str = "Unknown",
                        notes: str = "") -> dict:
    """Create a minimal, universal source entry.
    
    The pipeline stages handle all the heavy lifting:
    - Stage 03: Diarization identifies speakers automatically
    - Stage 04: Dominant speaker extraction works for any content
    - Stage 05: VAD segmentation is content-agnostic
    - Stage 07: ECAPA-TDNN verifies speaker consistency
    - Stage 08: Quality filter removes noisy/short clips
    
    So we only need: video_id, url, language. Everything else is optional.
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from: {url}")

    return {
        "video_id": video_id,
        "url": normalize_url(url),
        "title": "",                      # auto-filled by yt-dlp during download
        "channel": "",                    # auto-filled by yt-dlp during download
        "language": language,
        "speaker_name": speaker_name,
        "genre": genre,                   # talk/interview/comedy/skit/lecture/story/podcast/news
        "dominant_speaker_pct": 80,       # conservative default, pipeline verifies this
        "background_music": False,        # Demucs handles this regardless
        "pre_listen_quality": 3,          # neutral default
        "register": "conversational",     # default; overridden by LLM in Stage 09
        "emotions_heard": [],             # auto-detected in Stage 09
        "notes": notes,
    }


def load_existing_sources(sources_path: str) -> set:
    """Load existing video IDs to avoid duplicates."""
    existing = set()
    path = Path(sources_path)
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        entry = json.loads(line)
                        existing.add(entry.get("video_id", ""))
                    except json.JSONDecodeError:
                        pass
    return existing


def add_sources_interactive(sources_path: str, default_lang: str = "en-IN"):
    """Interactive mode: paste URLs one by one."""
    existing = load_existing_sources(sources_path)
    added = 0

    print("=" * 60)
    print("UNIVERSAL SOURCE ADDER")
    print("=" * 60)
    print(f"Default language: {default_lang}")
    print("Paste YouTube URLs one per line. Empty line or 'done' to finish.")
    print("Optional: add language after URL separated by space/tab")
    print("  Example: https://youtube.com/watch?v=XXXX hi-IN")
    print("=" * 60)

    with open(sources_path, "a") as f:
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not line or line.lower() == "done":
                break

            # Parse optional language from line
            parts = line.split()
            url = parts[0]
            lang = parts[1] if len(parts) > 1 else default_lang

            video_id = extract_video_id(url)
            if not video_id:
                print(f"  ✗ Invalid URL: {url}")
                continue

            if video_id in existing:
                print(f"  ⚠ Already exists: {video_id}")
                continue

            entry = create_source_entry(url, language=lang)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            existing.add(video_id)
            added += 1
            print(f"  ✓ Added: {video_id} ({lang})")

    print(f"\n{'=' * 60}")
    print(f"Added {added} new sources to {sources_path}")
    print(f"Total sources: {len(existing)}")


def add_sources_batch(sources_path: str, batch_file: str, default_lang: str = "en-IN"):
    """Batch mode: read URLs from a file."""
    existing = load_existing_sources(sources_path)
    added = 0
    skipped = 0

    with open(batch_file) as bf:
        lines = [l.strip() for l in bf if l.strip() and not l.startswith("#")]

    with open(sources_path, "a") as f:
        for line in lines:
            parts = line.split()
            url = parts[0]
            lang = parts[1] if len(parts) > 1 else default_lang

            video_id = extract_video_id(url)
            if not video_id:
                print(f"  ✗ Invalid: {url}")
                skipped += 1
                continue

            if video_id in existing:
                print(f"  ⚠ Duplicate: {video_id}")
                skipped += 1
                continue

            entry = create_source_entry(url, language=lang)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            existing.add(video_id)
            added += 1

    print(f"\nBatch import: {added} added, {skipped} skipped")
    print(f"Total sources: {len(existing)}")


def add_single_url(sources_path: str, url: str, lang: str = "en-IN",
                   genre: str = "talk", speaker: str = "Unknown",
                   notes: str = ""):
    """Add a single URL programmatically."""
    existing = load_existing_sources(sources_path)
    video_id = extract_video_id(url)

    if not video_id:
        print(f"✗ Invalid URL: {url}")
        return False

    if video_id in existing:
        print(f"⚠ Already exists: {video_id}")
        return False

    entry = create_source_entry(url, language=lang, genre=genre,
                                speaker_name=speaker, notes=notes)

    with open(sources_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✓ Added: {video_id} ({lang}, {genre})")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add YouTube videos of any genre to the TTS pipeline"
    )
    parser.add_argument("--sources", default="sources/sources.jsonl",
                        help="Path to sources.jsonl")
    parser.add_argument("--url", help="Single YouTube URL to add")
    parser.add_argument("--batch", help="File with one URL per line")
    parser.add_argument("--lang", default="en-IN",
                        help="Default language code (en-IN or hi-IN)")
    parser.add_argument("--genre", default="talk",
                        help="Genre: talk/interview/comedy/skit/lecture/story/podcast/news")
    parser.add_argument("--speaker", default="Unknown",
                        help="Speaker name (optional)")
    parser.add_argument("--notes", default="", help="Notes about the video")
    args = parser.parse_args()

    if args.url:
        add_single_url(args.sources, args.url, args.lang, args.genre,
                        args.speaker, args.notes)
    elif args.batch:
        add_sources_batch(args.sources, args.batch, args.lang)
    else:
        add_sources_interactive(args.sources, args.lang)
