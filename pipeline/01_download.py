#!/usr/bin/env python3
"""
Stage 01 — Programmatic Download

Downloads audio from YouTube videos listed in sources.jsonl.
Converts to 16kHz mono WAV for processing.

Usage:
    python pipeline/01_download.py
    python pipeline/01_download.py --sources sources/sources.jsonl
"""

from typing import Optional
import yt_dlp
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm


def download_audio(source: dict, raw_dir: Path) -> Optional[str]:
    """Download a single video's audio as 16kHz mono WAV."""
    video_id = source["video_id"]
    output_path = raw_dir / f"{video_id}.wav"

    if output_path.exists():
        print(f"  ✓ Already downloaded: {video_id}")
        # Still try to enrich metadata from existing .info.json
        _enrich_from_info_json(source, raw_dir / f"{video_id}.info.json")
        return str(output_path)

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "postprocessor_args": [
            "-ar", "16000",     # 16kHz for processing
            "-ac", "1",         # mono
            "-sample_fmt", "s16",  # 16-bit
        ],
        "outtmpl": str(raw_dir / f"{video_id}.%(ext)s"),
        "writeinfojson": True,
        "quiet": True,
        "no_warnings": True,
        # "cookiesfrombrowser": ("chrome",),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([source["url"]])
        # Auto-fill title and channel from yt-dlp info.json
        _enrich_from_info_json(source, raw_dir / f"{video_id}.info.json")
        print(f"  ✓ Downloaded: {video_id} ({source.get('title', 'Unknown')})")
        return str(output_path)
    except Exception as e:
        print(f"  yt-dlp failed for {video_id}: {e}")
        return download_audio_pytubefix(source, raw_dir, output_path)


def download_audio_pytubefix(source: dict, raw_dir: Path, output_path: Path) -> Optional[str]:
    """Fallback downloader for YouTube bot/format failures in yt-dlp."""
    video_id = source["video_id"]
    try:
        from pytubefix import YouTube

        yt = YouTube(source["url"])
        stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
        if stream is None:
            raise RuntimeError("No audio-only streams found")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(stream.download(output_path=tmpdir, filename=f"{video_id}_source"))
            cmd = [
                "ffmpeg", "-y", "-i", str(tmp_path),
                "-ar", "16000",
                "-ac", "1",
                "-sample_fmt", "s16",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)

        info = {
            "id": video_id,
            "webpage_url": source.get("url"),
            "title": yt.title or source.get("title", ""),
            "channel": yt.author or source.get("channel", ""),
            "duration": yt.length or source.get("estimated_duration_min", 0) * 60,
            "downloader": "pytubefix",
        }
        with open(raw_dir / f"{video_id}.info.json", "w") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        _enrich_from_info_json(source, raw_dir / f"{video_id}.info.json")
        print(f"  ✓ Downloaded with pytubefix: {video_id} ({source.get('title', 'Unknown')})")
        return str(output_path)
    except Exception as e:
        print(f"  ✗ FAILED {video_id}: {e}")
        return None


def _enrich_from_info_json(source: dict, info_path: Path):
    """Auto-fill empty title/channel/duration from yt-dlp's .info.json."""
    if not info_path.exists():
        return
    try:
        with open(info_path) as f:
            info = json.load(f)
        # Only fill if currently empty or placeholder
        if not source.get("title"):
            source["title"] = info.get("title", "")
        if not source.get("channel"):
            source["channel"] = info.get("channel", info.get("uploader", ""))
        if not source.get("duration_raw_s"):
            source["duration_raw_s"] = info.get("duration", 0)
        if not source.get("speaker_name") or source.get("speaker_name") == "Unknown":
            source["speaker_name"] = info.get("channel", info.get("uploader", "Unknown"))
    except (json.JSONDecodeError, OSError):
        pass


def run(sources_path: str = "sources/sources.jsonl",
        raw_dir: str = "data/raw",
        output_path: str = "data/metadata/01_downloaded.jsonl"):
    """Download all sources and save updated metadata."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(sources_path) as f:
        sources = [json.loads(line) for line in f if line.strip() and not line.startswith("#")]

    # Filter out example entries
    sources = [s for s in sources if not s.get("video_id", "").startswith("EXAMPLE")]

    if not sources:
        print("ERROR: No valid sources found in sources.jsonl")
        print("Add real YouTube video entries first!")
        return []

    print(f"Downloading {len(sources)} videos...")
    print("=" * 50)

    results = []
    success_count = 0

    for source in tqdm(sources, desc="Downloading"):
        path = download_audio(source, raw_dir)
        source["local_path"] = path
        source["download_success"] = path is not None
        results.append(source)
        if path:
            success_count += 1

    # Save results
    with open(output_path, "w") as f:
        for s in results:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 50}")
    print(f"Downloaded: {success_count}/{len(sources)} videos")
    print(f"Metadata saved to: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube audio")
    parser.add_argument("--sources", default="sources/sources.jsonl")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output", default="data/metadata/01_downloaded.jsonl")
    args = parser.parse_args()
    run(args.sources, args.raw_dir, args.output)
