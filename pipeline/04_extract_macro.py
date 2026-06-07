#!/usr/bin/env python3
"""
Stage 04 — Universal Macro Chunk Extraction

Generic extraction that works for ALL video types:
  - Lectures → dominant speaker only (original behavior)
  - Interviews → keeps all speakers or only the dominant one, configurable
  - Comedy/Skit → handles audience reactions, keeps performer speech
  - Talks/Stories → extracts primary speaker, filters applause segments
  - Podcasts → can extract all or specific speakers

Key improvement over the NPTEL-specific version:
  Instead of assuming a single lecturer, this stage:
  1. Analyzes speaker distribution to auto-detect content type
  2. Merges consecutive same-speaker segments with configurable gap
  3. Applies minimum duration filter to skip very short interjections
  4. Optionally keeps all speakers for interview/multi-speaker content

Usage:
    python pipeline/04_extract_macro.py
    python pipeline/04_extract_macro.py --keep-all-speakers
"""

import json
import argparse
import soundfile as sf
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def analyze_speaker_distribution(diarized_transcript: dict) -> dict:
    """Analyze speaker distribution to auto-detect content type.
    
    Returns:
        {
            "speakers": {"SPEAKER_00": 120.5, "SPEAKER_01": 30.2, ...},
            "dominant": "SPEAKER_00",
            "dominant_pct": 0.80,
            "num_speakers": 2,
            "content_type": "monologue" | "interview" | "panel" | "mixed"
        }
    """
    entries = diarized_transcript.get("diarized_transcript", [])
    if not entries:
        return {"speakers": {}, "dominant": None, "dominant_pct": 0, 
                "num_speakers": 0, "content_type": "unknown"}
    
    speaker_duration = defaultdict(float)
    for entry in entries:
        sid = entry.get("speaker_id", "SPEAKER_00")
        start = entry.get("start_time_seconds", 0)
        end = entry.get("end_time_seconds", 0)
        dur = end - start
        if dur > 0:
            speaker_duration[sid] += dur
    
    if not speaker_duration:
        return {"speakers": {}, "dominant": None, "dominant_pct": 0,
                "num_speakers": 0, "content_type": "unknown"}
    
    total = sum(speaker_duration.values())
    dominant = max(speaker_duration, key=speaker_duration.get)
    dominant_pct = speaker_duration[dominant] / total if total > 0 else 0
    num_speakers = len(speaker_duration)
    
    # Auto-detect content type
    if num_speakers == 1 or dominant_pct >= 0.90:
        content_type = "monologue"      # lecture, talk, storytelling, standup
    elif num_speakers == 2 and dominant_pct >= 0.55:
        content_type = "interview"       # interview, conversation
    elif num_speakers >= 3 or dominant_pct < 0.55:
        content_type = "panel"           # panel discussion, group conversation
    else:
        content_type = "mixed"
    
    return {
        "speakers": dict(speaker_duration),
        "dominant": dominant,
        "dominant_pct": dominant_pct,
        "num_speakers": num_speakers,
        "content_type": content_type,
    }


def words_for_span(diarized: dict, start_s: float, end_s: float) -> list:
    """Return timestamped words that fall inside a diarized speaker span."""
    timestamps = diarized.get("timestamps") or {}
    words = timestamps.get("words") or []
    starts = timestamps.get("start_time_seconds") or timestamps.get("start") or []
    ends = timestamps.get("end_time_seconds") or timestamps.get("end") or []

    span_words = []
    for word, start, end in zip(words, starts, ends):
        if start is None or end is None:
            continue
        if float(start) >= start_s - 0.05 and float(end) <= end_s + 0.05:
            span_words.append({
                "word": str(word),
                "start": float(start),
                "end": float(end),
            })
    return span_words


def extract_macro_chunks(audio_path: str, transcript_path: str, 
                         output_dir: Path, source_meta: dict,
                         keep_all_speakers: bool = False,
                         merge_gap_s: float = 0.5,
                         min_chunk_duration_s: float = 2.0) -> list:
    """Extract and merge speech segments — works for any video type.
    
    Args:
        audio_path: Path to the enhanced audio WAV.
        transcript_path: Path to the diarized transcript JSON.
        output_dir: Directory to save macro chunks.
        source_meta: Source metadata dict.
        keep_all_speakers: If True, extracts all speakers (useful for interviews).
                           If False, extracts only the dominant speaker.
        merge_gap_s: Max gap in seconds between same-speaker segments to merge.
        min_chunk_duration_s: Minimum duration for a chunk to be kept.
    
    Returns:
        List of chunk metadata dicts.
    """
    with open(transcript_path) as f:
        diarized = json.load(f)
    
    # Analyze speaker distribution
    distribution = analyze_speaker_distribution(diarized)
    
    if distribution["dominant"] is None:
        return []
    
    audio, sr = sf.read(audio_path)
    video_id = source_meta["video_id"]
    genre = source_meta.get("genre", "talk")
    
    # Decide which speakers to extract
    if keep_all_speakers:
        # Keep all speakers (interview/podcast mode)
        target_speakers = set(distribution["speakers"].keys())
    elif genre in ("interview", "podcast", "debate") and distribution["content_type"] == "interview":
        # For interview-type genres, keep the top 2 speakers
        sorted_speakers = sorted(distribution["speakers"].items(), 
                                 key=lambda x: x[1], reverse=True)
        target_speakers = {s[0] for s in sorted_speakers[:2]}
    else:
        # Default: only the dominant speaker (works for lectures, talks, 
        # comedy monologues, storytelling, skits with narrator, etc.)
        target_speakers = {distribution["dominant"]}
    
    entries = []
    for entry in diarized.get("diarized_transcript", []):
        normalized = dict(entry)
        start_s = float(normalized.get("start_time_seconds", 0) or 0)
        end_s = float(normalized.get("end_time_seconds", 0) or 0)
        normalized.setdefault("transcript", "")
        if not normalized.get("words"):
            normalized["words"] = words_for_span(diarized, start_s, end_s)
        entries.append(normalized)
    
    # Filter entries to target speakers
    filtered = [
        e for e in entries
        if e.get("speaker_id", "SPEAKER_00") in target_speakers
    ]
    
    if not filtered:
        return []
    
    # Sort by start time
    filtered.sort(key=lambda x: x.get("start_time_seconds", 0))
    
    # Merge consecutive SAME-SPEAKER segments within merge_gap_s
    merged = []
    for entry in filtered:
        if (merged 
            and entry.get("speaker_id") == merged[-1].get("speaker_id")
            and entry.get("start_time_seconds", 0) - merged[-1].get("end_time_seconds", 0) < merge_gap_s):
            # Merge with previous
            merged[-1]["end_time_seconds"] = entry.get("end_time_seconds", 0)
            merged[-1]["transcript"] += " " + entry.get("transcript", "")
            merged[-1]["words"] = merged[-1].get("words", []) + entry.get("words", [])
        else:
            merged.append(dict(entry))
    
    # Filter out very short chunks (e.g., "hmm", "okay", audience reactions)
    merged = [c for c in merged 
              if c.get("end_time_seconds", 0) - c.get("start_time_seconds", 0) >= min_chunk_duration_s]
    
    chunk_metas = []
    
    for i, chunk in enumerate(merged):
        start_s = chunk.get("start_time_seconds", 0)
        end_s = chunk.get("end_time_seconds", 0)
        
        start_sample = int(start_s * sr)
        end_sample = int(end_s * sr)
        segment = audio[start_sample:end_sample]
        
        chunk_path = output_dir / f"{video_id}_macro_{i:04d}.wav"
        sf.write(chunk_path, segment, sr)
        
        meta = {
            "path": str(chunk_path),
            "video_id": video_id,
            "chunk_id": i,
            "raw_audio_path": source_meta.get("local_path"),
            "processed_source_path": audio_path,
            "enhanced_path": source_meta.get("enhanced_path"),
            "speaker_id": chunk.get("speaker_id", "SPEAKER_00"),
            "start_time_seconds": start_s,
            "end_time_seconds": end_s,
            "source_start_time_seconds": start_s,
            "source_end_time_seconds": end_s,
            "duration_s": end_s - start_s,
            "transcript": chunk.get("transcript", ""),
            "words": chunk.get("words", []),
            "language": source_meta.get("language"),
            "genre": genre,
            "register": source_meta.get("register"),
            "emotions_heard": source_meta.get("emotions_heard", []),
            "source_notes": source_meta.get("notes"),
            "source_speaker_name": source_meta.get("speaker_name"),
            "source_title": source_meta.get("title"),
            "source_channel": source_meta.get("channel"),
            "source_url": source_meta.get("url"),
            "content_type": distribution["content_type"],
            "speaker_distribution": distribution,
        }
        chunk_metas.append(meta)
    
    return chunk_metas


def run(input_metadata: str = "data/metadata/03_transcribed.jsonl",
        macro_dir: str = "data/macro_chunks",
        output_metadata: str = "data/metadata/04_macro.jsonl",
        keep_all_speakers: bool = False,
        merge_gap_s: float = 0.5,
        min_chunk_duration_s: float = 2.0):
    
    macro_dir = Path(macro_dir)
    macro_dir.mkdir(parents=True, exist_ok=True)
    Path(output_metadata).parent.mkdir(parents=True, exist_ok=True)
    
    # Load extraction config from pipeline_config.yaml if available
    try:
        import yaml
        with open("config/pipeline_config.yaml") as f:
            config = yaml.safe_load(f)
        ext_cfg = config.get("extraction", {})
        # Config values override defaults but CLI args override config
        if not keep_all_speakers:
            keep_all_speakers = ext_cfg.get("keep_all_speakers", False)
        if merge_gap_s == 0.5:  # still default
            merge_gap_s = ext_cfg.get("merge_gap_s", 0.5)
        if min_chunk_duration_s == 2.0:  # still default
            min_chunk_duration_s = ext_cfg.get("min_chunk_duration_s", 2.0)
    except (ImportError, FileNotFoundError):
        pass
    
    with open(input_metadata) as f:
        sources = [json.loads(line) for line in f]
    
    all_chunks = []
    
    for source in tqdm(sources, desc="Extracting Macro Chunks"):
        audio_path = source.get("enhanced_path") or source.get("local_path")
        transcript_path = source.get("transcript_path")
        
        if not audio_path or not transcript_path:
            continue
        
        chunks = extract_macro_chunks(
            audio_path, transcript_path, macro_dir, source,
            keep_all_speakers=keep_all_speakers,
            merge_gap_s=merge_gap_s,
            min_chunk_duration_s=min_chunk_duration_s,
        )
        all_chunks.extend(chunks)
    
    with open(output_metadata, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    
    print(f"\nExtracted {len(all_chunks)} total macro chunks.")
    print(f"Metadata saved to: {output_metadata}")
    return all_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/metadata/03_transcribed.jsonl")
    parser.add_argument("--macro-dir", default="data/macro_chunks")
    parser.add_argument("--output", default="data/metadata/04_macro.jsonl")
    parser.add_argument("--keep-all-speakers", action="store_true",
                        help="Extract all speakers, not just the dominant one")
    parser.add_argument("--merge-gap", type=float, default=0.5,
                        help="Max gap (seconds) between same-speaker segments to merge")
    parser.add_argument("--min-chunk-duration", type=float, default=2.0,
                        help="Minimum chunk duration in seconds")
    args = parser.parse_args()
    run(args.input, args.macro_dir, args.output,
        keep_all_speakers=args.keep_all_speakers,
        merge_gap_s=args.merge_gap,
        min_chunk_duration_s=args.min_chunk_duration)
