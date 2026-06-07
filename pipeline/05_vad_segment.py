#!/usr/bin/env python3
"""
Stage 05 — VAD Segmentation

Uses Silero VAD to fragment macro-chunks into micro-segments (3-30s).
Maps ASR word timestamps to generate approximate transcripts for segments.

Usage:
    python pipeline/05_vad_segment.py
"""

import json
import argparse
import torch
import soundfile as sf
import yaml
from pathlib import Path
from tqdm import tqdm


def load_silero_vad():
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True
    )
    get_speech_timestamps, _, read_audio, *_ = utils
    return model, get_speech_timestamps, read_audio


def pack_speech_timestamps(timestamps: list, min_dur: float, max_dur: float,
                           merge_gap_s: float) -> list:
    """Pack short VAD speech events into stable TTS training segments."""
    packed = []
    current = None

    for ts in timestamps:
        start = float(ts["start"])
        end = float(ts["end"])
        if current is None:
            current = {"start": start, "end": end}
            continue

        gap = start - current["end"]
        merged_duration = end - current["start"]
        if gap <= merge_gap_s and merged_duration <= max_dur:
            current["end"] = end
            continue

        if current["end"] - current["start"] >= min_dur:
            packed.append(current)
        current = {"start": start, "end": end}

    if current and current["end"] - current["start"] >= min_dur:
        packed.append(current)

    return packed


def segment_chunk(chunk_meta: dict, output_dir: Path, 
                  model, get_timestamps, read_audio, config: dict) -> list:
    """Fragment a macro-chunk via VAD."""
    chunk_path = chunk_meta["path"]
    if not Path(chunk_path).exists():
        return []
        
    audio, sr = sf.read(chunk_path)
    wav = read_audio(chunk_path, sampling_rate=16000)
    
    seg_cfg = config.get("segmentation", {})
    min_dur = seg_cfg.get("min_duration_s", 3.0)
    max_dur = seg_cfg.get("max_duration_s", 30.0)
    
    raw_timestamps = get_timestamps(
        wav, model,
        threshold=seg_cfg.get("vad_threshold", 0.5),
        min_speech_duration_ms=seg_cfg.get("vad_min_speech_ms", 250),
        max_speech_duration_s=max_dur,
        min_silence_duration_ms=seg_cfg.get("min_silence_ms", 300),
        speech_pad_ms=seg_cfg.get("speech_pad_ms", 30),
        return_seconds=True
    )
    timestamps = pack_speech_timestamps(
        raw_timestamps,
        min_dur=min_dur,
        max_dur=max_dur,
        merge_gap_s=seg_cfg.get("merge_speech_gap_s", 0.8),
    )
    
    segments = []
    words = chunk_meta.get("words", [])
    video_id = chunk_meta["video_id"]
    chunk_id = chunk_meta["chunk_id"]
    
    for i, ts in enumerate(timestamps):
        t_start, t_end = ts["start"], ts["end"]
        duration = t_end - t_start
        
        if not (min_dur <= duration <= max_dur):
            continue
            
        # Extract word timestamps that fall into this segment
        # We adjust the global word start time to be relative to the chunk start
        # Actually, if words in macro chunk are relative to original audio, we need to map them properly.
        # Assuming words list start_time is relative to original audio, and chunk has start_time_seconds
        chunk_start_s = chunk_meta.get("start_time_seconds", 0.0)
        
        seg_words = []
        for w in words:
            # Word times might be relative to original file depending on extraction logic
            # Let's assume word start/end are relative to original file. 
            # So within chunk, word local start = w["start"] - chunk_start_s
            w_start = w.get("start", w.get("start_time_seconds", 0))
            w_end = w.get("end", w.get("end_time_seconds", 0))
            w_start_local = w_start - chunk_start_s
            w_end_local = w_end - chunk_start_s
            
            # Allow slight tolerance
            if w_start_local >= t_start - 0.1 and w_end_local <= t_end + 0.1:
                seg_words.append({
                    "word": w.get("word", ""),
                    "start": w_start_local,
                    "end": w_end_local
                })
                
        approx_transcript = " ".join(w["word"] for w in seg_words)
        if not approx_transcript and len(timestamps) == 1:
            approx_transcript = chunk_meta.get("transcript", "")
        
        start_sample = int(t_start * sr)
        end_sample = int(t_end * sr)
        segment_audio = audio[start_sample:end_sample]
        
        out_path = output_dir / f"{video_id}_c{chunk_id:04d}_s{i:03d}.wav"
        sf.write(out_path, segment_audio, sr)
        
        segments.append({
            "path": str(out_path),
            "chunk_path": chunk_path,
            "video_id": video_id,
            "chunk_id": chunk_id,
            "segment_id": i,
            "raw_audio_path": chunk_meta.get("raw_audio_path"),
            "processed_source_path": chunk_meta.get("processed_source_path"),
            "enhanced_path": chunk_meta.get("enhanced_path"),
            "macro_start_time_seconds": chunk_start_s,
            "macro_end_time_seconds": chunk_meta.get("end_time_seconds"),
            "source_start_time_seconds": chunk_start_s + t_start,
            "source_end_time_seconds": chunk_start_s + t_end,
            "vad_start": t_start,
            "vad_end": t_end,
            "duration_s": duration,
            "approx_transcript": approx_transcript,
            "words": seg_words,
            "speaker_id": chunk_meta.get("speaker_id"),
            "language": chunk_meta.get("language"),
            "register": chunk_meta.get("register"),
            "emotions_heard": chunk_meta.get("emotions_heard", []),
            "source_notes": chunk_meta.get("source_notes"),
            "source_speaker_name": chunk_meta.get("source_speaker_name"),
            "source_title": chunk_meta.get("source_title"),
            "source_channel": chunk_meta.get("source_channel"),
            "source_url": chunk_meta.get("source_url"),
        })
        
    return segments


def run(input_metadata: str = "data/metadata/04_macro.jsonl",
        segments_dir: str = "data/segments",
        output_metadata: str = "data/metadata/05_segments.jsonl"):
    
    segments_dir = Path(segments_dir)
    segments_dir.mkdir(parents=True, exist_ok=True)
    Path(output_metadata).parent.mkdir(parents=True, exist_ok=True)
    
    with open("config/pipeline_config.yaml") as f:
        config = yaml.safe_load(f)
        
    with open(input_metadata) as f:
        chunks = [json.loads(line) for line in f]
        
    model, get_timestamps, read_audio = load_silero_vad()
    
    all_segments = []
    for chunk in tqdm(chunks, desc="VAD Segmentation"):
        segs = segment_chunk(chunk, segments_dir, model, get_timestamps, read_audio, config)
        all_segments.extend(segs)
        
    with open(output_metadata, "w") as f:
        for s in all_segments:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"\nCreated {len(all_segments)} micro-segments.")
    print(f"Metadata saved to: {output_metadata}")
    return all_segments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/metadata/04_macro.jsonl")
    parser.add_argument("--segments-dir", default="data/segments")
    parser.add_argument("--output", default="data/metadata/05_segments.jsonl")
    args = parser.parse_args()
    run(args.input, args.segments_dir, args.output)
