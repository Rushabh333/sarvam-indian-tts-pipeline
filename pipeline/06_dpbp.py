#!/usr/bin/env python3
"""
Stage 06 — Dynamic Prosodic Boundary Padding (DPBP)

Applies DPBP to segment boundaries to prevent clipping trailing phonemes.

Usage:
    python pipeline/06_dpbp.py
"""

import json
import argparse
import yaml
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm


STOP_FINALS = set("bBdDgGpPtTkKcC")
FRICATIVE_FINALS = set("sSfFvVzZ")
NASAL_FINALS = set("mMnN")
VOWEL_CHARS_DEVANAGARI = "अआइईउऊएऐओऔ"
VOWEL_CHARS_LATIN = "aeiouAEIOU"
DANDA = "।"


def classify_final_phoneme(final_word: str) -> str:
    """Classify the final phoneme of a word for padding amount."""
    if not final_word:
        return "default"
        
    clean_word = final_word.rstrip("।.,?! ").rstrip()
    if not clean_word:
        return "default"
        
    last_char = clean_word[-1]
    
    if last_char in VOWEL_CHARS_LATIN or last_char in VOWEL_CHARS_DEVANAGARI:
        return "vowel"
    elif last_char in NASAL_FINALS:
        return "nasal"
    elif last_char in FRICATIVE_FINALS:
        return "fricative"
    elif last_char in STOP_FINALS:
        return "stop"
    return "default"


def find_nearest_zero_crossing(audio: np.ndarray, sample_idx: int, 
                               search_ms: int = 50, sr: int = 16000) -> int:
    search_samples = int((search_ms / 1000) * sr)
    start = max(0, sample_idx - search_samples)
    end = min(len(audio), sample_idx + search_samples)
    
    region = audio[start:end]
    zero_crossings = np.where(np.diff(np.sign(region)))[0]
    
    if len(zero_crossings) == 0:
        return sample_idx
        
    center = sample_idx - start
    closest_zc = zero_crossings[np.argmin(np.abs(zero_crossings - center))]
    return start + closest_zc


def process_segment(seg: dict, dpbp_dir: Path, config: dict) -> dict:
    """Apply DPBP logic to a segment and re-extract audio."""
    words = seg.get("words", [])
    vad_end = seg["vad_end"]
    
    seg["dpbp_applied"] = False
    seg["dpbp_padding_ms"] = 0
    seg["dpbp_end"] = vad_end
    
    dpbp_cfg = config.get("dpbp", {})
    
    if not words:
        return seg
        
    last_word_end = max(w.get("end", 0) for w in words)
    last_word_text = words[-1].get("word", "") if words else ""
    has_danda = DANDA in last_word_text
    
    proximity_ms = (vad_end - last_word_end) * 1000
    
    if abs(proximity_ms) <= dpbp_cfg.get("proximity_threshold_ms", 150):
        phoneme_class = classify_final_phoneme(last_word_text)
        pad_key = f"padding_{phoneme_class}_ms"
        base_padding_ms = dpbp_cfg.get(pad_key, dpbp_cfg.get("padding_default_ms", 250))
        
        if has_danda or last_word_text.rstrip().endswith((".", "?", "!")):
            padding_ms = base_padding_ms + dpbp_cfg.get("sentence_final_extra_ms", 50)
        else:
            padding_ms = base_padding_ms
            
        new_end_s = last_word_end + (padding_ms / 1000)
        
        # We need to re-extract from the MACRO CHUNK audio
        # The segment is already cut, so we load the original macro chunk and cut again
        
        # Get chunk path from metadata (requires chunk audio to exist)
        # But wait, we only have segment's path. We should load the macro chunk path.
        # We need to ensure we have the macro chunk path or pass it down.
        # Let's modify pipeline 05 to include chunk path or reconstruct it.
        chunk_path = Path("data/macro_chunks") / f"{seg['video_id']}_macro_{seg['chunk_id']:04d}.wav"
        
        if chunk_path.exists():
            chunk_audio, sr = sf.read(str(chunk_path))
            
            new_end_sample = int(new_end_s * sr)
            new_end_sample = min(new_end_sample, len(chunk_audio) - 1)
            
            final_sample = find_nearest_zero_crossing(
                chunk_audio, new_end_sample, search_ms=dpbp_cfg.get("zero_cross_search_ms", 50), sr=sr
            )
            
            seg["dpbp_end"] = final_sample / sr
            seg["dpbp_applied"] = True
            seg["dpbp_padding_ms"] = int((final_sample / sr - vad_end) * 1000)
            seg["dpbp_phoneme_class"] = phoneme_class
            
            # Re-extract
            start_s = seg["vad_start"]
            end_s = seg["dpbp_end"]
            
            start_sample = int(start_s * sr)
            end_sample = int(end_s * sr)
            new_audio = chunk_audio[start_sample:end_sample]
            
            dpbp_path = dpbp_dir / f"dpbp_{Path(seg['path']).name}"
            sf.write(str(dpbp_path), new_audio, sr)
            
            seg["pre_dpbp_path"] = seg.get("path")
            seg["path"] = str(dpbp_path)
            seg["duration_s"] = end_s - start_s
            if seg.get("macro_start_time_seconds") is not None:
                seg["source_end_time_seconds"] = seg["macro_start_time_seconds"] + end_s
            
    return seg


def run(input_metadata: str = "data/metadata/05_segments.jsonl",
        dpbp_dir: str = "data/dpbp_segments",
        output_metadata: str = "data/metadata/06_dpbp.jsonl"):
    
    dpbp_dir = Path(dpbp_dir)
    dpbp_dir.mkdir(parents=True, exist_ok=True)
    Path(output_metadata).parent.mkdir(parents=True, exist_ok=True)
    
    with open("config/pipeline_config.yaml") as f:
        config = yaml.safe_load(f)
        
    with open(input_metadata) as f:
        segments = [json.loads(line) for line in f]
        
    updated = []
    applied_count = 0
    
    for seg in tqdm(segments, desc="Applying DPBP"):
        # Default copy in case DPBP not applied
        if not dpbp_dir.exists():
            dpbp_dir.mkdir(parents=True)
            
        res = process_segment(seg, dpbp_dir, config)
        if res.get("dpbp_applied"):
            applied_count += 1
        updated.append(res)
        
    with open(output_metadata, "w") as f:
        for s in updated:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"\nApplied DPBP to {applied_count}/{len(segments)} segments.")
    print(f"Metadata saved to: {output_metadata}")
    return updated

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/metadata/05_segments.jsonl")
    parser.add_argument("--dpbp-dir", default="data/dpbp_segments")
    parser.add_argument("--output", default="data/metadata/06_dpbp.jsonl")
    args = parser.parse_args()
    run(args.input, args.dpbp_dir, args.output)
