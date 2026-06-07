#!/usr/bin/env python3
"""
Stage 08 — Quality Filtering (SNR & Duration)

Filters segments based on SNR and duration, logs rejected segments.

Usage:
    python pipeline/08_quality_filter.py
"""

import json
import argparse
import yaml
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from quality_guardrails import audio_quality_flags, audio_quality_metrics, text_quality_flags


def compute_snr(audio_path: str) -> float:
    try:
        audio, _ = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        signal_rms = np.sqrt(np.mean(audio ** 2))
        noise_est = np.percentile(np.abs(audio), 10) + 1e-9
        return float(20 * np.log10(signal_rms / noise_est))
    except:
        return 0.0

def run(input_metadata: str = "data/metadata/07_verified.jsonl",
        output_metadata: str = "data/metadata/08_filtered.jsonl",
        rejection_log: str = "analysis/rejection_log.jsonl"):
    
    Path(output_metadata).parent.mkdir(parents=True, exist_ok=True)
    Path(rejection_log).parent.mkdir(parents=True, exist_ok=True)
    
    with open("config/pipeline_config.yaml") as f:
        config = yaml.safe_load(f)
        
    cfg = config.get("quality", {})
    snr_min = cfg.get("snr_db_min", 15.0)
    dur_min = cfg.get("duration_min_s", 3.0)
    dur_max = cfg.get("duration_max_s", 30.0)
    content_cfg = config.get("content_guardrails", {})
    
    with open(input_metadata) as f:
        segments = [json.loads(line) for line in f]
        
    kept, rejected = [], []
    
    for seg in tqdm(segments, desc="Quality Filtering"):
        reasons = []
        
        # Check speaker verification
        if not seg.get("speaker_verified", False):
            reasons.append(f"speaker_sim={seg.get('ecapa_similarity', 0):.2f} < threshold")
            
        # Check duration
        dur = seg.get("duration_s", 0)
        if not (dur_min <= dur <= dur_max):
            reasons.append(f"duration={dur:.1f}s outside [{dur_min},{dur_max}]")
            
        # Calculate SNR and check
        snr = compute_snr(seg["path"])
        seg["snr_db"] = snr
        if snr < snr_min:
            reasons.append(f"snr={snr:.1f}dB < {snr_min}")

        try:
            audio_metrics = audio_quality_metrics(seg["path"])
            seg.update(audio_metrics)
            for flag in audio_quality_flags(audio_metrics, config):
                reasons.append(flag)
        except Exception as e:
            reasons.append(f"audio_metrics_error={e}")

        text_hard_flags, text_review_flags = text_quality_flags(seg, config)
        seg["quality_flags"] = sorted(set(text_hard_flags + text_review_flags))
        seg["review_flags"] = sorted(set(text_review_flags))

        for flag in text_hard_flags:
            reasons.append(flag)

        if content_cfg.get("reject_mid_sentence_boundaries", False):
            for flag in text_review_flags:
                if flag.startswith("possibly_"):
                    reasons.append(flag)
            
        if reasons:
            seg["rejection_reasons"] = reasons
            rejected.append(seg)
        else:
            kept.append(seg)
            
    # Round-Robin Balancing selection to get approx 30 minutes of highest-quality data per language
    # and select approx 3-4 clips per video.
    from collections import defaultdict
    lang_segments = {"en-IN": [], "hi-IN": []}
    for seg in kept:
        lang = seg.get("language", "en-IN")
        if lang in lang_segments:
            lang_segments[lang].append(seg)
        else:
            lang_segments["en-IN"].append(seg)
            
    selected_kept = []
    
    for lang, segs in lang_segments.items():
        # Group segments by video_id
        video_groups = defaultdict(list)
        for seg in segs:
            video_groups[seg["video_id"]].append(seg)
            
        # Sort each video's segments by a quality score: (similarity * 100) + SNR
        for vid, v_segs in video_groups.items():
            v_segs.sort(key=lambda x: (x.get("ecapa_similarity", 0.0) * 100.0) + x.get("snr_db", 0.0), reverse=True)
            
        # target seconds from config
        target_min = config.get("targets", {}).get("english_minutes" if lang == "en-IN" else "hindi_minutes", 30.0)
        target_sec = target_min * 60.0
        
        selected_segs = []
        total_duration = 0.0
        
        # Determine maximum passes
        max_passes = max(len(v_segs) for v_segs in video_groups.values()) if video_groups else 0
        
        for pass_idx in range(max_passes):
            # Round-robin selection: pick one clip per video at each pass
            for vid in sorted(video_groups.keys()):
                v_segs = video_groups[vid]
                if pass_idx < len(v_segs):
                    seg = v_segs[pass_idx]
                    selected_segs.append(seg)
                    total_duration += seg.get("duration_s", 0.0)
                    
            # Stop if target duration is met
            if total_duration >= target_sec:
                break
                
        print(f"\n[{lang}] Target: {target_min} mins ({target_sec}s). Selected {len(selected_segs)} clips, total duration: {total_duration/60:.2f} mins.")
        video_clip_counts = Counter(s["video_id"] for s in selected_segs)
        if video_clip_counts:
            print(f"[{lang}] Clips per video: Min={min(video_clip_counts.values())}, Max={max(video_clip_counts.values())}, Avg={sum(video_clip_counts.values())/len(video_clip_counts):.1f}")
        else:
            print(f"[{lang}] No clips selected.")
            
        selected_kept.extend(selected_segs)
            
    with open(output_metadata, "w") as f:
        for s in selected_kept:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    with open(rejection_log, "w") as f:
        for s in rejected:
            f.write(json.dumps({
                "path": s.get("path"),
                "video_id": s.get("video_id"),
                "reasons": s.get("rejection_reasons"),
                "quality_flags": s.get("quality_flags"),
                "review_flags": s.get("review_flags"),
                "snr_db": s.get("snr_db"),
                "duration_s": s.get("duration_s"),
                "ecapa_similarity": s.get("ecapa_similarity"),
                "source_title": s.get("source_title"),
                "source_url": s.get("source_url"),
                "transcript": s.get("approx_transcript")
            }, ensure_ascii=False) + "\n")
            
    print(f"\nFiltering: {len(selected_kept)}/{len(segments)} kept ({len(selected_kept)/max(1, len(segments)):.1%} retention)")
    
    all_reasons = []
    for s in rejected:
        all_reasons.extend([r.split("=")[0] for r in s.get("rejection_reasons", [])])
        
    print("Rejection breakdown:", dict(Counter(all_reasons)))
    print(f"Metadata saved to: {output_metadata}")
    
    return selected_kept

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/metadata/07_verified.jsonl")
    parser.add_argument("--output", default="data/metadata/08_filtered.jsonl")
    args = parser.parse_args()
    run(args.input, args.output)
