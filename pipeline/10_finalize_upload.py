#!/usr/bin/env python3
"""
Stage 10 — Finalize Format and HuggingFace Upload

Upsamples audio to 24kHz, loudness normalizes to -23 LUFS,
builds HuggingFace dataset, and pushes to Hub.

Usage:
    python pipeline/10_finalize_upload.py
"""

import os
import json
import argparse
import subprocess
import yaml
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from quality_guardrails import matched_raw_segment_path

try:
    from datasets import Dataset, DatasetDict, Audio
except ImportError:
    print("Warning: 'datasets' package not found. HF upload won't work.")

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")


def finalize_audio(input_path: str, output_path: str, config: dict) -> bool:
    """Upsample and loudness normalize via ffmpeg."""
    cfg = config.get("finalize", {})
    sr = cfg.get("target_sr", 24000)
    lufs = cfg.get("target_lufs", -23.0)
    tp = cfg.get("true_peak_dbtp", -1.5)
    lra = cfg.get("lra", 11)
    
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", str(sr),
        "-af", f"loudnorm=I={lufs}:TP={tp}:LRA={lra}",
        "-ac", "1",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error on {input_path}: {e.stderr.decode()}")
        return False


def finalized_audio_is_current(output_path: Path, expected_duration_s: float) -> bool:
    """Return true only when an existing final WAV still matches metadata."""
    if not output_path.exists():
        return False
    try:
        info = sf.info(str(output_path))
    except Exception:
        return False
    if info.samplerate != 24000:
        return False
    if expected_duration_s and abs(float(info.duration) - float(expected_duration_s)) > 0.25:
        return False
    return True


def export_raw_segment(seg: dict, raw_segments_dir: Path) -> str:
    """Export the exact pre-pipeline source span for before/after review."""
    raw_path = seg.get("raw_audio_path")
    start_s = seg.get("source_start_time_seconds")
    end_s = seg.get("source_end_time_seconds")
    if not raw_path or start_s is None or end_s is None:
        return ""
    raw_path = Path(raw_path)
    if not raw_path.exists() or float(end_s) <= float(start_s):
        return ""

    raw_segments_dir.mkdir(parents=True, exist_ok=True)
    out_path = matched_raw_segment_path(raw_segments_dir, seg)
    if out_path.exists():
        return str(out_path)

    audio, sr = sf.read(str(raw_path), always_2d=False)
    start_sample = max(0, int(float(start_s) * sr))
    end_sample = min(len(audio), int(float(end_s) * sr))
    if end_sample <= start_sample:
        return ""
    sf.write(str(out_path), audio[start_sample:end_sample], sr)
    return str(out_path)


def build_and_upload(metadata: list, repo_name: str):
    """Build HF dataset and push."""
    english = [s for s in metadata if s.get("language") == "en-IN"]
    hindi = [s for s in metadata if s.get("language") == "hi-IN"]
    
    def make_split(data):
        n_val = max(1, int(len(data) * 0.1))
        
        # Format the data dict for HuggingFace
        dict_data = {
            "id": [], "audio": [], "text": [], "language": [],
            "raw_transcript": [],
            "duration_seconds": [], "emotion": [], "style": [],
            "style_description": [], "snr_db": [], "ecapa_similarity": [],
            "speaker_id": [], "speaker_key": [],
            "source_speaker_name": [], "source_video_id": [], "source_url": [],
            "source_title": [], "source_channel": [], "split": [],
            "raw_audio_path": [], "raw_segment_path": [],
            "processed_source_path": [], "processed_segment_path": [],
            "final_path": [], "source_start_time_seconds": [],
            "source_end_time_seconds": [], "quality_flags": [], "review_flags": [],
            "devanagari_ratio": [], "latin_ratio": [],
            "emotion_status": [], "emotion_failure_reason": [],
            "emotion_review_required": []
        }
        
        for i, s in enumerate(data):
            dict_data["id"].append(Path(s.get("final_path", f"{s.get('video_id')}_{i}")).stem)
            dict_data["audio"].append(s.get("final_path"))
            dict_data["text"].append(s.get("normalized_transcript", ""))
            dict_data["raw_transcript"].append(s.get("approx_transcript", ""))
            dict_data["language"].append(s.get("language", ""))
            dict_data["duration_seconds"].append(s.get("duration_s", 0))
            dict_data["emotion"].append(s.get("emotion", "neutral"))
            dict_data["style"].append(s.get("style", "conversational"))
            dict_data["style_description"].append(s.get("style_description", ""))
            dict_data["snr_db"].append(s.get("snr_db", 0))
            dict_data["ecapa_similarity"].append(s.get("ecapa_similarity", 0))
            dict_data["speaker_id"].append(s.get("speaker_id", ""))
            dict_data["speaker_key"].append(s.get("speaker_key", ""))
            dict_data["source_speaker_name"].append(s.get("source_speaker_name", ""))
            dict_data["source_video_id"].append(s.get("video_id", ""))
            dict_data["source_url"].append(s.get("source_url", ""))
            dict_data["source_title"].append(s.get("source_title", ""))
            dict_data["source_channel"].append(s.get("source_channel", ""))
            dict_data["split"].append("validation" if i < n_val else "train")
            dict_data["raw_audio_path"].append(s.get("raw_audio_path", ""))
            dict_data["raw_segment_path"].append(s.get("raw_segment_path", ""))
            dict_data["processed_source_path"].append(s.get("processed_source_path", ""))
            dict_data["processed_segment_path"].append(s.get("path", ""))
            dict_data["final_path"].append(s.get("final_path", ""))
            dict_data["source_start_time_seconds"].append(s.get("source_start_time_seconds", 0))
            dict_data["source_end_time_seconds"].append(s.get("source_end_time_seconds", 0))
            dict_data["quality_flags"].append(s.get("quality_flags", []))
            dict_data["review_flags"].append(s.get("review_flags", []))
            dict_data["devanagari_ratio"].append(s.get("devanagari_ratio", 0))
            dict_data["latin_ratio"].append(s.get("latin_ratio", 0))
            dict_data["emotion_status"].append(s.get("emotion_status", "unknown"))
            dict_data["emotion_failure_reason"].append(s.get("emotion_failure_reason", ""))
            dict_data["emotion_review_required"].append(bool(s.get("emotion_review_required", False)))
            
        ds = Dataset.from_dict(dict_data)
        # decode=False avoids local audio decoding (no torchcodec needed);
        # HF Hub will store the raw WAV bytes and serve them correctly.
        return ds.cast_column("audio", Audio(sampling_rate=24000, decode=False))

    try:
        dataset_dict = DatasetDict({
            "english_train": make_split(english).filter(lambda x: x["split"] == "train"),
            "english_val": make_split(english).filter(lambda x: x["split"] == "validation"),
            "hindi_train": make_split(hindi).filter(lambda x: x["split"] == "train"),
            "hindi_val": make_split(hindi).filter(lambda x: x["split"] == "validation"),
        })
        
        full_repo_id = f"{HF_USERNAME}/{repo_name}"
        print(f"Pushing dataset to HuggingFace Hub: {full_repo_id}...")
        dataset_dict.push_to_hub(
            full_repo_id,
            token=HF_TOKEN,
            private=False
        )
        card_path = Path("dataset_card.md")
        if card_path.exists():
            from huggingface_hub import HfApi
            HfApi().upload_file(
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                repo_id=full_repo_id,
                repo_type="dataset",
                token=HF_TOKEN,
            )
        print(f"Successfully uploaded to: https://huggingface.co/datasets/{full_repo_id}")
    except Exception as e:
        print(f"Failed to upload to HuggingFace: {e}")


def run(input_metadata: str = "data/metadata/09_enriched.jsonl",
        final_dir: str = "data/final",
        output_metadata: str = "data/metadata/10_final.jsonl",
        skip_upload: bool = False):
    
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    Path(output_metadata).parent.mkdir(parents=True, exist_ok=True)
    
    with open("config/pipeline_config.yaml") as f:
        config = yaml.safe_load(f)
    before_after_cfg = config.get("before_after", {})
    raw_segments_dir = Path(before_after_cfg.get("raw_segments_dir", "data/raw_segments"))
    export_raw_segments = before_after_cfg.get("export_raw_segments", True)
        
    with open(input_metadata) as f:
        segments = [json.loads(line) for line in f]
        
    results = []
    success_count = 0
    
    for seg in tqdm(segments, desc="Finalizing Audio (ffmpeg)"):
        input_path = seg.get("path")
        if not input_path or not Path(input_path).exists():
            continue
            
        out_path = final_dir / Path(input_path).name
        
        expected_duration = seg.get("duration_s", 0)
        if finalized_audio_is_current(out_path, expected_duration) or finalize_audio(input_path, str(out_path), config):
            info = sf.info(str(out_path))
            seg["final_path"] = str(out_path)
            seg["duration_s"] = float(info.duration)
            seg["final_sample_rate"] = info.samplerate
            if export_raw_segments:
                seg["raw_segment_path"] = export_raw_segment(seg, raw_segments_dir)
            results.append(seg)
            success_count += 1
            
    with open(output_metadata, "w") as f:
        for s in results:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"\nFinalized audio for {success_count} segments.")
    
    if not skip_upload and HF_TOKEN:
        repo_name = config.get("hf", {}).get("repo_name", "sarvam-indian-tts-60min")
        build_and_upload(results, repo_name)
    elif not skip_upload:
        print("\nSkipping HuggingFace upload (HF_TOKEN not set).")
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/metadata/09_enriched.jsonl")
    parser.add_argument("--final-dir", default="data/final")
    parser.add_argument("--output", default="data/metadata/10_final.jsonl")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()
    run(args.input, args.final_dir, args.output, args.skip_upload)
