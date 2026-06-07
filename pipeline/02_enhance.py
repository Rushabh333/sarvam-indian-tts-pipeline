#!/usr/bin/env python3
"""
Stage 02 — Enhancement (GPU recommended)

Applies Demucs (htdemucs_ft) to remove background noise/music.
Computes SNR before and after. If SNR decreases (enhancement artifacts),
it rolls back and keeps the original audio.

Usage:
    python pipeline/02_enhance.py
"""

import subprocess
import sys
import soundfile as sf
import numpy as np
from pathlib import Path
import json
import shutil
from tqdm import tqdm
import argparse
import yaml


def compute_snr(audio: np.ndarray) -> float:
    """Estimate SNR from RMS of signal vs noise floor."""
    signal_rms = np.sqrt(np.mean(audio ** 2))
    noise_floor = np.percentile(np.abs(audio), 10) + 1e-8
    return float(20 * np.log10(signal_rms / noise_floor))


def run_demucs_separation(input_path: str, output_dir: str) -> str:
    """Run Demucs htdemucs_ft for residual transient cleanup, chunking long files if necessary."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    stem = input_path.stem
    
    # Read audio metadata to check duration
    info = sf.info(str(input_path))
    duration_sec = info.duration
    sr = info.samplerate
    
    vocals_path = output_dir / "htdemucs_ft" / "vocals" / f"{stem}.wav"
    vocals_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Threshold for chunking: 10 minutes (600 seconds)
    chunk_length_sec = 600
    
    if duration_sec <= chunk_length_sec:
        # Run Demucs directly on the file
        subprocess.run([
            sys.executable, "-m", "demucs",
            "--two-stems=vocals",
            "-n", "htdemucs_ft",
            "--segment", "7",
            "-o", str(output_dir),
            "--filename", "{stem}/{track}.wav",
            str(input_path)
        ], check=True, capture_output=True)
        # Resample direct Demucs output to original sample rate
        v_data, demucs_sr = sf.read(str(vocals_path))
        if len(v_data.shape) > 1:
            v_data = np.mean(v_data, axis=1)
        if demucs_sr != sr:
            import librosa
            v_data = librosa.resample(v_data, orig_sr=demucs_sr, target_sr=sr)
        sf.write(str(vocals_path), v_data, sr)
        return str(vocals_path)
    
    print(f"  -> Audio is {duration_sec/60:.1f} min long. Splitting into {chunk_length_sec/60:.0f}-min chunks for Demucs to prevent OOM...")
    
    # Load and chunk
    audio, sr = sf.read(str(input_path))
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        
    chunk_samples = chunk_length_sec * sr
    total_samples = len(audio)
    
    temp_chunks_dir = output_dir / f"chunks_{stem}"
    temp_chunks_dir.mkdir(parents=True, exist_ok=True)
    
    vocals_chunks = []
    
    for i in range(0, total_samples, chunk_samples):
        chunk_data = audio[i : i + chunk_samples]
        chunk_idx = i // chunk_samples
        chunk_file = temp_chunks_dir / f"chunk_{chunk_idx}.wav"
        
        # Save temp chunk
        sf.write(str(chunk_file), chunk_data, sr)
        
        # Run Demucs on this chunk
        chunk_output_dir = temp_chunks_dir / f"out_{chunk_idx}"
        subprocess.run([
            sys.executable, "-m", "demucs",
            "--two-stems=vocals",
            "-n", "htdemucs_ft",
            "--segment", "7",
            "-o", str(chunk_output_dir),
            "--filename", "{stem}/{track}.wav",
            str(chunk_file)
        ], check=True, capture_output=True)
        
        # Read the vocals output
        vocals_chunk_file = chunk_output_dir / "htdemucs_ft" / "vocals" / f"chunk_{chunk_idx}.wav"
        v_data, demucs_sr = sf.read(str(vocals_chunk_file))
        if len(v_data.shape) > 1:
            v_data = np.mean(v_data, axis=1)
        if demucs_sr != sr:
            import librosa
            v_data = librosa.resample(v_data, orig_sr=demucs_sr, target_sr=sr)
        vocals_chunks.append(v_data)
        
    # Concatenate all vocal chunks
    full_vocals = np.concatenate(vocals_chunks)
    
    # Save the merged vocal track
    sf.write(str(vocals_path), full_vocals, sr)
    
    # Clean up temp chunks
    shutil.rmtree(temp_chunks_dir)
    
    return str(vocals_path)



def enhance_audio(source: dict, enhanced_dir: Path) -> dict:
    """Enhance single audio, fallback if degrade."""
    raw_path = source.get("local_path")
    if not raw_path or not Path(raw_path).exists():
        return source

    video_id = source["video_id"]
    final_path = enhanced_dir / f"{video_id}_enhanced.wav"
    
    if final_path.exists():
        source["enhanced_path"] = str(final_path)
        return source

    try:
        audio_orig, sr = sf.read(raw_path)
        snr_orig = compute_snr(audio_orig)
        source["snr_original_db"] = snr_orig
        
        # Run Demucs
        demucs_dir = str(enhanced_dir / "demucs_tmp")
        demucs_vocals = run_demucs_separation(raw_path, demucs_dir)
        
        # Check SNR after
        audio_enh, _ = sf.read(demucs_vocals)
        if len(audio_enh.shape) > 1:
            audio_enh = np.mean(audio_enh, axis=1)
        snr_enh = compute_snr(audio_enh)
        source["snr_enhanced_db"] = snr_enh
        
        delta = snr_enh - snr_orig
        source["snr_delta_db"] = delta
        
        if delta >= 0.5:  # Slight threshold to ensure it actually helped
            sf.write(str(final_path), audio_enh, sr)
            source["enhanced"] = True
            source["enhancement_note"] = f"Enhanced (delta={delta:+.2f}dB)"
        else:
            shutil.copy(raw_path, final_path)
            source["enhanced"] = False
            source["enhancement_note"] = f"Kept original (delta={delta:+.2f}dB)"
            
        source["enhanced_path"] = str(final_path)
        
    except Exception as e:
        print(f"Error enhancing {video_id}: {e}")
        source["enhanced_path"] = raw_path
        source["enhanced"] = False
        source["enhancement_note"] = f"Error: {e}"
        
    return source


def copy_without_enhancement(source: dict, enhanced_dir: Path) -> dict:
    """Use clean source audio directly while preserving stage-2 metadata."""
    raw_path = source.get("local_path")
    if not raw_path or not Path(raw_path).exists():
        return source

    video_id = source["video_id"]
    final_path = enhanced_dir / f"{video_id}_enhanced.wav"
    if not final_path.exists():
        shutil.copy(raw_path, final_path)

    audio_orig, _ = sf.read(raw_path)
    snr_orig = compute_snr(audio_orig)
    source["snr_original_db"] = snr_orig
    source["snr_enhanced_db"] = snr_orig
    source["snr_delta_db"] = 0.0
    source["enhanced"] = False
    source["enhancement_note"] = "Skipped enhancement by config; using clean source audio"
    source["enhanced_path"] = str(final_path)
    return source


def run(input_metadata: str = "data/metadata/01_downloaded.jsonl",
        enhanced_dir: str = "data/enhanced",
        output_metadata: str = "data/metadata/02_enhanced.jsonl"):
    
    enhanced_dir = Path(enhanced_dir)
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    Path(output_metadata).parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_metadata) as f:
        sources = [json.loads(line) for line in f]

    try:
        with open("config/pipeline_config.yaml") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config = {}
    enhancement_enabled = config.get("enhancement", {}).get("enabled", True)
        
    valid_sources = [s for s in sources if s.get("download_success")]
    if not valid_sources:
        print("No downloaded sources to enhance.")
        return []
        
    if enhancement_enabled:
        print(f"Enhancing {len(valid_sources)} files (this will take time on GPU...)")
    else:
        print(f"Copying {len(valid_sources)} clean files without Demucs enhancement...")
    
    results = []
    progress_label = "Enhancing" if enhancement_enabled else "Copying"
    for source in tqdm(valid_sources, desc=progress_label):
        if enhancement_enabled:
            res = enhance_audio(source, enhanced_dir)
        else:
            res = copy_without_enhancement(source, enhanced_dir)
        results.append(res)
        
    # Clean up temp
    tmp_dir = enhanced_dir / "demucs_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
        
    with open(output_metadata, "w") as f:
        for s in results:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"\nMetadata saved to: {output_metadata}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/metadata/01_downloaded.jsonl")
    parser.add_argument("--enhanced-dir", default="data/enhanced")
    parser.add_argument("--output", default="data/metadata/02_enhanced.jsonl")
    args = parser.parse_args()
    run(args.input, args.enhanced_dir, args.output)
