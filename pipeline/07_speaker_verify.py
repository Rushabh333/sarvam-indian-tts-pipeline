#!/usr/bin/env python3
"""
Stage 07 — Speaker Verification (ECAPA-TDNN)

Verifies speaker identity against reference clips using SpeechBrain.

Usage:
    python pipeline/07_speaker_verify.py
"""

import json
import argparse
import yaml
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
from speechbrain.inference.speaker import SpeakerRecognition

ECAPA_MODEL = None

def get_ecapa_model(model_name: str):
    global ECAPA_MODEL
    if ECAPA_MODEL is None:
        ECAPA_MODEL = SpeakerRecognition.from_hparams(
            source=model_name,
            savedir="pretrained_models/ecapa"
        )
    return ECAPA_MODEL

def get_embedding(audio_path: str, model) -> np.ndarray:
    signal, sr = sf.read(audio_path)
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)
    if sr != 16000:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=16000)
        
    signal_tensor = torch.FloatTensor(signal).unsqueeze(0)
    embedding = model.encode_batch(signal_tensor)
    return embedding.squeeze().detach().cpu().numpy()

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def run(input_metadata: str = "data/metadata/06_dpbp.jsonl",
        output_metadata: str = "data/metadata/07_verified.jsonl"):
    
    Path(output_metadata).parent.mkdir(parents=True, exist_ok=True)
    
    with open("config/pipeline_config.yaml") as f:
        config = yaml.safe_load(f)
        
    cfg = config.get("speaker_verification", {})
    threshold = cfg.get("cosine_threshold", 0.82)
    model_name = cfg.get("model", "speechbrain/spkrec-ecapa-voxceleb")
    
    with open(input_metadata) as f:
        segments = [json.loads(line) for line in f]
        
    if not segments:
        return []
        
    model = get_ecapa_model(model_name)
    
    # Simple reference building: average embeddings of 3 longest segments per video
    # In a real setup, these would be manually verified.
    
    # Group by video and diarized speaker. Mixing interview speakers into one
    # reference embedding makes speaker verification meaningless.
    speaker_groups = {}
    for s in segments:
        speaker_key = f"{s['video_id']}::{s.get('speaker_id', 'unknown')}"
        s["speaker_key"] = speaker_key
        if speaker_key not in speaker_groups:
            speaker_groups[speaker_key] = []
        speaker_groups[speaker_key].append(s)
        
    results = []
    
    for speaker_key, segs in tqdm(speaker_groups.items(), desc="Verifying Speakers"):
        # Build reference
        sorted_segs = sorted(segs, key=lambda x: x.get("duration_s", 0), reverse=True)
        ref_clips = [s["path"] for s in sorted_segs[:cfg.get("num_reference_clips", 3)]]
        
        if not ref_clips:
            continue
            
        ref_embs = [get_embedding(c, model) for c in ref_clips]
        ref_embedding = np.mean(ref_embs, axis=0)
        ref_embedding /= np.linalg.norm(ref_embedding)
        
        for seg in segs:
            try:
                emb = get_embedding(seg["path"], model)
                sim = cosine_similarity(ref_embedding, emb)
                seg["ecapa_similarity"] = float(sim)
                seg["speaker_verified"] = sim >= threshold
            except Exception as e:
                seg["speaker_verified"] = False
                seg["ecapa_error"] = str(e)
                
            results.append(seg)
            
    with open(output_metadata, "w") as f:
        for s in results:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    verified_count = sum(1 for s in results if s.get("speaker_verified"))
    print(f"\nSpeaker verified: {verified_count}/{len(results)} segments.")
    print(f"Metadata saved to: {output_metadata}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/metadata/06_dpbp.jsonl")
    parser.add_argument("--output", default="data/metadata/07_verified.jsonl")
    args = parser.parse_args()
    run(args.input, args.output)
