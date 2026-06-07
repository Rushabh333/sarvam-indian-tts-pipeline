#!/usr/bin/env python3
"""
Sarvam TTS Pipeline Orchestrator

Runs all pipeline stages sequentially.
Usage:
    python run_pipeline.py
    python run_pipeline.py --start-stage 3 --end-stage 5
"""

import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Import pipeline modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import sys
import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def run_stage(script_name: str, skip_upload: bool = False):
    cmd = [sys.executable, f"pipeline/{script_name}"]
    if script_name == "10_finalize_upload.py" and skip_upload:
        cmd.append("--skip-upload")
        
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Stage {script_name} failed with exit code {result.returncode}")

def main():
    parser = argparse.ArgumentParser(description="Sarvam TTS Pipeline Orchestrator")
    parser.add_argument("--start-stage", type=int, default=0, help="Stage to start from")
    parser.add_argument("--end-stage", type=int, default=10, help="Stage to end at (inclusive)")
    parser.add_argument("--skip-upload", action="store_true", help="Skip HuggingFace upload in Stage 10")
    args = parser.parse_args()

    load_dotenv()
    
    stages = [
        (0, "Source Validation", "00_curate_sources.py"),
        (1, "Download", "01_download.py"),
        (2, "Enhancement", "02_enhance.py"),
        (3, "ASR & Diarization", "03_diarize_transcribe.py"),
        (4, "Macro Chunking", "04_extract_macro.py"),
        (5, "VAD Segmentation", "05_vad_segment.py"),
        (6, "DPBP Boundary Padding", "06_dpbp.py"),
        (7, "Speaker Verification", "07_speaker_verify.py"),
        (8, "Quality Filter", "08_quality_filter.py"),
        (9, "Emotion & Text Norm", "09_emotion_normalize.py"),
        (10, "Finalize & Upload", "10_finalize_upload.py"),
    ]

    print("=" * 60)
    print("SARVAM TTS PIPELINE")
    print(f"Running stages {args.start_stage} to {args.end_stage}")
    print("=" * 60)

    for step_num, name, script_name in stages:
        if args.start_stage <= step_num <= args.end_stage:
            print(f"\n[{step_num}/10] {name}")
            print("-" * 40)
            try:
                run_stage(script_name, skip_upload=args.skip_upload)
            except Exception as e:
                print(f"\n❌ Pipeline failed at Stage {step_num}: {name}")
                print(f"Error: {e}")
                sys.exit(1)
            
    print("\n✅ Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
