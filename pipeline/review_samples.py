#!/usr/bin/env python3
"""
Prep for manual review by sampling 20% of the final dataset.
Creates a CSV for you to grade.

Usage:
    python pipeline/review_samples.py
"""

import json
import csv
import random
from pathlib import Path

def run():
    Path("analysis").mkdir(exist_ok=True)
    
    try:
        with open("data/metadata/10_final.jsonl") as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("Final dataset not found. Run the full pipeline first.")
        return
        
    sample_size = max(1, int(len(data) * 0.2))
    samples = random.sample(data, sample_size)
    
    csv_path = "analysis/manual_review.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "File", "Language", "Duration(s)", "Emotion", "Style", 
            "Transcript", "Clear Audio? (1-5)", "Accurate Transcript? (Y/N)", 
            "Accurate Emotion? (Y/N)", "Notes"
        ])
        
        for s in samples:
            writer.writerow([
                Path(s.get("final_path", "")).name,
                s.get("language"),
                round(s.get("duration_s", 0), 1),
                s.get("emotion"),
                s.get("style"),
                s.get("normalized_transcript"),
                "", "", "", ""
            ])
            
    print(f"Sampled {sample_size} segments for review.")
    print(f"Please open {csv_path} and fill in the grading columns.")

if __name__ == "__main__":
    run()
