#!/usr/bin/env python3
"""
Generate quality statistics and analysis for the PDF report.
Reads final metadata and prints tables and histograms.
"""

import json
from collections import Counter
import statistics

def run():
    with open("data/metadata/10_final.jsonl") as f:
        data = [json.loads(line) for line in f]
        
    en = [d for d in data if d.get("language") == "en-IN"]
    hi = [d for d in data if d.get("language") == "hi-IN"]
    
    def print_stats(name, split_data):
        if not split_data:
            return
            
        durations = [d.get("duration_s", 0) for d in split_data]
        snrs = [d.get("snr_db", 0) for d in split_data]
        emotions = Counter([d.get("emotion") for d in split_data])
        styles = Counter([d.get("style") for d in split_data])
        
        total_dur = sum(durations) / 60
        avg_dur = statistics.mean(durations)
        avg_snr = statistics.mean(snrs)
        
        print(f"\n=== {name} ({len(split_data)} segments) ===")
        print(f"Total Duration: {total_dur:.2f} minutes")
        print(f"Avg Segment Dur: {avg_dur:.2f} seconds")
        print(f"Avg SNR: {avg_snr:.2f} dB")
        
        print("\nEmotions:")
        for k, v in emotions.most_common():
            print(f"  {k}: {v}")
            
        print("\nStyles:")
        for k, v in styles.most_common():
            print(f"  {k}: {v}")

    print_stats("Indian English", en)
    print_stats("Hindi", hi)
    
    total_min = sum([d.get("duration_s", 0) for d in data]) / 60
    print(f"\nGrand Total: {total_min:.2f} minutes")

if __name__ == "__main__":
    run()
