"""
⚠️ DEPRECATED — Use the universal pipeline instead!

This script was NPTEL-specific. The pipeline now handles ALL video types:
  - Talks, interviews, comedy, skits, storytelling, podcasts, news, etc.

Quick start:
  1. Add sources:  python pipeline/add_sources.py --url "URL" --lang en-IN
  2. Run pipeline: python run_pipeline.py --start-stage 1 --end-stage 8

For details see: pipeline/add_sources.py and pipeline/04_extract_macro.py
"""

import os
import time
import json
import yt_dlp
from pydub import AudioSegment
from sarvamai import SarvamAI
from dotenv import load_dotenv

# Configuration
load_dotenv()
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "your_sarvam_api_key_here")
OUTPUT_DIR = "extracted_clips"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize the Sarvam client
client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

def download_youtube_audio(youtube_url, output_filename="temp_audio"):
    """Downloads the cleanest audio track from YouTube in WAV format."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_filename,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            # Resample to 16kHz to be optimal for the API and TTS
            'preferredquality': '192',
        }],
        'postprocessor_args': [
            '-ar', '16000',
            '-ac', '1'
        ],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return f"{output_filename}.wav"

def get_sarvam_diarization(audio_path):
    """
    Sends audio to Sarvam Batch API to get speaker timestamps.
    NPTEL videos are long (usually 30+ mins). The REST API will timeout, 
    so we must use the Batch API job queue.
    """
    print("  Submitting batch job for diarization...")
    job = client.speech_to_text_job.create_job(
        model="saaras:v3",
        language_code="en-IN",
        mode="verbatim",
        with_diarization=True
    )
    
    job.upload_files(file_paths=[audio_path])
    job.start()
    
    print(f"  Waiting for job {job.job_id} to complete (this may take a few minutes for NPTEL videos)...")
    job.wait_until_complete()
    
    tmp_dir = "temp_diarization_results"
    os.makedirs(tmp_dir, exist_ok=True)
    job.download_outputs(output_dir=tmp_dir)
    
    import glob
    results = glob.glob(f"{tmp_dir}/*.json")
    
    if results and len(results) > 0:
        result_file = results[0]
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Clean up the downloaded file
        if os.path.exists(result_file):
            os.remove(result_file)
            
        # Normalize diarized_transcript to always be a list of segment entries
        if isinstance(data.get("diarized_transcript"), dict) and "entries" in data["diarized_transcript"]:
            data["diarized_transcript"] = data["diarized_transcript"]["entries"]
            
        return data
    else:
        raise Exception("Failed to download diarization results from Batch API.")

import numpy as np

def compute_snr(audio_segment: AudioSegment) -> float:
    """Estimates the Signal-to-Noise Ratio to detect background noise."""
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    if len(samples) == 0:
        return 0.0
    signal_rms = np.sqrt(np.mean(samples ** 2))
    noise_est = np.percentile(np.abs(samples), 10) + 1e-9
    return float(20 * np.log10(signal_rms / noise_est))

def extract_clean_chunks(audio_path, diarization_data, video_id, clip_duration_sec=15, max_clips=3):
    """Parses diarization logs, identifies single-speaker blocks, and exports up to max_clips."""
    audio = AudioSegment.from_wav(audio_path)
    clip_length_ms = clip_duration_sec * 1000
    clip_counter = 0

    # Sarvam's response stores segments under "diarized_transcript"
    segments = diarization_data.get("diarized_transcript", [])

    # Group segments by speaker to find the primary speaker (the lecturer)
    speaker_durations = {}
    for seg in segments:
        spk = seg.get("speaker_id", "SPEAKER_00")
        dur = seg.get("end_time_seconds", 0) - seg.get("start_time_seconds", 0)
        speaker_durations[spk] = speaker_durations.get(spk, 0) + dur

    if not speaker_durations:
        print("No distinct speakers detected.")
        return

    primary_speaker = max(speaker_durations, key=speaker_durations.get)
    print(f"  Identified Primary Lecturer: {primary_speaker}")

    # Process segments belonging to the primary speaker
    for seg in segments:
        if clip_counter >= max_clips:
            break
            
        if seg.get("speaker_id", "SPEAKER_00") != primary_speaker:
            continue
        
        start_ms = int(seg.get("start_time_seconds", 0) * 1000)
        end_ms = int(seg.get("end_time_seconds", 0) * 1000)

        # Extract continuous 15-second clips from long speech segments
        current_start = start_ms
        while current_start + clip_length_ms <= end_ms:
            if clip_counter >= max_clips:
                break
                
            chunk = audio[current_start : current_start + clip_length_ms]
            
            # Simple RMS energy check to ensure the clip isn't just silence/dead air
            if chunk.rms > 500:  
                snr = compute_snr(chunk)
                if snr > 15.0: # Minimum 15 dB SNR to ensure low background noise
                    output_path = os.path.join(OUTPUT_DIR, f"{video_id}_spk_{clip_counter}.wav")
                    chunk.export(output_path, format="wav")
                    print(f"  Saved clean {clip_duration_sec}s clip (SNR {snr:.1f}dB): {output_path}")
                    clip_counter += 1
                else:
                    print(f"  Skipped 15s clip due to high background noise (SNR {snr:.1f}dB)")
            
            # Move window forward without overlap
            current_start += clip_length_ms 

def process_pipeline(youtube_url):
    video_id = youtube_url.split("v=")[-1].split("&")[0]
    print(f"\nProcessing Video ID: {video_id}")
    
    raw_audio = download_youtube_audio(youtube_url, video_id)
    try:
        print("Analyzing audio structure with Sarvam Diarization...")
        diarization_results = get_sarvam_diarization(raw_audio)
        
        print(f"Slicing high-quality target segments...")
        extract_clean_chunks(raw_audio, diarization_results, video_id, clip_duration_sec=15, max_clips=3)
    finally:
        # Cleanup large raw files to save disk space
        if os.path.exists(raw_audio):
            os.remove(raw_audio)

# Example Usage with multiple links
if __name__ == "__main__":
    urls = [
        "https://www.youtube.com/watch?v=RhS1PB3gDgU"
    ]
    for url in urls:
        try:
            process_pipeline(url)
        except Exception as e:
            print(f"Failed to process {url}: {e}")
