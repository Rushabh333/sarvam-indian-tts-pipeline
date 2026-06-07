#!/usr/bin/env python3
"""
Stage 03 — Diarization and Transcription

Uses Sarvam AI API to transcribe audio and diarize speakers.
Handles smart routing: files >30s use batch API, files <30s use REST API.

Usage:
    python pipeline/03_diarize_transcribe.py
"""

import os
import json
import time
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import soundfile as sf
from sarvamai import SarvamAI

load_dotenv()
SARVAM_KEY = os.environ.get("SARVAM_API_KEY")
if not SARVAM_KEY:
    raise ValueError("SARVAM_API_KEY not found in environment or .env file")

client = SarvamAI(api_subscription_key=SARVAM_KEY)

def get_audio_duration(path: str) -> float:
    try:
        f = sf.SoundFile(path)
        return float(f.frames) / f.samplerate
    except:
        return 0.0

def process_batch(audio_path: str, is_hindi: bool, output_path: str) -> dict:
    """Process long audio via Batch API (supports diarization)."""
    lang = "hi-IN" if is_hindi else "en-IN"
    mode = "codemix" if is_hindi else "verbatim"
    
    print(f"Submitting batch job for {Path(audio_path).name} ({mode})...")
    
    job = client.speech_to_text_job.create_job(
        model="saaras:v3",
        language_code=lang,
        mode=mode,
        with_diarization=True,
        with_timestamps=True
    )
    
    job.upload_files(file_paths=[audio_path])
    job.start()
    
    print(f"Waiting for job {job.job_id} to complete...")
    job.wait_until_complete()
    
    # Download results to temp dir
    tmp_dir = Path("data/tmp_results") / Path(output_path).stem
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    job.download_outputs(output_dir=str(tmp_dir))
    
    import glob
    results = glob.glob(f"{tmp_dir}/*.json")
    
    # Assuming one file uploaded, getting first result
    if results and len(results) > 0:
        result_file = results[0]
        with open(result_file, 'r') as f:
            data = json.load(f)
            
        # Normalize diarized_transcript to always be a list of segment entries
        if isinstance(data.get("diarized_transcript"), dict) and "entries" in data["diarized_transcript"]:
            data["diarized_transcript"] = data["diarized_transcript"]["entries"]
            
        with open(output_path, 'w') as out_f:
            json.dump(data, out_f, ensure_ascii=False, indent=2)
            
        # Cleanup temp
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return data
    else:
        raise Exception("No results downloaded from batch job")

def process_rest(audio_path: str, is_hindi: bool, output_path: str) -> dict:
    """
    Process short audio via REST API. 
    NOTE: REST API might not fully support diarization the same way as Batch,
    but we use it as a fallback for short clips.
    """
    lang = "hi-IN" if is_hindi else "en-IN"
    mode = "codemix" if is_hindi else "verbatim"
    
    with open(audio_path, "rb") as f:
        response = client.speech_to_text.transcribe(
            file=f,
            model="saaras:v3",
            mode=mode,
            language_code=lang
        )
        
    # Standardize output to match batch format for downstream
    # (Mocking single speaker for short files if diarization missing)
    data = response.dict() if hasattr(response, 'dict') else response
    
    if "diarized_transcript" not in data:
        # Wrap the single transcript into a diarized format
        words = []
        if "words" in data:
            words = data["words"]
            
        data = {
            "diarized_transcript": [{
                "speaker_id": "SPEAKER_00",
                "transcript": data.get("transcript", ""),
                "start_time_seconds": 0.0,
                "end_time_seconds": get_audio_duration(audio_path),
                "words": words
            }]
        }
        
    with open(output_path, 'w') as out_f:
        json.dump(data, out_f, ensure_ascii=False, indent=2)
        
    return data

def run(input_metadata: str = "data/metadata/02_enhanced.jsonl",
        transcripts_dir: str = "data/transcripts",
        output_metadata: str = "data/metadata/03_transcribed.jsonl",
        force: bool = False):
    
    transcripts_dir = Path(transcripts_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    Path(output_metadata).parent.mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_metadata)
    if not input_path.exists() and input_metadata == "data/metadata/02_enhanced.jsonl":
        fallback = Path("data/metadata/01_downloaded.jsonl")
        if fallback.exists():
            print(f"{input_metadata} not found; falling back to {fallback} and using raw audio paths.")
            input_path = fallback

    with open(input_path) as f:
        sources = [json.loads(line) for line in f]
        
    results = []
    
    to_transcribe = []
    for source in sources:
        audio_path = source.get("enhanced_path") or source.get("local_path")
        if not audio_path or not Path(audio_path).exists():
            continue
            
        video_id = source["video_id"]
        output_path = transcripts_dir / f"{video_id}_transcript.json"
        
        if output_path.exists() and not force:
            print(f"Skipping {video_id}, already transcribed.")
            source["transcript_path"] = str(output_path)
            results.append(source)
        else:
            to_transcribe.append((source, audio_path, output_path))

    batch_size = 10
    for i in range(0, len(to_transcribe), batch_size):
        batch = to_transcribe[i:i+batch_size]
        print(f"\n--- Submitting transcription batch {i//batch_size + 1} ({len(batch)} files) ---")
        
        jobs_in_batch = []
        for source, audio_path, output_path in batch:
            is_hindi = source.get("language") == "hi-IN"
            lang = "hi-IN" if is_hindi else "en-IN"
            mode = "codemix" if is_hindi else "verbatim"
            video_id = source["video_id"]
            
            try:
                print(f"Submitting job for {video_id}...")
                job = client.speech_to_text_job.create_job(
                    model="saaras:v3",
                    language_code=lang,
                    mode=mode,
                    with_diarization=True,
                    with_timestamps=True
                )
                job.upload_files(file_paths=[audio_path])
                job.start()
                jobs_in_batch.append((source, job, output_path))
            except Exception as e:
                print(f"Failed to submit/start job for {video_id}: {e}")
                source["transcript_path"] = None
                results.append(source)

        print(f"\n--- Waiting for batch {i//batch_size + 1} to complete ---")
        for source, job, output_path in jobs_in_batch:
            video_id = source["video_id"]
            try:
                print(f"Waiting for job {job.job_id} ({video_id}) to complete...")
                job.wait_until_complete()
                
                tmp_dir = Path("data/tmp_results") / Path(output_path).stem
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                job.download_outputs(output_dir=str(tmp_dir))
                
                import glob
                glob_results = glob.glob(f"{tmp_dir}/*.json")
                if glob_results and len(glob_results) > 0:
                    result_file = glob_results[0]
                    with open(result_file, 'r') as f_res:
                        data = json.load(f_res)
                    if isinstance(data.get("diarized_transcript"), dict) and "entries" in data["diarized_transcript"]:
                        data["diarized_transcript"] = data["diarized_transcript"]["entries"]
                    with open(output_path, 'w') as out_f:
                        json.dump(data, out_f, ensure_ascii=False, indent=2)
                    
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    source["transcript_path"] = str(output_path)
                else:
                    raise Exception("No results downloaded from batch job")
            except Exception as e:
                print(f"Failed transcription/download for {video_id}: {e}")
                source["transcript_path"] = None
                
            results.append(source)

    # Sort results to match original sources order
    source_order = {s["video_id"]: idx for idx, s in enumerate(sources)}
    results.sort(key=lambda x: source_order.get(x["video_id"], 999))
    
    with open(output_metadata, "w") as f:
        for s in results:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"\nMetadata saved to: {output_metadata}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/metadata/02_enhanced.jsonl")
    parser.add_argument("--transcripts-dir", default="data/transcripts")
    parser.add_argument("--output", default="data/metadata/03_transcribed.jsonl")
    parser.add_argument("--force", action="store_true",
                        help="Re-run Sarvam transcription even when transcript JSON exists")
    args = parser.parse_args()
    run(args.input, args.transcripts_dir, args.output, force=args.force)
