#!/usr/bin/env python3
"""
Stage 09 — Emotion Tagging and Text Normalization

Uses Sarvam LLM to simultaneously:
1. Detect emotion and speaking style based on transcript and context.
2. Normalize the transcript for TTS (expand abbreviations, numbers).

Usage:
    python pipeline/09_emotion_normalize.py
"""

import os
import json
import argparse
import time
import requests
import yaml
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from sarvamai import SarvamAI
import concurrent.futures
from threading import Lock
from emotion_fallback import apply_heuristic_annotation, normalize_transcript_text

load_dotenv()
SARVAM_KEY = os.environ.get("SARVAM_API_KEY")
client = SarvamAI(api_subscription_key=SARVAM_KEY) if SARVAM_KEY else None

PROMPT_TEMPLATE = """You are an expert TTS data annotator for {language} speech.

Raw ASR transcript: "{transcript}"

Perform two tasks:
1. Emotion & Style: Determine the most likely emotion and speaking style from the text.
2. Normalization: Normalize the text for TTS reading (expand numbers to words, expand abbreviations, keep fillers like um/ah, add sentence-ending punctuation where appropriate). Keep code-mixed words intact.

Answer EXACTLY in this JSON format and nothing else:
{{
  "emotion": "choose ONE: happy, sad, angry, neutral, excited, formal, concerned, sarcastic",
  "style": "choose ONE: formal, conversational, authoritative, expressive, monotone",
  "style_description": "A short 1-sentence descriptive tag like 'Calm, measured lecture tone with occasional emphasis'",
  "normalized_transcript": "the fully normalized text here"
}}
"""

def query_sarvam_llm(transcript: str, language: str, model: str = "sarvam-105b") -> dict:
    if not SARVAM_KEY:
        return {"_error": "SARVAM_API_KEY is not set", "_status": "llm_failed"}

    lang_name = "Hindi/Code-mixed" if language == "hi-IN" else "Indian English"
    prompt = PROMPT_TEMPLATE.format(language=lang_name, transcript=transcript)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.sarvam.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {SARVAM_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                    "temperature": 0.1
                },
                timeout=90
            )
            response.raise_for_status()
            resp_json = response.json()
            choices = resp_json.get("choices", [])
            if not choices:
                raise ValueError("No choices returned in response: " + str(resp_json))
            msg = choices[0].get("message", {})
            content = msg.get("content")
            if content is None:
                raise ValueError("Message content is None (might be blocked by safety filters or empty): " + str(resp_json))
                
            content = content.strip()
            
            # Clean markdown code blocks if any
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                start_idx = content.find("{")
                end_idx = content.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    cleaned_content = content[start_idx:end_idx+1]
                    return json.loads(cleaned_content)
                raise
                
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to query LLM after {max_retries} attempts: {e}")
                return {"_error": str(e), "_status": "llm_failed"}
            time.sleep(2)

def transcribe_segment(audio_path: str, language: str) -> tuple[str, str]:
    """Transcribe a short segment using the REST API."""
    if client is None:
        return "", "SARVAM_API_KEY is not set"

    lang = "hi-IN" if language == "hi-IN" else "en-IN"
    mode = "codemix" if language == "hi-IN" else "verbatim"
    try:
        with open(audio_path, "rb") as f:
            response = client.speech_to_text.transcribe(
                file=f,
                model="saaras:v3",
                mode=mode,
                language_code=lang
            )
        data = response.dict() if hasattr(response, 'dict') else response
        return data.get("transcript", "").strip(), ""
    except Exception as e:
        print(f"ASR transcription failed for {audio_path}: {e}")
        return "", str(e)


def apply_default_annotation(seg: dict, transcript: str, status: str, reason: str = "") -> None:
    """Mark fallback labels explicitly so they cannot masquerade as model output."""
    seg["emotion"] = "neutral"
    seg["style"] = "conversational"
    seg["style_description"] = "Default fallback; manual review required"
    seg["normalized_transcript"] = transcript
    seg["emotion_status"] = status
    seg["emotion_failure_reason"] = reason
    seg["emotion_review_required"] = True

def run(input_metadata: str = "data/metadata/08_filtered.jsonl",
        output_metadata: str = "data/metadata/09_enriched.jsonl",
        max_workers: int = 8):
    
    Path(output_metadata).parent.mkdir(parents=True, exist_ok=True)

    try:
        with open("config/pipeline_config.yaml") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config = {}
    llm_model = config.get("sarvam", {}).get("llm_model", "sarvam-105b")
    
    with open(input_metadata) as f:
        segments = [json.loads(line) for line in f]
        
    # Load already enriched paths to resume
    enriched_dict = {}
    if Path(output_metadata).exists():
        try:
            with open(output_metadata) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if "path" in item:
                            enriched_dict[item["path"]] = item
            print(f"Loaded {len(enriched_dict)} existing items from {output_metadata} to resume.")
        except Exception as e:
            print(f"Failed to load existing metadata for resume: {e}")
            
    # Separate segments into already processed and remaining
    results_map = {path: item for path, item in enriched_dict.items()}
    to_process = [s for s in segments if s["path"] not in results_map]
    
    print(f"Total segments: {len(segments)}. Already processed: {len(results_map)}. Remaining to process: {len(to_process)}")
    
    if not to_process:
        print("All segments already processed.")
        # Re-write the output file in the correct order of the input segments
        results_in_order = [results_map[s["path"]] for s in segments if s["path"] in results_map]
        with open(output_metadata, "w") as f:
            for s in results_in_order:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return results_in_order

    file_lock = Lock()
    
    # Save helper to write all segments in order
    def save_all_results():
        results_in_order = []
        for s in segments:
            path = s["path"]
            if path in results_map:
                results_in_order.append(results_map[path])
            else:
                # If we don't have it yet, keep it unprocessed or add standard keys
                results_in_order.append(s)
        with open(output_metadata, "w") as f:
            for s in results_in_order:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    def process_one_segment(seg):
        transcript = seg.get("approx_transcript", "")
        lang = seg.get("language", "en-IN")
        
        if not transcript:
            # Try to transcribe using REST API as a fallback
            print(f"\nNo approx_transcript for {Path(seg['path']).name}, running ASR fallback...")
            transcript, asr_error = transcribe_segment(seg["path"], lang)
            if transcript:
                print(f"  ASR Transcript: {transcript}")
                seg["approx_transcript"] = transcript
                seg["asr_fallback_status"] = "success"
            else:
                seg["asr_fallback_status"] = "failed"
                seg["asr_fallback_error"] = asr_error
        
        if not transcript:
            apply_heuristic_annotation(
                seg,
                reason=seg.get("asr_fallback_error", "missing transcript"),
            )
            with file_lock:
                results_map[seg["path"]] = seg
                save_all_results()
            return
            
        llm_res = query_sarvam_llm(transcript, lang, model=llm_model)

        if llm_res.get("_status") == "llm_failed":
            seg["approx_transcript"] = transcript
            apply_heuristic_annotation(
                seg,
                reason=llm_res.get("_error", "LLM request failed"),
            )
        else:
            seg["emotion"] = llm_res.get("emotion", "neutral")
            seg["style"] = llm_res.get("style", "conversational")
            seg["style_description"] = llm_res.get("style_description", "")
            seg["normalized_transcript"] = normalize_transcript_text(llm_res.get("normalized_transcript", transcript))
            missing = [
                key for key in ("emotion", "style", "normalized_transcript")
                if not llm_res.get(key)
            ]
            if missing:
                seg["emotion_status"] = "llm_partial"
                seg["emotion_failure_reason"] = "missing keys: " + ",".join(missing)
                seg["emotion_review_required"] = True
            else:
                seg["emotion_status"] = "llm_success"
                seg["emotion_failure_reason"] = ""
                seg["emotion_review_required"] = False
        
        with file_lock:
            results_map[seg["path"]] = seg
            save_all_results()

    # Use ThreadPoolExecutor to parallelize API requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(
            executor.map(process_one_segment, to_process),
            total=len(to_process),
            desc="Emotion & Normalization (Parallel)"
        ))
        
    print(f"\nEnriched segments. Total: {len(segments)}")
    print(f"Metadata saved to: {output_metadata}")
    
    # Final write to make sure everything is in order
    save_all_results()
    
    return [results_map[s["path"]] for s in segments if s["path"] in results_map]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/metadata/08_filtered.jsonl")
    parser.add_argument("--output", default="data/metadata/09_enriched.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.input, args.output, args.workers)
