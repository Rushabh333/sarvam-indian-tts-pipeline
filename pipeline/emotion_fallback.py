#!/usr/bin/env python3
"""Deterministic fallback annotation when LLM emotion tagging is unavailable."""

from __future__ import annotations

import re


VALID_EMOTIONS = {"happy", "sad", "angry", "neutral", "excited", "formal", "concerned", "sarcastic"}
VALID_STYLES = {"formal", "conversational", "authoritative", "expressive", "monotone"}

EMOTION_HINT_MAP = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "excited": "excited",
    "passionate": "excited",
    "energetic": "excited",
    "expressive": "excited",
    "sarcastic": "sarcastic",
    "concerned": "concerned",
    "thoughtful": "concerned",
    "earnest": "concerned",
    "authoritative": "formal",
    "confident": "formal",
    "formal": "formal",
    "calm": "neutral",
    "casual": "neutral",
    "neutral": "neutral",
}


def normalize_transcript_text(text: str) -> str:
    """Minimal no-LLM TTS text cleanup. Does not rewrite content."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if text and text[-1] not in ".?!।":
        text += "."
    return text


def source_hints_for_segment(seg: dict, source_meta: dict | None = None) -> tuple[str, list[str], str]:
    source_meta = source_meta or {}
    register = seg.get("register") or source_meta.get("register") or ""
    emotions = seg.get("emotions_heard") or source_meta.get("emotions_heard") or []
    notes = seg.get("notes") or source_meta.get("notes") or ""
    return str(register), list(emotions or []), str(notes)


def infer_emotion_style(seg: dict, source_meta: dict | None = None) -> dict:
    """Infer coarse emotion/style from curated source hints and transcript cues."""
    register, emotions, notes = source_hints_for_segment(seg, source_meta)
    hints = [str(item).lower() for item in emotions]
    notes_l = notes.lower()
    title_l = str(seg.get("source_title") or source_meta.get("title") or "").lower()
    transcript_l = str(seg.get("approx_transcript") or "").lower()
    combined = " ".join(hints + [notes_l, title_l, transcript_l])

    emotion = "neutral"
    for hint in hints:
        mapped = EMOTION_HINT_MAP.get(hint)
        if mapped and mapped != "neutral":
            emotion = mapped
            break

    if emotion == "neutral":
        for keyword, mapped in EMOTION_HINT_MAP.items():
            if keyword != "neutral" and re.search(rf"\b{re.escape(keyword)}\b", combined):
                emotion = mapped
                break

    if emotion not in VALID_EMOTIONS:
        emotion = "neutral"

    register_l = register.lower()
    if "authoritative" in combined or emotion == "formal":
        style = "authoritative"
    elif register_l == "formal":
        style = "formal"
    elif emotion in {"happy", "sad", "angry", "excited", "sarcastic", "concerned"}:
        style = "expressive"
    elif "calm" in combined:
        style = "conversational"
    else:
        style = "conversational"

    if style not in VALID_STYLES:
        style = "conversational"

    return {
        "emotion": emotion,
        "style": style,
        "style_description": (
            f"Heuristic fallback from curated source hints: register={register or 'unknown'}, "
            f"emotions={','.join(emotions) if emotions else 'unknown'}"
        ),
    }


def apply_heuristic_annotation(seg: dict, source_meta: dict | None = None, reason: str = "") -> None:
    inferred = infer_emotion_style(seg, source_meta)
    transcript = normalize_transcript_text(seg.get("approx_transcript", ""))
    seg.update(inferred)
    seg["normalized_transcript"] = transcript
    if transcript:
        seg["emotion_status"] = "heuristic_fallback_with_transcript"
        seg["emotion_review_required"] = True
    else:
        seg["emotion_status"] = "heuristic_source_only_no_transcript"
        seg["emotion_review_required"] = True
    seg["emotion_failure_reason"] = reason or "LLM unavailable; used curated source-level hints"
