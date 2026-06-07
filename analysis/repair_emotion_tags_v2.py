#!/usr/bin/env python3
"""Repair emotion/style tags using transcript, source hints, and clip features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


EMOTIONS = ["neutral", "happy", "sad", "angry", "excited", "formal", "concerned", "sarcastic"]

SOURCE_HINT_MAP = {
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

LEXICON = {
    "happy": [
        "happy", "happiness", "joy", "smile", "laugh", "funny", "beautiful", "amazing",
        "खुश", "हँस", "हंस", "मुस्कुर", "अच्छा", "बहुत अच्छा",
    ],
    "sad": [
        "sad", "pain", "hurt", "loss", "cry", "tears", "alone", "broken", "struggle",
        "दुख", "दर्द", "रो", "आंसू", "अकेल", "टूट",
    ],
    "angry": [
        "angry", "anger", "wrong", "fight", "hate", "shout", "blame", "corrupt",
        "गुस्सा", "गलत", "लड़", "नफरत", "भ्रष्ट",
    ],
    "excited": [
        "success", "impossible", "possible", "power", "dream", "start", "begin",
        "must", "do it", "energy", "achieve", "goal", "motivation", "मुमकिन",
        "सफल", "सपना", "करना है", "जोश", "ऊर्जा",
    ],
    "formal": [
        "question", "answer", "system", "education", "geography", "heart", "anatomy",
        "strategy", "chapter", "example", "therefore", "concept", "dynamic",
        "प्रश्न", "उत्तर", "व्यवस्था", "रणनीति", "उदाहरण", "अध्याय",
    ],
    "concerned": [
        "problem", "mistake", "difficult", "hard", "fear", "risk", "truth", "better",
        "why", "how", "क्या", "क्यों", "कैसे", "समस्या", "गलती", "मुश्किल", "डर",
        "सच", "बेहतर", "चिंता",
    ],
    "sarcastic": [
        "myth", "superhero", "superheroes", "broken english", "joke", "sarcasm", "funny",
        "seriously", "really", "मिथ", "मजाक", "व्यंग",
    ],
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_metrics(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    return {row["id"]: row for row in rows}


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[\w\u0900-\u097F]+", text or ""))


def add_score(scores: dict[str, float], evidence: list[str], emotion: str, amount: float, reason: str) -> None:
    if emotion not in scores:
        return
    scores[emotion] += amount
    evidence.append(f"{emotion}+{amount:g}:{reason}")


def score_emotions(row: dict, metrics: dict | None = None) -> tuple[str, float, float, list[str], dict[str, float]]:
    metrics = metrics or {}
    text = norm_text(row.get("normalized_transcript") or row.get("approx_transcript") or "")
    title = norm_text(row.get("source_title", ""))
    notes = norm_text(row.get("source_notes", ""))
    combined = " ".join([text, title, notes])
    scores = {emotion: 0.05 for emotion in EMOTIONS}
    evidence: list[str] = []

    add_score(scores, evidence, "neutral", 0.35, "default prior")

    register = norm_text(row.get("register", ""))
    if register == "formal":
        add_score(scores, evidence, "formal", 0.8, "source register=formal")
    elif register == "conversational":
        add_score(scores, evidence, "neutral", 0.25, "source register=conversational")

    for hint in row.get("emotions_heard") or []:
        mapped = SOURCE_HINT_MAP.get(norm_text(str(hint)))
        if mapped:
            amount = 0.9 if mapped != "neutral" else 0.25
            add_score(scores, evidence, mapped, amount, f"source emotion hint={hint}")

    for emotion, keywords in LEXICON.items():
        hits = 0
        for keyword in keywords:
            if keyword in combined:
                hits += 1
        if hits:
            add_score(scores, evidence, emotion, min(1.2, 0.35 * hits), f"{hits} transcript/source keyword hits")

    duration = float(row.get("duration_s") or 0)
    wc = word_count(text)
    speech_rate = wc / duration if duration > 0 else 0.0
    if speech_rate >= 2.8:
        add_score(scores, evidence, "excited", 0.35, f"fast speech_rate={speech_rate:.2f}wps")
    elif 0 < speech_rate <= 1.25:
        add_score(scores, evidence, "concerned", 0.20, f"slow speech_rate={speech_rate:.2f}wps")

    active_ratio = float(row.get("active_ratio") or metrics.get("final_active_ratio") or 0)
    rms_dbfs = float(row.get("rms_dbfs") or metrics.get("final_rms_dbfs") or -120)
    peak_dbfs = float(row.get("peak_dbfs") or metrics.get("final_peak_dbfs") or -120)
    if active_ratio >= 0.70 and rms_dbfs >= -23.0:
        add_score(scores, evidence, "excited", 0.25, f"high activity/loudness active={active_ratio:.2f},rms={rms_dbfs:.1f}")
    if peak_dbfs > -2.0:
        add_score(scores, evidence, "excited", 0.15, f"high peak={peak_dbfs:.1f}dBFS")

    if "?" in (row.get("normalized_transcript") or row.get("approx_transcript") or ""):
        add_score(scores, evidence, "concerned", 0.15, "question punctuation")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_emotion, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(max(score, 0.0) for score in scores.values())
    confidence = top_score / total if total else 0.0
    margin = top_score - second_score
    return top_emotion, confidence, margin, evidence[:8], scores


def style_for(row: dict, emotion: str, confidence: float) -> str:
    register = norm_text(row.get("register", ""))
    if emotion == "formal":
        return "authoritative" if confidence >= 0.40 else "formal"
    if register == "formal" and emotion in {"neutral", "concerned"}:
        return "formal"
    if emotion in {"happy", "sad", "angry", "excited", "sarcastic", "concerned"}:
        return "expressive"
    return "conversational"


def repair_rows(rows: list[dict], metrics_by_id: dict[str, dict]) -> tuple[list[dict], Counter, Counter]:
    before = Counter(row.get("emotion", "missing") for row in rows)
    repaired = []
    for row in rows:
        item = dict(row)
        final_stem = Path(item.get("final_path") or item.get("path") or "").stem
        emotion, confidence, margin, evidence, scores = score_emotions(item, metrics_by_id.get(final_stem))
        old_emotion = item.get("emotion", "")
        old_style = item.get("style", "")
        item["previous_emotion"] = old_emotion
        item["previous_style"] = old_style
        item["emotion"] = emotion
        item["style"] = style_for(item, emotion, confidence)
        item["emotion_confidence"] = round(confidence, 4)
        item["emotion_score_margin"] = round(margin, 4)
        item["emotion_evidence"] = evidence
        item["emotion_scores"] = {key: round(value, 4) for key, value in sorted(scores.items())}
        item["emotion_status"] = "heuristic_text_source_prosody_v2"
        item["emotion_failure_reason"] = "LLM unavailable; repaired with transcript/source/prosody heuristic v2"
        item["emotion_review_required"] = confidence < 0.42 or margin < 0.35
        item["emotion_changed_by_v2"] = old_emotion != emotion or old_style != item["style"]
        repaired.append(item)
    after = Counter(row.get("emotion", "missing") for row in repaired)
    return repaired, before, after


def save_distribution_plot(path: Path, before: Counter, after: Counter) -> None:
    labels = sorted(set(before) | set(after))
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - 0.2 for i in x], [before.get(label, 0) for label in labels], width=0.4, label="before", color="#6B7280")
    ax.bar([i + 0.2 for i in x], [after.get(label, 0) for label in labels], width=0.4, label="after v2", color="#2563EB")
    ax.set_xticks(list(x), labels, rotation=25, ha="right")
    ax.set_ylabel("Clips")
    ax.set_title("Emotion distribution before/after v2 repair")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(
    input_path: str,
    output_path: str,
    metrics_csv: str = "analysis/before_after_metrics/before_after_metrics.csv",
    output_dir: str = "analysis/emotion_tag_repair",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(Path(input_path))
    metrics = load_metrics(Path(metrics_csv))
    repaired, before, after = repair_rows(rows, metrics)
    write_jsonl(Path(output_path), repaired)

    summary = {
        "rows": len(repaired),
        "before_emotion_distribution": dict(before),
        "after_emotion_distribution": dict(after),
        "style_distribution": dict(Counter(row.get("style", "missing") for row in repaired)),
        "changed_rows": sum(bool(row.get("emotion_changed_by_v2")) for row in repaired),
        "review_required": sum(bool(row.get("emotion_review_required")) for row in repaired),
        "mean_confidence": sum(float(row.get("emotion_confidence", 0)) for row in repaired) / max(1, len(repaired)),
    }
    (out / "emotion_tag_repair_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_distribution_plot(out / "emotion_distribution_before_after_v2.png", before, after)

    fields = [
        "final_path", "language", "source_title", "previous_emotion", "emotion", "style",
        "emotion_confidence", "emotion_score_margin", "emotion_review_required",
        "emotion_evidence", "normalized_transcript",
    ]
    with (out / "emotion_manual_review_v2.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(repaired, key=lambda r: (not r.get("emotion_review_required"), float(r.get("emotion_confidence", 0)))):
            writer.writerow({field: row.get(field, "") for field in fields})

    report = f"""# Emotion Tag Repair v2

## What Changed

The previous repair assigned many clips from whole-video source hints. v2 retags each clip using:

- transcript keyword evidence
- curated source emotion/register hints
- clip-level speech rate and audio activity/loudness
- confidence and margin thresholds

These are still heuristic labels, not human labels and not trained emotion-model predictions.

## Results

| Item | Value |
| --- | ---: |
| Clips retagged | {summary['rows']} |
| Rows changed | {summary['changed_rows']} |
| Manual review required | {summary['review_required']} |
| Mean confidence | {summary['mean_confidence']:.3f} |

## Before Emotion Distribution

{json.dumps(summary['before_emotion_distribution'], indent=2, ensure_ascii=False)}

## After Emotion Distribution

{json.dumps(summary['after_emotion_distribution'], indent=2, ensure_ascii=False)}

## Style Distribution

{json.dumps(summary['style_distribution'], indent=2, ensure_ascii=False)}

## Files

- `emotion_manual_review_v2.csv`
- `emotion_tag_repair_summary.json`
- `emotion_distribution_before_after_v2.png`
"""
    (out / "emotion_tag_repair_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-csv", default="analysis/before_after_metrics/before_after_metrics.csv")
    parser.add_argument("--output-dir", default="analysis/emotion_tag_repair")
    args = parser.parse_args()
    run(args.input, args.output, args.metrics_csv, args.output_dir)
