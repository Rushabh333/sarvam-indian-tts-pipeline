#!/usr/bin/env python3
"""Create a 37-clip manual review pack and Google Form generator.

The script prepares:
- a balanced, risk-prioritized 37-clip manifest
- final/raw audio files staged for upload to HuggingFace
- a Google Apps Script that creates the review form from public URLs
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review_form"
PUBLIC_BASE = (
    "https://huggingface.co/datasets/Rushabh3/sarvam-indian-tts-60min"
    "/resolve/main/review_form_37"
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_low_asr_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f) if row.get("id")}


def clip_id(row: dict) -> str:
    return Path(row["final_path"]).stem


def risk_score(row: dict, low_asr_ids: set[str]) -> float:
    score = 0.0
    cid = clip_id(row)
    if row.get("emotion_review_required"):
        score += 3.0
    if cid in low_asr_ids:
        score += 3.0
    if row.get("language") == "hi-IN":
        score += 0.5
    try:
        score += max(0.0, 0.9 - float(row.get("ecapa_similarity", 1.0))) * 10.0
    except Exception:
        pass
    try:
        duration = float(row.get("duration_s", 0.0))
        if duration < 5 or duration > 25:
            score += 0.75
    except Exception:
        pass
    return score


def select_review_rows(rows: list[dict], count: int = 37) -> list[dict]:
    low_asr_ids = load_low_asr_ids(ROOT / "analysis/before_after_metrics/low_asr_similarity_review.csv")
    rows = sorted(rows, key=lambda r: (-risk_score(r, low_asr_ids), r.get("language", ""), clip_id(r)))

    target_hi = 21
    target_en = count - target_hi
    source_cap = 5
    selected: list[dict] = []
    selected_ids: set[str] = set()
    source_counts: dict[str, int] = {}

    for language, target in (("hi-IN", target_hi), ("en-IN", target_en)):
        for row in rows:
            cid = clip_id(row)
            source = row.get("source_title", "")
            if (
                row.get("language") == language
                and cid not in selected_ids
                and source_counts.get(source, 0) < source_cap
            ):
                selected.append(row)
                selected_ids.add(cid)
                source_counts[source] = source_counts.get(source, 0) + 1
                if sum(1 for r in selected if r.get("language") == language) >= target:
                    break

    # If the cap prevented reaching 37 because the source pool is small, fill
    # the remainder by risk score while preserving the language target if possible.
    for row in rows:
        if len(selected) >= count:
            break
        cid = clip_id(row)
        if cid not in selected_ids:
            selected.append(row)
            selected_ids.add(cid)

    return selected[:count]


def copy_audio(selected: list[dict]) -> None:
    final_dir = OUT / "public_upload/review_form_37/final"
    raw_dir = OUT / "public_upload/review_form_37/raw"
    final_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for row in selected:
        final = ROOT / row["final_path"]
        raw = ROOT / row["raw_segment_path"]
        shutil.copy2(final, final_dir / final.name)
        shutil.copy2(raw, raw_dir / raw.name)


def build_manifest(selected: list[dict]) -> list[dict]:
    manifest = []
    for i, row in enumerate(selected, start=1):
        final_name = Path(row["final_path"]).name
        raw_name = Path(row["raw_segment_path"]).name
        transcript = row.get("normalized_transcript") or row.get("transcript") or ""
        manifest.append(
            {
                "review_id": f"R{i:02d}",
                "clip_id": clip_id(row),
                "language": row.get("language", ""),
                "duration_s": round(float(row.get("duration_s", 0.0)), 2),
                "emotion": row.get("emotion", ""),
                "style": row.get("style", ""),
                "emotion_review_required": bool(row.get("emotion_review_required")),
                "ecapa_similarity": row.get("ecapa_similarity", ""),
                "snr_db": row.get("snr_db", ""),
                "source_title": row.get("source_title", ""),
                "source_url": row.get("source_url", ""),
                "transcript": transcript,
                "final_audio_url": f"{PUBLIC_BASE}/final/{final_name}?download=true",
                "raw_before_audio_url": f"{PUBLIC_BASE}/raw/{raw_name}?download=true",
            }
        )
    return manifest


def write_manifest(manifest: list[dict]) -> None:
    csv_path = OUT / "selected_37_review_manifest.csv"
    json_path = OUT / "selected_37_review_manifest.json"
    fields = list(manifest[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def apps_script(manifest: list[dict]) -> str:
    data = json.dumps(manifest, ensure_ascii=False, indent=2)
    return f"""/**
 * Google Form generator for Sarvam TTS manual audio review.
 *
 * Usage:
 * 1. Open https://script.google.com/
 * 2. Create a new Apps Script project.
 * 3. Paste this whole file.
 * 4. Run createSarvamReviewForm().
 * 5. Approve permissions. The form URL is printed in the execution log.
 */

const REVIEW_CLIPS = {data};

function addChoiceQuestion(form, title, choices, required) {{
  const item = form.addMultipleChoiceItem();
  item.setTitle(title);
  item.setChoiceValues(choices);
  item.setRequired(required);
  return item;
}}

function createSarvamReviewForm() {{
  const form = FormApp.create('Sarvam TTS Manual Review - 37 Clips');
  form.setDescription(
    'Manual quality review for 37 clips from the Sarvam Indian TTS dataset. ' +
    'Open each public audio link, listen to the final clip, optionally compare with raw-before audio, then answer the questions.'
  );
  form.setCollectEmail(false);
  form.setProgressBar(true);
  form.setLimitOneResponsePerUser(false);

  REVIEW_CLIPS.forEach((clip, index) => {{
    form.addPageBreakItem()
      .setTitle(`${{clip.review_id}} - ${{clip.language}} - ${{clip.duration_s}}s`)
      .setHelpText(
        `Final audio: ${{clip.final_audio_url}}\\n` +
        `Raw before audio: ${{clip.raw_before_audio_url}}\\n` +
        `Source: ${{clip.source_title}}\\n${{clip.source_url}}\\n` +
        `Pipeline label: ${{clip.emotion}} / ${{clip.style}}\\n` +
        `Transcript: ${{clip.transcript}}`
      );

    addChoiceQuestion(form, `${{clip.review_id}} Q1 - Audio clarity`, [
      '5 - Studio quality / very clean',
      '4 - Clean, minor room tone only',
      '3 - Acceptable, minor issues',
      '2 - Noticeable noise, distracting',
      '1 - Heavy noise/music bleed, unusable'
    ], true);

    addChoiceQuestion(form, `${{clip.review_id}} Q2 - Speaker consistency`, [
      'Yes, clearly one speaker',
      'Mostly one speaker, brief overlap at boundary',
      'No, two or more speakers audible'
    ], true);

    addChoiceQuestion(form, `${{clip.review_id}} Q3 - Boundary quality`, [
      'Clean start and clean end',
      'One side is cut',
      'Both sides are cut mid-sentence',
      'Unnatural abrupt cut with click/pop'
    ], true);

    addChoiceQuestion(form, `${{clip.review_id}} Q4 - Speech naturalness`, [
      'Natural spontaneous speech',
      'Slightly read/scripted but acceptable',
      'Clearly read aloud/audiobook-style',
      'Synthetic or heavily processed voice'
    ], true);

    addChoiceQuestion(form, `${{clip.review_id}} Q5 - Dominant emotion/register`, [
      'Neutral / informational',
      'Happy / excited',
      'Sad / somber',
      'Angry / frustrated',
      'Formal / authoritative',
      'Conversational / casual',
      'Expressive / dramatic',
      'Cannot determine'
    ], true);

    addChoiceQuestion(form, `${{clip.review_id}} Q6 - Final decision`, [
      'Keep',
      'Keep after trimming boundary',
      'Reject',
      'Unsure / needs second reviewer'
    ], true);

    form.addParagraphTextItem()
      .setTitle(`${{clip.review_id}} Notes`)
      .setHelpText('Optional: mention noise, language mismatch, bad boundary, wrong emotion, or transcript issue.')
      .setRequired(false);
  }});

  Logger.log('Edit URL: ' + form.getEditUrl());
  Logger.log('Public response URL: ' + form.getPublishedUrl());
}}
"""


def write_readme() -> None:
    (OUT / "README.md").write_text(
        """# Manual Review Google Form Pack

This folder contains the 37-clip manual review setup.

Files:
- `selected_37_review_manifest.csv`: reviewer/debug manifest with public audio URLs.
- `selected_37_review_manifest.json`: same manifest in JSON.
- `create_google_form.gs`: paste into Google Apps Script and run `createSarvamReviewForm()`.
- `public_upload/review_form_37/`: audio files uploaded to the public HuggingFace dataset.

The Google Form uses public HuggingFace audio links because this environment does not have Google account OAuth configured. Once the Apps Script runs from your Google account, it creates the actual Google Form and prints both the edit URL and response URL in the Apps Script logs.
""",
        encoding="utf-8",
    )


def main() -> None:
    rows = load_jsonl(ROOT / "data/metadata/10_final.jsonl")
    selected = select_review_rows(rows)
    copy_audio(selected)
    manifest = build_manifest(selected)
    write_manifest(manifest)
    (OUT / "create_google_form.gs").write_text(apps_script(manifest), encoding="utf-8")
    write_readme()

    by_lang: dict[str, int] = {}
    for row in manifest:
        by_lang[row["language"]] = by_lang.get(row["language"], 0) + 1
    print(f"wrote {len(manifest)} review clips")
    print(by_lang)
    print(OUT / "selected_37_review_manifest.csv")
    print(OUT / "create_google_form.gs")


if __name__ == "__main__":
    main()
