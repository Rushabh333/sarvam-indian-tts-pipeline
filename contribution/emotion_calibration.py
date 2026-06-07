#!/usr/bin/env python3
"""
Post-hoc emotion calibration for Hindi/English emotion models.

Scalar temperature scaling calibrates confidence but usually cannot change the
predicted class. This module therefore supports a stronger but still lightweight
calibrator:

    calibrated_logits = logits / T + class_bias

The class bias term can correct a NEUTRAL prior collapse; the scalar T handles
confidence calibration. Both are fitted on a small manually labeled set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import classification_report, f1_score


EMOTION_LABELS = [
    "happy",
    "sad",
    "angry",
    "neutral",
    "excited",
    "formal",
    "concerned",
    "sarcastic",
]


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)
    ece = 0.0
    for idx in range(n_bins):
        lo = idx / n_bins
        hi = (idx + 1) / n_bins
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def logits_from_sample(sample: dict) -> np.ndarray:
    if "logits" in sample:
        return np.asarray(sample["logits"], dtype=np.float64)
    if "probs" in sample:
        probs = np.asarray(sample["probs"], dtype=np.float64)
        probs = np.clip(probs, 1e-9, 1.0)
        return np.log(probs)
    raise ValueError("Each sample must contain either 'logits' or 'probs'")


def load_labeled_logits(path: str) -> list[dict]:
    samples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        item["label_idx"] = EMOTION_LABELS.index(item["label"])
        item["logits_arr"] = logits_from_sample(item)
        samples.append(item)
    return samples


def fit_bias_temperature(samples: list[dict]) -> dict:
    logits = np.stack([sample["logits_arr"] for sample in samples])
    labels = np.asarray([sample["label_idx"] for sample in samples], dtype=np.int64)
    n_classes = logits.shape[1]

    def unpack(params: np.ndarray) -> tuple[float, np.ndarray]:
        log_t = params[0]
        temp = float(np.exp(log_t))
        bias = params[1:]
        bias = bias - bias.mean()
        return temp, bias

    def nll(params: np.ndarray) -> float:
        temp, bias = unpack(params)
        probs = stable_softmax((logits / temp) + bias)
        return float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-9)))

    initial = np.zeros(n_classes + 1, dtype=np.float64)
    result = minimize(nll, initial, method="BFGS")
    temp, bias = unpack(result.x)
    return {
        "temperature": temp,
        "bias": bias.tolist(),
        "success": bool(result.success),
        "loss": float(result.fun),
    }


def evaluate(samples: list[dict], calibration: dict | None = None) -> dict:
    logits = np.stack([sample["logits_arr"] for sample in samples])
    labels = np.asarray([sample["label_idx"] for sample in samples], dtype=np.int64)
    if calibration:
        temp = float(calibration["temperature"])
        bias = np.asarray(calibration["bias"], dtype=np.float64)
        probs = stable_softmax((logits / temp) + bias)
    else:
        probs = stable_softmax(logits)
    preds = probs.argmax(axis=1)
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "ece": expected_calibration_error(probs, labels),
        "pred_distribution": {
            EMOTION_LABELS[idx]: int(count)
            for idx, count in Counter(preds).items()
        },
        "report": classification_report(
            labels,
            preds,
            labels=list(range(len(EMOTION_LABELS))),
            target_names=EMOTION_LABELS,
            zero_division=0,
            output_dict=True,
        ),
    }


def run(input_path: str, output_path: str = "contribution/results/emotion_calibration_results.json") -> dict:
    samples = load_labeled_logits(input_path)
    by_lang: dict[str, list[dict]] = {}
    for sample in samples:
        by_lang.setdefault(sample.get("language", "unknown"), []).append(sample)

    results = {"labels": EMOTION_LABELS, "languages": {}}
    for language, lang_samples in sorted(by_lang.items()):
        if len(lang_samples) < 5:
            print(f"Skipping {language}: need at least 5 labeled samples, got {len(lang_samples)}")
            continue
        split = max(1, int(len(lang_samples) * 0.7))
        calib_samples = lang_samples[:split]
        eval_samples = lang_samples[split:] or lang_samples
        calibration = fit_bias_temperature(calib_samples)
        baseline = evaluate(eval_samples)
        calibrated = evaluate(eval_samples, calibration)
        results["languages"][language] = {
            "n_samples": len(lang_samples),
            "n_calibration": len(calib_samples),
            "n_eval": len(eval_samples),
            "calibration": calibration,
            "baseline": baseline,
            "calibrated": calibrated,
            "delta_macro_f1": calibrated["macro_f1"] - baseline["macro_f1"],
            "delta_ece": calibrated["ece"] - baseline["ece"],
        }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote calibration results: {out}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL with audio_path, language, label, and logits or probs")
    parser.add_argument("--output", default="contribution/results/emotion_calibration_results.json")
    args = parser.parse_args()
    run(args.input, args.output)
