import os
from pathlib import Path
from datasets import load_dataset

# Define dataset directory
DATASET_ROOT = Path(os.path.expanduser("~/sarvam-indian-tts-60min/data"))

# Splits and corresponding parquet filenames
splits = {
    "english_train": "english_train-00000-of-00001.parquet",
    "english_val": "english_val-00000-of-00001.parquet",
    "hindi_train": "hindi_train-00000-of-00001.parquet",
    "hindi_val": "hindi_val-00000-of-00001.parquet",
}

# Load each split
datasets = {}
for name, filename in splits.items():
    ds = load_dataset("parquet", data_files=str(DATASET_ROOT / filename), split="train")
    datasets[name] = ds

# Output directory for extracted wavs
OUT_DIR = Path("extracted_wav")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Iterate and write wavs
for split_name, ds in datasets.items():
    split_dir = OUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for idx, record in enumerate(ds):
        audio = record["audio"]
        # audio dict may contain 'bytes' or a 'path'
        if isinstance(audio, dict) and "bytes" in audio:
            wav_bytes = audio["bytes"]
        elif isinstance(audio, dict) and "path" in audio:
            wav_path = Path(audio["path"]).expanduser()
            wav_bytes = wav_path.read_bytes()
        else:
            # fallback: treat as raw bytes
            wav_bytes = audio
        out_path = split_dir / f"{split_name}_{idx:05d}.wav"
        out_path.write_bytes(wav_bytes)
    print(f"Extracted {len(ds)} wavs to {split_dir}")

