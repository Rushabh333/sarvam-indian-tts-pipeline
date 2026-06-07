# Transcript Coverage and Preservation Report

## Coverage

| Item | Value |
| --- | ---: |
| Final clips | 188 |
| Normalized transcripts present | 188 |
| Approx transcripts present | 188 |
| Local ASR-filled transcripts | 110 |
| Original pipeline transcripts | 78 |

## Raw vs Final ASR Preservation

These scores were computed by independently transcribing each raw before-clip and final after-clip on the AWS GPU with faster-whisper `base`, then comparing the two ASR strings.

| Split | Clips | Char similarity median | Char similarity mean | Word F1 median | Word F1 mean | Word F1 < 0.90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 188 | 0.981 | 0.797 | 0.966 | 0.830 | 58 |
| en-IN | 81 | 1.000 | 0.949 | 1.000 | 0.985 | 2 |
| hi-IN | 107 | 0.900 | 0.682 | 0.884 | 0.714 | 56 |

## Interpretation

Transcript coverage is now complete in `data/metadata/10_final.jsonl`: every final clip has `approx_transcript` and `normalized_transcript`.

English preservation is strong: median raw/final ASR word F1 is 1.000. Hindi/code-mixed preservation is less reliable by this ASR metric: median word F1 is 0.884, and many low-score rows show ASR script/language instability rather than obvious audio corruption. For Hindi, this metric should be used as a manual-review priority signal, not as a hard failure label.

Low-similarity rows are written to `low_asr_similarity_review.csv`.

## Plots

- `asr_word_f1_hist_by_language.png`
- `asr_char_similarity_hist_by_language.png`
- `asr_similarity_by_language.png`
