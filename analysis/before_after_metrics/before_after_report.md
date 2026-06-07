# Before/After Clip Quality Metrics

Measured matched raw/final clip pairs from `data/metadata/10_final.jsonl`.

## Dataset

- Pairs measured: 188
- Language counts: {'en-IN': 81, 'hi-IN': 107}
- Raw clips at 16 kHz: 188/188
- Final clips at 24 kHz: 188/188
- Transcript available: 188/188

## Mean Results

| Metric | Before/raw | After/final | Delta |
| --- | ---: | ---: | ---: |
| SNR dB | 39.45 | 39.20 | -0.25 |
| Clipping % | 0.0000 | 0.0000 | +0.0000 |
| Integrated loudness LUFS | -20.84 | -23.33 | -2.49 |
| LUFS target error vs -23 | +2.16 | -0.33 | error delta -3.37 |
| RMS dBFS | -21.29 | -23.73 | -2.45 |
| Peak dBFS | -3.34 | -5.56 | -2.22 |
| Active speech ratio | 0.618 | 0.613 | -0.004 |
| Silence ratio | 0.382 | 0.387 | +0.004 |
| Loudness range dB | 36.90 | 36.62 | -0.27 |
| Crest factor dB | 17.95 | 18.17 | +0.22 |
| Spectral flatness | 0.032338 | 0.022527 | -0.009810 |
| Low-frequency rumble ratio | 0.005989 | 0.006140 | +0.000151 |
| High-frequency noise ratio | 0.002478 | 0.002260 | -0.000218 |
| Speech-band energy ratio | 0.991533 | 0.991600 | +0.000067 |
| Duration mismatch seconds | - | 0.0000 | lower is better |
| Acoustic similarity proxy | - | 0.9912 | higher is better |

## Gate Summary

| Gate | Result |
| --- | ---: |
| SNR improved by >0.5 dB | 4.3% |
| SNR unchanged within +/-0.5 dB | 71.8% |
| SNR worsened by >0.5 dB | 23.9% |
| Raw LUFS within 2 dB of -23 | 36.2% |
| Final LUFS within 2 dB of -23 | 96.8% |
| Duration delta under 50 ms | 100.0% |
| Acoustic similarity >= 0.98 | 81.9% |
| Final clips with zero clipping | 100.0% |

## Interpretation

The before/after evidence does not support a claim that enhancement improved acoustic quality. Mean proxy SNR moved by -0.25 dB, so enhancement was effectively neutral/slightly negative by this metric. Acoustic similarity is high, which means the output is very close to the input.

What the pipeline did measurably improve is dataset standardization: final files are 24 kHz, duration alignment is effectively exact, clipping remains zero, peak levels are safer, and loudness is moved toward the -23 LUFS target. This is a standardization/export-readiness win, not a denoising win.

`acoustic_similarity` is a mel-spectrogram cosine proxy for speaker/content preservation. It is not a replacement for a dedicated ECAPA raw-vs-final speaker model, but it is useful for confirming that finalization did not materially change the spoken content.

`snr_db` remains a proxy based on percentile noise floor and should not be presented as DNSMOS or perceptual MOS. The stronger added checks here are LUFS target error, clipping, active/silence ratio, spectral flatness, low-frequency rumble, high-frequency noise, speech-band energy, duration alignment, and acoustic preservation.

## Plots

- `quality_means_before_after.png`
- `snr_raw_vs_final.png`
- `rms_raw_vs_final.png`
- `snr_delta_hist.png`
- `lufs_target_error_delta_hist.png`
- `high_freq_noise_delta_hist.png`
- `spectral_flatness_delta_hist.png`
- `duration_delta_hist.png`
- `acoustic_similarity_hist.png`
