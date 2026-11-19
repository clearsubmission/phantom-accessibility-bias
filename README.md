# Phantom Accessibility Bias

Code for "Phantom Accessibility Bias: How Recording and Evaluation Artifacts Masquerade as Discrimination in Audio Deepfake Detection".

## Data (not included)
- TORGO: http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html
- ASVspoof 2019 LA: https://datashare.ed.ac.uk/handle/10283/3336
Place them under `data/` and build the manifest with `make_manifest.py`.

## Detectors (not included)
- AASIST / AASIST-L: https://github.com/clovaai/aasist (clone into `aasist/`)
- W2V2-AASIST and XLSR-SLS: public checkpoints, exported to ONNX

## Pipeline
1. `make_manifest.py` — build out/manifest.csv
2. `score.py` (AASIST, AASIST-L) and `score_onnx.py` (W2V2-AASIST, XLSR-SLS)
   Env options: `NORM=1` gain normalization, `TRIM=1` silence trimming, `GAIN_DB=x`, `PAD=zero`, `CKPT=path`
3. `analyze_report.py` — speaker-level audit with exact permutation tests
4. `score_gain.py` — gain swap and dose-response
5. `render_spoofs.py`, `tradeoff_matched.py` — loudness-matched security cost
6. `finetune_aug.py` — augmentation fine-tuning
7. `make_figs.py` — figures
