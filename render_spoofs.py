import argparse, os, csv, numpy as np, soundfile as sf, librosa
from concurrent.futures import ThreadPoolExecutor
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="out/manifest.csv"); ap.add_argument("--group", default="severe")
ap.add_argument("--mic", default="wav_arrayMic"); ap.add_argument("--outdir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
rng = np.random.default_rng(a.seed); os.makedirs(a.outdir, exist_ok=True)
def read(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True); x = x.mean(1)
    return librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
def lvl(x): return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
rows = list(csv.DictReader(open(a.manifest)))
gen = [r for r in rows if r["source"] == "torgo" and r["mic"] == a.mic and r["group"] == a.group]
spf = [r for r in rows if r["source"] != "torgo" and r["label"] == "spoof"]
def glvl(r):
    try: x = read(r["path"]); return lvl(x) if len(x) > 1600 else None
    except Exception: return None
with ThreadPoolExecutor(16) as ex: levels = np.array([v for v in ex.map(glvl, gen) if v is not None])
print(f"{a.group}/{a.mic}: {len(levels)} genuine files, RMS level median {np.median(levels):.1f} dBFS (IQR {np.percentile(levels,25):.1f} to {np.percentile(levels,75):.1f})")
targets = rng.choice(levels, size=len(spf))
def render(i):
    r = dict(spf[i])
    try:
        x = read(r["path"]); x = x * 10 ** ((targets[i] - lvl(x)) / 20)
        pk = np.abs(x).max(); x = x * (0.99 / pk) if pk > 0.99 else x
        p = os.path.join(a.outdir, os.path.splitext(os.path.basename(r["path"]))[0] + ".wav"); sf.write(p, x, 16000); r["path"] = p
        return r
    except Exception: return None
with ThreadPoolExecutor(16) as ex: out = [r for r in ex.map(render, range(len(spf))) if r is not None]
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(out)
print("wrote", len(out), "spoofs ->", a.out)
