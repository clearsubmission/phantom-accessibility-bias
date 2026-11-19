import os, csv, argparse, numpy as np, soundfile as sf, librosa, onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
try: ort.preload_dlls()
except Exception: pass
ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--bs", type=int, default=16); a = ap.parse_args()
CUT = 64600; NORM = os.environ.get("NORM") == "1"; GAIN = float(os.environ.get("GAIN_DB", "0"))
def pad(x): return x[:CUT] if len(x) >= CUT else (np.pad(x, (0, CUT - len(x))) if os.environ.get("PAD") == "zero" else np.tile(x, CUT // len(x) + 1)[:CUT])
def load(p):
    try:
        x, sr = sf.read(p, dtype="float32", always_2d=True); x = x.mean(1)
        if sr != 16000: x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        if os.environ.get("TRIM") == "1": x = librosa.effects.trim(x, top_db=30)[0]
        if len(x) <= 1600: return None, 0.0
        if NORM:
            x = x * (10 ** (-26 / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-8)); pk = np.abs(x).max(); x = x * (0.99 / pk) if pk > 0.99 else x
        return pad(x * 10 ** (GAIN / 20)).astype(np.float32), len(x) / 16000
    except Exception: return None, 0.0
sess = ort.InferenceSession(a.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print("providers:", sess.get_providers()); IN, OUT = sess.get_inputs()[0].name, sess.get_outputs()[0].name
if "CUDAExecutionProvider" not in sess.get_providers() and not os.environ.get("ALLOW_CPU"):
    raise SystemExit("CUDA provider not active -- refusing to run on CPU (set ALLOW_CPU=1 to override)")
rows = list(csv.DictReader(open(a.manifest)))
with ThreadPoolExecutor(8) as ex:
    for i in tqdm(range(0, len(rows), a.bs)):
        chunk = rows[i:i + a.bs]; res = list(ex.map(lambda r: load(r["path"]), chunk))
        xs = [x for x, _ in res if x is not None]
        lg = sess.run([OUT], {IN: np.stack(xs)})[0] if xs else None; k = 0
        for r, (x, d) in zip(chunk, res):
            r.update(dur=d, ok=x is not None, score=float(lg[k, 1]) if x is not None else np.nan); k += x is not None
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", a.out)
