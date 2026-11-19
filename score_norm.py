import argparse, json, os, sys, csv, importlib
import numpy as np, soundfile as sf, librosa, torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
CUT = 64600  # AASIST input length (~4 s at 16 kHz), repeat-padded like the official eval
def pad(x):
    if len(x) >= CUT: return x[:CUT]
    return np.tile(x, CUT // len(x) + 1)[:CUT]
class DS(Dataset):
    def __init__(s, rows): s.rows = rows
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        try:
            x, sr = sf.read(s.rows[i]["path"], dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            if sr != 16000: x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            ok = len(x) > 1600
            if ok:
                r = np.sqrt(np.mean(x ** 2)) + 1e-8; x = x * (10 ** (-26 / 20) / r)
                pk = np.abs(x).max(); x = x * (0.99 / pk) if pk > 0.99 else x
        except Exception:
            x, ok = np.zeros(0, np.float32), False
        dur = len(x) / 16000
        x = pad(x) if ok else np.zeros(CUT, np.float32)
        return torch.from_numpy(x.astype(np.float32)), i, dur, ok
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True); ap.add_argument("--aasist", required=True)
ap.add_argument("--model", default="AASIST"); ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
sys.path.insert(0, a.aasist)
cfg = json.load(open(os.path.join(a.aasist, "config", f"{a.model}.conf")))
mc = cfg["model_config"]
Model = importlib.import_module(f"models.{mc['architecture']}").Model
model = Model(mc).to(dev)
model.load_state_dict(torch.load(os.path.join(a.aasist, cfg["model_path"]), map_location=dev))
model.eval()
rows = list(csv.DictReader(open(a.manifest)))
dl = DataLoader(DS(rows), batch_size=a.bs, num_workers=a.workers, multiprocessing_context="fork")
with torch.no_grad():
    for x, idx, dur, ok in tqdm(dl):
        _, out = model(x.to(dev))
        s = out[:, 1].float().cpu().numpy()  # index 1 = bonafide; higher = more genuine
        for j, i in enumerate(idx.tolist()):
            rows[i].update(score=float(s[j]), dur=float(dur[j]), ok=bool(ok[j]))
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", a.out, "on", dev)
