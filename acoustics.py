import numpy as np, pandas as pd, soundfile as sf, librosa
from multiprocessing import get_context
def feats(path):
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True); x = x.mean(1)
        if sr != 16000: x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        if len(x) < 1600: return (path,) + (np.nan,) * 4
        db = 20 * np.log10(librosa.feature.rms(y=x, frame_length=400, hop_length=160)[0] + 1e-8)
        iv = librosa.effects.split(x, top_db=30, frame_length=400, hop_length=160)
        return (path, float(np.percentile(db, 95)), float(np.percentile(db, 95) - np.percentile(db, 10)),
                float(sum(e - s for s, e in iv) / len(x)), float(len(iv) / (len(x) / 16000)))
    except Exception: return (path,) + (np.nan,) * 4
if __name__ == "__main__":
    m = pd.read_csv("out/manifest.csv"); paths = m[m.source == "torgo"].path.tolist()
    with get_context("fork").Pool(16) as pool: res = pool.map(feats, paths, chunksize=64)
    pd.DataFrame(res, columns=["path", "level_db", "snr_db", "voiced_frac", "segs_per_s"]).to_csv("out/acoustics.csv", index=False)
    print("wrote out/acoustics.csv", len(res))
