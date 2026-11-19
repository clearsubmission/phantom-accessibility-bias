import sys, numpy as np, pandas as pd
df = pd.read_csv(sys.argv[1]); df = df[(df.ok == True) & (df.source == "torgo")].merge(pd.read_csv("out/acoustics.csv"), on="path")
thr = {m: np.quantile(g[g.group == "control"].score, 0.05) for m, g in df.groupby("mic")}
df["flag"] = (df.score < df.mic.map(thr)).astype(int)
print(f"\n##### {sys.argv[1]}\n== acoustics by mic x group")
print(df.groupby(["mic", "group"])[["level_db", "snr_db", "voiced_frac", "segs_per_s"]].median().round(2).to_string())
a = df[df.mic == "wav_arrayMic"].copy(); a["snr_bin"] = pd.qcut(a.snr_db, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
print("\n== ARRAY mic: FPR by SNR quartile x group (is severe still higher at matched SNR?)")
print(a.pivot_table(index="snr_bin", columns="group", values="flag", aggfunc="mean", observed=True).round(3).to_string())
print(a.pivot_table(index="snr_bin", columns="group", values="flag", aggfunc="size", observed=True).to_string())
