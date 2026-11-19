import sys, itertools, numpy as np, pandas as pd
df = pd.read_csv(sys.argv[1]); df = df[(df.ok == True) & (df.source == "torgo")].copy()
p = df.path.str.split("/"); df["key"] = df.speaker + "/" + p.str[-3] + "/" + p.str[-1]
thr = {m: np.quantile(g[g.group == "control"].score, 0.05) for m, g in df.groupby("mic")}
df["flag"] = (df.score < df.mic.map(thr)).astype(int)
print(f"\n##### {sys.argv[1]} (threshold = 5% quantile of same-mic control scores)")
for mic, g in df.groupby("mic"):
    s = g.groupby(["speaker", "group"]).flag.mean().reset_index()
    s = s[s.group.isin(["control", "severe"])]
    v = s.flag.values; sev = s.group.values == "severe"; obs = v[sev].mean()
    null = np.array([v[list(c)].mean() for c in itertools.combinations(range(len(v)), sev.sum())])
    print(f"{mic}: severe {obs:.3f} vs control {v[~sev].mean():.3f} | exact one-sided p = {np.mean(null >= obs - 1e-12):.4f} ({len(null)} perms)")
w = df.pivot_table(index=["key", "group"], columns="mic", values="flag", aggfunc="first").dropna().reset_index()
w["array_only"] = (w.wav_arrayMic == 1) & (w.wav_headMic == 0)
w["head_only"] = (w.wav_arrayMic == 0) & (w.wav_headMic == 1)
w["both"] = (w.wav_arrayMic == 1) & (w.wav_headMic == 1)
print(f"\npaired utterances (same recording, both mics): {len(w)}")
print(w.groupby("group")[["wav_arrayMic", "wav_headMic", "array_only", "head_only", "both"]].mean().round(3).to_string())
