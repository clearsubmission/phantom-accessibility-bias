import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_curve
df = pd.read_csv(sys.argv[1]); df = df[df.ok == True]
asv = df[df.source == "asvspoof"]
y = (asv.label == "bonafide").astype(int).values; s = asv.score.values
fpr, tpr, thr = roc_curve(y, s); fnr = 1 - tpr
k = np.nanargmin(np.abs(fnr - fpr)); t_eer = thr[k]
t_1 = np.quantile(asv[asv.label == "bonafide"].score, 0.01)  # rejects 1% of ASVspoof bona fide
print(f"SANITY  ASVspoof EER = {100*(fpr[k]+fnr[k])/2:.2f}%  (AASIST paper reports ~0.8%)")
t = df[df.source == "torgo"].copy()
t["flag_eer"] = t.score < t_eer; t["flag_1"] = t.score < t_1
t["len"] = np.where(t.dur < 1.5, "short<1.5s", "long>=1.5s")
order = ["control", "mild", "moderate", "severe", "unknown"]
agg = dict(n=("score", "size"), spk=("speaker", "nunique"), fpr_eer=("flag_eer", "mean"),
           fpr_1pct=("flag_1", "mean"), mean_score=("score", "mean"))
for keys in [["group"], ["mic", "group"], ["len", "group"]]:
    g = t.groupby(keys).agg(**agg).reset_index()
    g["o"] = g.group.map({k: i for i, k in enumerate(order)})
    print("\n== FPR on GENUINE TORGO speech by", keys); print(g.sort_values(keys[:-1] + ["o"]).drop(columns="o").round(3).to_string(index=False))
sp = t.groupby(["group", "speaker"]).agg(n=("score", "size"), fpr_eer=("flag_eer", "mean"),
                                        fpr_1pct=("flag_1", "mean")).round(3)
print("\n== per speaker"); print(sp.to_string())
sp.to_csv(sys.argv[1].replace(".csv", "_per_speaker.csv"))
