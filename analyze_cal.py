import sys, numpy as np, pandas as pd
df = pd.read_csv(sys.argv[1]); df = df[(df.ok == True) & (df.source == "torgo")]
for alpha in [0.05, 0.01]:
    rows = []
    for mic, g in df.groupby("mic"):
        t = np.quantile(g[g.group == "control"].score, alpha)  # threshold calibrated on same-mic controls
        for (grp, spk), h in g.groupby(["group", "speaker"]):
            rows.append(dict(mic=mic, group=grp, speaker=spk, n=len(h), fpr=(h.score < t).mean()))
    o = pd.DataFrame(rows)
    print(f"\n=== {sys.argv[1]} | threshold = {alpha:.0%} quantile of same-mic CONTROL scores ===")
    print(o.groupby(["mic", "group"]).agg(spk=("speaker", "nunique"), fpr_mean_spk=("fpr", "mean"),
          fpr_min=("fpr", "min"), fpr_max=("fpr", "max")).round(3).to_string())
    print(o.pivot_table(index=["group", "speaker"], columns="mic", values="fpr").round(3).to_string())
