import sys, numpy as np, pandas as pd
def thr(cal, a):  # split-conformal: P(genuine flagged) <= a under exchangeability
    s = np.sort(cal); k = int(np.floor(a * (len(s) + 1)))
    return -np.inf if k < 1 else s[k - 1]
df = pd.read_csv(sys.argv[1]); df = df[df.ok == True]
spoof = df[(df.source == "asvspoof") & (df.label == "spoof")].score.values
t = df[df.source == "torgo"]
for a in [0.05, 0.01]:
    rows = []
    for mic, g in t.groupby("mic"):
        for spk, h in g.groupby("speaker"):
            grp = h.group.iloc[0]
            tp = thr(g[(g.group == "control") & (g.speaker != spk)].score.values, a)
            others = g[(g.group == grp) & (g.speaker != spk)]
            r = dict(mic=mic, group=grp, speaker=spk, fpr_pooled=(h.score < tp).mean(), spoof_pass_pooled=(spoof >= tp).mean())
            if others.speaker.nunique() >= 2:
                tm = thr(others.score.values, a)
                r.update(fpr_group=(h.score < tm).mean(), spoof_pass_group=(spoof >= tm).mean())
            rows.append(r)
    o = pd.DataFrame(rows)
    print(f"\n##### {sys.argv[1]} | target FPR {a:.0%} | leave-one-speaker-out")
    print(o.groupby(["mic", "group"]).mean(numeric_only=True).round(3).to_string())
    print(o.set_index(["mic", "group", "speaker"]).round(3).to_string())
