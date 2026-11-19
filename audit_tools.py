import itertools, numpy as np, pandas as pd
MICS = ["wav_arrayMic", "wav_headMic"]

def load(f):
    d = pd.read_csv(f)
    if "ok" in d: d = d[d.ok.astype(str).str.lower() == "true"]
    return d.dropna(subset=["score"])

def eer(d):
    a = d[d.source != "torgo"]
    if len(a) == 0 or a.label.nunique() < 2: return float("nan")
    y = (a.label == "bonafide").values; s = a.score.values
    o = np.argsort(s); y = y[o]
    frr = np.cumsum(y) / y.sum(); far = 1 - np.cumsum(~y) / (~y).sum()
    i = np.argmin(np.abs(frr - far)); return 100 * (frr[i] + far[i]) / 2

def thr(d, mic):
    t = d[(d.source == "torgo") & (d.mic == mic)]
    return np.quantile(t[t.group == "control"].score, 0.05)

def spk_fpr(d, mic, th, std=False):
    t = d[(d.source == "torgo") & (d.mic == mic) & d.group.isin(["control", "severe"])].copy()
    t["rej"] = t.score < th
    if not std: return t.groupby(["group", "speaker"]).rej.mean().reset_index(name="fpr")
    t["band"] = pd.cut(t.dur, [0, 1.5, 3, 1e9], labels=[0, 1, 2])
    w = t[t.group == "control"].band.value_counts(normalize=True)
    rows = []
    for (g, s), h in t.groupby(["group", "speaker"]):
        v = 0.0
        for b, wb in w.items():
            hb = h[h.band == b]; v += wb * (hb.rej.mean() if len(hb) else h.rej.mean())
        rows.append(dict(group=g, speaker=s, fpr=v))
    return pd.DataFrame(rows)

def perm(s):
    v = s.fpr.values; sev = (s.group == "severe").values
    null = np.array([v[list(c)].mean() for c in itertools.combinations(range(len(v)), sev.sum())])
    return v[~sev].mean(), v[sev].mean(), np.mean(null >= v[sev].mean() - 1e-12)

def short_long(d, mic, th):
    t = d[(d.source == "torgo") & (d.mic == mic) & (d.group == "control")]
    return (t[t.dur < 1.5].score < th).mean(), (t[t.dur >= 1.5].score < th).mean()

def report(f):
    d = load(f); print(f"== {f}   ASVspoof EER {eer(d):.2f}%")
    for m in MICS:
        th = thr(d, m); c, s, p = perm(spk_fpr(d, m, th)); cs, ss, ps = perm(spk_fpr(d, m, th, std=True)); sh, lo = short_long(d, m, th)
        print(f"  {m:13s} raw: sev {s:.3f} ctl {c:.3f} p={p:.3f} | len-std: sev {ss:.3f} ctl {cs:.3f} p={ps:.3f} | ctl FPR short {sh:.3f} long {lo:.3f}")
