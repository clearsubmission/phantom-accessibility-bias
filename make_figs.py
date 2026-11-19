import os, itertools, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
os.makedirs("out/figs", exist_ok=True); plt.rcParams.update({"font.size": 8})
DET = ["AASIST", "AASIST-L", "W2V2"]; NAME = {"AASIST": "AASIST", "AASIST-L": "AASIST-L", "W2V2": "W2V2-AASIST"}
ORDER = ["control", "mild", "moderate", "severe"]; COL = {"control": "#4C72B0", "mild": "#DD8452", "moderate": "#937860", "severe": "#C44E52"}
MICS = {"wav_arrayMic": "array mic", "wav_headMic": "head mic"}
def load(f): d = pd.read_csv(f); return d[d.ok == True]
def thr5(c, a=0.05):
    s = np.sort(c); k = int(np.floor(a * (len(s) + 1))); return -np.inf if k < 1 else s[k - 1]
def spk_fpr(f, mic):
    d = load(f); t = d[(d.source == "torgo") & (d.mic == mic)]; thr = np.quantile(t[t.group == "control"].score, 0.05)
    return t.groupby(["group", "speaker"]).score.apply(lambda s: (s < thr).mean()).reset_index(name="fpr")
def perm_p(f, mic):
    s = spk_fpr(f, mic); s = s[s.group.isin(["control", "severe"])]; v = s.fpr.values; sev = (s.group == "severe").values
    null = np.array([v[list(c)].mean() for c in itertools.combinations(range(len(v)), sev.sum())])
    return v[~sev].mean(), v[sev].mean(), np.mean(null >= v[sev].mean() - 1e-12)

# Table 1: headline numbers
rows = []
for M in DET:
    for cond, f in [("raw", f"out/scores_{M}.csv"), ("normalized", f"out/scores_{M}_norm.csv")]:
        for mic in MICS:
            c, s, p = perm_p(f, mic); rows.append(dict(detector=NAME[M], audio=cond, mic=MICS[mic], control_fpr=c, severe_fpr=s, exact_p=p))
tab = pd.DataFrame(rows).round(3); tab.to_csv("out/figs/table1.csv", index=False); print(tab.to_string(index=False))

# Fig 1: per-speaker FPR, raw vs normalized, 3 detectors x 2 mics
fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.0), sharey="row")
for i, M in enumerate(DET):
    for j, mic in enumerate(MICS):
        ax = axes[i, j]
        d = spk_fpr(f"out/scores_{M}.csv", mic).merge(spk_fpr(f"out/scores_{M}_norm.csv", mic), on=["group", "speaker"], suffixes=("_raw", "_norm"))
        d["o"] = d.group.map({g: k for k, g in enumerate(ORDER)}); d = d.sort_values(["o", "speaker"]); x = np.arange(len(d)); w = 0.4
        ax.bar(x - w / 2, d.fpr_raw, w, color=[COL[g] for g in d.group])
        ax.bar(x + w / 2, d.fpr_norm, w, color=[COL[g] for g in d.group], alpha=0.35, hatch="//", edgecolor="k", lw=0.3)
        ax.axhline(0.05, ls="--", c="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(d.speaker, rotation=90, fontsize=6)
        ax.set_title(f"{NAME[M]}, {MICS[mic]}"); ax.set_ylim(0, 0.5 if M == "W2V2" else 1.05)
        for b in np.cumsum([int((d.group == g).sum()) for g in ORDER])[:-1]: ax.axvline(b - 0.5, c="gray", lw=0.5, alpha=0.5)
    axes[i, 0].set_ylabel("FPR on genuine speech")
axes[0, 0].legend(handles=[Patch(color=COL[g], label=g) for g in ORDER] + [Patch(facecolor="w", edgecolor="k", label="raw"),
                  Patch(facecolor="w", edgecolor="k", hatch="//", label="normalized")], frameon=False, fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig("out/figs/fig1_per_speaker.pdf")

# Fig 2: dose-response (raw-waveform detectors)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
STY = {"arrayMic/ctrl": ("#4C72B0", "-", "array / control"), "arrayMic/sev": ("#C44E52", "-", "array / severe"),
       "headMic/ctrl": ("#4C72B0", ":", "head / control"), "headMic/sev": ("#C44E52", ":", "head / severe")}
for ax, M in zip(axes, ["AASIST", "AASIST-L"]):
    o = pd.read_csv(f"out/dose_{M}_table.csv", index_col=0)
    for c, (col, ls, lab) in STY.items(): ax.plot(o.index, o[c], ls, color=col, marker="o", ms=3, label=lab)
    ax.axhline(0.05, ls="--", c="k", lw=0.8); ax.axvline(14, c="gray", lw=0.8, alpha=0.6); ax.text(14.4, 0.45, "measured\ngap ≈14 dB", fontsize=6, color="gray"); ax.set_xlabel("applied gain (dB)"); ax.set_title(M)
axes[0].set_ylabel("FPR on genuine speech"); axes[0].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig("out/figs/fig2_dose_response.pdf")

# Fig 3: causal gain swap, 3 detectors
fig, ax = plt.subplots(figsize=(3.6, 2.6)); w = 0.27
for i, M in enumerate(DET):
    r = load(f"out/scores_{M}.csv"); r = r[(r.source == "torgo") & (r.mic == "wav_arrayMic")]; t = np.quantile(r[r.group == "control"].score, 0.05)
    v = [(r[r.group == "control"].score < t).mean(), (pd.read_csv(f"out/inj_{M}_ctrl_p14.csv").score < t).mean(),
         (r[r.group == "severe"].score < t).mean(), (pd.read_csv(f"out/inj_{M}_sev_m14.csv").score < t).mean()]
    ax.bar_label(ax.bar(np.arange(4) + (i - 1) * w, v, w, label=NAME[M]), fmt="%.2f", fontsize=5)
ax.set_xticks(range(4)); ax.set_xticklabels(["controls\nraw", "controls\n+14 dB", "severe\nraw", "severe\n−14 dB"])
ax.axhline(0.05, ls="--", c="k", lw=0.8); ax.set_ylabel("FPR (array mic)"); ax.legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig("out/figs/fig3_gain_swap.pdf")

# Fig 4: fairness-security tradeoff, pooled vs group-conditional (LOSO, array mic, 5%)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
for i, M in enumerate(DET):
    df = load(f"out/scores_{M}.csv"); sp = df[(df.source == "asvspoof") & (df.label == "spoof")].score.values
    t = df[(df.source == "torgo") & (df.mic == "wav_arrayMic")]; tp = thr5(t[t.group == "control"].score.values); res = {"pooled": [], "group-conditional": []}
    for spk, h in t[t.group == "severe"].groupby("speaker"):
        tm = thr5(t[(t.group == "severe") & (t.speaker != spk)].score.values)
        res["pooled"].append(((h.score < tp).mean(), (sp >= tp).mean())); res["group-conditional"].append(((h.score < tm).mean(), (sp >= tm).mean()))
    for j, (k, v) in enumerate(res.items()):
        v = np.array(v).mean(0); c = ["#4C72B0", "#C44E52"][j]
        axes[0].bar_label(axes[0].bar(i * 2.5 + j, v[0], color=c, label=k if i == 0 else None), fmt="%.2f", fontsize=6); axes[1].bar_label(axes[1].bar(i * 2.5 + j, v[1], color=c), fmt="%.2f", fontsize=6)
for ax, yl in zip(axes, ["severe speakers' FPR", "deepfakes accepted"]):
    ax.set_xticks([0.5, 3.0, 5.5]); ax.set_xticklabels([NAME[m] for m in DET]); ax.set_ylabel(yl)
axes[0].axhline(0.05, ls="--", c="k", lw=0.8); axes[0].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig("out/figs/fig4_tradeoff.pdf")
print("figures:", sorted(os.listdir("out/figs")))
