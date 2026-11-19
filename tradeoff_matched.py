import sys, os, numpy as np
from audit_tools import load
d = load(sys.argv[1]); a = d[(d.source == "torgo") & (d.mic == "wav_arrayMic")]
sets = {"original": d[(d.source != "torgo") & (d.label == "spoof")].score.values}
for f in sys.argv[2:]: sets[os.path.basename(f)] = load(f).score.values
tp = np.quantile(a[a.group == "control"].score, 0.05)
res = {"pooled": [], "group-conditional": []}
for spk in sorted(a[a.group == "severe"].speaker.unique()):
    h = a[(a.group == "severe") & (a.speaker == spk)].score.values
    tm = np.quantile(a[(a.group == "severe") & (a.speaker != spk)].score, 0.05)
    for name, t in [("pooled", tp), ("group-conditional", tm)]:
        res[name].append([(h < t).mean()] + [(s >= t).mean() for s in sets.values()])
print(f"== {sys.argv[1]}")
print("  threshold           severe FPR  " + "  ".join(f"accept[{k}]" for k in sets))
for name, v in res.items():
    v = np.array(v).mean(0); print(f"  {name:18s}  {v[0]:.3f}       " + "  ".join(f"{x:.3f}" for x in v[1:]))
