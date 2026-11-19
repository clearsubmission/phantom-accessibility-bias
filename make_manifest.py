import argparse, glob, os, re, csv, random
# TORGO severity labels: VERIFY against TORGO docs / the papers you cite
SEVERITY = {"F01":"severe","M01":"severe","M02":"severe","M04":"severe",
            "M05":"moderate","F03":"mild","F04":"mild","M03":"mild"}
ap = argparse.ArgumentParser()
ap.add_argument("--torgo", required=True); ap.add_argument("--la", required=True)
ap.add_argument("--asv_spoof_n", type=int, default=10000)
ap.add_argument("--out", required=True); a = ap.parse_args()
rows = []
for p in glob.glob(os.path.join(a.torgo, "**", "wav_*Mic", "*.wav"), recursive=True):
    parts = p.split(os.sep)
    spk = next((x for x in parts if re.fullmatch(r"[FM]C?\d+", x)), None)
    mic = next((x for x in parts if x.startswith("wav_")), "")
    if spk is None: continue
    ctrl = "C" in spk
    rows.append(dict(path=p, source="torgo", speaker=spk, mic=mic,
                     group="control" if ctrl else SEVERITY.get(spk, "unknown"),
                     label="bonafide", attack="-"))
proto = os.path.join(a.la, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt")
asv = []
for line in open(proto):
    spk, utt, _, attack, key = line.split()
    asv.append(dict(path=os.path.join(a.la, "ASVspoof2019_LA_eval", "flac", utt + ".flac"),
                    source="asvspoof", speaker=spk, mic="-",
                    group="asv_" + key, label=key, attack=attack))
random.seed(0)
bona = [r for r in asv if r["label"] == "bonafide"]
spoof = [r for r in asv if r["label"] == "spoof"]
rows += bona + random.sample(spoof, min(a.asv_spoof_n, len(spoof)))
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
from collections import Counter
print(Counter(r["group"] for r in rows)); print("unknown TORGO speakers:",
      sorted({r["speaker"] for r in rows if r["group"] == "unknown"}))
print("TORGO per speaker/mic:", Counter((r["speaker"], r["mic"]) for r in rows if r["source"] == "torgo"))
