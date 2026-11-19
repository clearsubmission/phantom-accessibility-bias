import os, sys, json, random, argparse, importlib, numpy as np, soundfile as sf, librosa, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
ap = argparse.ArgumentParser()
ap.add_argument("--aasist", default="aasist"); ap.add_argument("--model", default="AASIST")
ap.add_argument("--la", default="data/LA"); ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--dev_n", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
CUT = 64600
def tile(x): return x[:CUT] if len(x) >= CUT else np.tile(x, CUT // len(x) + 1)[:CUT]
def proto(split):
    fn = os.path.join(a.la, "ASVspoof2019_LA_cm_protocols", f"ASVspoof2019.LA.cm.{split}.{'trn' if split == 'train' else 'trl'}.txt")
    root = os.path.join(a.la, f"ASVspoof2019_LA_{split}", "flac")
    return [(os.path.join(root, p[1] + ".flac"), 1 if p[4] == "bonafide" else 0) for p in (l.split() for l in open(fn))]
def read(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True); x = x.mean(1)
    return librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
class DS(Dataset):
    def __init__(s, rows, aug): s.rows, s.aug = rows, aug
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        p, y = s.rows[i]; x = read(p)
        if s.aug:
            if random.random() < 0.5:
                xt = librosa.effects.trim(x, top_db=30)[0]
                if len(xt) > 1600: x = xt
            if random.random() < 0.3:
                L = int(random.uniform(0.5, 1.5) * 16000)
                if len(x) > L: st = random.randint(0, len(x) - L); x = x[st:st + L]
            if len(x) > CUT: st = random.randint(0, len(x) - CUT); x = x[st:st + CUT]
            x = tile(x) * 10 ** (random.uniform(-20, 20) / 20)
            pk = np.abs(x).max(); x = x * (0.99 / pk) if pk > 0.99 else x
        else:
            x = tile(x)
        return torch.from_numpy(x.astype(np.float32)), y
def eer(s, y):
    o = np.argsort(s); y = y[o].astype(bool); frr = np.cumsum(y) / y.sum(); far = 1 - np.cumsum(~y) / (~y).sum()
    i = np.argmin(np.abs(frr - far)); return 100 * (frr[i] + far[i]) / 2
dev = "cuda"
sys.path.insert(0, a.aasist)
cfg = json.load(open(os.path.join(a.aasist, "config", f"{a.model}.conf"))); mc = cfg["model_config"]
model = importlib.import_module(f"models.{mc['architecture']}").Model(mc).to(dev)
model.load_state_dict(torch.load(os.path.join(a.aasist, cfg["model_path"]), map_location=dev))
tr = proto("train"); dv = proto("dev"); random.Random(a.seed).shuffle(dv); dv = dv[:a.dev_n]
dl_tr = DataLoader(DS(tr, True), batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True, multiprocessing_context="fork")
dl_dv = DataLoader(DS(dv, False), batch_size=128, num_workers=a.workers, multiprocessing_context="fork")
def evaluate():
    model.eval(); S, Y = [], []
    with torch.no_grad():
        for x, y in dl_dv: S.append(model(x.to(dev))[1][:, 1].float().cpu().numpy()); Y.append(y.numpy())
    model.train(); return eer(np.concatenate(S), np.concatenate(Y))
opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
crit = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9], device=dev))
best = evaluate(); print(f"epoch 0 (pretrained) dev EER {best:.3f}%", flush=True)
torch.save(model.state_dict(), a.out.replace(".pth", "_ep0.pth"))
model.train()
for ep in range(1, a.epochs + 1):
    tot = 0.0
    for x, y in tqdm(dl_tr, desc=f"epoch {ep}", mininterval=30):
        loss = crit(model(x.to(dev))[1], y.to(dev)); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    e = evaluate(); print(f"epoch {ep} loss {tot / len(dl_tr):.4f} dev EER {e:.3f}%", flush=True)
    torch.save(model.state_dict(), a.out.replace(".pth", f"_ep{ep}.pth"))
    torch.save(model.state_dict(), a.out)
print("done")
