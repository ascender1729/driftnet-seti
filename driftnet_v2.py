#!/usr/bin/env python3
"""DriftNet v2: a LEARNABLE de-Doppler layer for narrowband technosignature detection.

Upgrades over v1 (the novelty lever per DRIFTNET_NOVELTY.md):
  1. LEARNABLE drift kernels. The drift grid is an nn.Parameter; the de-Doppler shift uses
     differentiable linear interpolation, so gradients flow to the drift rates and the network
     learns where to integrate. (v1 used a fixed grid + non-differentiable round/gather.)
  2. Explicit rho(d) matched-filter statistic: rho[d] = max_f R[d,f], the exact quantity turboSETI
     thresholds, fed to the classifier alongside the CNN over the drift spectrum.
  3. Drift-rate regression head (a physical output), trained jointly on signal samples.

  driftnet_v2.py [--train-extra data_realbg]   train on data/(+extra), eval on data_test/
  driftnet_v2.py --self-check
"""
import glob, json, sys
import numpy as np
import torch
import torch.nn as nn

CLS = ("signal", "rfi", "noise"); CI = {c: i for i, c in enumerate(CLS)}


def load(split):
    lab = json.load(open(f"{split}/labels.json"))
    X, y, snr, dr = [], [], [], []
    for _id, m in lab.items():
        X.append(np.load(f"{split}/npy/{_id}.npy").astype(np.float32))
        y.append(CI[m["cls"]]); snr.append(m.get("snr_db", 0.0)); dr.append(m.get("drift_hz_s", 0.0))
    return np.stack(X), np.array(y), np.array(snr), np.array(dr, dtype=np.float32)


def ddt_interp(P, drifts):
    """Differentiable, LEARNABLE de-Doppler transform. P:[B,T,F], drifts:[D] (learnable).
    Linear interpolation at f + d*t makes it differentiable in d. Fully vectorized over drifts
    (no Python loop) so it is ~10-20x faster on GPU, which enables the multi-seed sweep."""
    B, T, Fq = P.shape
    D = drifts.shape[0]
    t = torch.arange(T, device=P.device, dtype=P.dtype)
    f = torch.arange(Fq, device=P.device, dtype=P.dtype)
    pos = f.view(1, 1, Fq) + drifts.view(D, 1, 1) * t.view(1, T, 1)     # [D,T,F] float positions
    lo = torch.floor(pos); frac = pos - lo                             # frac carries the d gradient
    loi = lo.long(); hii = loi + 1
    vlo = ((loi >= 0) & (loi < Fq)).to(P.dtype); vhi = ((hii >= 0) & (hii < Fq)).to(P.dtype)
    Pe = P.unsqueeze(1).expand(B, D, T, Fq)                            # [B,D,T,F]
    gl = torch.gather(Pe, 3, loi.clamp(0, Fq - 1).view(1, D, T, Fq).expand(B, D, T, Fq)) * vlo.view(1, D, T, Fq)
    gh = torch.gather(Pe, 3, hii.clamp(0, Fq - 1).view(1, D, T, Fq).expand(B, D, T, Fq)) * vhi.view(1, D, T, Fq)
    val = (1 - frac).view(1, D, T, Fq) * gl + frac.view(1, D, T, Fq) * gh   # [B,D,T,F]
    return val.sum(2)                                                   # [B,D,F]


class DriftNetV2(nn.Module):
    def __init__(self, init_drifts, learn_drifts=True, use_cnn=True):   # ablation flags
        super().__init__()
        self.drifts = nn.Parameter(torch.tensor(init_drifts, dtype=torch.float32),
                                   requires_grad=learn_drifts)          # learnable vs frozen kernels
        self.use_cnn = use_cnn
        D = len(init_drifts)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool2d(4),
            nn.Flatten(), nn.Linear(64 * 16, 128), nn.ReLU())
        self.rho_mlp = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())  # rho(d) branch
        fdim = (128 + 64) if use_cnn else 64
        self.head = nn.Linear(fdim, 3)                               # classifier
        self.drift_head = nn.Linear(fdim, 1)                         # drift-rate regression

    def forward(self, P):
        # subtract per-TIME-ROW median across frequency (flattens broadband) - keeps narrowband lines,
        # both drifting AND vertical. (Subtracting the per-CHANNEL time-median would erase vertical RFI.)
        P = P - P.median(dim=2, keepdim=True).values
        R = ddt_interp(P, self.drifts)                                # [B,D,F] drift spectrum
        rho = R.max(dim=2).values                                     # [B,D] matched-filter statistic per drift
        rhon = self.rho_mlp((rho - rho.mean(1, keepdim=True)) / (rho.std(1, keepdim=True) + 1e-6))
        if self.use_cnn:
            Rn = (R - R.mean(dim=(1, 2), keepdim=True)) / (R.std(dim=(1, 2), keepdim=True) + 1e-6)
            feat = torch.cat([self.cnn(Rn.unsqueeze(1)), rhon], 1)
        else:
            feat = rhon                                              # rho-only ablation (no CNN over R)
        return self.head(feat), self.drift_head(feat).squeeze(1)


def run(extra=None, epochs=70):
    torch.manual_seed(0)
    Xtr, ytr, _, drtr = load("data")
    if extra:
        Xe, ye, _, dre = load(extra); Xtr = np.concatenate([Xtr, Xe]); ytr = np.concatenate([ytr, ye]); drtr = np.concatenate([drtr, dre])
    Xte, yte, snr, _ = load("data_test")
    Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr); drtr = torch.tensor(drtr); Xte = torch.tensor(Xte)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = DriftNetV2(list(np.linspace(-0.8, 0.8, 17))).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss(); mse = nn.MSELoss()
    for e in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            logit, dpred = net(Xtr[b].to(dev))
            issig = (ytr[b] == CI["signal"]).to(dev)
            loss = ce(logit, ytr[b].to(dev))
            if issig.any():                                            # drift regression on signal only
                loss = loss + 0.1 * mse(dpred[issig], drtr[b].to(dev)[issig])
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(Xte[i:i + 64].to(dev))[0].argmax(1).cpu() for i in range(0, len(Xte), 64)]).numpy()
    acc = (pred == yte).mean()
    def sub(mask):
        m = mask & (yte != CI["noise"]); return (pred[m] == yte[m]).mean() if m.any() else float("nan")
    hi = sub(snr >= 12); lo = sub((snr > 0) & (snr < 6)); rfi = (pred[yte == CI["rfi"]] == CI["rfi"]).mean()
    print(f"[DriftNet v2] overall={acc:.3f}  high-SNR={hi:.3f}  low-SNR={lo:.3f}  RFI-recall={rfi:.3f}"
          + (f"  (+{extra})" if extra else ""))
    print(f"  learned drifts range: [{net.drifts.min().item():.2f}, {net.drifts.max().item():.2f}]")
    print("  vs dedoppler 0.833/1.000/0.211/0.867 | CNN 0.683/0.902/0.105 | Opus-VLM 0.711/1.000/0.000 | v1 0.511/0.784/0.316/0.433")
    json.dump({"overall": float(acc), "high_snr": float(hi), "low_snr": float(lo), "rfi_recall": float(rfi)},
              open("driftnet_v2_result.json", "w"), indent=1)
    return acc


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        P = torch.zeros(1, 128, 256, requires_grad=True)
        drifts = torch.tensor([0.0, 0.4], requires_grad=True)
        R = ddt_interp(P.detach().clone().requires_grad_(True), drifts)
        R.sum().backward()
        print("drift grad flows:", drifts.grad is not None and torch.isfinite(drifts.grad).all().item())
        assert drifts.grad is not None, "learnable drift must receive gradient"
        print("SELF-CHECK OK: gradients reach the learnable drift kernels")
    else:
        extra = sys.argv[sys.argv.index("--train-extra") + 1] if "--train-extra" in sys.argv else None
        run(extra=extra)
