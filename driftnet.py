#!/usr/bin/env python3
"""DriftNet: a differentiable de-Doppler transform layer + CNN backbone for narrowband
technosignature detection. The DDT layer turns turboSETI's incoherent de-Doppler search into a
trainable operation: it integrates power along a bank of Doppler-drift lines, concentrating a faint
drifting signal into a bright peak (the sqrt(N_t) matched-filter gain the VLMs lack) while keeping
gradients flowing so the whole detector trains end to end.

  driftnet.py                 (train on data/, eval on data_test/, compare vs baselines/VLMs)
  driftnet.py --self-check
"""
import glob, json, sys
import numpy as np
import torch
import torch.nn as nn

CLS = ("signal", "rfi", "noise"); CI = {c: i for i, c in enumerate(CLS)}


def load(split):
    lab = json.load(open(f"{split}/labels.json"))
    X, y, snr = [], [], []
    for _id, m in lab.items():
        X.append(np.load(f"{split}/npy/{_id}.npy").astype(np.float32))
        y.append(CI[m["cls"]]); snr.append(m["snr_db"])
    return np.stack(X), np.array(y), np.array(snr)


def ddt(P, drifts):
    """Differentiable de-Doppler transform. P:[B,T,F] -> R:[B,D,F]. For each drift d (channels per
    time step) it rolls row t by round(d*t) and sums over time; gather makes it differentiable in P."""
    B, T, F = P.shape
    t = torch.arange(T, device=P.device)
    outs = []
    for d in drifts:
        sh = torch.round(d * t).long()                                  # per-row shift
        raw = torch.arange(F, device=P.device).view(1, F) + sh.view(T, 1)   # [T,F], may be out of range
        valid = ((raw >= 0) & (raw < F)).float().view(1, T, F)          # mask instead of wrap-around
        idx = raw.clamp(0, F - 1).view(1, T, F).expand(B, T, F)
        outs.append((torch.gather(P, 2, idx) * valid).sum(1))          # [B,F] integrated line, edges masked
    return torch.stack(outs, 1)                                         # [B,D,F] drift spectrum


class DriftNet(nn.Module):
    def __init__(self, drifts):
        super().__init__()
        self.register_buffer("drifts", torch.tensor(drifts, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool2d(4),
            nn.Flatten(), nn.Linear(32 * 16, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, P):                                               # P:[B,T,F]
        P = P - P.median(dim=1, keepdim=True).values                    # per-channel bandpass removal
        R = ddt(P, self.drifts.tolist())                               # [B,D,F] matched-filter response
        R = (R - R.mean(dim=(1, 2), keepdim=True)) / (R.std(dim=(1, 2), keepdim=True) + 1e-6)
        return self.net(R.unsqueeze(1))


def run(epochs=45):
    torch.manual_seed(0)
    Xtr, ytr, _ = load("data"); Xte, yte, snr = load("data_test")
    Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr); Xte = torch.tensor(Xte)
    drifts = list(np.linspace(-0.8, 0.8, 17))                          # dense grid incl 0 (RFI ridge)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = DriftNet(drifts).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss()
    for e in range(epochs):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            b = perm[i:i + 32]
            opt.zero_grad(); out = net(Xtr[b].to(dev)); loss = lossf(out, ytr[b].to(dev))
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(Xte[i:i + 64].to(dev)).argmax(1).cpu() for i in range(0, len(Xte), 64)]).numpy()
    acc = (pred == yte).mean()
    def sub(mask):
        m = mask & (yte != CI["noise"]); return (pred[m] == yte[m]).mean() if m.any() else float("nan")
    hi = sub(snr >= 12); lo = sub((snr > 0) & (snr < 6))
    rfi = (pred[yte == CI["rfi"]] == CI["rfi"]).mean()
    print(f"[DriftNet] overall={acc:.3f}  high-SNR={hi:.3f}  low-SNR={lo:.3f}  RFI-recall={rfi:.3f}")
    print("  compare: dedoppler 0.833/1.000/0.211/0.867 | CNN 0.683/0.902/0.105 | Opus-VLM 0.711/1.000/0.000")
    json.dump({"overall": float(acc), "high_snr": float(hi), "low_snr": float(lo), "rfi_recall": float(rfi)},
              open("driftnet_result.json", "w"), indent=1)
    return acc


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        # DDT must concentrate a clean drift into a sharper peak than the raw sum at d=0
        P = torch.zeros(1, 128, 256)
        for t in range(128):
            f = 60 + int(0.4 * t); P[0, t, f] = 10.0                    # a drift-0.4 line
        R = ddt(P, [0.0, 0.4])
        peak0, peak4 = R[0, 0].max().item(), R[0, 1].max().item()
        print(f"peak at d=0: {peak0:.1f}, peak at d=0.4 (matched): {peak4:.1f}")
        assert peak4 > peak0 * 3, "matched drift must concentrate the line"
        print("SELF-CHECK OK: DDT concentrates the matched drift")
    else:
        run()
