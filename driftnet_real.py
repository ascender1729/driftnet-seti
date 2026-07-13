#!/usr/bin/env python3
"""DriftNet on REAL GBT data (everything real, on the instance).

Train: setigen 3-class injections (signal drifting / rfi zero-drift / noise) into REAL Green Bank
off-target backgrounds (cp3_inject_real). Evaluate: held-out real-background injections
(completeness vs SNR) AND the real 3-class set (Voyager signal + Enriquez RFI + real noise,
cp3_real3class). Baseline: the classical de-Doppler classifier on the same real data.
"""
import json, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from driftnet_v2 import DriftNetV2, load, CI, CLS
import baselines as B

INJ = ["--snrs=3,6,9,12,15,20", "--drifts=-0.4,-0.2,0.2,0.4"]     # small Hz/s: line stays in the 16-sample window


def build():
    if not os.path.exists("data_realbg/labels.json"):
        subprocess.run([sys.executable, "cp3_inject_real.py", *INJ, "--no-png", "--bg-per-cell=25", "--out=data_realbg"], check=True)
    if not os.path.exists("data_realbg_test/labels.json"):
        subprocess.run([sys.executable, "cp3_inject_real.py", *INJ, "--no-png", "--bg-per-cell=8", "--out=data_realbg_test"], check=True)
    try:
        subprocess.run([sys.executable, "cp3_real3class.py"], check=True)
    except Exception as e:
        print("real3class build failed:", e, flush=True)


def completeness(pred, y, snr):
    out = {}
    for s in sorted(set(float(v) for v in snr[y == CI["signal"]])):
        m = (snr == s) & (y == CI["signal"])
        out[s] = float((pred[m] == CI["signal"]).mean()) if m.any() else None
    return out


def eval_net(net, split, dev):
    X, y, snr, _ = load(split); X = torch.tensor(X); net.eval()
    with torch.no_grad():
        pred = torch.cat([net(X[i:i + 128].to(dev))[0].argmax(1).cpu() for i in range(0, len(X), 128)]).numpy()
    rfi = float((pred[y == CI["rfi"]] == CI["rfi"]).mean()) if (y == CI["rfi"]).any() else None
    return {"overall": float((pred == y).mean()), "rfi_recall": rfi, "n": len(y),
            "completeness_vs_snr": completeness(pred, y, snr)}


def eval_dedoppler(train_split, test_split):
    Xtr, ytr, _, _ = load(train_split); thr = B.fit_dedoppler(Xtr, ytr)
    X, y, snr, _ = load(test_split)
    pred = np.array([B._dd_classify(*B.dedoppler_features(x), thr) for x in X])
    rfi = float((pred[y == CI["rfi"]] == CI["rfi"]).mean()) if (y == CI["rfi"]).any() else None
    return {"overall": float((pred == y).mean()), "rfi_recall": rfi, "n": len(y),
            "completeness_vs_snr": completeness(pred, y, snr), "threshold": float(thr)}


def main():
    os.makedirs("results", exist_ok=True)
    build()
    Xtr, ytr, _, drtr = load("data_realbg")
    Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr); drtr = torch.tensor(drtr)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} real-bg train n={len(Xtr)} shape={tuple(Xtr.shape[1:])}", flush=True)
    net = DriftNetV2(list(np.linspace(-3, 3, 17))).to(dev)         # ch/step grid for real 16-sample drift
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 120)
    ce = nn.CrossEntropyLoss(); mse = nn.MSELoss()
    for e in range(120):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            b = perm[i:i + 64]; opt.zero_grad()
            logit, dp = net(Xtr[b].to(dev)); iss = (ytr[b] == CI["signal"]).to(dev)
            loss = ce(logit, ytr[b].to(dev))
            if iss.any():
                loss = loss + 0.1 * mse(dp[iss], drtr[b].to(dev)[iss])
            loss.backward(); opt.step()
        sched.step()
    torch.save(net.state_dict(), "results/driftnet_real.pt")
    # THE KEY RESULT: DriftNet vs classical de-Doppler in real GBT noise. Save + print FIRST.
    res = {"driftnet_realbg": eval_net(net, "data_realbg_test", dev),
           "dedoppler_realbg": eval_dedoppler("data_realbg", "data_realbg_test")}
    json.dump(res, open("results/driftnet_real_results.json", "w"), indent=1)
    d, c = res["driftnet_realbg"], res["dedoppler_realbg"]
    print(f"REAL-BG: DriftNet overall={d['overall']:.3f} rfi={d['rfi_recall']} | dedoppler overall={c['overall']:.3f} rfi={c['rfi_recall']}", flush=True)
    print("DriftNet completeness vs SNR (real bg):", {k: round(v, 2) for k, v in d["completeness_vs_snr"].items() if v is not None}, flush=True)
    print("dedoppler completeness vs SNR (real bg):", {k: round(v, 2) for k, v in c["completeness_vs_snr"].items() if v is not None}, flush=True)
    try:                                                              # optional, incomplete real-3class eval
        if os.path.isdir("real_data/real3class/npy"):
            res["driftnet_real3class"] = eval_net(net, "real_data/real3class", dev)
            json.dump(res, open("results/driftnet_real_results.json", "w"), indent=1)
    except Exception as e:
        print("real3class eval skipped:", type(e).__name__, str(e)[:80], flush=True)
    print("DRIFTNET_REAL_DONE", flush=True)


if __name__ == "__main__":
    main()
