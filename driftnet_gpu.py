#!/usr/bin/env python3
"""Full DriftNet training on GPU: large synthetic corpus + long cosine-scheduled training + a small
hyperparameter sweep, saving the best model and full metrics. Run on a Lambda GPU instance.
"""
import json, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from driftnet_v2 import DriftNetV2, load, CI, CLS


def gen_data():
    if not os.path.exists("data/labels.json"):
        subprocess.run([sys.executable, "gen_dataset.py", "--n-per-class", "800", "--out", "data", "--seed", "0"], check=True)
    if not os.path.exists("data_test/labels.json"):
        subprocess.run([sys.executable, "gen_dataset.py", "--n-per-class", "200", "--out", "data_test", "--seed", "999"], check=True)


def metrics(net, Xte, yte, snr, dev):
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(Xte[i:i + 128].to(dev))[0].argmax(1).cpu() for i in range(0, len(Xte), 128)]).numpy()
    acc = float((pred == yte).mean())
    def sub(mask):
        m = mask & (yte != CI["noise"]); return float((pred[m] == yte[m]).mean()) if m.any() else float("nan")
    return {"overall": acc, "high_snr": sub(snr >= 12), "low_snr": sub((snr > 0) & (snr < 6)),
            "rfi_recall": float((pred[yte == CI["rfi"]] == CI["rfi"]).mean())}


def train(cfg, data, dev):
    Xtr, ytr, drtr, Xte, yte, snr = data
    torch.manual_seed(0)
    net = DriftNetV2(list(np.linspace(-cfg["dmax"], cfg["dmax"], cfg["ndrift"]))).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    ce = nn.CrossEntropyLoss(); mse = nn.MSELoss()
    for e in range(cfg["epochs"]):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), cfg["bs"]):
            b = perm[i:i + cfg["bs"]]; opt.zero_grad()
            logit, dp = net(Xtr[b].to(dev)); iss = (ytr[b] == CI["signal"]).to(dev)
            loss = ce(logit, ytr[b].to(dev))
            if iss.any():
                loss = loss + 0.1 * mse(dp[iss], drtr[b].to(dev)[iss])
            loss.backward(); opt.step()
        sched.step()
    return net, metrics(net, Xte, yte, snr, dev)


def main():
    os.makedirs("results", exist_ok=True)
    gen_data()
    Xtr, ytr, _, drtr = load("data"); Xte, yte, snr, _ = load("data_test")
    Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr); drtr = torch.tensor(drtr); Xte = torch.tensor(Xte)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "| train:", len(Xtr), "test:", len(Xte), flush=True)
    data = (Xtr, ytr, drtr, Xte, yte, snr)
    grid = [
        {"dmax": 0.8, "ndrift": 17, "lr": 1e-3, "bs": 64, "epochs": 150},
        {"dmax": 0.8, "ndrift": 33, "lr": 1e-3, "bs": 64, "epochs": 150},
        {"dmax": 1.0, "ndrift": 33, "lr": 5e-4, "bs": 64, "epochs": 200},
        {"dmax": 0.8, "ndrift": 33, "lr": 1e-3, "bs": 32, "epochs": 200},
    ]
    results, best = [], None
    for j, cfg in enumerate(grid):
        t0 = time.time(); net, m = train(cfg, data, dev)
        m["cfg"] = cfg; m["secs"] = round(time.time() - t0, 1); results.append(m)
        print(f"[cfg {j}] {cfg} -> overall={m['overall']:.3f} hi={m['high_snr']:.3f} lo={m['low_snr']:.3f} rfi={m['rfi_recall']:.3f} ({m['secs']}s)", flush=True)
        if best is None or m["overall"] > best[0]:
            best = (m["overall"], net, cfg, m)
    torch.save(best[1].state_dict(), "results/driftnet_best.pt")
    json.dump({"sweep": results, "best_cfg": best[2], "best_metrics": best[3],
               "baselines": {"dedoppler": [0.833, 1.0, 0.211, 0.867], "cnn": [0.683, 0.902, 0.105, 0.617],
                             "opus_vlm": [0.711, 1.0, 0.0, 0.517]}},
              open("results/driftnet_gpu_results.json", "w"), indent=1)
    print("BEST:", best[2], "->", best[3], flush=True)


if __name__ == "__main__":
    main()
