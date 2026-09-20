#!/usr/bin/env python3
"""Comprehensive, publishable-grade study on real GBT noise, at scale.

Panel: full (learnable kernels + CNN), fixed-kernel + CNN, rho-only (no CNN), raw-pixel CNN
(Ma-2023 style, no de-Doppler), and the classical thresholded de-Doppler baseline.
Design: train on Voyager X-band OFF noise; evaluate IN-DISTRIBUTION (held-out) and OUT-OF-DISTRIBUTION
on EACH Enriquez L-band cadence separately (multi-band OOD). SEEDS repeated for mean+-SD. Reports
McNemar (pooled over seeds) with Benjamini-Hochberg correction, and completeness-vs-SNR with Wilson CIs.
"""
import glob, json, math, os, subprocess, sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from driftnet_v2 import DriftNetV2, load, CI, CLS
import baselines as B

SEEDS = [0, 1, 2, 3, 4]
INJ = ["--snrs=3,6,9,12,15,20", "--drifts=-0.4,-0.2,0.2,0.4", "--no-png"]


def build_sets():
    if not os.path.exists("data_train/labels.json"):
        subprocess.run([sys.executable, "cp3_inject_real.py", *INJ, "--cad-dir=real_data/cadence",
                        "--bg-per-cell=30", "--out=data_train"], check=True)
    ood = {}
    for cad in sorted(d for d in glob.glob("real_data/rfi/HIP*") if os.path.isdir(d)):
        name = os.path.basename(cad); out = f"data_ood_{name}"
        if not os.path.exists(f"{out}/labels.json"):
            r = subprocess.run([sys.executable, "cp3_inject_real.py", *INJ, f"--cad-dir={cad}",
                                "--bg-per-cell=12", f"--out={out}"], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"OOD {name} failed:", (r.stderr or "")[-200:].replace("\n", " "), flush=True); continue
        ood[name] = out
    return ood


class RawCNN(nn.Module):                                              # Ma-2023-style raw-pixel baseline
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool2d(4),
            nn.Flatten(), nn.Linear(64 * 16, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, P):
        P = P - P.median(dim=2, keepdim=True).values
        P = (P - P.mean((1, 2), keepdim=True)) / (P.std((1, 2), keepdim=True) + 1e-6)
        return self.net(P.unsqueeze(1)), torch.zeros(P.shape[0], device=P.device)


def make(variant):
    if variant == "raw_cnn": return RawCNN()
    learn = variant != "fixed_kernel"; cnn = variant != "rho_only"
    return DriftNetV2(list(np.linspace(-3, 3, 17)), learn_drifts=learn, use_cnn=cnn)


def wilson(p, n):
    if n == 0: return 0.0
    z = 1.96; d = 1 + z * z / n
    return round(z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d, 3)


def train(variant, seed, tr, dev):
    Xtr, ytr, drtr = tr; torch.manual_seed(seed)
    net = make(variant).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
    ce = nn.CrossEntropyLoss(); mse = nn.MSELoss()
    for e in range(60):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            b = perm[i:i + 64]; opt.zero_grad()
            logit, dp = net(Xtr[b].to(dev)); iss = (ytr[b] == CI["signal"]).to(dev)
            loss = ce(logit, ytr[b].to(dev)) + (0.1 * mse(dp[iss], drtr[b].to(dev)[iss]) if iss.any() and dp.requires_grad else 0)
            loss.backward(); opt.step()
        sched.step()
    return net


def predict(net, X, dev):
    """Return (argmax preds, signal-class softmax score) for ROC/PR/FAR."""
    net.eval(); ps, sc = [], []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            prob = torch.softmax(net(X[i:i + 128].to(dev))[0], 1).cpu()
            ps.append(prob.argmax(1).numpy()); sc.append(prob[:, CI["signal"]].numpy())
    return np.concatenate(ps), np.concatenate(sc)


def det_metrics(pred, score, y):
    """SETI-practitioner detection metrics: ROC-AUC and PR-AP for signal-vs-rest, and the false-alarm
    rate (fraction of non-signal predicted signal)."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    ybin = (y == CI["signal"]).astype(int)
    auc = float(roc_auc_score(ybin, score)) if len(set(ybin.tolist())) > 1 else float("nan")
    ap = float(average_precision_score(ybin, score)) if len(set(ybin.tolist())) > 1 else float("nan")
    nonsig = y != CI["signal"]
    far = float((pred[nonsig] == CI["signal"]).mean()) if nonsig.any() else 0.0
    return round(auc, 3), round(ap, 3), round(far, 3)


def mcnemar(pa, pb, y):
    b = int(((pa == y) & (pb != y)).sum()); c = int(((pa != y) & (pb == y)).sum())
    if b + c == 0: return 1.0
    from scipy.stats import chi2
    return float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))


def bh(pvals):                                                       # Benjamini-Hochberg
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out = {}
    for i, (k, p) in enumerate(items, 1):
        out[k] = round(min(p * m / i, 1.0), 4)
    return out


def main():
    os.makedirs("results", exist_ok=True)
    ood = build_sets()
    Xtr0, ytr0, snr0, dr0 = load("data_train")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    testsets = {"in_dist": None}                                     # in_dist = held-out split of train
    for name, path in ood.items():
        testsets[f"ood_{name}"] = load(path)
    print(f"device={dev} train={len(Xtr0)} testsets={list(testsets)}", flush=True)
    variants = ["full", "fixed_kernel", "rho_only", "raw_cnn"]
    # accumulate per (variant, testset): list of overall acc over seeds; and pooled preds for McNemar
    acc = {v: {ts: [] for ts in testsets} for v in variants + ["classical"]}
    pooled = {v: {ts: ([], []) for ts in testsets} for v in variants + ["classical"]}  # (preds, ys)
    mets = {v: {ts: [] for ts in testsets} for v in variants + ["classical"]}  # per-seed (auc,ap,far)
    learned = {}                                                     # seed -> trained drift grid of "full"
    init_grid = [float(d) for d in np.linspace(-3, 3, 17)]           # same as make()
    for seed in SEEDS:
        rng = np.random.default_rng(seed); idx = rng.permutation(len(Xtr0)); cut = int(len(Xtr0) * 0.85)
        tri, tei = idx[:cut], idx[cut:]
        Xtr = torch.tensor(Xtr0[tri]); ytr = torch.tensor(ytr0[tri]); drtr = torch.tensor(dr0[tri])
        tr = (Xtr, ytr, drtr)
        te_local = {"in_dist": (torch.tensor(Xtr0[tei]), ytr0[tei], snr0[tei])}
        for ts in ood: te_local[f"ood_{os.path.basename(ts)}" if False else f"ood_{ts}"] = None
        for name, path in ood.items():
            Xo, yo, so, _ = testsets[f"ood_{name}"]; te_local[f"ood_{name}"] = (torch.tensor(Xo), yo, so)
        thr = B.fit_dedoppler(Xtr0[tri], ytr0[tri])
        for v in variants:
            net = train(v, seed, tr, dev)
            if v == "full":
                fin = net.drifts.detach().cpu().tolist()
                learned[seed] = {"final": fin, "init": init_grid,
                                 "mean_abs_move": float(np.mean(np.abs(np.array(fin) - np.array(init_grid)))),
                                 "min": float(min(fin)), "max": float(max(fin))}
            for ts, dat in te_local.items():
                X, y, _ = dat; p, sc = predict(net, X, dev)
                acc[v][ts].append(float((p == y).mean()))
                mets[v][ts].append(det_metrics(p, sc, y))
                pooled[v][ts][0].append(p); pooled[v][ts][1].append(y)
        for ts, dat in te_local.items():                            # classical (deterministic; still per-seed threshold)
            X, y, _ = dat; Xn = X.numpy()
            feats = [B.dedoppler_features(x) for x in Xn]
            pc = np.array([B._dd_classify(*ff, thr) for ff in feats]); scc = np.array([ff[0] for ff in feats])
            acc["classical"][ts].append(float((pc == y).mean()))
            mets["classical"][ts].append(det_metrics(pc, scc, y))
            pooled["classical"][ts][0].append(pc); pooled["classical"][ts][1].append(y)
        print(f"seed {seed} done: full in_dist={acc['full']['in_dist'][-1]:.3f} "
              f"ood_mean={np.mean([acc['full'][t][-1] for t in testsets if t.startswith('ood')]):.3f}", flush=True)

    summary = {"seeds": SEEDS, "acc_mean_sd": {}, "det_mean_sd": {}, "mcnemar_bh": {},
               "learned_drifts": learned,
               "classical_grid": {"max_drift": B.DD_MAX_DRIFT, "n": B.DD_N_DRIFTS}}
    for v in acc:
        summary["acc_mean_sd"][v] = {ts: [round(float(np.mean(a)), 3), round(float(np.std(a)), 3)]
                                     for ts, a in acc[v].items()}
        summary["det_mean_sd"][v] = {ts: {"auc": round(float(np.nanmean([m[0] for m in mm])), 3),
                                          "ap": round(float(np.nanmean([m[1] for m in mm])), 3),
                                          "far": round(float(np.mean([m[2] for m in mm])), 3)}
                                     for ts, mm in mets[v].items() if mm}
    # McNemar (pooled preds across seeds) vs classical and full-vs-fixed, per testset, BH-corrected
    for ts in testsets:
        pv = {}
        yf = np.concatenate(pooled["full"][ts][1])
        for v in ["fixed_kernel", "rho_only", "raw_cnn", "classical"]:
            pv[f"full_vs_{v}"] = mcnemar(np.concatenate(pooled["full"][ts][0]),
                                         np.concatenate(pooled[v][ts][0]), yf)
        summary["mcnemar_bh"][ts] = bh(pv)
    json.dump(summary, open("results/comprehensive_results.json", "w"), indent=1)
    print("=== SUMMARY (mean acc) ===", flush=True)
    for v in acc:
        print(f"  {v:14s}", {ts: summary['acc_mean_sd'][v][ts][0] for ts in testsets}, flush=True)
    print("McNemar BH p (in_dist):", summary["mcnemar_bh"]["in_dist"], flush=True)
    print("COMPREHENSIVE_DONE", flush=True)


if __name__ == "__main__":
    main()
