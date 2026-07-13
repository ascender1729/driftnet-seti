#!/usr/bin/env python3
"""Reverse cross-band generalization: TRAIN on L-band (Enriquez), TEST out-of-distribution on X-band
(Voyager). The main paper trains X and tests L; a referee will ask whether the transfer is directional.
This runs the identical five-variant panel and metrics in the reverse direction, reusing the
driftnet_comprehensive machinery.

  reverse_ood.py
"""
import glob, json, os, subprocess, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from driftnet_v2 import load, CI
import baselines as B
import driftnet_comprehensive as DC     # reuse make/train/predict/det_metrics/mcnemar/bh/SEEDS

INJ = ["--snrs=3,6,9,12,15,20", "--drifts=-0.4,-0.2,0.2,0.4", "--no-png"]


def build():
    """Train sets = each Enriquez L cadence (pooled); OOD test = the Voyager X-band cadence."""
    Ldirs = sorted(d for d in glob.glob("real_data/rfi/HIP*") if os.path.isdir(d))
    if not Ldirs:
        raise RuntimeError("no L-band HIP cadences under real_data/rfi")
    trainsets = []
    for cad in Ldirs:
        name = os.path.basename(cad); out = f"data_L_{name}"
        if not os.path.exists(f"{out}/labels.json"):
            r = subprocess.run([sys.executable, "cp3_inject_real.py", *INJ, f"--cad-dir={cad}",
                                "--bg-per-cell=18", f"--out={out}"], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"L set {name} failed:", (r.stderr or "")[-200:].replace("\n", " "), flush=True); continue
        trainsets.append(out)
    if not os.path.exists("data_X_ood/labels.json"):
        subprocess.run([sys.executable, "cp3_inject_real.py", *INJ, "--cad-dir=real_data/cadence",
                        "--bg-per-cell=30", "--out=data_X_ood"], check=True)
    return trainsets, "data_X_ood"


def main():
    os.makedirs("results", exist_ok=True)
    trainsets, xood = build()
    Xs, ys, ss, ds = [], [], [], []
    for ts in trainsets:
        X, y, s, d = load(ts); Xs.append(X); ys.append(y); ss.append(s); ds.append(d)
    X0 = np.concatenate(Xs); y0 = np.concatenate(ys); s0 = np.concatenate(ss); d0 = np.concatenate(ds)
    Xx, yx, sx, _ = load(xood)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[reverse-ood L->X] train(L pooled)={len(X0)} from {len(trainsets)} cadences; ood(X)={len(Xx)} dev={dev}", flush=True)
    variants = ["full", "fixed_kernel", "rho_only", "raw_cnn"]
    testsets = ["in_dist", "ood_X"]
    acc = {v: {t: [] for t in testsets} for v in variants + ["classical"]}
    mets = {v: {t: [] for t in testsets} for v in variants + ["classical"]}
    pooled = {v: {t: ([], []) for t in testsets} for v in variants + ["classical"]}
    for seed in DC.SEEDS:
        rng = np.random.default_rng(seed); idx = rng.permutation(len(X0)); cut = int(len(X0) * 0.85)
        tri, tei = idx[:cut], idx[cut:]
        Xtr = torch.tensor(X0[tri]); ytr = torch.tensor(y0[tri]); drtr = torch.tensor(d0[tri]); tr = (Xtr, ytr, drtr)
        te = {"in_dist": (torch.tensor(X0[tei]), y0[tei], s0[tei]), "ood_X": (torch.tensor(Xx), yx, sx)}
        thr = B.fit_dedoppler(X0[tri], y0[tri])
        for v in variants:
            net = DC.train(v, seed, tr, dev)
            for t, (X, y, _) in te.items():
                p, sc = DC.predict(net, X, dev)
                acc[v][t].append(float((p == y).mean())); mets[v][t].append(DC.det_metrics(p, sc, y))
                pooled[v][t][0].append(p); pooled[v][t][1].append(y)
        for t, (X, y, _) in te.items():
            Xn = X.numpy(); feats = [B.dedoppler_features(x) for x in Xn]
            pc = np.array([B._dd_classify(*ff, thr) for ff in feats]); scc = np.array([ff[0] for ff in feats])
            acc["classical"][t].append(float((pc == y).mean())); mets["classical"][t].append(DC.det_metrics(pc, scc, y))
            pooled["classical"][t][0].append(pc); pooled["classical"][t][1].append(y)
        print(f"seed {seed}: full in_dist(L)={acc['full']['in_dist'][-1]:.3f} ood(X)={acc['full']['ood_X'][-1]:.3f}", flush=True)

    summary = {"seeds": DC.SEEDS, "direction": "L_to_X", "acc_mean_sd": {}, "det_mean_sd": {}, "mcnemar_bh": {}}
    for v in acc:
        summary["acc_mean_sd"][v] = {t: [round(float(np.mean(a)), 3), round(float(np.std(a)), 3)] for t, a in acc[v].items()}
        summary["det_mean_sd"][v] = {t: {"auc": round(float(np.nanmean([m[0] for m in mm])), 3),
                                         "ap": round(float(np.nanmean([m[1] for m in mm])), 3),
                                         "far": round(float(np.mean([m[2] for m in mm])), 3)} for t, mm in mets[v].items() if mm}
    for t in testsets:
        yf = np.concatenate(pooled["full"][t][1]); pv = {}
        for v in ["fixed_kernel", "rho_only", "raw_cnn", "classical"]:
            pv[f"full_vs_{v}"] = DC.mcnemar(np.concatenate(pooled["full"][t][0]), np.concatenate(pooled[v][t][0]), yf)
        summary["mcnemar_bh"][t] = DC.bh(pv)
    json.dump(summary, open("results/reverse_ood_results.json", "w"), indent=1)
    print("=== REVERSE-OOD (L->X) SUMMARY (mean acc) ===", flush=True)
    for v in acc:
        print(f"  {v:14s}", {t: summary['acc_mean_sd'][v][t][0] for t in testsets}, flush=True)
    print("REVERSE_OOD_DONE", flush=True)


if __name__ == "__main__":
    main()
