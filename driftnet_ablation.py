#!/usr/bin/env python3
"""Ablation + OOD test answering the two blocking review points:
(1) does LEARNABILITY earn its keep? -> full (learn+CNN) vs fixed-kernel(+CNN) vs rho-only(learn,noCNN).
(2) does it GENERALIZE? -> train on Voyager X-band OFF noise, test OUT-OF-DISTRIBUTION on Enriquez
    L-band OFF noise (different band, target, session). McNemar tests + per-SNR Wilson CIs.
Everything on real GBT background noise, on the instance.
"""
import json, math, os, subprocess, sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from driftnet_v2 import DriftNetV2, load, CI, CLS
import baselines as B

INJ = ["--snrs=3,6,9,12,15,20", "--drifts=-0.4,-0.2,0.2,0.4", "--no-png"]


def build():
    """Build train (Voyager X-band OFFs). Try each Enriquez L-band cadence for the OOD test until one
    succeeds; return True if an OOD set was built, else False (caller falls back to in-distribution)."""
    if not os.path.exists("data_train/labels.json"):
        subprocess.run([sys.executable, "cp3_inject_real.py", *INJ, "--cad-dir=real_data/cadence",
                        "--bg-per-cell=25", "--out=data_train"], check=True)
    if os.path.exists("data_ood/labels.json"):
        return True
    for cad in sorted(d for d in __import__("glob").glob("real_data/rfi/HIP*") if os.path.isdir(d)):
        r = subprocess.run([sys.executable, "cp3_inject_real.py", *INJ, f"--cad-dir={cad}",
                            "--bg-per-cell=10", "--out=data_ood"], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists("data_ood/labels.json"):
            print("OOD built from", os.path.basename(cad), flush=True); return True
        open(f"results/ood_err_{os.path.basename(cad)}.log", "w").write(
            f"rc={r.returncode}\n--STDOUT--\n{r.stdout or ''}\n--STDERR--\n{r.stderr or ''}")   # full error to S3
        print("OOD build failed for", os.path.basename(cad), "rc", r.returncode, ":",
              (r.stderr or "")[-400:].replace("\n", " "), flush=True)
    return False


def wilson(p, n):
    if n == 0: return 0.0
    z = 1.96; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(h, 3)


def completeness(pred, y, snr):
    out = {}
    for s in sorted(set(float(v) for v in snr[y == CI["signal"]])):
        m = (snr == s) & (y == CI["signal"]); n = int(m.sum())
        if n: out[s] = {"c": round(float((pred[m] == CI["signal"]).mean()), 3), "n": n,
                        "ci": wilson(float((pred[m] == CI["signal"]).mean()), n)}
    return out


def train_variant(name, learn, cnn, tr, te, dev):
    Xtr, ytr, drtr = tr; Xte, yte, snr = te
    torch.manual_seed(0)
    net = DriftNetV2(list(np.linspace(-3, 3, 17)), learn_drifts=learn, use_cnn=cnn).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
    ce = nn.CrossEntropyLoss(); mse = nn.MSELoss()
    for e in range(60):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            b = perm[i:i + 64]; opt.zero_grad()
            logit, dp = net(Xtr[b].to(dev)); iss = (ytr[b] == CI["signal"]).to(dev)
            loss = ce(logit, ytr[b].to(dev)) + (0.1 * mse(dp[iss], drtr[b].to(dev)[iss]) if iss.any() else 0)
            loss.backward(); opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(Xte[i:i + 128].to(dev))[0].argmax(1).cpu() for i in range(0, len(Xte), 128)]).numpy()
    drift_moved = float((net.drifts.detach().cpu() - torch.linspace(-3, 3, 17)).abs().mean())
    return pred, {"name": name, "overall": round(float((pred == yte).mean()), 3),
                  "overall_ci": wilson(float((pred == yte).mean()), len(yte)),
                  "rfi_recall": round(float((pred[yte == CI["rfi"]] == CI["rfi"]).mean()), 3),
                  "kernels_moved_meanabs": round(drift_moved, 4),
                  "completeness": completeness(pred, yte, snr)}


def mcnemar(pa, pb, y):
    b = int(((pa == y) & (pb != y)).sum()); c = int(((pa != y) & (pb == y)).sum())
    if b + c == 0: return {"b": b, "c": c, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2d
    return {"b": b, "c": c, "chi2": round(chi2, 2), "p": round(float(chi2d.sf(chi2, 1)), 4)}


def main():
    os.makedirs("results", exist_ok=True)
    ood_ok = build()
    Xtr, ytr, snr_tr, drtr = load("data_train")
    if ood_ok:
        Xte, yte, snr, _ = load("data_ood"); test_kind = "OOD (train Voyager X-band, test Enriquez L-band)"
    else:                                                            # in-distribution held-out split
        rng = np.random.default_rng(0); idx = rng.permutation(len(Xtr)); cut = int(len(Xtr) * 0.8)
        tri, tei = idx[:cut], idx[cut:]
        Xte, yte, snr = Xtr[tei], ytr[tei], snr_tr[tei]
        Xtr, ytr, drtr = Xtr[tri], ytr[tri], drtr[tri]
        test_kind = "in-distribution held-out (OOD build unavailable)"
    tr = (torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(drtr))
    te = (torch.tensor(Xte), yte, snr)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} test_kind={test_kind} train={len(Xtr)} test={len(yte)}", flush=True)
    preds, rows = {}, {}
    for name, learn, cnn in [("full_learn+cnn", True, True), ("fixed_kernel+cnn", False, True),
                             ("rho_only_learn", True, False)]:
        preds[name], rows[name] = train_variant(name, learn, cnn, tr, te, dev)
        r = rows[name]; print(f"[{name}] OOD overall={r['overall']}+-{r['overall_ci']} rfi={r['rfi_recall']} kernels_moved={r['kernels_moved_meanabs']}", flush=True)
    # classical de-Doppler baseline (train tau on data_train, eval OOD)
    thr = B.fit_dedoppler(Xtr, ytr)
    pc = np.array([B._dd_classify(*B.dedoppler_features(x), thr) for x in Xte]); preds["classical"] = pc
    rows["classical"] = {"name": "classical_dedoppler", "overall": round(float((pc == yte).mean()), 3),
                         "overall_ci": wilson(float((pc == yte).mean()), len(yte)),
                         "rfi_recall": round(float((pc[yte == CI["rfi"]] == CI["rfi"]).mean()), 3),
                         "completeness": completeness(pc, yte, snr)}
    print(f"[classical] OOD overall={rows['classical']['overall']} rfi={rows['classical']['rfi_recall']}", flush=True)
    tests = {"full_vs_fixed": mcnemar(preds["full_learn+cnn"], preds["fixed_kernel+cnn"], yte),
             "full_vs_classical": mcnemar(preds["full_learn+cnn"], preds["classical"], yte),
             "fixed_vs_classical": mcnemar(preds["fixed_kernel+cnn"], preds["classical"], yte),
             "full_vs_rhoonly": mcnemar(preds["full_learn+cnn"], preds["rho_only_learn"], yte)}
    json.dump({"test_kind": test_kind, "variants": rows, "mcnemar": tests}, open("results/ablation_results.json", "w"), indent=1)
    print("McNemar full_vs_fixed:", tests["full_vs_fixed"], flush=True)
    print("VERDICT: learnability earns its keep IF full > fixed by a significant McNemar; else the win is the classifier.", flush=True)
    print("ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
