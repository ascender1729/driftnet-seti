#!/usr/bin/env python3
"""Two baselines for the 3-class technosignature benchmark, scored on the held-out test set.

A) dedoppler  - a turboSETI-style Doppler-drift matched filter. For each integer drift rate it
   sums intensity along the drifted line (best starting channel), normalised by the noise floor.
   Classify: weak everywhere -> noise; strongest line drifts -> signal; strongest line is vertical
   (zero-drift) -> rfi. Two thresholds are fit on the TRAIN set (grid search for best accuracy).
B) cnn        - a small supervised CNN on the [T,F] arrays. Trains on TRAIN, scored on TEST.

Outputs baseline_dedoppler.json and baseline_cnn.json: {"acc","per":{id:{true,pred,snr_db}}}.

Usage: baselines.py            (fit+score both, write jsons)
       baselines.py --self-check
"""
import json, glob, os, sys
import numpy as np

CLS = ("signal", "rfi", "noise")
CI = {c: i for i, c in enumerate(CLS)}


def load(split):
    lab = json.load(open(f"{split}/labels.json"))
    X, y, snr, ids = [], [], [], []
    for _id, m in lab.items():
        X.append(np.load(f"{split}/npy/{_id}.npy").astype(np.float32))
        y.append(CI[m["cls"]]); snr.append(m["snr_db"]); ids.append(_id)
    return np.stack(X), np.array(y), np.array(snr), ids


# ---------- A) dedoppler matched filter ----------
def dedoppler_features(img, max_drift=1):
    """Return (drift_score, vertical_score): best mean-subtracted line SNR over drifting vs zero-drift.
    max_drift is channels-per-row; the injected signals drift 0.15..0.6 ch/step so +-1 covers them."""
    T, F = img.shape
    # subtract per-TIME-ROW mean across frequency (flattens broadband) - keeps narrowband lines,
    # both drifting AND vertical. (Subtracting the per-CHANNEL time-mean would erase vertical RFI.)
    m = img - img.mean(axis=1, keepdims=True)
    noise = m.std() + 1e-9
    drifts = np.linspace(-max_drift, max_drift, 9)
    best_drift, best_vert = 0.0, 0.0
    for d in drifts:
        shifts = np.round(d * np.arange(T)).astype(int)
        acc = np.zeros(F)
        for t in range(T):
            acc += np.roll(m[t], -shifts[t])
        score = acc.max() / (noise * np.sqrt(T))   # integrated SNR of the best line at this drift
        if abs(d) < 1e-6:
            best_vert = max(best_vert, score)
        else:
            best_drift = max(best_drift, score)
    return best_drift, best_vert


def fit_dedoppler(Xtr, ytr):
    feats = np.array([dedoppler_features(x) for x in Xtr])   # [n,2] = (drift, vertical)
    # grid-search a detection threshold t (line present?) and margin for drift-vs-vertical
    best = (-1, 1.0)
    for t in np.linspace(1.0, 6.0, 26):
        pred = np.array([_dd_classify(dr, ve, t) for dr, ve in feats])
        acc = (pred == ytr).mean()
        if acc > best[0]:
            best = (acc, t)
    return best[1]


def _dd_classify(drift, vert, thr):
    if max(drift, vert) < thr:
        return CI["noise"]
    return CI["signal"] if drift > vert else CI["rfi"]


def run_dedoppler(Xtr, ytr, Xte, yte, snr_te, ids_te):
    thr = fit_dedoppler(Xtr, ytr)
    per, correct = {}, 0
    for x, yt, s, _id in zip(Xte, yte, snr_te, ids_te):
        dr, ve = dedoppler_features(x)
        p = _dd_classify(dr, ve, thr)
        per[_id] = {"true": CLS[yt], "pred": CLS[p], "snr_db": float(s)}
        correct += int(p == yt)
    return {"method": "dedoppler", "threshold": float(thr), "acc": correct / len(yte), "per": per}


# ---------- B) small CNN ----------
def run_cnn(Xtr, ytr, Xte, yte, snr_te, ids_te, epochs=25):
    import torch, torch.nn as nn
    torch.manual_seed(0)
    def norm(X):
        X = (X - X.mean(axis=(1, 2), keepdims=True)) / (X.std(axis=(1, 2), keepdims=True) + 1e-6)
        return torch.tensor(X[:, None], dtype=torch.float32)
    Xtr_t, Xte_t = norm(Xtr), norm(Xte)
    ytr_t = torch.tensor(ytr)
    # MAX-pool throughout: the class signal is a thin bright streak (sparse). Average-pooling
    # washes it into the noise floor; max-pooling preserves the line and its orientation.
    net = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(4),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool2d(4),
        nn.Flatten(), nn.Linear(16 * 16, 32), nn.ReLU(), nn.Linear(32, 3))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr_t))          # SHUFFLE - train set is class-ordered; unshuffled
        Xs, ys = Xtr_t[perm], ytr_t[perm]           # single-class batches collapse the net to one class
        for i in range(0, len(Xs), 32):
            opt.zero_grad()
            out = net(Xs[i:i + 32])
            loss = lossf(out, ys[i:i + 32]); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(Xte_t).argmax(1).numpy()
    per = {i: {"true": CLS[t], "pred": CLS[p], "snr_db": float(s)}
           for i, t, p, s in zip(ids_te, yte, pred, snr_te)}
    return {"method": "cnn", "acc": float((pred == yte).mean()), "per": per}


def main():
    Xtr, ytr, _, _ = load("data")
    Xte, yte, snr_te, ids_te = load("data_test")
    dd = run_dedoppler(Xtr, ytr, Xte, yte, snr_te, ids_te)
    json.dump(dd, open("baseline_dedoppler.json", "w"), indent=1)
    print(f"[dedoppler] acc={dd['acc']:.3f} thr={dd['threshold']:.2f}")
    cnn = run_cnn(Xtr, ytr, Xte, yte, snr_te, ids_te)
    json.dump(cnn, open("baseline_cnn.json", "w"), indent=1)
    print(f"[cnn]       acc={cnn['acc']:.3f}")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        # tiny sanity: a clean drift, a vertical, and noise must be separable by the features
        rng = np.random.default_rng(0)
        def mk(kind):
            a = rng.chisquare(2, (128, 256)).astype(np.float32) * 5
            if kind == "s":
                for t in range(128):
                    f = 60 + int(0.4 * t)
                    if f < 256: a[t, f] += 40
            elif kind == "v":
                a[:, 120] += 40
            return a
        ds, vs = dedoppler_features(mk("s")); print("signal feats drift/vert:", round(ds, 1), round(vs, 1))
        dv, vv = dedoppler_features(mk("v")); print("rfi    feats drift/vert:", round(dv, 1), round(vv, 1))
        dn, vn = dedoppler_features(mk("n")); print("noise  feats drift/vert:", round(dn, 1), round(vn, 1))
        assert ds > vs and vv > dv and max(dn, vn) < ds, "dedoppler features not separating"
        print("SELF-CHECK OK")
    else:
        main()
