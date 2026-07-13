#!/usr/bin/env python3
"""CP3: completeness-vs-SNR/drift benchmark by injecting into REAL backgrounds.

The adversarial gate's fix Y: do NOT test on pure-synthetic noise. Take REAL off-target cadence
background (real GBT noise + bandpass), inject a controlled drifting narrowband signal at a grid of
(SNR, drift), and measure each method's completeness across the plane. Bridges sim-to-real: the
signal is controlled (so we know ground truth and SNR) but the background is real telescope data.

Output: grid/npy + grid/png + grid/labels.json  (cls signal/noise, snr_db, drift, realization).
Then analyze_grid.py (CP3b) runs turboSETI/CNN/VLMs on it for completeness curves.

Usage: cp3_grid.py --snrs 3,6,9,12,15,18 --drifts 0.15,0.3,0.5 --reps 20 --out grid
       cp3_grid.py --self-check
"""
import argparse, glob, json, os, re
import numpy as np

CAD = "real_data/cadence"
NT = 16


def real_backgrounds(n, half_hz=400, seed=0):
    """Cut n REAL background tiles (16 x F) from OFF cadence files at signal-free frequencies."""
    from blimpy import Waterfall
    rng = np.random.default_rng(seed)
    offs = sorted(f.replace("\\", "/") for f in glob.glob(f"{CAD}/*.rawspec.0000.h5"))[1::2]  # OFF = even idx
    assert offs, "no OFF cadence files"
    half = half_hz / 1e6
    bgs = []
    # sample random signal-free centers well away from the Voyager tones (~8419.54-8419.57)
    for _ in range(n):
        f = offs[rng.integers(len(offs))]
        wf = Waterfall(f)                                  # header only fast; then a window
        fch1 = wf.header["fch1"]; foff = wf.header["foff"]; nch = wf.header["nchans"]
        # pick a center at least 5 kHz from the known signal band
        while True:
            ci = int(rng.integers(int(nch * 0.15), int(nch * 0.85)))
            c = fch1 + ci * foff
            if abs(c - 8419.555) > 0.005:                  # >5 kHz from the Voyager tones
                break
        w = Waterfall(f, f_start=min(c - half, c + half), f_stop=max(c - half, c + half))
        d = np.squeeze(w.data).astype(np.float32)
        if d.shape[0] >= NT:
            bgs.append(d[:NT])
    return bgs


CHAN_HZ = 2.7939677238464355     # cadence channel width (Hz); TSAMP below is s/sample
TSAMP_S = 18.253611008


def inject(bg, snr_db, drift_hz_s, rng):
    """Add a drifting narrowband line into a copy of the real background at the target SNR.
    drift_hz_s is a PHYSICAL drift rate; convert to channels-per-time-sample with the real cadence
    sampling so injected signals drift like real technosignatures (real Voyager ~-0.36 Hz/s)."""
    img = bg.copy()
    T, F = img.shape
    noise = img.std() + 1e-9
    amp = noise * (10 ** (snr_db / 20.0))
    drift = drift_hz_s * TSAMP_S / CHAN_HZ                # channels per time sample
    f0 = rng.integers(F // 4, 3 * F // 4)
    for t in range(T):
        f = int(f0 + drift * t)
        if 0 <= f < F:
            img[t, f] += amp
            if f + 1 < F:
                img[t, f + 1] += amp * 0.6
    return img


def render(arr, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    a = arr.astype(np.float32); lo, hi = np.percentile(a, 5), np.percentile(a, 99.5)
    plt.imsave(path, np.clip((a - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray", origin="lower")


def build(snrs, drifts, reps, out, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(f"{out}/npy", exist_ok=True); os.makedirs(f"{out}/png", exist_ok=True)
    need = len(snrs) * len(drifts) * reps + reps * len(snrs)      # signals + matched noise negatives
    bgs = real_backgrounds(max(need // 4 + 8, 12), seed=seed)     # reuse a pool of real backgrounds
    assert bgs, "no real backgrounds cut"
    labels = {}; i = 0
    def bg(): return bgs[rng.integers(len(bgs))]
    for snr in snrs:
        for drift in drifts:
            for _ in range(reps):
                d = int(np.sign(rng.standard_normal())) or 1
                arr = inject(bg(), snr, drift * d, rng)
                _id = f"sig_{i:05d}"; np.save(f"{out}/npy/{_id}.npy", arr); render(arr, f"{out}/png/{_id}.png")
                labels[_id] = {"cls": "signal", "snr_db": snr, "drift": round(drift * d, 3)}; i += 1
        for _ in range(reps):                                     # matched noise negatives (real bg, no inject)
            arr = bg().copy()
            _id = f"noi_{i:05d}"; np.save(f"{out}/npy/{_id}.npy", arr); render(arr, f"{out}/png/{_id}.png")
            labels[_id] = {"cls": "noise", "snr_db": 0.0, "drift": 0.0}; i += 1
    json.dump(labels, open(f"{out}/labels.json", "w"), indent=0)
    print(f"[cp3] {i} tiles ({len([v for v in labels.values() if v['cls']=='signal'])} signal / "
          f"{len([v for v in labels.values() if v['cls']=='noise'])} noise) into REAL backgrounds -> {out}/")
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", default="3,6,9,12,15,18")
    ap.add_argument("--drifts", default="0.1,0.2,0.4")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", default="grid")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        lab = build([6, 15], [0.3], 2, "grid_smoke")
        assert any(v["cls"] == "signal" for v in lab.values()) and any(v["cls"] == "noise" for v in lab.values())
        print("SELF-CHECK OK")
    else:
        build([float(x) for x in a.snrs.split(",")], [float(x) for x in a.drifts.split(",")], a.reps, a.out)
