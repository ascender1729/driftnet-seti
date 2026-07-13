#!/usr/bin/env python3
"""KEDD-S on REAL GBT noise: the matched-false-alarm sensitivity ordering for accelerating narrowband
signals, validated on real non-Gaussian Breakthrough Listen off-source noise (not white Gaussian).

Three detectors, MATCHED false alarm calibrated on REAL OFF frames:
  linear (a=0), free-box curved (wide accel grid), manifold (thin physical accel band).
Injections (setigen-style manual curved line into REAL noise): linear (a=0) and high-acceleration
signals (the 2026 chirp-up regime, where a few-channel bend appears even within a short dwell).
Guard against a tautology: the injected accelerations are drawn INDEPENDENTLY (uniform in the manifold
band), not from the detector's grid; drift is placed off-grid so the linear detector is not unfairly
penalized by quantization.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cp3_inject_real as C3

FCHANS = 256
rng = np.random.default_rng(7)


def real_offs(cad_dir):
    fs = sorted(f.replace("\\", "/") for f in glob.glob(f"{cad_dir}/**/*.0000.h5", recursive=True))
    if not fs:
        raise RuntimeError(f"no *.0000.h5 under {cad_dir}")
    return fs


def real_noise(cad_dir):
    """A real GBT off-noise window [T, F], per-time-row median subtracted (bandpass removed)."""
    import h5py
    offs = real_offs(cad_dir); h5 = offs[rng.integers(len(offs))]
    with h5py.File(h5, "r") as hf:
        nch = hf["data"].shape[2]
    fr = C3.bg_frame(h5, int(rng.integers(int(nch * 0.2), int(nch * 0.8))), fchans=FCHANS)
    d = np.asarray(fr.data, dtype=np.float64)
    d = d - np.median(d, axis=1, keepdims=True)          # per-time-row bandpass removal
    return d / (np.std(d) + 1e-9)                         # unit-ish scale


def inject(P, d, a, snr):
    """Add a narrowband line along f(t)=f0+d*t+0.5*a*t^2 into a copy of real noise P at nominal snr."""
    T, F = P.shape; Q = P.copy(); t = np.arange(T)
    amp = snr / np.sqrt(T)                                # integrated-SNR normalization (noise is unit-std)
    f0 = int(rng.integers(int(F * 0.35), int(F * 0.65))) + float(rng.uniform(-0.5, 0.5))  # off-grid
    ch = f0 + d * t + 0.5 * a * t * t
    for ti in range(T):
        c = int(round(ch[ti]))
        if 0 <= c < F:
            Q[ti, c] += amp
    return Q


def rho(P, drifts, accels):
    T, F = P.shape; cols = np.arange(F); t = np.arange(T); sig = np.sqrt(T); best = -1e9
    for dd in drifts:
        for a in accels:
            shift = np.rint(dd * t + 0.5 * a * t * t).astype(int)
            idx = (cols[None, :] + shift[:, None]) % F
            m = np.take_along_axis(P, idx, axis=1).sum(0).max()
            if m > best:
                best = m
    return best / sig


def main():
    cad_dir = sys.argv[sys.argv.index("--cad-dir") + 1] if "--cad-dir" in sys.argv else "real_data/cadence"
    C3.SKIP_PNG = True
    T = 16                                               # real BL fine-res dwell length
    DRIFTS = np.linspace(-0.6, 0.6, 13)
    A_MAN = np.linspace(-0.03, 0.03, 7)                  # thin physical band (a few-channel bend over T=16)
    A_BOX = np.linspace(-0.12, 0.12, 25)                 # wide box (chase RFI / completeness) -> big trials
    print(f"[kedd-real] T={T} F={FCHANS} trials lin={len(DRIFTS)} man={len(DRIFTS)*len(A_MAN)} box={len(DRIFTS)*len(A_BOX)}", flush=True)

    def thr(accels, n=200, fa=0.05):
        return float(np.quantile([rho(real_noise(cad_dir)[:T], DRIFTS, accels) for _ in range(n)], 1 - fa))
    tL, tM, tB = thr([0.0]), thr(A_MAN), thr(A_BOX)
    print(f"[kedd-real] matched-FA thresholds on REAL noise: linear={tL:.2f} manifold={tM:.2f} free-box={tB:.2f}", flush=True)

    def compl(accels, t, afn, snr, n=100):
        return float(np.mean([rho(inject(real_noise(cad_dir)[:T], float(rng.choice(DRIFTS)) + float(rng.uniform(-0.05, 0.05)), afn(), snr), DRIFTS, accels) > t for _ in range(n)]))

    onman = lambda: float(rng.uniform(-0.03, 0.03))
    out = {"T": T, "F": FCHANS, "thr": {"linear": tL, "manifold": tM, "freebox": tB}, "curves": {}}
    print(f"{'snr':>5} | {'linear-sig lin/box/man':>26} | {'accel-sig lin/box/man':>26}", flush=True)
    for snr in [4, 6, 8, 10, 12]:
        lin = [compl([0.0], tL, lambda: 0.0, snr), compl(A_BOX, tB, lambda: 0.0, snr), compl(A_MAN, tM, lambda: 0.0, snr)]
        acc = [compl([0.0], tL, onman, snr), compl(A_BOX, tB, onman, snr), compl(A_MAN, tM, onman, snr)]
        out["curves"][snr] = {"linear_sig": lin, "accel_sig": acc}
        print(f"{snr:>5} | {lin[0]:>7.2f}{lin[1]:>8.2f}{lin[2]:>8.2f}   | {acc[0]:>7.2f}{acc[1]:>8.2f}{acc[2]:>8.2f}", flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/kedd_real.json", "w"), indent=1)
    print("KEDD_REAL_DONE", flush=True)


if __name__ == "__main__":
    main()
