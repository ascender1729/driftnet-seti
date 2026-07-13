#!/usr/bin/env python3
"""CP3: completeness benchmark - inject setigen signals at an SNR x drift grid into REAL GBT noise.

The adversarial gate demanded this: a synthetic line dropped into a REAL Breakthrough Listen
off-target background (real bandpass, real fine-channel statistics) so the VLM/CNN cannot win on
setigen's idealized texture. We load clean windows from the real OFF cadence files as setigen
Frames, inject a narrowband drifting line at a known SNR and drift rate, and render. The label is
(signal at snr,drift) or (noise). This yields the detection-completeness-vs-SNR and vs-drift curves
that are the scientifically meaningful axis (turboSETI's matched filter is near-optimal at the faint
end; VLMs should fall off the visual floor).

  cp3_inject_real.py --snrs 3,6,9,12,15,20 --drifts -4,-2,-1,1,2,4 --bg-per-cell 3 --out data_realbg
  cp3_inject_real.py --validate    (tiny grid, render one panel to eyeball injection-in-real-noise)
"""
import argparse, glob, json, os, re
import numpy as np

CAD = "real_data/cadence"
OFF_IDX = (1, 3, 5)                      # OFF observations = real signal-free GBT backgrounds
SKIP_PNG = False                          # --no-png: skip the slow matplotlib render (DriftNet uses .npy only)


def off_files():
    # recursive (Enriquez zips can nest) + tolerant of file count; OFF = every other (ABACAD) file
    fs = sorted(f.replace("\\", "/") for f in glob.glob(f"{CAD}/**/*.0000.h5", recursive=True))
    offs = [fs[i] for i in OFF_IDX if i < len(fs)] or fs[1::2] or fs[:1]
    if not offs:
        raise RuntimeError(f"no *.0000.h5 files under {CAD}")
    return offs


def bg_frame(h5, ch_center, fchans=256):
    """A clean real-noise window from a real OFF file, as a DETACHED setigen Frame. Reads via h5py
    directly (works for BOTH Voyager .rawspec and Enriquez .gpuspec files; blimpy rejects the latter
    for a missing HDF5 CLASS attribute)."""
    import hdf5plugin, h5py, setigen as stg                       # hdf5plugin registers the BL filters
    with h5py.File(h5, "r") as hf:
        a = hf["data"].attrs
        foff, fch1, tsamp = float(a["foff"]), float(a["fch1"]), float(a["tsamp"])
        nch = hf["data"].shape[2]
        lo = min(max(0, ch_center - fchans // 2), nch - fchans); hi = lo + fchans
        data = hf["data"][:, 0, lo:hi].astype(np.float64)        # [T, F] real GBT noise window
    T, F = data.shape
    fr = stg.Frame(fchans=F, tchans=T, df=abs(foff) * 1e6, dt=tsamp, fch1=(fch1 + foff * lo) * 1e6)
    fr.data = data                                               # inject real noise as the background
    return fr


def _fresh(frame):
    import setigen as stg
    fr = stg.Frame(fchans=frame.fchans, tchans=frame.tchans, df=frame.df, dt=frame.dt, fch1=frame.fch1)
    fr.data = frame.data.copy()
    return fr


def inject(frame, snr, drift_hz_s):
    import setigen as stg
    fr = _fresh(frame)                                           # detached copy (no h5py -> no pickle error)
    # get_intensity(snr) needs setigen-registered noise; we injected REAL noise, so set the level
    # directly from the real background std: integration SNR = level*sqrt(tchans)/noise_std (nominal).
    noise_std = float(np.std(fr.data))
    level = snr * noise_std / np.sqrt(fr.tchans)
    start = fr.get_frequency(index=fr.fchans // 2)
    fr.add_signal(stg.constant_path(f_start=start, drift_rate=drift_hz_s),
                  stg.constant_t_profile(level=level),
                  stg.gaussian_f_profile(width=2 * float(fr.df)),
                  stg.constant_bp_profile(level=1))
    return np.asarray(fr.data, dtype=np.float32)


def render(arr, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    a = arr.astype(np.float32)
    a = a - np.median(a, axis=0, keepdims=True)          # subtract per-channel bandpass (drifting line survives)
    lo, hi = np.percentile(a, 5), np.percentile(a, 99.5)
    plt.imsave(path, np.clip((a - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray", origin="lower")


def build(snrs, drifts, bg_per_cell, out, seed=0):
    os.makedirs(f"{out}/npy", exist_ok=True)
    if not SKIP_PNG: os.makedirs(f"{out}/png", exist_ok=True)
    rng = np.random.default_rng(seed); offs = off_files()
    labels, i = {}, 0
    def a_bg():
        import hdf5plugin, h5py
        h5 = offs[rng.integers(len(offs))]
        with h5py.File(h5, "r") as hf:
            nch = hf["data"].shape[2]                                # window range scales to the file
        return bg_frame(h5, int(rng.integers(int(nch * 0.15), int(nch * 0.85))))
    for snr in snrs:
        for dr in drifts:
            for _ in range(bg_per_cell):
                arr = inject(a_bg(), snr, dr); _id = f"sig_{i:05d}"
                np.save(f"{out}/npy/{_id}.npy", arr); (None if SKIP_PNG else render(arr, f"{out}/png/{_id}.png"))
                labels[_id] = {"cls": "signal", "snr_db": float(snr), "drift_hz_s": float(dr), "real_bg": True}
                i += 1
    for snr in snrs:                                             # RFI = zero-drift line injected into real noise
        for _ in range(bg_per_cell * len(drifts) // 2):
            arr = inject(a_bg(), snr, 0.0); _id = f"rfi_{i:05d}"
            np.save(f"{out}/npy/{_id}.npy", arr); (None if SKIP_PNG else render(arr, f"{out}/png/{_id}.png"))
            labels[_id] = {"cls": "rfi", "snr_db": float(snr), "drift_hz_s": 0.0, "real_bg": True}
            i += 1
    for _ in range(len(snrs) * bg_per_cell * len(drifts) // 2):  # matched noise-only negatives
        arr = np.asarray(a_bg().data, dtype=np.float32); _id = f"noise_{i:05d}"
        np.save(f"{out}/npy/{_id}.npy", arr); (None if SKIP_PNG else render(arr, f"{out}/png/{_id}.png"))
        labels[_id] = {"cls": "noise", "snr_db": 0.0, "drift_hz_s": 0.0, "real_bg": True}
        i += 1
    json.dump(labels, open(f"{out}/labels.json", "w"), indent=0)
    print(f"[cp3] {i} samples ({len(snrs)}x{len(drifts)} grid, {bg_per_cell}/cell) into REAL GBT noise -> {out}/")
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", default="3,6,9,12,15,20")
    ap.add_argument("--drifts", default="-4,-2,-1,1,2,4")
    ap.add_argument("--bg-per-cell", type=int, default=3)
    ap.add_argument("--out", default="data_realbg")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--cad-dir")
    a = ap.parse_args()
    if a.no_png: SKIP_PNG = True
    if a.cad_dir: CAD = a.cad_dir
    if a.validate:
        offs = off_files(); print("real OFF backgrounds:", [os.path.basename(f) for f in offs])
        bg = bg_frame(offs[0], 500000)
        hi = inject(bg, 20, -2.0); lo = inject(bg, 4, -2.0)
        os.makedirs("data_realbg_val", exist_ok=True)
        render(hi, "data_realbg_val/inject_snr20.png"); render(lo, "data_realbg_val/inject_snr4.png")
        render(np.asarray(bg.data, dtype=np.float32), "data_realbg_val/noise_only.png")
        print("wrote data_realbg_val/{inject_snr20,inject_snr4,noise_only}.png  (line should be obvious@20, faint@4)")
    else:
        build([float(x) for x in a.snrs.split(",")], [float(x) for x in a.drifts.split(",")],
              a.bg_per_cell, a.out)
