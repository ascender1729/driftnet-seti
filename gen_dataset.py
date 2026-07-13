#!/usr/bin/env python3
"""Labeled dynamic-spectrogram dataset for the technosignature-detection benchmark.

Three balanced classes:
  signal - a narrowband line that DRIFTS (Doppler) on a noise floor  -> the technosignature
  rfi    - a narrowband line at CONSTANT frequency on a noise floor   -> interference
  noise  - pure noise floor                                          -> nothing

Primary generator is setigen (the field-standard injector, for credibility). If setigen is
not importable yet, a faithful numpy generator with the SAME output contract is used, so the
pipeline never blocks on install. Output contract (identical either way):
  data/npy/<id>.npy   float32 [T,F] dynamic spectrum
  data/png/<id>.png   rendered grayscale image (no axes/labels - nothing that leaks the class)
  data/labels.json    {id: {"cls","snr_db","drift_hz_s","gen"}}

Usage: gen_dataset.py --n-per-class 200 --out data --seed 0
       (a --n-per-class 4 --self-check smoke run asserts the contract)
"""
import argparse, json, os
import numpy as np

T, F = 128, 256                      # time rows, freq channels -> the image is F wide, T tall
CLASSES = ("signal", "rfi", "noise")

try:
    import setigen as stg            # noqa
    from astropy import units as u   # noqa
    HAVE_STG = True
except Exception:
    HAVE_STG = False


def _render_png(arr, path):
    # ponytail: matplotlib imsave, grayscale, no axes - the raw image is all the VLM/CNN should see.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = arr - arr.min()
    a = a / (a.max() + 1e-9)
    plt.imsave(path, a, cmap="gray", origin="lower")


def _gen_numpy(cls, snr_db, drift, rng):
    """Faithful fallback: chi2-like noise floor + optional thin (drifting or constant) line."""
    base = 10.0
    img = rng.chisquare(2, size=(T, F)).astype(np.float32) * (base / 2.0)   # chi2 noise, mean~base
    noise_std = img.std()
    if cls != "noise":
        amp = noise_std * (10 ** (snr_db / 20.0))          # signal amplitude from SNR (dB, amplitude ratio)
        f0 = rng.integers(F // 4, 3 * F // 4)
        for t in range(T):
            f = int(f0 + (drift * t if cls == "signal" else 0))
            if 0 <= f < F:
                img[t, f] += amp
                if f + 1 < F:                              # 2-wide line, like a real narrowband
                    img[t, f + 1] += amp * 0.6
    return img


def _gen_setigen(cls, snr_db, drift, rng):
    # setigen 2.7: df/dt/get_frequency are plain floats in Hz/s; drift_rate is Hz/s.
    frame = stg.Frame(fchans=F, tchans=T, df=2.7939677238464355, dt=18.253611008, fch1=6095.214842353016e6)
    frame.add_noise(x_mean=10, noise_type="chi2")
    if cls != "noise":
        snr_lin = 10 ** (snr_db / 10.0)                    # get_intensity takes a power SNR
        start = frame.get_frequency(index=int(rng.integers(F // 4, 3 * F // 4)))
        # drift (channels/step) * df (Hz/channel) / dt (s/step) = Hz/s ; RFI is zero-drift
        dr = (drift * float(frame.df) / float(frame.dt)) if cls == "signal" else 0.0
        frame.add_signal(
            stg.constant_path(f_start=start, drift_rate=dr),
            stg.constant_t_profile(level=frame.get_intensity(snr=snr_lin)),
            stg.gaussian_f_profile(width=2 * float(frame.df)),
            stg.constant_bp_profile(level=1))
    return np.asarray(frame.data, dtype=np.float32)


def build(n_per_class, out, seed):
    rng = np.random.default_rng(seed)
    os.makedirs(f"{out}/npy", exist_ok=True)
    os.makedirs(f"{out}/png", exist_ok=True)
    labels, i = {}, 0
    gen_name = "setigen" if HAVE_STG else "numpy"
    for cls in CLASSES:
        for _ in range(n_per_class):
            snr_db = float(rng.uniform(3, 20)) if cls != "noise" else 0.0   # SNR sweep 3..20 dB
            drift = float(rng.uniform(0.15, 0.6)) * (1 if rng.random() < 0.5 else -1)  # channels/step
            arr = (_gen_setigen if HAVE_STG else _gen_numpy)(cls, snr_db, drift, rng)
            _id = f"{cls}_{i:05d}"
            np.save(f"{out}/npy/{_id}.npy", arr)
            _render_png(arr, f"{out}/png/{_id}.png")
            labels[_id] = {"cls": cls, "snr_db": round(snr_db, 2),
                           "drift_hz_s": round(drift, 3) if cls == "signal" else 0.0, "gen": gen_name}
            i += 1
    json.dump(labels, open(f"{out}/labels.json", "w"), indent=0)
    print(f"[gen] {i} samples ({n_per_class}/class) via {gen_name} -> {out}/  (T={T},F={F})")
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        lab = build(4, a.out + "_smoke", a.seed)
        assert len(lab) == 12, len(lab)
        import glob
        assert len(glob.glob(a.out + "_smoke/png/*.png")) == 12
        assert {v["cls"] for v in lab.values()} == set(CLASSES)
        print("SELF-CHECK OK")
    else:
        build(a.n_per_class, a.out, a.seed)
