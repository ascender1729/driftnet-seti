#!/usr/bin/env python3
"""Assemble a REAL 3-class labeled waterfall set from real GBT data only (no synthetic).

  signal : real narrowband technosignatures = turboSETI hits in the Voyager ON cadence panels
  rfi    : real interference = the Enriquez-2017 top-11 hits (pipeline-attributed RFI)
  noise  : real signal-free backgrounds = OFF panels / clean windows

All reads go through h5py directly (the Enriquez files lack the HDF5 CLASS attribute blimpy
requires; h5py reads them fine). Output: real_data/real3class/{png,labels.json}, VLM-ready.

Usage: cp3_real3class.py
"""
import csv, glob, json, os, re
import numpy as np
import hdf5plugin  # registers the bitshuffle/LZ4 filters BL uses; must precede h5py reads
import h5py

OUT = "real_data/real3class"
FCH = 256                                            # channels per cut window


def hdr(path):
    with h5py.File(path, "r") as h:
        a = h["data"].attrs
        return float(a["fch1"]), float(a["foff"]), int(h["data"].shape[2])


def cut(path, f_center_mhz):
    """h5py window of FCH channels centered on a target frequency (MHz)."""
    fch1, foff, nch = hdr(path)
    ch = int(round((f_center_mhz - fch1) / foff))
    lo = min(max(0, ch - FCH // 2), nch - FCH); hi = lo + FCH
    with h5py.File(path, "r") as h:
        return h["data"][:, 0, lo:hi].astype(np.float64)      # [16, FCH]


def render(arr, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    a = arr - np.median(arr, axis=0, keepdims=True)           # bandpass subtract (keep drifting line)
    lo, hi = np.percentile(a, 5), np.percentile(a, 99.5)
    plt.imsave(path, np.clip((a - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray", origin="lower")


def brightest_freq(path, guard=200):
    """Frequency (MHz) of the brightest narrowband channel after bandpass detrend - the RFI line."""
    from scipy.ndimage import uniform_filter1d
    fch1, foff, nch = hdr(path)
    with h5py.File(path, "r") as h:
        spec = h["data"][:, 0, :].mean(axis=0)
    ex = spec - uniform_filter1d(spec, 2001)
    ex[:guard] = 0; ex[-guard:] = 0
    ch = int(np.argmax(ex))
    return fch1 + foff * ch


def voyager_signal(labels, i):
    """Real signals: top turboSETI hits in each Voyager ON cadence file."""
    onfiles = sorted(f.replace("\\", "/") for f in glob.glob("real_data/cadence/*0011*.h5") +
                     glob.glob("real_data/cadence/*0013*.h5") + glob.glob("real_data/cadence/*0015*.h5"))
    for f in onfiles:
        dat = os.path.splitext(f)[0] + ".dat"
        if not os.path.exists(dat):
            continue
        hits = [l.split() for l in open(dat) if l.strip() and not l.startswith("#")]
        for hcols in sorted(hits, key=lambda c: -float(c[2]))[:3]:      # top 3 by SNR
            fmhz = float(hcols[3])
            _id = f"signal_{i:04d}"; render(cut(f, fmhz), f"{OUT}/png/{_id}.png")
            labels[_id] = {"cls": "signal", "src": os.path.basename(f), "freq_mhz": fmhz}; i += 1
    return i


def enriquez_rfi_noise(labels, i):
    """Real RFI: brightest hit in each Enriquez ON file. Real noise: a clean window in each OFF file."""
    for cad in sorted(glob.glob("real_data/rfi/HIP*/")):
        fs = sorted(f.replace("\\", "/") for f in glob.glob(cad + "*.0000.h5"))
        if len(fs) < 6:
            continue
        for j, f in enumerate(fs):
            if j % 2 == 0:                                     # ON files: cut the RFI hit
                fmhz = brightest_freq(f)
                _id = f"rfi_{i:04d}"; render(cut(f, fmhz), f"{OUT}/png/{_id}.png")
                labels[_id] = {"cls": "rfi", "src": os.path.basename(f), "freq_mhz": round(fmhz, 4)}; i += 1
            else:                                              # OFF files: a clean window = noise
                fch1, foff, nch = hdr(f)
                fmhz = fch1 + foff * (nch // 3)                # a fixed off-hit region
                _id = f"noise_{i:04d}"; render(cut(f, fmhz), f"{OUT}/png/{_id}.png")
                labels[_id] = {"cls": "noise", "src": os.path.basename(f), "freq_mhz": round(fmhz, 4)}; i += 1
    return i


def voyager_noise(labels, i):
    """Real noise: OFF panels of the Voyager cadence, a window near (but off) the carrier."""
    for f in sorted(glob.glob("real_data/cadence/*0012*.h5") + glob.glob("real_data/cadence/*0014*.h5") +
                    glob.glob("real_data/cadence/*0016*.h5")):
        f = f.replace("\\", "/")
        _id = f"noise_{i:04d}"; render(cut(f, 8419.30), f"{OUT}/png/{_id}.png")
        labels[_id] = {"cls": "noise", "src": os.path.basename(f), "freq_mhz": 8419.30}; i += 1
    return i


def main():
    os.makedirs(f"{OUT}/png", exist_ok=True)
    labels, i = {}, 0
    i = voyager_signal(labels, i)
    i = enriquez_rfi_noise(labels, i)
    i = voyager_noise(labels, i)
    json.dump(labels, open(f"{OUT}/labels.json", "w"), indent=1)
    from collections import Counter
    c = Counter(v["cls"] for v in labels.values())
    print(f"[real3class] {i} REAL samples: {dict(c)} -> {OUT}/")


if __name__ == "__main__":
    main()
