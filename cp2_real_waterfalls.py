#!/usr/bin/env python3
"""CP2 (extraction): cut CLEAN real waterfall panels around the real cadence events.

Fixes the CP1 montage: centers each panel's window on the drift-EXTRAPOLATED carrier frequency for
that panel's time (the signal drifts ~-0.36 Hz/s across the 30-min cadence), and sigma-clips the
render so the bandpass/DC artifact does not dominate. Produces labeled real images:
  ON panels  (telescope on-source) -> the real technosignature is present  -> label 'signal'
  OFF panels (telescope off-source) -> the signal is absent                -> label 'noise'
plus a clean 6-panel cadence montage (the ON-OFF persistence view).

Usage: cp2_real_waterfalls.py
"""
import csv, glob, json, os, re
import numpy as np

CAD = "real_data/cadence"
OUT = "real_data/cp2_out"
os.makedirs(OUT, exist_ok=True)
TSAMP = 18.253611008          # s per time sample
NT = 16                       # time samples per file


def cadence_files():
    fs = sorted(f.replace("\\", "/") for f in glob.glob(f"{CAD}/*.rawspec.0000.h5"))
    assert len(fs) == 6, len(fs)
    return fs


def panel_tstart(path):
    return int(re.search(r"_59046_(\d+)_DIAG", path).group(1))      # start second from the filename


def top_event():
    """Highest-SNR event from the filter_threshold=3 CSV (real Voyager carrier)."""
    rows = list(csv.DictReader(open("real_data/cp1_out/events_f3.csv")))
    rows.sort(key=lambda r: -float(r["SNR"]))
    e = rows[0]
    return {"freq": float(e["Freq"]), "drift": float(e["DriftRate"]),
            "snr": float(e["SNR"]), "src_tstart": panel_tstart(e["FileID"])}


def render(arr, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, 5), np.percentile(a, 99.5)          # sigma-clip the bandpass/DC tail
    a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    plt.imsave(path, a, cmap="gray", origin="lower")


def cut(files, ev, half_hz=400):
    from blimpy import Waterfall
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    half = half_hz / 1e6
    labels, panels = {}, []
    fig, axes = plt.subplots(6, 1, figsize=(5, 8))
    for i, (f, ax) in enumerate(zip(files, axes)):
        dt = panel_tstart(f) - ev["src_tstart"]                  # seconds since the detection panel
        center = ev["freq"] + ev["drift"] * dt / 1e6             # drift-extrapolated MHz for this panel
        wf = Waterfall(f, f_start=center - half, f_stop=center + half)
        d = np.squeeze(wf.data)
        role = "signal" if i % 2 == 0 else "noise"               # ON=signal present, OFF=noise
        p = f"{OUT}/{role}_panel{i}.png"; render(d, p)
        labels[f"panel{i}"] = {"cls": role, "on_off": "ON" if i % 2 == 0 else "OFF",
                               "center_mhz": round(center, 6)}
        panels.append((p, role))
        a = d.astype(np.float32); lo, hi = np.percentile(a, 5), np.percentile(a, 99.5)
        ax.imshow(np.clip((a - lo) / (hi - lo + 1e-9), 0, 1), aspect="auto", cmap="viridis", origin="lower")
        ax.set_ylabel("ON" if i % 2 == 0 else "OFF", rotation=0, ha="right", va="center", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_title(f"Real Voyager cadence, drift-tracked ({ev['freq']:.4f} MHz, {ev['drift']:.3f} Hz/s)", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/real_cadence_montage.png", dpi=110); plt.close()
    return labels, panels


def main():
    files = cadence_files()
    ev = top_event()
    print(f"top event: {ev['freq']:.5f} MHz  drift {ev['drift']:.3f} Hz/s  SNR {ev['snr']:.1f}")
    labels, panels = cut(files, ev)
    json.dump({"event": ev, "panels": labels}, open(f"{OUT}/cp2_labels.json", "w"), indent=1)
    print("panels:", {k: v["cls"] for k, v in labels.items()})
    print(f"-> {OUT}/real_cadence_montage.png + {len(panels)} labeled panels")


if __name__ == "__main__":
    main()
