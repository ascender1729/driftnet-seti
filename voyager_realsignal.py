#!/usr/bin/env python3
"""Real-signal positive control for Part 1 (DriftNet, the linear detector).

Part 2 already showed the curved detector recovers the real Voyager carrier. This does the analogous
control for the DriftNet classifier: run the learned decision on the GENUINE Voyager-1 carrier in a real
ON observation (not a setigen injection) and show it fires as 'signal', while a carrier-blanked copy of
the same window scores as 'noise'. This partially answers "would the learned decision work on a real
narrowband signal, not just synthetic ones?"

2026-09-20 fix: carrier channel from the turboSETI .dat beside the .h5 (highest-SNR hit), not the
time-median argmax (which hit the DC spike at 2^19). The carrier drifts ~ -2.3 ch/samp, just outside
the frozen [-2,+2] grid, so the window is scored twice: with the as-is model (scores_grid2) and with
the same trainer at grid_max=3 (scores_grid3). Blanking follows the drifting track, not a column.

  voyager_realsignal.py --cad-dir real_data/cadence
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cp4_cadence_roc as CP4
from driftnet_v2 import CLS


def turboseti_carrier(h5):
    """Carrier channel + drift from the turboSETI .dat beside the .h5 (same basename): the highest-SNR
    hit. Index is the channel index in file order. Returns (chan, drift_hz_s, snr) or None if no .dat."""
    dat = os.path.splitext(h5)[0] + ".dat"
    if not os.path.exists(dat):
        return None
    best = None
    for line in open(dat):
        p = line.split()
        if not p or p[0].startswith("#"):
            continue
        drift, snr, idx = float(p[1]), float(p[2]), int(p[5])
        if best is None or snr > best[2]:
            best = (idx, drift, snr)
    return best


def track_check(W, d_chs, halfw=3):
    """Does the per-row argmax (after per-row median removal) follow c(t) = F//2 + d_chs*t?"""
    T, F = W.shape; t = np.arange(T)
    am = (W - np.median(W, axis=1, keepdims=True)).argmax(axis=1)
    res = {}
    for name, d in (("as_given", d_chs), ("flipped", -d_chs)):
        c = np.rint(F // 2 + d * t).astype(int)
        res[name] = {"frac_within": float(np.mean(np.abs(am - c) <= halfw)),
                     "max_abs_resid": int(np.max(np.abs(am - c)))}
    res["row_argmax"] = am.tolist()
    res["sign_ok"] = res["as_given"]["frac_within"] >= res["flipped"]["frac_within"]
    return res


def blank_track(W, d_chs, halfw=3):
    """Blank +-halfw channels around the drifting track c(t) = F//2 + d_chs*t in every row."""
    T, F = W.shape; Wn = W.copy(); fill = np.median(W)
    for t in range(T):
        c = int(np.rint(F // 2 + d_chs * t))
        Wn[t, max(0, c - halfw):min(F, c + halfw + 1)] = fill
    return Wn


def carrier_window(cad_dir, fchans=512):
    """Load a real Voyager ON file, find the carrier channel (turboSETI .dat; argmax fallback), and
    return a [T, fchans] window centered on it, read via the SAME loader used for training frames."""
    import h5py
    fs = sorted(f.replace("\\", "/") for f in glob.glob(f"{cad_dir}/**/*VOYAGER*.0000.h5", recursive=True))
    if not fs:
        fs = sorted(f.replace("\\", "/") for f in glob.glob(f"{cad_dir}/**/*.0000.h5", recursive=True))
    on = fs[0]
    with h5py.File(on, "r") as hf:
        foff = float(hf["data"].attrs["foff"]); tsamp = float(hf["data"].attrs["tsamp"])
        fch1 = float(hf["data"].attrs["fch1"])
        hit = turboseti_carrier(on)
        if hit is not None:
            cpk, dat_drift, dat_snr = hit; src = "turboseti_dat"
            print(f"[voyager-realsignal] carrier from {os.path.splitext(on)[0] + '.dat'}: Index={cpk} drift={dat_drift:+.6f} Hz/s SNR={dat_snr:.1f}", flush=True)
        else:
            data = hf["data"][:, 0, :].astype(np.float64)
            prof = np.median(data, axis=0); prof = prof - np.median(prof)
            cpk = int(np.argmax(prof)); dat_drift = None; dat_snr = None; src = "argmax_fallback"
            print(f"[voyager-realsignal] no .dat beside {on}; argmax fallback chan {cpk}", flush=True)
    fr = CP4.C3.bg_frame(on, cpk, fchans=fchans)          # same loader/orientation as train_driftnet
    W = np.asarray(fr.data, dtype=np.float32)
    # expected drift in channels per sample, sign from the file's channel order (foff<0: -Hz/s -> +chan)
    d_chs = (dat_drift * tsamp / (foff * 1e6)) if dat_drift is not None else 0.0
    meta = {"carrier_source": src, "dat_drift_hz_s": dat_drift, "dat_snr": dat_snr,
            "d_dat_chan_per_samp": float(d_chs), "carrier_mhz": fch1 + foff * cpk}
    return W, cpk, on, meta


def main():
    cad_dir = sys.argv[sys.argv.index("--cad-dir") + 1] if "--cad-dir" in sys.argv else "real_data/cadence"
    CP4.C3.SKIP_PNG = True
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = {"grid2": CP4.train_driftnet(cad_dir, dev),                # as-is: frozen [-2,+2] grid
            "grid3": CP4.train_driftnet(cad_dir, dev, grid_max=3.0)}  # same trainer, grid reaches the carrier
    W, cpk, on, meta = carrier_window(cad_dir)
    T, F = W.shape
    print(f"[voyager-realsignal] real ON file={os.path.basename(on)} window T={T} F={F} carrier_chan={cpk} ({meta['carrier_source']})", flush=True)
    chk = track_check(W, meta["d_dat_chan_per_samp"])
    print(f"[voyager-realsignal] track check d_dat={meta['d_dat_chan_per_samp']:+.3f} ch/samp: "
          f"as_given within3={chk['as_given']['frac_within']:.2f} maxres={chk['as_given']['max_abs_resid']} | "
          f"flipped within3={chk['flipped']['frac_within']:.2f} maxres={chk['flipped']['max_abs_resid']} "
          f"-> sign {'OK' if chk['sign_ok'] else 'WRONG'}", flush=True)
    print(f"[voyager-realsignal] row argmax: {chk['row_argmax']}", flush=True)
    Wn = blank_track(W, meta["d_dat_chan_per_samp"])       # blank the drifting carrier track (+-3 ch)

    def score(net, arr):
        with torch.no_grad():
            prob = torch.softmax(net(torch.tensor(arr.astype(np.float32)[None]).to(dev))[0], 1)[0]
        return {CLS[i]: round(float(prob[i]), 4) for i in range(len(CLS))}

    res = {}
    for name, net in nets.items():
        real, blank = score(net, W), score(net, Wn)
        res[f"scores_{name}"] = {"real_carrier_scores": real, "carrier_blanked_scores": blank,
                                 "pred_real": max(real, key=real.get), "pred_blanked": max(blank, key=blank.get),
                                 "drift_grid_max": 2.0 if name == "grid2" else 3.0}
        print(f"  [{name}] real carrier   -> {res[f'scores_{name}']['pred_real']:6s} {real}", flush=True)
        print(f"  [{name}] track blanked  -> {res[f'scores_{name}']['pred_blanked']:6s} {blank}", flush=True)
    g2 = res["scores_grid2"]
    out = {"cadence": cad_dir, "on_file": os.path.basename(on), "T": T, "F": F, "carrier_chan": cpk,
           "real_carrier_scores": g2["real_carrier_scores"], "carrier_blanked_scores": g2["carrier_blanked_scores"],
           "pred_real": g2["pred_real"], "pred_blanked": g2["pred_blanked"],
           "blank_mode": "track", "track_check": chk, **meta, **res}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/voyager_realsignal.json", "w"), indent=1)
    print("VOYAGER_REALSIGNAL_DONE", flush=True)


if __name__ == "__main__":
    main()
