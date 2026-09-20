#!/usr/bin/env python3
"""Voyager-1 real-signal positive control for the curvature-constrained detector.

Not an injection: run the curved de-Doppler over (drift, acceleration) on the REAL Voyager-1 carrier in
a real GBT ON observation. Two things to show: (1) the detector recovers the real carrier (it works on a
genuine narrowband signal, not just setigen injections); (2) its best-fit acceleration a_hat is small --
the real quiescent carrier sits in the low-acceleration manifold, so the physics prior's placement is
validated on real data rather than assumed.

2026-09-20 fix: the carrier channel is taken from the authors' turboSETI hit table (.dat next to the
.h5, highest-SNR row). The previous time-median argmax landed on the coarse-channel DC spike at
2^19 (8419.921875 MHz), not the carrier at 8419.542731 MHz (Index 659989, -0.357 Hz/s).
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def turboseti_carrier(h5):
    """Carrier channel + drift from the turboSETI .dat beside the .h5 (same basename): the highest-SNR
    hit. Index is the channel index in file order (fch1 + foff*Index reproduces the frequency).
    Returns (chan, drift_hz_s, snr) or None if no .dat."""
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
    """Does the per-row argmax follow c(t) = F//2 + d_chs*t? Returns dict with the fraction of rows
    within halfw channels for the given sign and for the flipped sign (sign sanity for the blanking)."""
    T, F = W.shape; t = np.arange(T); am = W.argmax(axis=1)
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


def load_on_window(cad_dir, fchans=512):
    """Read a real Voyager ON file, locate the carrier (turboSETI .dat; argmax fallback), return a
    [T, fchans] window centered on its start channel (per-time-row median subtracted)."""
    import h5py, hdf5plugin
    fs = sorted(f.replace("\\", "/") for f in glob.glob(f"{cad_dir}/*.0000.h5"))
    on = fs[0]                                            # first ON observation of the ABACAD cadence
    with h5py.File(on, "r") as hf:
        data = hf["data"][:, 0, :].astype(np.float64)    # [T, Nch]
        foff = float(hf["data"].attrs["foff"]); fch1 = float(hf["data"].attrs["fch1"])
        tsamp = float(hf["data"].attrs["tsamp"])
    hit = turboseti_carrier(on)
    if hit is not None:
        cpk, dat_drift, dat_snr = hit; src = "turboseti_dat"
        print(f"[voyager] carrier from {os.path.splitext(on)[0] + '.dat'}: Index={cpk} drift={dat_drift:+.6f} Hz/s SNR={dat_snr:.1f}", flush=True)
    else:
        prof = np.median(data, axis=0)                    # time-median spectrum: fallback only
        prof = prof - np.median(prof)
        cpk = int(np.argmax(prof)); dat_drift = None; dat_snr = None; src = "argmax_fallback"
        print(f"[voyager] no .dat beside {on}; argmax fallback chan {cpk}", flush=True)
    lo = min(max(0, cpk - fchans // 2), data.shape[1] - fchans); hi = lo + fchans
    W = data[:, lo:hi]
    W = W - np.median(W, axis=1, keepdims=True)           # per-time-row bandpass removal
    W = W / (np.std(W) + 1e-9)
    carrier_mhz = (fch1 + foff * cpk)                     # already MHz in BL headers
    meta = {"carrier_source": src, "carrier_chan": int(cpk), "dat_drift_hz_s": dat_drift, "dat_snr": dat_snr,
            "on_file": os.path.basename(on), "foff_mhz": foff}
    return W, carrier_mhz, foff, tsamp, cpk, meta


def curved_peak(W, drifts, accels):
    """Return (rho, d_hat, a_hat) at the peak of the curved de-Doppler response over the grid."""
    T, F = W.shape; cols = np.arange(F); t = np.arange(T); sig = np.sqrt(T)
    best = (-1e9, 0.0, 0.0)
    for dd in drifts:
        for a in accels:
            shift = np.rint(dd * t + 0.5 * a * t * t).astype(int)
            idx = (cols[None, :] + shift[:, None]) % F
            m = np.take_along_axis(W, idx, axis=1).sum(0).max() / sig
            if m > best[0]:
                best = (float(m), float(dd), float(a))
    return best


def main():
    cad_dir = sys.argv[sys.argv.index("--cad-dir") + 1] if "--cad-dir" in sys.argv else "real_data/cadence"
    W, carrier_mhz, foff, tsamp, cpk, meta = load_on_window(cad_dir)
    T, F = W.shape
    df_hz = abs(foff) * 1e6                               # Hz per channel
    print(f"[voyager] real ON window T={T} F={F}; carrier near {carrier_mhz:.6f} MHz (chan {cpk}, {meta['carrier_source']})", flush=True)
    drifts = np.linspace(-3.0, 3.0, 61)                  # channels per sample (carrier is ~ -2.3 ch/samp)
    accels = np.linspace(-0.04, 0.04, 41)                # curvature grid straddling zero
    rho, dhat, ahat = curved_peak(W, drifts, accels)
    rho_lin, dhat_lin, _ = curved_peak(W, drifts, [0.0])  # linear-only peak for comparison
    # expected drift in channels per sample, sign from the file's channel order (foff<0: -Hz/s -> +chan)
    d_dat_chs = (meta["dat_drift_hz_s"] * tsamp / (foff * 1e6)) if meta["dat_drift_hz_s"] is not None else dhat_lin
    chk = track_check(W, d_dat_chs)
    print(f"[voyager] track check d_dat={d_dat_chs:+.3f} ch/samp: as_given within3={chk['as_given']['frac_within']:.2f} "
          f"maxres={chk['as_given']['max_abs_resid']} | flipped within3={chk['flipped']['frac_within']:.2f} "
          f"maxres={chk['flipped']['max_abs_resid']} -> sign {'OK' if chk['sign_ok'] else 'WRONG'}", flush=True)
    print(f"[voyager] row argmax: {chk['row_argmax']}", flush=True)
    # noise floor: same search on the same window with the drifting carrier TRACK blanked (+-3 ch)
    rho_noise, _, _ = curved_peak(blank_track(W, d_dat_chs), drifts, accels)
    # physical units
    dhat_hzs = dhat * df_hz / tsamp                       # Hz/s
    ahat_hzs2 = ahat * df_hz / (tsamp * tsamp)            # Hz/s^2
    a_grid_span = float(accels.max())
    out = {"carrier_mhz": carrier_mhz, "T": T, "F": F, "rho_carrier": rho, "rho_linear": rho_lin,
           "rho_noise_floor": rho_noise, "blank_mode": "track",
           "d_hat_chan_per_samp": dhat, "a_hat_chan_per_samp2": ahat,
           "d_hat_hz_s": dhat_hzs, "a_hat_hz_s2": ahat_hzs2,
           "d_hat_linear_chan_per_samp": dhat_lin, "d_hat_linear_hz_s": dhat_lin * df_hz / tsamp,
           "d_dat_chan_per_samp": float(d_dat_chs), "track_check": chk,
           "a_hat_over_grid_span": abs(ahat) / a_grid_span, "tsamp_s": tsamp, "df_hz": df_hz,
           "drift_grid": [float(drifts.min()), float(drifts.max()), len(drifts)], **meta}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/voyager_control.json", "w"), indent=1)
    print(f"[voyager] carrier recovered: rho={rho:.1f} (linear-only rho={rho_lin:.1f}, track-blanked noise-floor rho={rho_noise:.1f})", flush=True)
    print(f"[voyager] best-fit drift d_hat={dhat_hzs:+.3f} Hz/s ({dhat:+.2f} ch/samp; .dat says {meta['dat_drift_hz_s']} Hz/s = {d_dat_chs:+.2f} ch/samp), "
          f"curvature a_hat={ahat_hzs2:+.2e} Hz/s^2", flush=True)
    print(f"[voyager] |a_hat| is {100*abs(ahat)/a_grid_span:.1f}% of the searched grid span -> "
          f"{'IN the low-acceleration manifold' if abs(ahat)/a_grid_span < 0.15 else 'OUTSIDE the manifold'}", flush=True)
    print("VOYAGER_CONTROL_DONE", flush=True)


if __name__ == "__main__":
    main()
