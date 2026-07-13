#!/usr/bin/env python3
"""Voyager-1 real-signal positive control for the curvature-constrained detector.

Not an injection: run the curved de-Doppler over (drift, acceleration) on the REAL Voyager-1 carrier in
a real GBT ON observation. Two things to show: (1) the detector recovers the real carrier (it works on a
genuine narrowband signal, not just setigen injections); (2) its best-fit acceleration a_hat is small --
the real quiescent carrier sits in the low-acceleration manifold, so the physics prior's placement is
validated on real data rather than assumed.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_on_window(cad_dir, fchans=512):
    """Read a real Voyager ON file, locate the brightest narrowband line (the carrier), return a
    [T, fchans] window centered on it (per-time-row median subtracted)."""
    import h5py, hdf5plugin
    fs = sorted(f.replace("\\", "/") for f in glob.glob(f"{cad_dir}/*.0000.h5"))
    on = fs[0]                                            # first ON observation of the ABACAD cadence
    with h5py.File(on, "r") as hf:
        data = hf["data"][:, 0, :].astype(np.float64)    # [T, Nch]
        foff = float(hf["data"].attrs["foff"]); fch1 = float(hf["data"].attrs["fch1"])
        tsamp = float(hf["data"].attrs["tsamp"])
    prof = np.median(data, axis=0)                        # time-median spectrum: carrier is a sharp peak
    prof = prof - np.median(prof)
    cpk = int(np.argmax(prof))
    lo = min(max(0, cpk - fchans // 2), data.shape[1] - fchans); hi = lo + fchans
    W = data[:, lo:hi]
    W = W - np.median(W, axis=1, keepdims=True)           # per-time-row bandpass removal
    W = W / (np.std(W) + 1e-9)
    carrier_mhz = (fch1 + foff * cpk)                     # already MHz in BL headers
    return W, carrier_mhz, foff, tsamp, cpk


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
    W, carrier_mhz, foff, tsamp, cpk = load_on_window(cad_dir)
    T, F = W.shape
    print(f"[voyager] real ON window T={T} F={F}; carrier near {carrier_mhz:.6f} MHz (chan {cpk})", flush=True)
    drifts = np.linspace(-1.0, 1.0, 41)                  # channels per sample
    accels = np.linspace(-0.04, 0.04, 41)                # curvature grid straddling zero
    rho, dhat, ahat = curved_peak(W, drifts, accels)
    # noise floor: same search on the same window with the carrier column blanked
    Wn = W.copy(); Wn[:, F // 2 - 3:F // 2 + 4] = np.median(Wn)
    rho_noise, _, _ = curved_peak(Wn, drifts, accels)
    # physical units
    df_hz = abs(foff) * 1e6                               # Hz per channel
    dhat_hzs = dhat * df_hz / tsamp                       # Hz/s
    ahat_hzs2 = ahat * df_hz / (tsamp * tsamp)            # Hz/s^2
    a_grid_span = float(accels.max())
    out = {"carrier_mhz": carrier_mhz, "T": T, "F": F, "rho_carrier": rho, "rho_noise_floor": rho_noise,
           "d_hat_chan_per_samp": dhat, "a_hat_chan_per_samp2": ahat,
           "d_hat_hz_s": dhat_hzs, "a_hat_hz_s2": ahat_hzs2,
           "a_hat_over_grid_span": abs(ahat) / a_grid_span, "tsamp_s": tsamp, "df_hz": df_hz}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/voyager_control.json", "w"), indent=1)
    print(f"[voyager] carrier recovered: rho={rho:.1f} (noise-floor rho={rho_noise:.1f})", flush=True)
    print(f"[voyager] best-fit drift d_hat={dhat_hzs:+.3f} Hz/s, curvature a_hat={ahat_hzs2:+.2e} Hz/s^2", flush=True)
    print(f"[voyager] |a_hat| is {100*abs(ahat)/a_grid_span:.1f}% of the searched grid span -> "
          f"{'IN the low-acceleration manifold' if abs(ahat)/a_grid_span < 0.15 else 'OUTSIDE the manifold'}", flush=True)
    print("VOYAGER_CONTROL_DONE", flush=True)


if __name__ == "__main__":
    main()
