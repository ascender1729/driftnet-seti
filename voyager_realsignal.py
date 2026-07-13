#!/usr/bin/env python3
"""Real-signal positive control for Part 1 (DriftNet, the linear detector).

Part 2 already showed the curved detector recovers the real Voyager carrier. This does the analogous
control for the DriftNet classifier: run the learned decision on the GENUINE Voyager-1 carrier in a real
ON observation (not a setigen injection) and show it fires as 'signal', while a carrier-blanked copy of
the same window scores as 'noise'. This partially answers "would the learned decision work on a real
narrowband signal, not just synthetic ones?"

  voyager_realsignal.py --cad-dir real_data/cadence
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cp4_cadence_roc as CP4
from driftnet_v2 import CLS


def carrier_window(cad_dir, fchans=512):
    """Load a real Voyager ON file, find the carrier channel (peak of the time-median spectrum), and
    return a [T, fchans] window centered on it, read via the SAME loader used for training frames."""
    import h5py
    fs = sorted(f.replace("\\", "/") for f in glob.glob(f"{cad_dir}/**/*VOYAGER*.0000.h5", recursive=True))
    if not fs:
        fs = sorted(f.replace("\\", "/") for f in glob.glob(f"{cad_dir}/**/*.0000.h5", recursive=True))
    on = fs[0]
    with h5py.File(on, "r") as hf:
        data = hf["data"][:, 0, :].astype(np.float64)
    prof = np.median(data, axis=0); prof = prof - np.median(prof)
    cpk = int(np.argmax(prof))
    fr = CP4.C3.bg_frame(on, cpk, fchans=fchans)          # same loader/orientation as train_driftnet
    W = np.asarray(fr.data, dtype=np.float32)
    return W, cpk, on


def main():
    cad_dir = sys.argv[sys.argv.index("--cad-dir") + 1] if "--cad-dir" in sys.argv else "real_data/cadence"
    CP4.C3.SKIP_PNG = True
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = CP4.train_driftnet(cad_dir, dev)                 # trained on setigen injections only
    W, cpk, on = carrier_window(cad_dir)
    T, F = W.shape
    print(f"[voyager-realsignal] real ON file={os.path.basename(on)} window T={T} F={F} carrier_chan={cpk}", flush=True)

    def score(arr):
        with torch.no_grad():
            prob = torch.softmax(net(torch.tensor(arr.astype(np.float32)[None]).to(dev))[0], 1)[0]
        return {CLS[i]: round(float(prob[i]), 4) for i in range(len(CLS))}

    real = score(W)
    Wn = W.copy(); Wn[:, F // 2 - 3:F // 2 + 4] = np.median(Wn)     # blank the carrier column
    blank = score(Wn)
    out = {"cadence": cad_dir, "on_file": os.path.basename(on), "T": T, "F": F, "carrier_chan": cpk,
           "real_carrier_scores": real, "carrier_blanked_scores": blank,
           "pred_real": max(real, key=real.get), "pred_blanked": max(blank, key=blank.get)}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/voyager_realsignal.json", "w"), indent=1)
    print(f"  real carrier   -> {out['pred_real']:6s} {real}", flush=True)
    print(f"  carrier blanked-> {out['pred_blanked']:6s} {blank}", flush=True)
    print("VOYAGER_REALSIGNAL_DONE", flush=True)


if __name__ == "__main__":
    main()
