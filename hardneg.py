#!/usr/bin/env python3
"""Hard-negative cadence evaluation (the referee's "natural strengthening experiment").

The main cp4 false-alarm floor rests on easy negatives (fully persistent RFI, vetoed by OFF presence;
pure noise). This adds the negatives that are the true source of survey false alarms: RFI that lands in
the ON pointings but not the OFF pointings by chance, mimicking the ON-present/OFF-absent cadence of a
real signal. Three classes, false-alarm rate of BOTH detectors on each:

  vert_on : VERTICAL (zero-drift) line in the 3 ON frames only. Mimics the ON-OFF cadence but is
            non-drifting terrestrial RFI. turboSETI's persistence filter sees only presence/absence and
            should flag it; DriftNet's orientation-aware readout should reject it (it reads a vertical
            line as rfi). This is where the learned decision can add a real, new capability.
  drift_on: DRIFTING line in the ON frames only. In a single cadence this is genuinely indistinguishable
            from an astrophysical signal, so we EXPECT both to flag it -- the honest limit that motivates
            multi-cadence ephemeris follow-up.
  partial : drifting line in a broken-persistence subset (2 ON + 1 OFF). Both persistence filters veto.

Performance: turboSETI (FindDoppler + find_event) is the CPU bottleneck and is embarrassingly parallel
across trials, so we run the turboSETI phase in a multiprocessing Pool over all trials, then score with
DriftNet on the GPU in the main process. A false alarm is ANY persistent event turboSETI reports
(inj_freq=None), regardless of frequency -- a drifting line's event drifts away from its start channel.

  hardneg.py --cad-dir real_data/cadence --band X [--quick]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")      # 1 BLAS thread/worker: we parallelize across trials
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cp4_cadence_roc as CP4

TAU = 0.5                                  # DriftNet operating point (inside the FA=0 plateau of cp4)
CLASSES = {
    "vert_on":  {"frames": [0, 2, 4], "drift": 0.0,  "desc": "vertical RFI, ON-only"},
    "drift_on": {"frames": [0, 2, 4], "drift": None, "desc": "drifting RFI, ON-only (ambiguous)"},
    "partial":  {"frames": [0, 2, 3], "drift": None, "desc": "drifting RFI, broken persistence"},
}


def build_hard(cad_files, trial_dir, snr, drift, inject_frames, rng):
    """Write a 6-file cadence with a line injected into exactly `inject_frames` (drift=0 -> vertical)."""
    import h5py
    os.makedirs(trial_dir, exist_ok=True)
    with h5py.File(cad_files[0], "r") as hf:
        nch = hf["data"].shape[2]
    center = CP4.edge_center(nch, rng)
    frames = [CP4.C3.bg_frame(f, center, fchans=CP4.FCHANS) for f in cad_files]
    inj_freq = frames[0].get_frequency(index=CP4.INJ_IDX)
    names = ["SYNTH_ON", "SYNTH_OFFa", "SYNTH_ON", "SYNTH_OFFb", "SYNTH_ON", "SYNTH_OFFc"]
    files = []
    for i in range(6):
        bg = frames[i]
        if i in inject_frames:
            data, _ = CP4.inject_line(bg, snr, drift, f_start=inj_freq)
        else:
            data = np.asarray(bg.data, dtype=np.float32)
        p = CP4.write_h5(data, bg, f"{trial_dir}/f{i}.h5", names[i], tstart=59000.0 + i * (1.0 / 86400.0))
        files.append(p)
    return files, inj_freq


def turbo_worker(arg):
    """CPU-only: build a hard-neg cadence and run turboSETI. inj_freq=None so ANY persistent event counts
    as a false alarm (a drifting line's event drifts off its start channel). Deterministic per-trial seed."""
    idx, td, cad_files, snr, drift, frames, seed = arg
    CP4.C3.SKIP_PNG = True
    try:
        rng = np.random.default_rng(seed)
        files, _ = build_hard(cad_files, td, snr, drift, frames, rng)
        turbo_snr = float(CP4.turbo_detect(files, None, td))
        return (idx, turbo_snr, files, "")
    except Exception as e:
        return (idx, -1.0, None, str(e)[:150])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cad-dir", default="real_data/cadence")
    ap.add_argument("--band", default="X")
    ap.add_argument("--out", default="results/hardneg")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    CP4.C3.SKIP_PNG = True
    os.makedirs(a.out, exist_ok=True)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ntr = 4 if a.quick else a.trials
    snrs = [15, 25]
    nworkers = a.workers or max(2, (os.cpu_count() or 4) - 2)
    cadences = CP4.get_cadences(a.cad_dir)
    print(f"[hardneg] band={a.band} dev={dev} trials/class/snr={ntr} snrs={snrs} cadences={len(cadences)} workers={nworkers}", flush=True)

    # ---- plan every trial deterministically (main-process RNG for reproducibility) ----
    rng = np.random.default_rng(11)
    nonzero = [d for d in CP4.DRIFTS if abs(d) > 0]
    plan = []
    idx = 0
    for cls, spec in CLASSES.items():
        for snr in snrs:
            for k in range(ntr):
                drift = spec["drift"] if spec["drift"] is not None else float(rng.choice(nonzero))
                cad = cadences[int(rng.integers(len(cadences)))]
                td = f"{a.out}/t_{a.band}_{cls}_{snr}_{k}"
                seed = int(rng.integers(1 << 31))
                plan.append({"idx": idx, "cls": cls, "snr": snr, "drift": drift, "td": td,
                             "frames": spec["frames"], "cad": cad, "seed": seed})
                idx += 1

    # ---- phase 1: turboSETI in parallel across cores (the CPU bottleneck) ----
    args = [(p["idx"], p["td"], p["cad"], p["snr"], p["drift"], p["frames"], p["seed"]) for p in plan]
    print(f"[hardneg] running {len(args)} turboSETI trials on {nworkers} cores ...", flush=True)
    with Pool(nworkers) as pool:
        turbo_out = pool.map(turbo_worker, args)      # ordered; turbo_out[i] <-> plan[i]

    # ---- phase 2: DriftNet scoring on GPU (main process, fast) ----
    net = CP4.train_driftnet(a.cad_dir, dev)
    records = []
    for p, (i2, turbo_snr, files, err) in zip(plan, turbo_out):
        if files is None:
            print(f"  {p['cls']} snr={p['snr']} ERR {err}", flush=True); continue
        scores = CP4.driftnet_scores(files, net, dev)
        dn_flag = bool(CP4.cadence_decision(scores, TAU))
        records.append({"cls": p["cls"], "snr": p["snr"], "drift": p["drift"],
                        "turbo_snr": turbo_snr, "dn_flag": dn_flag, "scores": scores})
        on = [round(scores[j], 2) for j in (0, 2, 4)]; off = [round(scores[j], 2) for j in (1, 3, 5)]
        print(f"  {p['cls']:8s} snr={p['snr']} : turboSNR={turbo_snr:5.1f} dnFlag={dn_flag} ON={on} OFF={off}", flush=True)

    # ---- aggregate: false-alarm rate per class ----
    agg = {}
    for cls in CLASSES:
        rs = [r for r in records if r["cls"] == cls]
        n = len(rs)
        if n == 0:
            continue
        agg[cls] = {
            "desc": CLASSES[cls]["desc"], "n": n,
            "turbo_fa_op25": float(np.mean([r["turbo_snr"] >= 25.0 for r in rs])),
            "turbo_fa_op10": float(np.mean([r["turbo_snr"] >= 10.0 for r in rs])),
            "driftnet_fa_tau0.5": float(np.mean([r["dn_flag"] for r in rs])),
            "dn_fa_by_tau": {str(round(t, 2)): float(np.mean([CP4.cadence_decision(r["scores"], t) for r in rs]))
                             for t in (0.3, 0.5, 0.7)},
            "mean_ON_signalsoftmax": float(np.mean([np.mean([r["scores"][i] for i in (0, 2, 4)]) for r in rs])),
            "mean_OFF_signalsoftmax": float(np.mean([np.mean([r["scores"][i] for i in (1, 3, 5)]) for r in rs])),
        }
    out = {"band": a.band, "tau": TAU, "snrs": snrs, "workers": nworkers, "n_total": len(records),
           "classes": agg, "records": records}
    json.dump(out, open(f"{a.out}/hardneg_results_{a.band}.json", "w"), indent=1)
    print("=== HARDNEG SUMMARY (false-alarm rate; lower is better) ===", flush=True)
    for cls, d in agg.items():
        print(f"  {cls:8s} ({d['desc']}): turboSETI FA@25={d['turbo_fa_op25']:.2f} FA@10={d['turbo_fa_op10']:.2f} "
              f"| DriftNet FA@tau0.5={d['driftnet_fa_tau0.5']:.2f}  [ON={d['mean_ON_signalsoftmax']:.2f} OFF={d['mean_OFF_signalsoftmax']:.2f}]", flush=True)
    print("HARDNEG_DONE", flush=True)


if __name__ == "__main__":
    main()
