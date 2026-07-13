#!/usr/bin/env python3
"""CP4: the matched-false-alarm, cadence-level comparison against REAL turboSETI (the referee's
top-priority experiment).

Design that makes the comparison fair: BOTH detectors run under the SAME ON-OFF spatial-persistence
filter, so the only thing that differs is the per-frame decision.
  - We build synthetic ABACAD cadences from REAL GBT off-noise windows (different windows = different
    sky positions). ON frames = indices 0,2,4; OFF = 1,3,5.
  - SIGNAL cadence: inject a faint drifting line into the 3 ON frames only (true positive; must pass
    persistence). RFI cadence: inject a drifting line into ALL 6 frames (must be vetoed by OFF presence).
    NOISE cadence: nothing injected (negative control).
  - turboSETI: FindDoppler at Enriquez's documented point (snr>=25, max_drift=2 Hz/s) on each frame,
    then find_event_pipeline(filter_threshold=3, ON first). A cadence is DETECTED if an event lands
    within tolerance of the injected frequency. This is a single operating point.
  - DriftNet: score each frame's signal-softmax; cadence DETECTED at threshold tau if all 3 ON frames
    score > tau and no OFF frame does. Sweeping tau traces a completeness-vs-false-alarm ROC.
  - Report completeness (recovered signal cadences vs injected SNR) and the false-alarm rate over the
    RFI+NOISE cadences, with the cadence as the unit, and DriftNet's completeness at turboSETI's FA.

  cp4_cadence_roc.py --cad-dir real_data/cadence --band X [--quick]
"""
import argparse, glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cp3_inject_real as C3

FCHANS, TCHANS = 512, 16
SNRS = [3, 6, 9, 12, 15, 20, 25, 30]
DRIFTS = [-0.3, -0.1, 0.1, 0.3]
TURBO_FLOOR, MAX_DRIFT = 8.0, 2.0         # FindDoppler floor (low, so hits are recorded with their SNR)
OPERATING_POINTS = [10.0, 25.0]           # Price 2020 (S/N>=10) and Enriquez 2017 (S/N>=25) cuts
FREQ_TOL_HZ = 400.0                        # event-to-injection match tolerance


def real_off_files(cad_dir):
    fs = sorted(f.replace("\\", "/") for f in glob.glob(f"{cad_dir}/**/*.0000.h5", recursive=True))
    if not fs:
        raise RuntimeError(f"no *.0000.h5 under {cad_dir}")
    return fs


def get_cadences(cad_dir):
    """A cadence = 6 sibling *.0000.h5 files in one directory, time-sorted (real ABACAD structure:
    same frequency band across all 6, ON at 0,2,4). Real RFI sits at a fixed frequency across the whole
    cadence, so ALL frames must share the band -- this is why we use the real 6-file cadences, not
    independent windows."""
    from collections import defaultdict
    groups = defaultdict(list)
    for f in real_off_files(cad_dir):
        groups[os.path.dirname(f)].append(f)
    cads = [sorted(v)[:6] for v in groups.values() if len(v) >= 6]
    if not cads:
        raise RuntimeError(f"no 6-file cadence under {cad_dir} (dirs: {list(groups)[:3]})")
    return cads


def edge_center(nch, rng):
    """A channel center in the band edges, away from the middle 20% (avoids the real Voyager carrier
    that sits near band center in the diagnostic cadence)."""
    lo_band = (int(nch * 0.12), int(nch * 0.38)); hi_band = (int(nch * 0.62), int(nch * 0.88))
    band = lo_band if rng.random() < 0.5 else hi_band
    return int(rng.integers(*band))


INJ_IDX = int(FCHANS * 0.30)              # off-center start channel: away from the DC bin (blanked by
                                          # turboSETI's blank_dc) and from the band edges (drift stays in)


def inject_line(frame, snr, drift, f_start=None, idx=INJ_IDX):
    """Inject a drifting narrowband line into a COPY of frame at the given nominal SNR. Returns
    (data[T,F], f_start_hz). Level set from the real background std (matches cp3). The start channel is
    OFF-center: injecting at the central DC bin makes turboSETI (blank_dc=True) miss the signal."""
    import setigen as stg
    fr = C3._fresh(frame)
    noise_std = float(np.std(fr.data))
    level = snr * noise_std / np.sqrt(fr.tchans)
    if f_start is None:
        f_start = fr.get_frequency(index=int(idx))
    fr.add_signal(stg.constant_path(f_start=f_start, drift_rate=drift),
                  stg.constant_t_profile(level=level),
                  stg.gaussian_f_profile(width=2 * float(fr.df)),
                  stg.constant_bp_profile(level=1))
    return np.asarray(fr.data, dtype=np.float32), float(f_start)


def write_h5(data, template_frame, path, source_name, tstart):
    """Write a [T,F] array to a blimpy-readable .h5 with ON/OFF metadata that find_event needs."""
    import setigen as stg, h5py
    fr = stg.Frame(fchans=template_frame.fchans, tchans=template_frame.tchans,
                   df=template_frame.df, dt=template_frame.dt, fch1=template_frame.fch1)
    fr.data = data.astype(np.float64)
    saver = next((getattr(fr, m) for m in ("save_h5", "save_hdf5", "save_data") if hasattr(fr, m)), None)
    if saver is None:
        raise RuntimeError(f"no setigen save method; have {[m for m in dir(fr) if 'save' in m]}")
    saver(path)                                            # blimpy-format .h5
    try:                                                   # stamp source/tstart for the persistence filter
        with h5py.File(path, "r+") as hf:
            hf["data"].attrs["source_name"] = np.bytes_(source_name)
            hf["data"].attrs["tstart"] = float(tstart)
    except Exception as e:
        print("  warn: source_name stamp failed:", str(e)[:80], flush=True)
    return path


def build_cadence(cad_files, trial_dir, snr, drift, mode, rng):
    """Write a 6-file ABACAD cadence from a REAL 6-file cadence: extract the SAME channel window from
    all 6 files (same frequency band, real different-time noise). mode in {signal, rfi, noise}.
    Returns (files, inj_freq_hz|None)."""
    import h5py
    os.makedirs(trial_dir, exist_ok=True)
    with h5py.File(cad_files[0], "r") as hf:
        nch = hf["data"].shape[2]
    center = edge_center(nch, rng)                          # SAME window for all 6 -> same freq band
    frames = [C3.bg_frame(f, center, fchans=FCHANS) for f in cad_files]
    inj_freq = frames[0].get_frequency(index=INJ_IDX) if mode != "noise" else None
    files = []
    names = ["SYNTH_ON", "SYNTH_OFFa", "SYNTH_ON", "SYNTH_OFFb", "SYNTH_ON", "SYNTH_OFFc"]
    for i in range(6):
        is_on = (i % 2 == 0)
        bg = frames[i]
        if mode == "signal" and is_on:                     # astrophysical: ON-only, drifting
            data, _ = inject_line(bg, snr, drift, f_start=inj_freq)
        elif mode == "rfi":                                # terrestrial: persistent line in EVERY frame
            data, _ = inject_line(bg, snr, drift, f_start=inj_freq)
        else:
            data = np.asarray(bg.data, dtype=np.float32)
        # closely-spaced tstart (1 s apart): the injected line sits at the SAME frequency in each ON
        # frame, so find_event's drift-corrected cross-frame match requires negligible inter-obs drift.
        p = write_h5(data, bg, f"{trial_dir}/f{i}.h5", names[i], tstart=59000.0 + i * (1.0 / 86400.0))
        files.append(p)
    return files, inj_freq


# ---------------- turboSETI branch ----------------
def turbo_detect(files, inj_freq_hz, work_dir):
    """Run FindDoppler at a low floor and find_event persistence; return the SNR of the best qualifying
    event (near the injection for a signal cadence; any event for a neg cadence), 0.0 if none. The
    operating-point cut (S/N>=10 or >=25) is applied later in aggregation, so one run traces both."""
    from turbo_seti.find_doppler.find_doppler import FindDoppler
    from turbo_seti.find_event.find_event_pipeline import find_event_pipeline
    dats = []
    for f in files:
        try:                                               # n_coarse_chan=1: treat the whole window as ONE
            FindDoppler(f, max_drift=MAX_DRIFT, snr=TURBO_FLOOR,   # coarse channel, else turboSETI chops it
                        n_coarse_chan=1, out_dir=os.path.dirname(f) + "/").search()  # into 8-ch pieces the
        except Exception as e:                             # drifting line crosses -> never integrated
            print("  turbo FindDoppler err:", str(e)[:120], flush=True)
        dats.append(os.path.splitext(f)[0] + ".dat")
    hlst, dlst = f"{work_dir}/c.lst", f"{work_dir}/d.lst"
    open(hlst, "w").write("\n".join(os.path.abspath(f) for f in files) + "\n")
    open(dlst, "w").write("\n".join(os.path.abspath(d) for d in dats if os.path.exists(d)) + "\n")
    csv = f"{work_dir}/events.csv"
    try:
        find_event_pipeline(dlst, h5_file_list_str=hlst, number_in_cadence=6,
                            filter_threshold=3, on_off_first="ON", csv_name=csv, saving=True)
    except TypeError:
        find_event_pipeline(dlst, hlst, 6, 3, "ON", csv, True)
    except Exception as e:
        print("  find_event err:", str(e)[:120], flush=True); return 0.0
    if not os.path.exists(csv):
        return 0.0
    import csv as _csv
    with open(csv) as fh:
        rows = list(_csv.DictReader(fh))
    best = 0.0
    for r in rows:
        sk = next((k for k in r if k.strip().lower() in ("snr", "maxsnr", "max_snr")), None)
        fk = next((k for k in r if "freq" in k.lower()), None)
        try:
            snr = float(r[sk]) if sk else 0.0
        except (ValueError, TypeError):
            snr = 0.0
        if inj_freq_hz is not None and fk:                 # signal cadence: require freq match
            try:
                if abs(float(r[fk]) * 1e6 - inj_freq_hz) >= FREQ_TOL_HZ:
                    continue
            except (ValueError, TypeError):
                continue
        best = max(best, snr)
    return best                                            # best persistent-event SNR (0 if none)


# ---------------- DriftNet branch ----------------
def train_driftnet(cad_dir, dev):
    """Train DriftNet on single-frame injections (drifting signal / zero-drift rfi / noise) built from
    the SAME real backgrounds, at the cadence frame size. Returns the trained net."""
    import torch, torch.nn as nn
    from driftnet_v2 import DriftNetV2, CI
    import h5py
    torch.manual_seed(0)                                   # deterministic init (was unseeded -> flaky)
    offs = real_off_files(cad_dir); rng = np.random.default_rng(0)
    X, y = [], []
    def bg():
        h5 = offs[rng.integers(len(offs))]
        with h5py.File(h5, "r") as hf:
            nch = hf["data"].shape[2]
        return C3.bg_frame(h5, edge_center(nch, rng), fchans=FCHANS)
    n = 80                                                 # BALANCED: 1 signal / 1 rfi / 1 noise per iter
    for _ in range(n):
        d = float(rng.choice([x for x in DRIFTS if abs(x) > 0]))
        arr, _ = inject_line(bg(), float(rng.choice([6, 10, 15, 25])), d); X.append(arr); y.append(CI["signal"])
        arr, _ = inject_line(bg(), float(rng.choice([6, 12, 25])), 0.0); X.append(arr); y.append(CI["rfi"])
        X.append(np.asarray(bg().data, dtype=np.float32)); y.append(CI["noise"])
    X = torch.tensor(np.stack(X)); y = torch.tensor(np.array(y))
    net = DriftNetV2(list(np.linspace(-2, 2, 17)), learn_drifts=False).to(dev)   # frozen (proven equal)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); ce = nn.CrossEntropyLoss()
    for e in range(70):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 48):
            b = perm[i:i + 48]; opt.zero_grad()
            logit, _ = net(X[b].to(dev)); ce(logit, y[b].to(dev)).backward(); opt.step()
    net.eval()
    with torch.no_grad():                                  # report train accuracy so bad convergence is visible
        pr = torch.cat([net(X[i:i+64].to(dev))[0].argmax(1).cpu() for i in range(0, len(X), 64)])
    print(f"  [driftnet] train acc={(pr==y).float().mean():.3f} (n={len(X)}, balanced 3-class)", flush=True)
    return net


def driftnet_scores(files, net, dev):
    import torch
    from blimpy import Waterfall
    sc = []
    for f in files:
        wf = Waterfall(f, load_data=True); d = np.squeeze(wf.data).astype(np.float32)
        d = np.ascontiguousarray(d[:, ::-1])               # blimpy reads back frequency-REVERSED (foff<0
        with torch.no_grad():                              # convention) vs the in-memory training frames;
            prob = torch.softmax(net(torch.tensor(d[None]).to(dev))[0], 1)[0]   # flip to match orientation
        sc.append(float(prob[0]))                          # signal-class softmax
    return sc                                              # [6] scores, indices 0..5 (ON at even)


def cadence_decision(scores, tau):
    on = [scores[i] for i in (0, 2, 4)]; off = [scores[i] for i in (1, 3, 5)]
    return all(s > tau for s in on) and all(s <= tau for s in off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cad-dir", default="real_data/cadence")
    ap.add_argument("--band", default="X")
    ap.add_argument("--out", default="results/cp4")
    ap.add_argument("--trials-per-cell", type=int, default=6)
    ap.add_argument("--neg", type=int, default=40)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    C3.SKIP_PNG = True
    os.makedirs(a.out, exist_ok=True)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    snrs = [6, 15, 30] if a.quick else SNRS
    tpc = 2 if a.quick else a.trials_per_cell
    nneg = 6 if a.quick else a.neg
    cadences = get_cadences(a.cad_dir)
    print(f"[cp4] band={a.band} dev={dev} snrs={snrs} trials/cell={tpc} neg={nneg} cadences={len(cadences)}", flush=True)
    net = train_driftnet(a.cad_dir, dev)
    rng = np.random.default_rng(7)

    records = []                                           # each: dict(mode, snr, turbo_snr, scores)
    def run_trial(mode, snr, k):
        td = f"{a.out}/trial_{mode}_{snr}_{k}"
        drift = float(rng.choice(DRIFTS))
        cad = cadences[rng.integers(len(cadences))]        # a real 6-file cadence
        files, inj = build_cadence(cad, td, snr, drift, mode, rng)
        turbo_snr = float(turbo_detect(files, inj, td))    # best persistent-event SNR (0 if none)
        scores = driftnet_scores(files, net, dev)
        records.append({"mode": mode, "snr": snr, "turbo_snr": turbo_snr, "scores": scores})
        print(f"  {mode} snr={snr} k={k}: turboSNR={turbo_snr:.1f} "
              f"driftON={[round(scores[i],2) for i in (0,2,4)]} driftOFF={[round(scores[i],2) for i in (1,3,5)]}", flush=True)

    for snr in snrs:
        for k in range(tpc):
            run_trial("signal", snr, k)
    for k in range(nneg):                                  # false-alarm population: rfi + noise
        run_trial("rfi", 20, k)
        if k < nneg // 2:
            run_trial("noise", 0, k)

    # ---- aggregate ----
    sig = [r for r in records if r["mode"] == "signal"]
    neg = [r for r in records if r["mode"] != "signal"]
    # turboSETI: detection at operating point OP iff best persistent-event SNR >= OP
    turbo = {}
    for op in OPERATING_POINTS:
        fa = float(np.mean([r["turbo_snr"] >= op for r in neg])) if neg else 0.0
        comp = {int(s): float(np.mean([r["turbo_snr"] >= op for r in sig if r["snr"] == s])) for s in snrs}
        turbo[f"op{int(op)}"] = {"false_alarm": fa, "completeness_by_snr": comp,
                                 "completeness_overall": float(np.mean([r["turbo_snr"] >= op for r in sig])) if sig else 0.0}
    # DriftNet ROC over tau
    taus = list(np.linspace(0.02, 0.98, 25))
    roc = []
    for tau in taus:
        comp = float(np.mean([cadence_decision(r["scores"], tau) for r in sig])) if sig else 0.0
        fa = float(np.mean([cadence_decision(r["scores"], tau) for r in neg])) if neg else 0.0
        roc.append({"tau": round(float(tau), 3), "completeness": comp, "fa": fa})
    # DriftNet completeness at each turboSETI operating point's FA (best comp with fa <= turbo fa)
    dn_at = {}
    for op in OPERATING_POINTS:
        tfa = turbo[f"op{int(op)}"]["false_alarm"]
        feas = [p for p in roc if p["fa"] <= tfa + 1e-9] or roc
        best = max(feas, key=lambda p: p["completeness"])
        by_snr = {int(s): float(np.mean([cadence_decision(r["scores"], best["tau"]) for r in sig if r["snr"] == s])) for s in snrs}
        dn_at[f"op{int(op)}"] = {"matched_fa": tfa, "tau": best["tau"], "fa": best["fa"],
                                 "completeness_overall": best["completeness"], "completeness_by_snr": by_snr}

    out = {"band": a.band, "n_signal": len(sig), "n_neg": len(neg), "snrs": snrs,
           "turbo_floor": TURBO_FLOOR, "max_drift": MAX_DRIFT, "operating_points": OPERATING_POINTS,
           "turbo": turbo, "driftnet": {"roc": roc, "at_turbo_fa": dn_at}, "records": records}
    json.dump(out, open(f"{a.out}/cp4_results_{a.band}.json", "w"), indent=1)
    print("=== CP4 SUMMARY ===", flush=True)
    for op in OPERATING_POINTS:
        t = turbo[f"op{int(op)}"]; d = dn_at[f"op{int(op)}"]
        print(f" OP S/N>={int(op)}: turboSETI FA={t['false_alarm']:.3f} comp={t['completeness_overall']:.3f} "
              f"{t['completeness_by_snr']} | DriftNet@FA<={d['matched_fa']:.3f} comp={d['completeness_overall']:.3f} {d['completeness_by_snr']}", flush=True)
    print("CP4_DONE", flush=True)


if __name__ == "__main__":
    main()
