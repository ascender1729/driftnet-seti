#!/usr/bin/env python3
"""CP1: turboSETI on a REAL BL ON-OFF cadence + cut labeled real waterfalls.

Runs the field-native detector on its home turf (a 6-file ABACAD cadence), applies the ON-OFF
spatial-persistence filter (find_event_pipeline), and cuts real waterfall panels around each event
so we get REAL labeled images for the VLM benchmark:
  - an event that persists in all ON panels and is absent in OFF = a real technosignature-like SIGNAL
  - the same event's OFF panels = NOISE (signal absent) -> real negative examples
  - hits that appear in OFF panels too = RFI-like

Honest note: this is the Voyager diagnostic cadence (same source labeled throughout), so we do NOT
assume the ON-OFF structure - we MEASURE it and report what find_event_pipeline actually returns.

Usage: cp1_real_cadence.py         (assumes real_data/cadence/*.h5 present)
"""
import glob, json, os
import numpy as np

CAD = "real_data/cadence"
OUT = "real_data/cp1_out"
os.makedirs(OUT, exist_ok=True)


def cadence_files():
    # forward slashes only - turbo_seti splits paths on '/' and leaks a 'cadence\\' component
    # from glob's native backslashes on Windows otherwise.
    fs = sorted(f.replace("\\", "/") for f in glob.glob(f"{CAD}/*.rawspec.0000.h5"))
    assert len(fs) == 6, f"expected 6 cadence files, found {len(fs)}"
    return fs


def run_turbo(files):
    from turbo_seti.find_doppler.find_doppler import FindDoppler
    dats = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        dat = f"{CAD}/{base}.dat"                          # write .dat next to the .h5 (turbo_seti
        if not os.path.exists(dat):                        # mangles out_dir when input is in a subdir)
            FindDoppler(f, max_drift=4, snr=10, out_dir=CAD + "/").search()
        dats.append(dat)
        print(f"  turbo: {os.path.basename(dat)} ({sum(1 for l in open(dat) if l.strip() and not l.startswith('#'))} hits)")
    return dats


def find_events(files, dats):
    """ON-OFF persistence filter. Defensive against kwarg drift across turbo_seti versions."""
    from turbo_seti.find_event.find_event_pipeline import find_event_pipeline
    hlst, dlst = f"{OUT}/cadence.lst", f"{OUT}/dat_files.lst"
    open(hlst, "w").write("\n".join(os.path.abspath(f) for f in files) + "\n")
    open(dlst, "w").write("\n".join(os.path.abspath(d) for d in dats) + "\n")
    results = {}
    for thr in (1, 2, 3):                                   # 1=any ON; 3=all ON & no OFF (strictest)
        csv = f"{OUT}/events_f{thr}.csv"
        try:
            find_event_pipeline(dlst, h5_file_list_str=hlst, number_in_cadence=6,
                                filter_threshold=thr, on_off_first="ON", csv_name=csv, saving=True)
        except TypeError:                                   # older/newer signature
            find_event_pipeline(dlst, hlst, 6, thr, "ON", csv, True)
        n = (sum(1 for _ in open(csv)) - 1) if os.path.exists(csv) else 0
        results[thr] = {"csv": csv, "n_events": max(n, 0)}
        print(f"  filter_threshold={thr}: {results[thr]['n_events']} events -> {csv}")
    return results


def cut_panels(files, freq_mhz, half_hz=1500, tag="event"):
    """Cut a small freq window around freq_mhz from each of the 6 cadence panels; render each and a
    stacked montage (the ON-OFF cadence view). Returns per-panel PNG paths."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from blimpy import Waterfall
    half = half_hz / 1e6
    panels = []
    fig, axes = plt.subplots(6, 1, figsize=(5, 8))
    for i, (f, ax) in enumerate(zip(files, axes)):
        wf = Waterfall(f, f_start=freq_mhz - half, f_stop=freq_mhz + half)
        d = np.squeeze(wf.data).astype(np.float32)
        a = (d - d.min()) / (d.max() - d.min() + 1e-9)
        p = f"{OUT}/{tag}_panel{i}_{'ON' if i % 2 == 0 else 'OFF'}.png"
        plt.imsave(p, a, cmap="gray", origin="lower"); panels.append(p)
        ax.imshow(a, aspect="auto", cmap="viridis", origin="lower")
        ax.set_ylabel(f"{'ON' if i%2==0 else 'OFF'}", rotation=0, ha="right", va="center", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_title(f"Real cadence around {freq_mhz:.5f} MHz (ABACAD)", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/{tag}_cadence_montage.png", dpi=100); plt.close()
    return panels


def main():
    files = cadence_files()
    print("=== turboSETI on each cadence file ===")
    dats = run_turbo(files)
    print("=== ON-OFF persistence filter ===")
    ev = find_events(files, dats)
    # cut panels around the Voyager carrier (known ~8419.297 MHz) as the worked example
    print("=== cut real cadence panels around the carrier ===")
    panels = cut_panels(files, 8419.29703, tag="carrier")
    summary = {"cadence_files": [os.path.basename(f) for f in files],
               "events_by_threshold": {k: v["n_events"] for k, v in ev.items()},
               "carrier_panels": [os.path.basename(p) for p in panels]}
    json.dump(summary, open(f"{OUT}/cp1_summary.json", "w"), indent=1)
    print("\nCP1 summary:", json.dumps(summary["events_by_threshold"]))
    print(f"-> {OUT}/cp1_summary.json + carrier_cadence_montage.png")


if __name__ == "__main__":
    main()
