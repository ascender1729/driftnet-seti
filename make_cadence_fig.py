#!/usr/bin/env python3
"""Cadence-level completeness figure: DriftNet vs REAL turboSETI at matched (zero) false-alarm, as a
function of injected SNR, for both bands. Reads results/cp4_results_{X,L}.json (from cp4_cadence_roc.py)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "mathtext.fontset": "stixsans", "axes.linewidth": 0.7,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})

DN, T25, T10 = "#2c6fbb", "#c0392b", "#d98a3d"           # DriftNet, turboSETI S/N>=25, turboSETI S/N>=10


def curve(d, key, op):
    by = d[key]["at_turbo_fa"][op]["completeness_by_snr"] if key == "driftnet" else d["turbo"][op]["completeness_by_snr"]
    snrs = sorted(int(s) for s in by)
    return snrs, [by[str(s)] if str(s) in by else by[s] for s in snrs]


def panel(ax, path, band):
    d = json.load(open(path))
    s, dn = curve(d, "driftnet", "op25")                  # DriftNet completeness identical across OPs (its own tau)
    _, t25 = curve(d, "turbo", "op25"); _, t10 = curve(d, "turbo", "op10")
    ax.plot(s, dn, "o-", color=DN, lw=2, ms=5, label="DriftNet (matched FA)")
    ax.plot(s, t10, "s--", color=T10, lw=1.6, ms=4, label=r"turboSETI, S/N$\geq$10 (Price 2020)")
    ax.plot(s, t25, "^--", color=T25, lw=1.6, ms=4, label=r"turboSETI, S/N$\geq$25 (Enriquez 2017)")
    tfa = d["turbo"]["op25"]["false_alarm"]; dfa = d["driftnet"]["at_turbo_fa"]["op25"]["fa"]
    ax.set_xlabel("injected SNR"); ax.set_ylabel("cadence completeness"); ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"({'a' if band=='X' else 'b'}) {band}-band  (all detectors at FA$=${max(tfa,dfa):.2f})", fontsize=10)
    ax.legend(fontsize=7.4, loc="lower right", framealpha=0.92); ax.grid(alpha=0.25, lw=0.5)


def main():
    if not (os.path.exists("results/cp4_results_X.json") and os.path.exists("results/cp4_results_L.json")):
        print("missing cp4 result json"); return
    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.0))
    panel(ax[0], "results/cp4_results_X.json", "X")
    panel(ax[1], "results/cp4_results_L.json", "L")
    plt.tight_layout()
    plt.savefig("tex/figs/cadence_roc.png", dpi=200); plt.savefig("tex/figs/cadence_roc.pdf"); plt.close()
    print("wrote cadence_roc.png + .pdf")
    for b in ("X", "L"):
        d = json.load(open(f"results/cp4_results_{b}.json"))
        print(f"{b}: turbo op25 comp={d['turbo']['op25']['completeness_overall']:.3f} "
              f"op10={d['turbo']['op10']['completeness_overall']:.3f} | "
              f"DriftNet={d['driftnet']['at_turbo_fa']['op25']['completeness_overall']:.3f}")


if __name__ == "__main__":
    main()
