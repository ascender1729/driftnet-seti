#!/usr/bin/env python3
"""Headline results figures from results/comprehensive_results.json (emitted by driftnet_comprehensive.py).

Two panels, publication style, matching make_figures.py palette:
  (1) accuracy per variant, in-distribution (Voyager X-band) vs OOD mean (Enriquez L-band cadences),
      grouped bars with +-SD error bars -> the cross-band robustness story.
  (2) detection metrics: false-alarm rate and ROC-AUC per variant (in-dist + OOD), the SETI-practitioner axis.
The classical fixed-threshold baseline is drawn in a distinct hue so the "learned rule transfers, threshold
does not" gap is visible at a glance.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "mathtext.fontset": "stixsans", "axes.linewidth": 0.7,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})

VARIANTS = ["full", "fixed_kernel", "rho_only", "raw_cnn", "classical"]
LABEL = {"full": "DriftNet\n(full)", "fixed_kernel": "frozen\nkernels", "rho_only": r"$\rho$-only",
         "raw_cnn": "raw-pixel\nCNN", "classical": "classical\nthreshold"}
IN_C, OOD_C, CLASSIC = "#5b7fa6", "#b8894b", "#c0392b"


def ood_mean_sd(d, key, variant, field=None):
    """Mean over all ood_* testsets for a variant. field=None -> acc_mean_sd [mean,sd]; else det field."""
    oods = [t for t in d[key][variant] if t.startswith("ood_")]
    if field is None:
        ms = [d[key][variant][t][0] for t in oods]; sd = [d[key][variant][t][1] for t in oods]
        return (float(np.mean(ms)) if ms else float("nan"), float(np.mean(sd)) if sd else 0.0)
    vs = [d[key][variant][t][field] for t in oods if d[key][variant][t].get(field) == d[key][variant][t].get(field)]
    return (float(np.mean(vs)) if vs else float("nan"), 0.0)


def main():
    p = "results/comprehensive_results.json"
    if not os.path.exists(p):
        print("no results yet:", p); return
    d = json.load(open(p))
    acc = d["acc_mean_sd"]; det = d.get("det_mean_sd", {})
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0))

    # ---- panel (a): accuracy, in-dist vs OOD ----
    x = np.arange(len(VARIANTS)); w = 0.38
    in_m = [acc[v]["in_dist"][0] for v in VARIANTS]; in_s = [acc[v]["in_dist"][1] for v in VARIANTS]
    od = [ood_mean_sd(d, "acc_mean_sd", v) for v in VARIANTS]
    od_m = [m for m, _ in od]; od_s = [s for _, s in od]
    ax[0].bar(x - w / 2, in_m, w, yerr=in_s, capsize=3, color=IN_C, label="in-distribution (X-band)")
    ax[0].bar(x + w / 2, od_m, w, yerr=od_s, capsize=3, color=OOD_C, label="OOD mean (L-band cadences)")
    # majority-class chance: the injection set is 2:1:1 (signal:rfi:noise), so always-"signal" scores 0.5, not 1/3
    ax[0].axhline(0.5, ls=":", lw=0.9, color="#999"); ax[0].text(len(VARIANTS) - 0.6, 0.515, "majority-class chance", fontsize=6.5, color="#999")
    ax[0].set_xticks(x); ax[0].set_xticklabels([LABEL[v] for v in VARIANTS], fontsize=7.5)
    ax[0].set_ylabel("3-class accuracy"); ax[0].set_ylim(0, 1.02); ax[0].set_title("(a) cross-band robustness", fontsize=9.5)
    ax[0].legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    for xi, (m, s) in zip(x - w / 2, zip(in_m, in_s)): ax[0].text(xi, m + s + 0.01, f"{m:.2f}", ha="center", fontsize=6.3)
    for xi, (m, s) in zip(x + w / 2, zip(od_m, od_s)): ax[0].text(xi, m + s + 0.01, f"{m:.2f}", ha="center", fontsize=6.3)

    # ---- panel (b): detection metrics (FAR + AUC), in-dist ----
    if det:
        far_in = [det[v]["in_dist"]["far"] for v in VARIANTS]
        far_od = [ood_mean_sd(d, "det_mean_sd", v, "far")[0] for v in VARIANTS]
        auc_in = [det[v]["in_dist"]["auc"] for v in VARIANTS]
        auc_od = [ood_mean_sd(d, "det_mean_sd", v, "auc")[0] for v in VARIANTS]
        ax[1].bar(x - w / 2, far_in, w, color=IN_C, label="FAR in-dist")
        ax[1].bar(x + w / 2, far_od, w, color=OOD_C, label="FAR OOD")
        ax[1].set_xticks(x); ax[1].set_xticklabels([LABEL[v] for v in VARIANTS], fontsize=7.5)
        ax[1].set_ylabel("false-alarm rate"); ax[1].set_title("(b) false alarms + ROC-AUC", fontsize=9.5)
        axb = ax[1].twinx()
        axb.plot(x - w / 2, auc_in, "o-", color="#2c3e50", ms=4, lw=1.2, label="AUC in-dist")
        axb.plot(x + w / 2, auc_od, "s--", color="#7a6b99", ms=4, lw=1.2, label="AUC OOD")
        axb.set_ylabel("ROC-AUC (signal vs rest)"); axb.set_ylim(0.4, 1.02)
        h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = axb.get_legend_handles_labels()
        ax[1].legend(h1 + h2, l1 + l2, fontsize=6.8, loc="upper right", framealpha=0.9)
    else:
        ax[1].text(0.5, 0.5, "det_mean_sd absent", ha="center", transform=ax[1].transAxes)

    plt.tight_layout()
    plt.savefig("tex/figs/results.png", dpi=200); plt.savefig("tex/figs/results.pdf"); plt.close()
    print("wrote results.png + .pdf")
    # also dump a compact LaTeX-ready summary line for the paper
    mc = d.get("mcnemar_bh", {}).get("in_dist", {})
    print("in_dist acc:", {v: acc[v]["in_dist"][0] for v in VARIANTS})
    print("OOD mean acc:", {v: round(ood_mean_sd(d, 'acc_mean_sd', v)[0], 3) for v in VARIANTS})
    print("McNemar BH (in_dist):", mc)


if __name__ == "__main__":
    main()
