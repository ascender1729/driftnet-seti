#!/usr/bin/env python3
"""Turn the baseline + VLM result jsons into the paper's numbers and figures.

Reads whatever exists of: baseline_dedoppler.json, baseline_cnn.json, vlm_zeroshot.json,
vlm_fewshot.json. Produces:
  results_summary.json   - per method: overall acc, high/low-SNR acc, per-class recall
  figures/acc_vs_snr.png - accuracy vs SNR, all methods (the headline honest-negative-result axis)
  figures/confusion_<m>.png - confusion matrix per method
  figures/rfi_recall.png - per-method RFI recall (the 'do VLMs see vertical RFI' finding)

Usage: analyze_results.py
"""
import json, os, glob
import numpy as np

CLASSES = ("signal", "rfi", "noise")
os.makedirs("figures", exist_ok=True)


def load_methods():
    """Return {display_name: {id: {true,pred,snr_db}}} from every result file present."""
    methods = {}
    for f, name in [("baseline_dedoppler.json", "dedoppler"), ("baseline_cnn.json", "cnn")]:
        if os.path.exists(f):
            methods[name] = json.load(open(f))["per"]
    for f, tag in [("vlm_zeroshot.json", "0shot"), ("vlm_fewshot.json", "fewshot")]:
        if os.path.exists(f):
            d = json.load(open(f))
            for model_id, res in d.items():
                short = model_id.split(".")[-1].split(":")[0].replace("-20251001", "")[:22]
                methods[f"{short}/{tag}"] = res["per"]
    return methods


def metrics(per):
    rows = list(per.values())
    def acc(R): return sum(x["true"] == x["pred"] for x in R) / max(len(R), 1)
    hi = [x for x in rows if x["snr_db"] >= 12 and x["true"] != "noise"]
    lo = [x for x in rows if 0 < x["snr_db"] < 6]
    recall = {}
    for c in CLASSES:
        Rc = [x for x in rows if x["true"] == c]
        recall[c] = round(acc(Rc), 3)
    return {"overall": round(acc(rows), 3), "high_snr": round(acc(hi), 3),
            "low_snr": round(acc(lo), 3), "recall": recall, "n": len(rows)}


def confusion(per):
    idx = {c: i for i, c in enumerate(CLASSES)}
    M = np.zeros((3, 3), int)
    for x in per.values():
        M[idx[x["true"]], idx[x["pred"]]] += 1
    return M


def plot_acc_vs_snr(methods):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    bins = [(3, 6), (6, 9), (9, 12), (12, 15), (15, 21)]
    centers = [(a + b) / 2 for a, b in bins]
    plt.figure(figsize=(7, 5))
    for name, per in methods.items():
        rows = [x for x in per.values() if x["true"] != "noise"]   # SNR only meaningful for signal/rfi
        ys = []
        for a, b in bins:
            R = [x for x in rows if a <= x["snr_db"] < b]
            ys.append(sum(x["true"] == x["pred"] for x in R) / len(R) if R else np.nan)
        plt.plot(centers, ys, marker="o", label=name, alpha=0.8)
    plt.xlabel("injected SNR (dB)"); plt.ylabel("accuracy (signal+rfi)")
    plt.title("Accuracy vs SNR - VLMs vs classical/CNN baselines")
    plt.legend(fontsize=7, loc="lower right"); plt.grid(alpha=0.3); plt.ylim(0, 1.02)
    plt.tight_layout(); plt.savefig("figures/acc_vs_snr.png", dpi=110); plt.close()


def plot_confusions(methods):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for name, per in methods.items():
        M = confusion(per)
        fig, ax = plt.subplots(figsize=(3.2, 3))
        ax.imshow(M, cmap="Blues")
        ax.set_xticks(range(3), CLASSES, fontsize=8); ax.set_yticks(range(3), CLASSES, fontsize=8)
        ax.set_xlabel("pred"); ax.set_ylabel("true"); ax.set_title(name, fontsize=8)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, M[i, j], ha="center", va="center",
                        color="white" if M[i, j] > M.max() / 2 else "black", fontsize=9)
        plt.tight_layout(); plt.savefig(f"figures/confusion_{name.replace('/', '_')}.png", dpi=110); plt.close()


def plot_rfi_recall(summary):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = list(summary); rfi = [summary[n]["recall"]["rfi"] for n in names]
    plt.figure(figsize=(8, 4)); plt.barh(names, rfi, color="#c0392b")
    plt.xlabel("RFI recall (fraction of vertical-line RFI correctly labeled)")
    plt.title("Do the methods recognize constant-frequency RFI?"); plt.xlim(0, 1)
    plt.tight_layout(); plt.savefig("figures/rfi_recall.png", dpi=110); plt.close()


def main():
    methods = load_methods()
    if not methods:
        print("no result files yet"); return
    summary = {name: metrics(per) for name, per in methods.items()}
    json.dump(summary, open("results_summary.json", "w"), indent=1)
    print(f"{'method':32s} {'overall':>8s} {'hiSNR':>7s} {'loSNR':>7s} {'rfiRec':>7s}")
    for name, m in summary.items():
        print(f"{name:32s} {m['overall']:8.3f} {m['high_snr']:7.3f} {m['low_snr']:7.3f} {m['recall']['rfi']:7.3f}")
    plot_acc_vs_snr(methods); plot_confusions(methods); plot_rfi_recall(summary)
    print(f"\nfigures -> figures/  ({len(glob.glob('figures/*.png'))} png)")


if __name__ == "__main__":
    main()
