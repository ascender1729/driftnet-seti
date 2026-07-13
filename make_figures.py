#!/usr/bin/env python3
"""Generate the paper's schematic artifacts: (1) the DriftNet architecture diagram with the ablation
branches marked, and (2) a concept figure showing a drifting line concentrating into a peak in the
de-Doppler drift spectrum. Pure matplotlib, no external tools."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
import matplotlib.colors as mcolors
import os
os.makedirs("tex/figs", exist_ok=True)

plt.rcParams.update({                                    # embed real fonts (no Type-3), muted look
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "stixsans", "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
# one desaturated hue per block TYPE; orange reserved ONLY for the trainable-ablation dimension
PAL = {"input": ("#dce6f0", "#5b7ca6"), "transform": ("#cde6e1", "#3e8e7e"),
       "cnn": ("#d6dbe0", "#5a6673"), "mlp": ("#f0e3c8", "#b8894b"),
       "fuse": ("#e6e0ec", "#7a6b99")}
LEARN = "#e07a2f"; FROZEN = "#b7bcc2"; MAIN = "#3a3f45"; OPT = "#9aa0a6"
CHIP = {"signal": "#4c8c64", "rfi": "#c6803b", "noise": "#8a9199"}


def box(ax, xy, w, h, text, fc="#eaf1fb", ec="#2c3e50", fs=8.5):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                linewidth=1.2, facecolor=fc, edgecolor=ec))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, ec="#2c3e50"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.2, color=ec))


def _thumbs():
    """Real arrays for the inline thumbnails: a strong drifting line so R shows a FOCUSED blob."""
    rng = np.random.default_rng(1); T, F = 64, 256
    P = rng.chisquare(2, (T, F)) * 3.0
    d_true = 0.9
    for t in range(T):
        f = 60 + int(round(d_true * t)); P[t, f:f + 2] += 26        # strong line -> clean concentration
    drifts = np.linspace(-1.5, 1.5, 61); R = np.zeros((len(drifts), F))
    Pd = P - np.median(P, axis=0, keepdims=True)
    for k, d in enumerate(drifts):
        for t in range(T): R[k] += np.roll(Pd[t], -int(round(d * t)))
    return P, R, drifts, R.max(axis=1)


def _shade(c, f):
    r, g, b = mcolors.to_rgb(c)
    return tuple(min(1, max(0, v * f + (0.06 if f > 1 else 0))) for v in (r, g, b))


def architecture():
    # DRAFTS-style 2.5D pipeline: real data as tilted image planes, network layers as isometric 3D
    # slabs, a single green flow arrow left-to-right, tensor dims at each base. The rho/MLP readout is a
    # lower parallel track that merges into the classifier. Vector PDF + PNG.
    P, R, drifts, rho = _thumbs()
    fig = plt.figure(figsize=(14.0, 5.6))
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
    FLOW = "#2f9e7f"; YC = 0.60                          # green flow color; main-stream centerline
    DX, DY = 0.010, 0.055                               # isometric depth (x,y), tuned to the 14x5.6 aspect

    def iso(x, y, w, h, fc, ec, lw=1.4, z=3):           # a 3D slab: front + top + right faces
        bg.add_patch(Polygon([(x, y + h), (x + w, y + h), (x + w + DX, y + h + DY), (x + DX, y + h + DY)],
                             closed=True, facecolor=_shade(fc, 1.14), edgecolor=ec, lw=lw, zorder=z))
        bg.add_patch(Polygon([(x + w, y), (x + w + DX, y + DY), (x + w + DX, y + h + DY), (x + w, y + h)],
                             closed=True, facecolor=_shade(fc, 0.82), edgecolor=ec, lw=lw, zorder=z))
        bg.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw, zorder=z + 1))

    def block(cx, w, h, key, title, sub="", dims="", y=YC, tfs=13):
        fc, ec = PAL[key]; x = cx - w / 2; yb = y - h / 2
        iso(x, yb, w, h, fc, ec)
        bg.text(cx + DX / 2, y + DY / 2, title, ha="center", va="center", fontsize=tfs, color="#14171c", zorder=6)
        if sub: bg.text(cx + DX / 2, y - h * 0.30, sub, ha="center", va="center", fontsize=8.5,
                        color="#4a5563", family="monospace", zorder=6)
        if dims: bg.text(cx, yb - 0.055, dims, ha="center", va="top", fontsize=8.5, color="#586472",
                         family="monospace", zorder=6)

    def plane(cx, w, h, data, title, dims, y=YC):        # tilted image "data plane" with a depth face
        x = cx - w / 2; yb = y - h / 2
        bg.add_patch(Polygon([(x + w, yb), (x + w + DX, yb + DY), (x + w + DX, yb + h + DY), (x + w, yb + h)],
                             closed=True, facecolor="#20242c", edgecolor="#20242c", lw=0.5, zorder=2))
        ax = fig.add_axes([x, yb, w, h]); ax.imshow(data, aspect="auto", cmap="cividis", origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_edgecolor("#20242c"); s.set_linewidth(1.2)
        bg.text(cx, y + h / 2 + 0.055, title, ha="center", va="bottom", fontsize=10.5, color="#20242c", zorder=6)
        bg.text(cx, yb - 0.055, dims, ha="center", va="top", fontsize=8.5, color="#586472",
                family="monospace", zorder=6)

    def flow(x0, x1, y0=YC, y1=YC, ec=FLOW, lw=2.6, rad=None, dashed=False):
        cs = dict(connectionstyle=f"arc3,rad={rad}") if rad is not None else {}
        bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22, lw=lw,
                                     color=ec, linestyle=(":" if dashed else "-"), zorder=5, **cs))

    # ---- main stream, left to right ----
    plane(0.065, 0.075, 0.34, P, "dynamic spectrum", r"$P\,[N_t\!\times\!N_f]$")
    flow(0.108, 0.163)
    # de-Doppler transform slab, with the frozen|learnable kernel bank shown as two thin sub-slabs on top
    block(0.235, 0.14, 0.30, "transform", "de-Doppler", "", tfs=13)
    bg.text(0.235 + DX / 2, YC + DY / 2 - 0.052, r"$R[k,f]=\sum_t P[t, f{+}d_k t]$", ha="center",
            va="center", fontsize=9.5, color="#2b564c", zorder=6)
    bg.add_patch(Rectangle((0.185, YC - 0.185), 0.045, 0.055, facecolor=FROZEN, edgecolor="#8a9199",
                           hatch="////", lw=0.8, zorder=6))
    bg.add_patch(Rectangle((0.24, YC - 0.185), 0.045, 0.055, facecolor=LEARN, edgecolor="#a85a1e", lw=0.8, zorder=6))
    bg.text(0.2075, YC - 0.205, "frozen", ha="center", va="top", fontsize=7.8, color="#5a6673", zorder=6)
    bg.text(0.2625, YC - 0.205, "learnable", ha="center", va="top", fontsize=7.8, color="#a85a1e", zorder=6)
    bg.text(0.235, YC - 0.255, r"kernels $d_k,\ k{=}1..D$", ha="center", va="top", fontsize=8.5,
            color="#586472", family="monospace", zorder=6)
    flow(0.31, 0.363)
    plane(0.425, 0.075, 0.34, R, "drift spectrum", r"$R\,[D\!\times\!F]$")

    # ---- fork: CNN main (up), rho->MLP lower track ----
    flow(0.47, 0.535, y1=YC + 0.02)
    # CNN as a stack of isometric conv slabs (16->32->64), DRAFTS-style
    conv_x, conv_w, gaps = 0.55, 0.028, 0.006
    for i, (hh, lab) in enumerate([(0.20, "32"), (0.24, "64"), (0.28, "64")]):
        cx = conv_x + i * (conv_w + gaps)
        iso(cx, YC - hh / 2, conv_w, hh, PAL["cnn"][0], PAL["cnn"][1])
        bg.text(cx + conv_w / 2, YC - hh / 2 - 0.05, lab, ha="center", va="top", fontsize=8, color="#586472",
                family="monospace", zorder=6)
    bg.text(0.585, YC + 0.265, "CNN over $R$", ha="center", fontsize=12.5, color="#14171c", zorder=6)
    bg.text(0.585, YC + 0.225, "conv stack -> feat[128]", ha="center", fontsize=8.5, color="#4a5563",
            family="monospace", zorder=6)
    # rho profile plane (lower) -> MLP slab
    axr = fig.add_axes([0.505, 0.135, 0.10, 0.14]); axr.plot(drifts, rho, color=PAL["mlp"][1], lw=1.8)
    axr.axvline(drifts[rho.argmax()], color=LEARN, lw=1.1, ls="--"); axr.set_xticks([]); axr.set_yticks([])
    for s in axr.spines.values(): s.set_edgecolor("#20242c")
    bg.text(0.555, 0.295, r"$\rho[k]=\max_f R[k,f]$", ha="center", fontsize=9.5, color="#20242c", zorder=6)
    flow(0.435, 0.50, y0=YC - 0.11, y1=0.205, ec=PAL["mlp"][1], rad=-0.2)
    flow(0.61, 0.66, y0=0.205, y1=0.205, ec=PAL["mlp"][1])
    block(0.715, 0.09, 0.13, "mlp", "MLP", "rho[D]->[32]", y=0.205, tfs=12)
    # ---- merge into classifier ----
    flow(0.62, 0.79, y0=YC + 0.05, y1=YC, ec=PAL["cnn"][1])
    flow(0.765, 0.79, y0=0.235, y1=YC - 0.06, ec=PAL["mlp"][1], rad=0.2)
    block(0.85, 0.11, 0.30, "fuse", "classifier", "[128+64]->softmax", tfs=12.5)
    bg.text(0.85 + DX / 2, YC + DY / 2 + 0.055, "concat", ha="center", fontsize=9.5, color="#4a5563", zorder=6)
    flow(0.905, 0.945)
    for i, c in enumerate(("signal", "rfi", "noise")):
        y = YC + 0.085 - i * 0.085
        bg.add_patch(Rectangle((0.95, y - 0.028), 0.032, 0.056, facecolor=CHIP[c], edgecolor="none", zorder=6))
        bg.text(0.988, y, c, ha="left", va="center", fontsize=11.5, color="#14171c", zorder=6)

    # ablation / baseline legend (bottom-left whitespace)
    lx, ly = 0.05, 0.14
    bg.text(lx, ly + 0.03, "ablations & baseline", fontsize=10.5, weight="bold", color="#4a5361")
    bg.add_patch(Rectangle((lx, ly - 0.02), 0.02, 0.028, facecolor=FROZEN, hatch="////", edgecolor="#8a9199", lw=0.6))
    bg.add_patch(Rectangle((lx + 0.026, ly - 0.02), 0.02, 0.028, facecolor=LEARN, edgecolor="#a85a1e", lw=0.6))
    bg.text(lx + 0.055, ly - 0.006, r"frozen vs learnable kernels: no measurable effect ($p{=}0.42$)",
            fontsize=9, va="center", color="#3a3f45")
    bg.plot([lx, lx + 0.046], [ly - 0.055, ly - 0.055], color=OPT, ls=":", lw=2.2)
    bg.text(lx + 0.055, ly - 0.055, r"$\rho$-only variant: drop the CNN branch", fontsize=9, va="center", color="#3a3f45")
    bg.plot([lx, lx + 0.046], [ly - 0.1, ly - 0.1], color=PAL["transform"][1], ls="-", lw=2.2)
    bg.text(lx + 0.055, ly - 0.1, r"classical baseline: fixed threshold on $\rho$ (no learned head)",
            fontsize=9, va="center", color=PAL["transform"][1])

    fig.suptitle("DriftNet: a learned decision rule over the classical de-Doppler statistic", fontsize=15, y=0.98)
    plt.savefig("tex/figs/architecture.png", dpi=200, bbox_inches="tight")
    plt.savefig("tex/figs/architecture.pdf", bbox_inches="tight"); plt.close()
    print("wrote architecture.png + .pdf")


def concept():
    rng = np.random.default_rng(0); T, F = 64, 256
    P = rng.chisquare(2, (T, F)).astype(float) * 3
    for t in range(T):                                   # a faint drifting line
        f = 70 + int(0.9 * t); P[t, f:f + 2] += 6
    # de-Doppler over a drift grid
    drifts = np.linspace(-1.5, 1.5, 61); R = np.zeros((len(drifts), F))
    Pd = P - np.median(P, axis=0, keepdims=True)
    for k, d in enumerate(drifts):
        for t in range(T):
            R[k] += np.roll(Pd[t], -int(round(d * t)))
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.3))
    ax[0].imshow(P, aspect="auto", cmap="gray", origin="lower"); ax[0].set_title("(a) dynamic spectrum (faint drift)", fontsize=9)
    ax[0].set_xlabel("frequency"); ax[0].set_ylabel("time"); ax[0].set_xticks([]); ax[0].set_yticks([])
    im = ax[1].imshow(R, aspect="auto", cmap="magma", origin="lower",
                      extent=[0, F, drifts[0], drifts[-1]])
    ax[1].set_title("(b) de-Doppler drift spectrum $R[d,f]$", fontsize=9)
    ax[1].set_xlabel("frequency"); ax[1].set_ylabel("drift rate $d$"); ax[1].set_xticks([])
    ky, kx = np.unravel_index(R.argmax(), R.shape)
    ax[1].plot(kx, drifts[ky], "o", mfc="none", mec="cyan", ms=12)
    ax[1].text(kx + 10, drifts[ky], "matched-filter peak", fontsize=7.5, color="cyan", va="center")
    plt.tight_layout(); plt.savefig("tex/figs/concept.png", dpi=300)
    plt.savefig("tex/figs/concept.pdf"); plt.close()
    print("wrote concept.png + .pdf")


if __name__ == "__main__":
    architecture(); concept()
