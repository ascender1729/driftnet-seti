#!/usr/bin/env python3
"""KEDD-S core claim (single-frame, defensible half): at MATCHED false alarm, constraining the
acceleration search to the thin astrophysical manifold recovers faint accelerating signals that both
(a) the linear search loses to curvature and (b) the free-box curved search loses to its own trials
floor -- because the free box must cover a WIDE acceleration range (to be complete / to chase the
chirp-RFI zoo), paying a large look-elsewhere penalty, while the manifold pays a small one.

Honest scope learned from the first run: the single-frame acceleration constraint does NOT by itself
reject a BRIGHT off-manifold chirp (partial alignment still triggers); RFI rejection is a CADENCE-joint
ephemeris-consistency mechanism, deferred to the real-data cadence experiment. Here we isolate the
clean, single-frame result: the matched-FA sensitivity ordering manifold >= free-box > linear.
"""
import numpy as np

T, F = 48, 160
DRIFTS = np.linspace(-0.7, 0.7, 13)
A_MANIFOLD = np.linspace(-0.004, 0.004, 5)          # thin astrophysical band
A_FREEBOX = np.linspace(-0.03, 0.03, 31)            # wide box (must chase RFI / be complete) -> big trials
rng = np.random.default_rng(3)
_t = np.arange(T); _cols = np.arange(F)


def frame(f0, d, a, amp):
    P = rng.normal(0, 1.0, (T, F))
    ch = f0 + d * _t + 0.5 * a * _t * _t
    for ti in range(T):
        c = int(round(ch[ti]))
        if 0 <= c < F:
            P[ti, c] += amp
    return P


def rho(P, accels):                                  # vectorized de-Doppler (no per-time python loop)
    sig = np.sqrt(T); best = -1e9
    for d in DRIFTS:
        for a in accels:
            shift = np.rint(d * _t + 0.5 * a * _t * _t).astype(int)
            idx = (_cols[None, :] + shift[:, None]) % F           # [T,F]
            m = np.take_along_axis(P, idx, axis=1).sum(0).max()
            if m > best:
                best = m
    return best / sig


def thr(accels, fa=0.05, n=200):
    return np.quantile([rho(rng.normal(0, 1.0, (T, F)), accels) for _ in range(n)], 1 - fa)


def rate(accels, t, afn, amp, n=90):
    f0 = lambda: rng.integers(55, 105); d0 = lambda: float(rng.choice(DRIFTS))    # on-grid drift (fair to linear)
    return np.mean([rho(frame(f0(), d0(), afn(), amp), accels) > t for _ in range(n)])


def main():
    LIN, MAN, BOX = [0.0], A_MANIFOLD, A_FREEBOX
    print(f"trials: linear={len(DRIFTS)}  manifold={len(DRIFTS)*len(MAN)}  free-box={len(DRIFTS)*len(BOX)}", flush=True)
    tL, tM, tB = thr(LIN), thr(MAN), thr(BOX)
    print(f"matched-FA thresholds (FA=0.05): linear={tL:.2f}  manifold={tM:.2f}  free-box={tB:.2f}\n", flush=True)
    onman = lambda: float(rng.uniform(-0.004, 0.004))
    print(f"{'':>5} {'linear signal (control)':>26} {'on-manifold accel signal':>28}")
    print(f"{'amp':>5} {'lin   box   man':>26} {'lin   box   man':>28}")
    print("-" * 62)
    for amp in [1.0, 1.3, 1.6, 2.0]:
        lL = rate(LIN, tL, lambda: 0.0, amp); lB = rate(BOX, tB, lambda: 0.0, amp); lM = rate(MAN, tM, lambda: 0.0, amp)
        aL = rate(LIN, tL, onman, amp); aB = rate(BOX, tB, onman, amp); aM = rate(MAN, tM, onman, amp)
        print(f"{amp:>5.1f} {lL:>8.2f}{lB:>8.2f}{lM:>8.2f}   {aL:>10.2f}{aB:>8.2f}{aM:>8.2f}", flush=True)
    print("\nEXPECTED: linear-signal columns roughly tie (no regression, linear detector fine on a=0).")
    print("On-manifold accel: linear LOSES (curvature); free-box recovers but its wide-search threshold")
    print("costs faint completeness; manifold recovers the most at the SAME false alarm (small Occam")
    print("penalty over the thin band beats the free box's wide look-elsewhere penalty).")
    print("KEDD_CONCEPT_DONE")


if __name__ == "__main__":
    main()
