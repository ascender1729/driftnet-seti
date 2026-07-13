#!/usr/bin/env python3
"""Concept test for a NEW algorithm: curved-path (accelerating-drift) de-Doppler.

Motivation from our own result: the linear de-Doppler integrated SNR is a near-sufficient statistic
ONLY for a constant drift rate. A real emitter with line-of-sight ACCELERATION has a drift rate that
changes during the observation, so its track curves: f(t) = f0 + d*t + 0.5*a*t^2. turboSETI searches
only a=0 (straight lines), so it loses SNR on accelerating signals -- and this is exactly the regime
where learning/searching the transform is NOT useless.

This test asks the make-or-break question: does searching the curvature a recover an accelerating
signal that the linear (a=0) search misses? If yes, the algorithm is real.
"""
import numpy as np

T, F = 128, 512
rng = np.random.default_rng(0)


def make_frame(f0, d, a, amp, noise=1.0):
    """Inject a narrowband line along f(t)=f0+d*t+0.5*a*t^2 into Gaussian noise."""
    P = rng.normal(0, noise, (T, F))
    t = np.arange(T)
    ch = f0 + d * t + 0.5 * a * t * t
    for ti in range(T):
        c = int(round(ch[ti]))
        if 0 <= c < F:
            P[ti, c] += amp
    return P


def dedoppler(P, d, a):
    """Integrate along one curved path (d, a); return the summed spectrum S(f)."""
    t = np.arange(T)
    shift = d * t + 0.5 * a * t * t
    S = np.zeros(F)
    for ti in range(T):
        S += np.roll(P[ti], -int(round(shift[ti])))
    return S


def best_rho(P, drifts, accels):
    """Peak integrated SNR over the (drift, accel) grid. accels=[0] gives the linear (turboSETI) search."""
    noise_sigma = np.sqrt(T)                       # incoherent sum of T unit-variance samples
    best = -1e9; arg = None
    for d in drifts:
        for a in accels:
            S = dedoppler(P, d, a)
            peak = S.max() / noise_sigma
            if peak > best:
                best, arg = peak, (d, a)
    return best, arg


def main():
    drifts = np.linspace(-1.0, 1.0, 41)            # channels per sample
    accels = np.linspace(-0.004, 0.004, 33)        # channels per sample^2 (curvature)
    d0, amp = 0.3, 6.0

    print(f"{'accel a0':>10} {'linear rho':>12} {'curved rho':>12} {'curved gain':>12}  recovered")
    print("-" * 62)
    for a0 in [0.0, 0.001, 0.002, 0.003, 0.004]:
        # curvature magnitude: how far the track bends off the best straight line, in channels
        t = np.arange(T); bend = 0.5 * a0 * t.max() ** 2
        lin, _ = best_rho(make_frame(200, d0, a0, amp), drifts, [0.0])
        cur, (dc, ac) = best_rho(make_frame(200, d0, a0, amp), drifts, accels)
        gain = cur / lin if lin > 0 else float("nan")
        rec = "curved only" if (cur > 5 and lin < 5) else ("both" if lin > 5 else "neither")
        print(f"{a0:>10.4f} {lin:>12.2f} {cur:>12.2f} {gain:>11.2f}x  {rec}   (bend={bend:.0f} ch, a_hat={ac:+.4f})")
    # noise floor check: pure noise should give rho ~ a few sigma for both
    nlin, _ = best_rho(rng.normal(0, 1, (T, F)), drifts, [0.0])
    ncur, _ = best_rho(rng.normal(0, 1, (T, F)), drifts, accels)
    print(f"\nnoise-only floor: linear rho={nlin:.2f}, curved rho={ncur:.2f} (curved searches more paths -> slightly higher floor)")
    print("CONCEPT_DONE")


if __name__ == "__main__":
    main()
