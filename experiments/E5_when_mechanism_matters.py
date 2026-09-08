"""E5 -- at what flexible share does the mechanism stop being redundant?

Why this experiment exists.  On the calibrated instances of E3, publishing a
static marginal factor removes essentially the whole equilibrium gap, and SHADE
removes the same gap.  That is not a failure of the mechanism; it is a
measurement about today's grid.  The congestion strength we measure is
kappa = 0.007-0.052, so the term a load-responsive correction exists to price is
one to two orders of magnitude smaller than the accounting wedge, and a signal
that is merely *correct in level* is already almost sufficient.

The question a reviewer will then ask, and the one an operator would ask, is
when that stops being true.  This script answers it directly rather than by
extrapolating kappa: it scales aggregate flexible load on the SAME measured
hourly factors and reports where a static marginal signal starts to leave a gap
that a load-responsive one closes.  It also reports the round count SHADE needs
to reach a given accuracy, which is the number an operator would budget for --
not the round count to machine precision.

PROVENANCE.  The hourly AEF, MEF and beta are measured [replay].  Everything
else is [sim], and the large-share end of the sweep is an EXTRAPOLATION: the
second-order expansion of Section 3.2 is fitted on the range of load variation
present in the data, and at a 16% flexible share the aggregate excursion is well
outside it.  Read those rows as a statement about the model, not about the grid.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from gh.core import nash, solve_sep_qp
import gh.baselines as B
from E1_calibrate import BAS, load, hourly_profile
from E3_main import build, conform, NICE
import E3_main

SEEDS = 8
SHARES = [0.0038, 0.01, 0.02, 0.04, 0.08, 0.16]
TARGETS = [1.001, 1.0001]
K_MAX = 1500


def shade_trace(inst, Cs, gamma=None, K=K_MAX, ftol=1e-12, ytol=1e-9,
                patience=5):
    """SHADE, returning the final ratio and the first round at which the ratio
    fell below each target."""
    n = inst.n
    gamma = gamma if gamma is not None else 1.8 / n
    x = inst.feasible_start()
    yhat = x.sum(0).copy()
    prev = yhat.copy()
    prevC = inst.social(x)
    hit = {t: None for t in TARGETS}
    flat = 0
    for k in range(1, K + 1):
        for i in range(n):
            ymi = yhat - x[i]
            q = inst.a + inst.b * ymi + ((inst.m - inst.a) + inst.b * ymi)
            x[i] = solve_sep_qp(q + inst.lam[i] * inst.psi[i], inst.b, inst.X[i])
        y = x.sum(0)
        yhat = np.maximum(prev + gamma * (y - prev), 0.0)
        dy = float(np.max(np.abs(yhat - prev))) / max(1.0, float(np.max(yhat)))
        prev = yhat
        cur = inst.social(x)
        r = cur / Cs
        for t in TARGETS:
            if hit[t] is None and r <= t:
                hit[t] = k
        dC = abs(prevC - cur) / max(1.0, abs(prevC))
        prevC = cur
        if dC <= ftol and dy <= ytol:
            flat += 1
            if flat >= patience:
                break
        else:
            flat = 0
    return inst.social(x) / Cs, hit


def one(a, m, b, dm, share, seed):
    E3_main.FLEX_SHARE = share
    inst = build(a, m, b, dm, seed)
    Cs = B.planner_value(inst)
    xn = nash(inst)[0]
    y = xn.sum(0)
    A = y > 1e-9
    kappa = (float(np.mean(inst.b[A] * y[A]
                           / np.maximum(inst.m[A] - inst.a[A], 1e-9)))
             if A.any() else float("nan"))
    sh, hit = shade_trace(inst, Cs)
    return {
        "nash": inst.social(xn) / Cs,
        "agnostic": inst.social(B.carbon_agnostic(inst)) / Cs,
        "mef_static": inst.social(B.mef_static(inst)) / Cs,
        "shade": sh,
        "kappa": kappa,
        "rounds_1e3": hit[1.001], "rounds_1e4": hit[1.0001],
    }


if __name__ == "__main__":
    print("loading EIA-930 ...")
    store = load()
    prof = {}
    for ba in BAS:
        a, m, bb, dm = hourly_profile(store, ba)
        m2, b2, _, _ = conform(a, m, bb)
        prof[ba] = (a, m2, b2, dm)

    print(f"\n[replay-grid / sim-workload]  ratio to the planner, mean over "
          f"{SEEDS} seeds\n")
    print(f"{'region':>7s} {'flex%':>7s} {'kappa':>7s} | {'agnostic':>9s} "
          f"{'Nash':>9s} {'staticMEF':>10s} {'SHADE':>9s} | "
          f"{'static leaves':>13s} {'rounds':>13s}")
    print("-" * 104)
    rows = []
    for ba in BAS:
        a, m, b, dm = prof[ba]
        for share in SHARES:
            acc = [one(a, m, b, dm, share, s) for s in range(SEEDS)]
            g = {k: float(np.mean([r[k] for r in acc]))
                 for k in ("nash", "agnostic", "mef_static", "shade", "kappa")}
            rr = {k: [r[k] for r in acc if r[k] is not None]
                  for k in ("rounds_1e3", "rounds_1e4")}
            g["rounds_1e3"] = float(np.median(rr["rounds_1e3"])) if rr["rounds_1e3"] else None
            g["rounds_1e4"] = float(np.median(rr["rounds_1e4"])) if rr["rounds_1e4"] else None
            # share of the equilibrium gap a static signal leaves that the
            # load-responsive one closes
            left = (g["mef_static"] - g["shade"]) / max(g["nash"] - 1.0, 1e-12)
            rows.append(dict(region=ba, share=share, static_leaves=left, **g))
            r3 = "-" if g["rounds_1e3"] is None else f"{g['rounds_1e3']:.0f}"
            r4 = "-" if g["rounds_1e4"] is None else f"{g['rounds_1e4']:.0f}"
            print(f"{NICE[ba]:>7s} {100*share:6.2f}% {g['kappa']:7.4f} | "
                  f"{g['agnostic']:9.5f} {g['nash']:9.5f} {g['mef_static']:10.5f} "
                  f"{g['shade']:9.5f} | {100*left:12.1f}% {r3:>6s}/{r4:>6s}")
        print()
    E3_main.FLEX_SHARE = 0.0038

    print("Flexible share (as a fraction of regional demand) at which a static")
    print("marginal signal leaves more than 1% / 5% of the equilibrium gap")
    print("unclosed, by linear interpolation between the sweep points:")
    thresholds = {}
    for ba in BAS:
        rr = [r for r in rows if r["region"] == ba]
        xs = np.array([r["share"] for r in rr])
        ys = np.array([r["static_leaves"] for r in rr])
        t = {}
        for lvl in (0.01, 0.05):
            hit = np.flatnonzero(ys > lvl)
            if hit.size == 0:
                t[lvl] = None
            elif hit[0] == 0:
                t[lvl] = float(xs[0])
            else:
                j = hit[0]
                w = (lvl - ys[j - 1]) / (ys[j] - ys[j - 1])
                t[lvl] = float(xs[j - 1] + w * (xs[j] - xs[j - 1]))
        thresholds[ba] = t
        def f(v):
            return "beyond 16%" if v is None else f"{100*v:.1f}% of demand"
        print(f"  {NICE[ba]:>6s}:  >1% at {f(t[0.01]):>17s}   "
              f">5% at {f(t[0.05]):>17s}")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "E5.json")
    json.dump({"provenance": "REPLAY-GRID / SIM-WORKLOAD; the large-share rows "
                             "EXTRAPOLATE the second-order expansion beyond the "
                             "load range it was fitted on",
               "seeds": SEEDS, "shares": SHARES, "targets": TARGETS,
               "rows": rows,
               "thresholds": {k: {str(a): b for a, b in v.items()}
                              for k, v in thresholds.items()}},
              open(out, "w"), indent=2)
    print(f"\nwrote {os.path.relpath(out)}")
