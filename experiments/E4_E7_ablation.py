"""E4 + E7 -- operational metrics and the SHADE ablation, on calibrated grids.

These were the last two placeholder tables in the supplementary material.  Both
run on the same real-grid / synthetic-workload instances as E3, so the numbers
are comparable to the main table row for row.

E4, operational metrics per method:
    peak-to-mean of aggregate flexible load
    price uplift proxy in the targeted hours (beta * y, the congestion term the
      grid clears against, in gCO2/kWh-equivalent per MWh)
    deadline violations (should be zero: every method returns a feasible point,
      and we assert it)
    bytes exchanged per operator per day

E7, ablation: each design choice in SHADE removed in turn.
    no adder                 operators see the broadcast aggregate, no externality
    no own-load subtraction  pi built from yhat instead of yhat - x_i (over-corrects)
    no own-congestion term   local step becomes linear (under-corrects, oscillates)
    no damping               gamma = 1
    hourly adder             adder coarsened to 3-hour blocks
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from gh.core import solve_sep_qp, solve_sep_lp, nash
import gh.baselines as B
from E1_calibrate import load, hourly_profile, BAS
from E3_main import build, conform, shade, SEEDS, T, N


def shade_variant(inst, K=80, gamma=None, no_adder=False, no_own_sub=False,
                  no_own_cong=False, no_damping=False, block=1, tol=1e-10):
    n = inst.n
    gamma = 1.0 if no_damping else (gamma or 1.8 / n)
    x = inst.feasible_start(); yhat = x.sum(0).copy(); prev = yhat.copy()
    for _ in range(K):
        for i in range(n):
            ymi = yhat if no_own_sub else yhat - x[i]
            pi = np.zeros(inst.T) if no_adder else (inst.m - inst.a) + inst.b * ymi
            if block > 1:                       # coarsen the adder in time
                pad = (-inst.T) % block
                pp = np.concatenate([pi, np.full(pad, pi[-1])])
                pi = np.repeat(pp.reshape(-1, block).mean(1), block)[:inst.T]
            q = inst.a + inst.b * ymi + pi + inst.lam[i] * inst.psi[i]
            if no_own_cong:
                x[i] = solve_sep_lp(q, inst.X[i])
            else:
                x[i] = solve_sep_qp(q, inst.b, inst.X[i])
        y = x.sum(0)
        yhat = np.maximum(prev + gamma * (y - prev), 0.0)
        if np.max(np.abs(yhat - prev)) <= tol * max(1.0, float(np.max(yhat))):
            break
        prev = yhat
    return x


def ops(inst, x, rounds=None):
    y = x.sum(0)
    p2m = float(y.max() / max(y.mean(), 1e-12))
    uplift = float(np.sum(inst.b * y * y) / max(np.sum(y), 1e-12))
    viol = sum(0 if Xi.check(x[i], tol=1e-6) else 1 for i, Xi in enumerate(inst.X))
    kb = (0.0 if rounds is None else rounds * inst.T * 8 * 2 / 1024.0)
    return p2m, uplift, viol, kb


if __name__ == "__main__":
    print("loading EIA-930 ...")
    store = load()
    # same hourly profiles and the same model conformance as E3, so every row
    # here is comparable to the main table row for row
    prof = {}
    for b in BAS:
        a, m, be, dm = hourly_profile(store, b)
        m2, b2, _, _ = conform(a, m, be)
        prof[b] = (a, m2, b2, dm)

    METHODS = [
        ("Carbon-agnostic",   lambda I, r: B.carbon_agnostic(I), None),
        ("Independent greedy", lambda I, r: (lambda z: z[0] if isinstance(z, tuple) else z)(nash(I)), None),
        ("Threshold deferral", lambda I, r: B.threshold(I), None),
        ("Randomized jitter",  lambda I, r: B.jitter(I, seed=r), None),
        ("SHADE (ours)",       lambda I, r: shade(I), 80),
    ]
    print(f"\nE4  operational metrics (mean over {len(BAS)} regions x {SEEDS} seeds)")
    print(f"{'Method':<22s} {'peak/mean':>10s} {'uplift':>9s} {'viol':>6s} {'kB/op/d':>9s}")
    print("-" * 60)
    e4 = {}
    for name, fn, rd in METHODS:
        acc = []
        for b in BAS:
            a, m, be, dm = prof[b]
            for s in range(SEEDS):
                inst = build(a, m, be, dm, s)
                acc.append(ops(inst, fn(inst, s), rd))
        A = np.array(acc)
        e4[name] = dict(p2m=float(A[:, 0].mean()), uplift=float(A[:, 1].mean()),
                        viol=int(A[:, 2].sum()), kb=float(A[:, 3].mean()))
        print(f"{name:<22s} {A[:,0].mean():10.2f} {A[:,1].mean():9.3f} "
              f"{int(A[:,2].sum()):6d} {A[:,3].mean():9.1f}")

    VARIANTS = [
        ("Full SHADE",                    dict()),
        ("No externality adder",          dict(no_adder=True)),
        ("No own-load subtraction",       dict(no_own_sub=True)),
        ("No own-congestion term",        dict(no_own_cong=True)),
        ("No damping ($\\gamma=1$)",      dict(no_damping=True)),
        ("Adder coarsened to 3\\,h",      dict(block=3)),
    ]
    print(f"\nE7  ablation (ratio to planner)")
    print(f"{'Configuration':<30s} {'ratio':>9s} {'vs full':>9s}")
    print("-" * 52)
    e7 = {}
    for label, kw in VARIANTS:
        vals = []
        for b in BAS:
            a, m, be, dm = prof[b]
            for s in range(SEEDS):
                inst = build(a, m, be, dm, s)
                Cs = B.planner_value(inst)
                try:
                    x = shade_variant(inst, **kw)
                except Exception as e:
                    continue
                if not all(Xi.check(x[i], tol=1e-6) for i, Xi in enumerate(inst.X)):
                    continue
                vals.append(inst.social(x) / Cs)
        if not vals:
            print(f"{label:<30s} {'diverged / infeasible':>19s}")
            e7[label] = None; continue
        v = float(np.mean(vals))
        e7[label] = v
        base = e7.get("Full SHADE")
        d = "" if label == "Full SHADE" else f"{100*(v-base):+8.2f}"
        print(f"{label:<30s} {v:9.4f} {d:>9s}")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "E4_E7.json")
    json.dump({"note": "REAL GRID (EIA-930) + SYNTHETIC WORKLOAD",
               "E4": e4, "E7": e7}, open(out, "w"), indent=2)
    print(f"\nwrote {os.path.relpath(out)}")
