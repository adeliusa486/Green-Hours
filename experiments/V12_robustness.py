"""V12 -- robustness of the equilibrium gap and of SHADE across the parameter
axes a referee will ask about.

One axis at a time, everything else held at the reference point.  Reports the
uncorrected equilibrium ratio, SHADE's ratio, and the share of the gap SHADE
removes, so the reader can see whether the mechanism's value is an artefact of
one lucky parameterisation.

Axes: number of operators; flexible-load share; congestion strength; deadline
slack; per-slot power envelope; market-share inequality; idiosyncratic signal
noise; aggregation error.

SYNTHETIC.  Smooth (quadratic) response throughout, so this is directly
comparable to the paper's main table; V10/V11 cover the piecewise response.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from gh.instances import make_instance
from gh.core import nash, solve_sep_qp
from gh.baselines import planner_value

T, SEEDS = 24, 3
REF = dict(n=16, T=T, kappa=1.0, lam=0.0)


def _ne(inst, signal=None):
    r = nash(inst, signal=signal) if signal is not None else nash(inst)
    return r[0] if isinstance(r, tuple) else r


def shade(inst, K=60, gamma=None, agg_err=0.0, rng=None, tol=1e-9):
    n = inst.n
    gamma = gamma or 1.8 / n
    x = inst.feasible_start()
    yhat = x.sum(axis=0).copy(); prev = yhat.copy()
    for _ in range(K):
        for i in range(n):
            y_mi = yhat - x[i]
            pi = (inst.m - inst.a) + inst.b * y_mi
            q = inst.a + inst.b * y_mi + pi + inst.lam[i] * inst.psi[i]
            x[i] = solve_sep_qp(q, inst.b, inst.X[i])
        y = x.sum(axis=0)
        if agg_err > 0.0:
            y = np.maximum(y + rng.normal(0.0, agg_err * y.mean(), inst.T), 0.0)
        yhat = np.maximum(prev + gamma * (y - prev), 0.0)
        if np.max(np.abs(yhat - prev)) <= tol * max(1.0, float(np.max(yhat))):
            break
        prev = yhat
    return x


def evaluate(label, value, mk, noise=0.0, agg_err=0.0):
    nr, sr = [], []
    for s in range(SEEDS):
        rng = np.random.default_rng(1000 + s)
        inst = mk(s)
        Cs = planner_value(inst)
        sig = None
        if noise > 0.0:
            sig = inst.a[None, :] * (1.0 + rng.normal(0, noise, (inst.n, inst.T)))
        nr.append(inst.social(_ne(inst, sig)) / Cs)
        sr.append(inst.social(shade(inst, agg_err=agg_err, rng=rng)) / Cs)
    nv, sv = np.mean(nr), np.mean(sr)
    rm = (nv - sv) / (nv - 1.0) if nv > 1.0 + 1e-9 else float("nan")
    print(f"{label:>22s} {str(value):>10s} | {nv:8.4f} {sv:8.4f} {100*rm:8.1f}%")
    return dict(axis=label, value=value, nash=nv, shade=sv, removed=rm)


print(f"{'axis':>22s} {'value':>10s} | {'nash':>8s} {'shade':>8s} {'gap rm':>9s}")
print("-" * 64)
rows = []

for v in [2, 4, 8, 16, 32]:
    rows.append(evaluate("operators n", v,
                         lambda s, v=v: make_instance(**{**REF, "n": v}, seed=s)))
print()
for v in [0.5, 2.0, 8.0]:
    rows.append(evaluate("flexible share x", v,
                         lambda s, v=v: make_instance(**REF, flex_scale=v, seed=s)))
print()
for v in [0.25, 1.0, 4.0, 8.0]:
    rows.append(evaluate("congestion kappa", v,
                         lambda s, v=v: make_instance(**{**REF, "kappa": v}, seed=s)))
print()
for v in [(1.0, 4.0), (2.0, 12.0), (2.0, 24.0), (8.0, 24.0)]:
    rows.append(evaluate("deadline slack (h)", f"{v[0]:g}-{v[1]:g}",
                         lambda s, v=v: make_instance(**REF, slack_hours=v, seed=s)))
print()
for v in [0.15, 0.35, 1.00]:
    rows.append(evaluate("envelope cap frac", v,
                         lambda s, v=v: make_instance(**REF, cap_frac=v, seed=s)))
print()
for v in ["equal", "lognormal", "heavytail"]:
    rows.append(evaluate("size distribution", v,
                         lambda s, v=v: make_instance(**REF, split=v, seed=s)))
print()
for v in [0.0, 0.05, 0.20]:
    rows.append(evaluate("signal noise sigma", v,
                         lambda s: make_instance(**REF, seed=s), noise=v))
print()
for v in [0.0, 0.05, 0.25]:
    rows.append(evaluate("aggregation error", v,
                         lambda s: make_instance(**REF, seed=s), agg_err=v))

print()
fin = [r["removed"] for r in rows if np.isfinite(r["removed"])]
print(f"SHADE removes {100*np.min(fin):.1f}%-{100*np.max(fin):.1f}% of the gap "
      f"across all {len(fin)} configurations (median {100*np.median(fin):.1f}%).")

out = os.path.join(os.path.dirname(__file__), "..", "results", "V12.json")
json.dump({"T": T, "seeds": SEEDS, "ref": REF, "rows": rows,
           "note": "SYNTHETIC; one axis at a time, smooth response"},
          open(out, "w"), indent=2)
print(f"wrote {os.path.relpath(out)}")
