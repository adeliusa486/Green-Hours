"""V9 -- where does the strategic share cross the Plan A threshold?

V6 measured the accounting/strategic decomposition at kappa in {0.5,1,2,4} and
found the strategic share running 2.4% -> 12.7%.  The improvement plan sets the
Plan A / Plan B gate at ~15%: above it the mechanism carries the paper, below it
the paper is reframed around measurement and accounting.

E1 will report a single kappa per region.  This sweep says what that number has
to be for the gate to open, so the calibration can be read against a threshold
instead of a vibe.  It also extends the range upward, since V5's addendum showed
the *total* gap is non-monotone while the *share* is monotone -- the share is the
quantity the gate is defined on, so it is the one to extrapolate.

Synthetic instances.  This does not replace E1; it parameterises the decision.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from gh.instances import make_instance
from gh.core import nash
from gh.baselines import planner_value

N, T, SEEDS = 16, 24, 3
KAPPAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def decompose(inst):
    """Total gap, and its split into the accounting wedge and the strategic
    residual.  The wedge is what a single operator would still suffer from
    average-factor accounting; the residual is what needs a mechanism."""
    Cs = planner_value(inst)
    r = nash(inst)
    x_ne = r[0] if isinstance(r, tuple) else r
    total = inst.social(x_ne) / Cs - 1.0

    # Accounting-only counterfactual: charge each agent the true marginal factor
    # but leave the congestion externality unpriced.  What remains of the gap
    # after this is the strategic component.
    inst_acc = make_instance.__self__ if False else None
    a_save = inst.a.copy()
    try:
        inst.a = inst.m.copy()          # wedge removed, congestion untouched
        r2 = nash(inst)
        x_acc = r2[0] if isinstance(r2, tuple) else r2
        strategic = inst.social(x_acc) / Cs - 1.0
    finally:
        inst.a = a_save
    accounting = total - strategic
    return total, accounting, strategic


print(f"{'kappa':>6s} {'total gap':>10s} {'accounting':>11s} {'strategic':>10s} "
      f"{'strat share':>12s} {'gate':>6s}")
print("-" * 62)
rows = []
for kap in KAPPAS:
    tots, accs, strs = [], [], []
    for s in range(SEEDS):
        inst = make_instance(n=N, T=T, seed=s, kappa=kap, lam=0.0)
        t, a, st = decompose(inst)
        tots.append(t); accs.append(a); strs.append(st)
    t, a, st = np.mean(tots), np.mean(accs), np.mean(strs)
    share = st / t if t > 1e-12 else float("nan")
    gate = "OPEN" if share >= 0.15 else ""
    print(f"{kap:6.1f} {100*t:9.2f}% {100*a:10.2f}% {100*st:9.2f}% "
          f"{100*share:11.1f}% {gate:>6s}")
    rows.append(dict(kappa=kap, total=t, accounting=a, strategic=st, share=share))

print()
sh = [r["share"] for r in rows]
if max(sh) >= 0.15:
    k = next(r["kappa"] for r in rows if r["share"] >= 0.15)
    print(f"Strategic share reaches the 15% Plan A gate at kappa ~ {k:g}.")
else:
    print(f"Strategic share does not reach 15% anywhere in "
          f"kappa <= {KAPPAS[-1]:g}; peak {100*max(sh):.1f}%. "
          f"E1 must deliver a kappa above this range for Plan A.")

out = os.path.join(os.path.dirname(__file__), "..", "results", "V9.json")
json.dump({"n": N, "T": T, "seeds": SEEDS, "rows": rows,
           "note": "SYNTHETIC; parameterises the Plan A/B gate, does not replace E1"},
          open(out, "w"), indent=2)
print(f"wrote {os.path.relpath(out)}")
