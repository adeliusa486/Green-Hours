"""V10 -- does the curtailment cliff raise the strategic share?

V9 found that under the smooth quadratic the strategic share reaches the 15%
Plan A gate only at kappa ~ 16, which is not a plausible calibration.  The
hypothesis (REVIEW2 section 5) is that this is an artefact of WHERE the model
puts the curvature, not of how much there is: a smooth quadratic spreads it
evenly over the horizon, so crossing the zero-carbon headroom is never an event,
whereas the physics concentrates all of it at the headroom boundary -- and
whether the agents collectively cross that boundary is exactly the joint
decision the paper is about.

Design.  For each instance we hold the AGENT side fixed (affine attributed
factor, so the potential, uniqueness and the local water-filling step are
untouched) and swap only the SOCIAL cost between:

    smooth   dE_t(y) = m_t*y + b_t*y^2                      (the paper today)
    cliff    dE_t(y) piecewise, near-flat inside headroom H_t, steep beyond,
             with the same operating-point marginal m_t

so the comparison isolates the redistribution of curvature.  Both are evaluated
at the SAME equilibrium and against their OWN planner.

Decomposition, as in V6/V9: the accounting wedge is what survives when agents
are charged the true marginal factor but the congestion externality is left
unpriced; the strategic residual is the rest, and it is the part a mechanism can
address.

SYNTHETIC.  This tests a modelling hypothesis; it does not calibrate anything.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from gh.instances import make_instance
from gh.core import nash
from gh.baselines import planner_value
from gh.cliff import cliff_from_instance, social_cliff, planner_cliff

N, T, SEEDS = 16, 24, 4
KAPPAS = [0.5, 1.0, 2.0, 4.0]
JUMPS = [3.0, 6.0, 12.0]


def _ne(inst):
    r = nash(inst)
    return r[0] if isinstance(r, tuple) else r


def split_smooth(inst):
    Cs = planner_value(inst)
    total = inst.social(_ne(inst)) / Cs - 1.0
    a_save = inst.a.copy()
    try:
        inst.a = inst.m.copy()              # wedge removed, congestion left
        strat = inst.social(_ne(inst)) / Cs - 1.0
    finally:
        inst.a = a_save
    return total, total - strat, strat


def split_cliff(inst, cl):
    Cs = planner_cliff(inst, cl)
    total = social_cliff(inst, cl, _ne(inst)) / Cs - 1.0
    a_save = inst.a.copy()
    try:
        inst.a = inst.m.copy()
        strat = social_cliff(inst, cl, _ne(inst)) / Cs - 1.0
    finally:
        inst.a = a_save
    return total, total - strat, strat


print(f"{'kappa':>6s} {'jump':>6s} | {'model':>7s} {'total':>8s} "
      f"{'account':>9s} {'strategic':>10s} {'share':>8s} {'gate':>6s}")
print("-" * 74)
rows = []
for kap in KAPPAS:
    # smooth reference
    ts, as_, ss = [], [], []
    for s in range(SEEDS):
        inst = make_instance(n=N, T=T, seed=s, kappa=kap, lam=0.0)
        t, a, st = split_smooth(inst)
        ts.append(t); as_.append(a); ss.append(st)
    t, a, st = np.mean(ts), np.mean(as_), np.mean(ss)
    sh = st / t if abs(t) > 1e-12 else float("nan")
    print(f"{kap:6.1f} {'--':>6s} | {'smooth':>7s} {100*t:7.2f}% {100*a:8.2f}% "
          f"{100*st:9.2f}% {100*sh:7.1f}%")
    rows.append(dict(kappa=kap, jump=None, model="smooth",
                     total=t, accounting=a, strategic=st, share=sh))

    for jump in JUMPS:
        ts, as_, ss = [], [], []
        for s in range(SEEDS):
            inst = make_instance(n=N, T=T, seed=s, kappa=kap, lam=0.0)
            cl = cliff_from_instance(inst, headroom_frac=0.6, jump=jump)
            t2, a2, st2 = split_cliff(inst, cl)
            ts.append(t2); as_.append(a2); ss.append(st2)
        t2, a2, st2 = np.mean(ts), np.mean(as_), np.mean(ss)
        sh2 = st2 / t2 if abs(t2) > 1e-12 else float("nan")
        gate = "OPEN" if sh2 >= 0.15 else ""
        print(f"{kap:6.1f} {jump:6.1f} | {'cliff':>7s} {100*t2:7.2f}% "
              f"{100*a2:8.2f}% {100*st2:9.2f}% {100*sh2:7.1f}% {gate:>6s}")
        rows.append(dict(kappa=kap, jump=jump, model="cliff",
                         total=t2, accounting=a2, strategic=st2, share=sh2))
    print()

sm = [r["share"] for r in rows if r["model"] == "smooth"]
cf = [r["share"] for r in rows if r["model"] == "cliff"]
print(f"strategic share, smooth: {100*np.nanmin(sm):.1f}%-{100*np.nanmax(sm):.1f}%")
print(f"strategic share, cliff : {100*np.nanmin(cf):.1f}%-{100*np.nanmax(cf):.1f}%")
if np.nanmax(cf) >= 0.15:
    print("=> the cliff clears the 15% Plan A gate somewhere in this range.")
else:
    print("=> the cliff does NOT clear the gate here; Plan B stands.")

out = os.path.join(os.path.dirname(__file__), "..", "results", "V10.json")
json.dump({"n": N, "T": T, "seeds": SEEDS, "rows": rows,
           "note": "SYNTHETIC; tests a modelling hypothesis, calibrates nothing"},
          open(out, "w"), indent=2)
print(f"wrote {os.path.relpath(out)}")
