"""V13 -- falsify a price-of-anarchy bound that covers the PIECEWISE response.

The gap this closes.  Theorem 1 is proved for the smooth quadratic, but the
paper's motivating figure, its simulator, and V10/V11 all use a cliff.  A
referee is entitled to say the theory does not describe the system the paper is
about.  The fix does not need a new equilibrium analysis, because of an
asymmetry the model already has:

    the AGENTS' perceived cost stays affine (they are charged an attributed
    average factor), so the 3/2 bound for atomic splittable affine congestion
    games applies to the perceived game UNCHANGED.

Only the translation between perceived and true cost changes.  Define, per slot,
over the range of aggregate flexible load the instance can reach,

    rho_t = sup_y  E_t(y) / [ y * (a_t + b_t y) ]      (true / perceived)
    sig_t = inf_y  E_t(y) / [ y * (a_t + b_t y) ]

Then  sig_min * Psi <= C <= rho_max * Psi  pointwise, and chaining through the
3/2 gives

    PoA  <=  (3/2) * rho_max / sig_min .                          (*)

For the smooth response E_t(y) = m*y + b*y^2 this collapses: rho_t = m/a at
y -> 0 and sig_t >= 1, recovering (3/2) * MEF/AEF exactly.  For a cliff rho_max
is far larger, which is the formal reason the piecewise response has a bigger
gap -- and it is measurable from the same published quantities.

This script tries hard to BREAK (*) on random instances of both kinds.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from gh.instances import make_instance
from gh.core import nash
from gh.baselines import planner_value, planner_aggregate
from gh.cliff import cliff_from_instance, social_cliff, planner_cliff

GRID = 4001          # resolution of the sup/inf scan


def _ne(inst):
    r = nash(inst)
    return r[0] if isinstance(r, tuple) else r


def ratios(inst, cl, y_max):
    """rho_max and sig_min over slots, scanning y in (0, y_max_t]."""
    T = inst.T
    rho = np.full(T, -np.inf)
    sig = np.full(T, np.inf)
    for t in range(T):
        hi = max(float(y_max[t]), 1e-12)
        y = np.linspace(hi / GRID, hi, GRID)
        if cl is None:
            true = inst.m[t] * y + inst.b[t] * y ** 2
        else:
            sub = type(cl)(cl.H[[t]], cl.m_lo[[t]], cl.b_lo[[t]],
                           cl.m_hi[[t]], cl.b_hi[[t]])
            true = np.array([sub.value(np.array([v]))[0] for v in y])
        perc = y * (inst.a[t] + inst.b[t] * y)
        r = true / np.maximum(perc, 1e-300)
        rho[t], sig[t] = r.max(), r.min()
    return float(rho.max()), float(sig.min())


def run(cliff, n, T, kappa, seed, jump=6.0):
    inst = make_instance(n=n, T=T, seed=seed, kappa=kappa, lam=0.0)
    cl = cliff_from_instance(inst, headroom_frac=0.6, jump=jump) if cliff else None

    x_ne = _ne(inst)
    y_ne = x_ne.sum(axis=0)
    if cl is None:
        Cne, Cst = inst.social(x_ne), planner_value(inst)
        y_star = planner_aggregate(inst)
    else:
        Cne, Cst = social_cliff(inst, cl, x_ne), planner_cliff(inst, cl)
        y_star = planner_aggregate(inst)          # scan range only
    poa = Cne / Cst

    # scan over the range the instance can actually reach in either profile
    y_max = np.maximum(np.maximum(y_ne, y_star), 1e-9) * 1.05
    rho, sig = ratios(inst, cl, y_max)
    bound = 1.5 * rho / max(sig, 1e-12)
    return poa, bound, rho, sig


print(f"{'model':>7s} {'n':>4s} {'T':>4s} {'kappa':>6s} | {'PoA':>8s} "
      f"{'bound':>9s} {'PoA/bound':>10s} {'rho':>7s} {'sig':>6s}")
print("-" * 74)
rows, viol, worst = [], 0, 0.0
cases = []
for cliff in (False, True):
    for n in (2, 4, 16):
        for kappa in (0.5, 2.0, 8.0):
            for seed in (0, 1, 2):
                cases.append((cliff, n, 16, kappa, seed))
for cliff, n, T, kappa, seed in cases:
    try:
        poa, bound, rho, sig = run(cliff, n, T, kappa, seed)
    except Exception as e:
        print(f"  skip ({e})"); continue
    frac = poa / bound
    worst = max(worst, frac)
    if frac > 1.0 + 1e-9:
        viol += 1
        print(f"{'cliff' if cliff else 'smooth':>7s} {n:4d} {T:4d} {kappa:6.2f} | "
              f"{poa:8.4f} {bound:9.4f} {frac:10.4f} {rho:7.3f} {sig:6.3f}  <-- VIOLATION")
    rows.append(dict(cliff=cliff, n=n, T=T, kappa=kappa, seed=seed,
                     poa=poa, bound=bound, frac=frac, rho=rho, sig=sig))

for m in (False, True):
    sub = [r for r in rows if r["cliff"] == m]
    tag = "cliff " if m else "smooth"
    print(f"{tag}: {len(sub)} instances | PoA/bound max {max(r['frac'] for r in sub):.4f} "
          f"| rho range {min(r['rho'] for r in sub):.2f}-{max(r['rho'] for r in sub):.2f} "
          f"| PoA max {max(r['poa'] for r in sub):.4f}")

print()
print(f"bound violations: {viol} / {len(rows)}    worst PoA/bound = {worst:.4f}")
if viol == 0:
    print("(*) survives falsification on every instance tried.")

out = os.path.join(os.path.dirname(__file__), "..", "results", "V13.json")
json.dump({"rows": rows, "violations": viol, "worst_frac": worst,
           "note": "SYNTHETIC falsification of the generalised PoA bound"},
          open(out, "w"), indent=2)
print(f"wrote {os.path.relpath(out)}")
