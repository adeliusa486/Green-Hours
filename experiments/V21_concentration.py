"""V21 -- the epsilon-Nash bound at a realistic market, not the trace's market.

Theorem 4 bounds the market-power term of epsilon by 2 beta_bar rho_bar E_bar^2.
Section 5.5 evaluates it at the Azure trace's structure -- n = 32 operators with
a largest share of rho_bar = 0.045 -- and observes that many comparable
operators are easier to coordinate than a duopoly, since n equal operators
sharing a fixed pool give epsilon = Theta(E_tot^2 / n^3).

That is correct and it cuts against the market the mechanism would actually be
deployed into.  Hyperscale cloud is three to five firms, and the relevant
rho_bar is nearer 0.3 than 0.045.  The bound is linear in rho_bar and quadratic
in E_bar, so both move the wrong way at once, and the paper should say by how
much rather than leave a reviewer to work it out.

This script evaluates the bound on the calibrated instances at a range of market
structures, holding the total flexible pool fixed, and reports epsilon both in
absolute terms and as a fraction of the equilibrium gap it is supposed to be
closing -- which is the comparison that decides whether the guarantee is
informative.

Writes results/V21.json.  Arithmetic on calibrated primitives; no new solve.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import gh.baselines as B
from E1_calibrate import BAS, load, hourly_profile
from E3_main import build, conform, _ne

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "V21.json")

# market structures to evaluate, as (label, n, rho_bar).  rho_bar is the largest
# operator's share of the total flexible pool.
MARKETS = [
    ("Azure trace", 32, 0.045),
    ("n=32, equal", 32, 1 / 32),
    ("n=8, equal", 8, 1 / 8),
    ("n=5, equal", 5, 1 / 5),
    ("n=5, concentrated", 5, 0.30),
    ("n=4, concentrated", 4, 0.30),
    ("n=3, hyperscale", 3, 0.40),
    ("n=2, duopoly", 2, 0.50),
]


if __name__ == "__main__":
    print("loading EIA-930 ...", flush=True)
    store = load()
    out = {"provenance": "REPLAY-GRID / SIM-WORKLOAD.  Theorem 4's market-power "
                         "term evaluated on the calibrated primitives at a "
                         "range of market structures, with the total flexible "
                         "pool held fixed.  Arithmetic, not a new solve.",
           "markets": [m[0] for m in MARKETS], "by_region": {}}

    for ba in BAS:
        a, m, be, dm = hourly_profile(store, ba)
        m2, b2, _, _ = conform(a, m, be)
        inst = build(a, m2, b2, dm, 0)
        beta_bar = float(np.max(inst.b))
        E_tot = float(sum(Xi.E for Xi in inst.X))
        # the quantities epsilon is judged against
        xn = _ne(inst)
        Cs = B.planner_value(inst)
        Cn = inst.social(xn)
        gap_abs = Cn - Cs                       # what a mechanism can recover
        rows = []
        for lab, n, rho in MARKETS:
            E_bar = rho * E_tot                 # largest operator's energy
            eps = 2.0 * beta_bar * rho * E_bar ** 2
            rows.append({
                "market": lab, "n": n, "rho_bar": rho, "E_bar": E_bar,
                "epsilon": eps,
                "n_epsilon": n * eps,           # Theorem 4's welfare loss bound
                "eps_over_gap": eps / gap_abs,
                "n_eps_over_gap": n * eps / gap_abs,
            })
        out["by_region"][ba] = {
            "beta_bar": beta_bar, "E_tot": E_tot,
            "C_planner": Cs, "C_nash": Cn, "gap_abs": gap_abs,
            "gap_rel": gap_abs / Cs, "rows": rows}

        print(f"\n{ba}: beta_max {beta_bar:.5g}, pool {E_tot:.1f}, "
              f"equilibrium gap {gap_abs:.4g} ({100*gap_abs/Cs:.2f}% of planner)")
        print(f"  {'market':<20s} {'n':>3s} {'rho':>6s} {'epsilon':>11s} "
              f"{'n*eps':>11s} {'n*eps/gap':>10s}")
        for r in rows:
            print(f"  {r['market']:<20s} {r['n']:>3d} {r['rho_bar']:>6.3f} "
                  f"{r['epsilon']:>11.4g} {r['n_epsilon']:>11.4g} "
                  f"{r['n_eps_over_gap']:>10.2f}")

    # headline: the trace's structure against a hyperscale one
    head = {}
    for ba in BAS:
        rows = {r["market"]: r for r in out["by_region"][ba]["rows"]}
        head[ba] = {
            "trace_n_eps_over_gap": rows["Azure trace"]["n_eps_over_gap"],
            "hyperscale_n_eps_over_gap": rows["n=3, hyperscale"]["n_eps_over_gap"],
            "ratio": (rows["n=3, hyperscale"]["n_eps_over_gap"]
                      / rows["Azure trace"]["n_eps_over_gap"]),
        }
    out["headline"] = head
    print("\nwelfare-loss bound n*epsilon as a multiple of the equilibrium gap")
    print(f"{'region':<7s} {'Azure n=32':>11s} {'n=3 hyper':>11s} {'ratio':>9s}")
    for ba in BAS:
        h = head[ba]
        print(f"{ba:<7s} {h['trace_n_eps_over_gap']:>11.3f} "
              f"{h['hyperscale_n_eps_over_gap']:>11.1f} {h['ratio']:>9.0f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT}")
