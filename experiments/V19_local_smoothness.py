"""V19 -- numerical check of the local-smoothness inequality behind Theorem 1.

An earlier version of the proof discharged the 3/2 factor by exhibiting each
X_i as the flow polytope of a series-parallel network and citing the tight
atomic-splittable bound.  The construction is correct but the citation is not
free: the cited bound is proved for uncapacitated flows, and X_i carries per-slot
capacities and prefix lower bounds, so "applies verbatim" was doing work it had
not earned.

The reduction turns out to be unnecessary.  The bound follows from local
smoothness alone, which needs nothing about the network -- only that each X_i is
a convex subset of the non-negative orthant.  Per resource, writing
c(y) = a + b y with a, b >= 0:

    LHS_t(x, x*) = a y* + b y y* + b sum_i x_i (x*_i - x_i)
                <= a y* + b y y* + b y*^2 / 4                        (i)
                <= (a y* + b y*^2) + (1/3)(a y + b y^2)              (ii)

(i) because x(x* - x) <= x*^2/4 termwise and sum_i x*_i^2 <= y*^2 for
non-negative shares; (ii) because (3/4)y*^2 - y y* + (1/3)y^2 is the perfect
square (sqrt3 y*/2 - y/sqrt3)^2 >= 0.  So the game is (1, 1/3)-locally smooth
and PoA <= 1/(1 - 1/3) = 3/2.

This script checks both inequalities and the resulting bound on random
instances, including the tight cases (y = 1.5 y*, x_i = x*_i / 2), so that a
sign error in the algebra would show up rather than being argued about.
"""
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "V19.json")

LAM, MU = 1.0, 1.0 / 3.0


def lhs_resource(a, b, x, xs):
    """Local-smoothness left side on one resource: the Nash cost plus the
    first-order deviation term, summed over players."""
    y, ys = x.sum(), xs.sum()
    return a * ys + b * y * ys + b * float(np.sum(x * (xs - x)))


def rhs_resource(a, b, x, xs):
    y, ys = x.sum(), xs.sum()
    return LAM * (a * ys + b * ys ** 2) + MU * (a * y + b * y ** 2)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    worst_gap = np.inf          # rhs - lhs; must stay >= 0
    worst_case = None
    n_checked = 0

    # (1) random instances over a wide range of n, curvature and asymmetry
    for trial in range(200000):
        n = int(rng.integers(1, 9))
        a = float(rng.uniform(0, 500))
        b = float(rng.uniform(0, 1)) * 10 ** rng.uniform(-4, 0)
        scale = 10 ** rng.uniform(-3, 3)
        x = rng.random(n) * scale
        xs = rng.random(n) * scale
        if rng.random() < 0.15:           # sometimes make one player dominant
            x[0] *= 20
        if rng.random() < 0.15:
            xs[0] *= 20
        g = rhs_resource(a, b, x, xs) - lhs_resource(a, b, x, xs)
        n_checked += 1
        if g < worst_gap:
            worst_gap = g
            worst_case = dict(n=n, a=a, b=b, x=x.tolist(), xs=xs.tolist())

    # (2) the cases where the two inequalities are tight simultaneously:
    #     x_i = x*_i / 2 makes (i) an equality, y = 1.5 y* makes (ii) one.
    #     Both at once needs n players with x_i = x*_i/2 and sum x = 1.5 sum x*,
    #     which is only consistent at x* = 0; so drive each to equality alone.
    tight = []
    for n in range(1, 9):
        xs = np.full(n, 1.0)
        # (i) tight: x_i = x*_i / 2 for every i
        x = xs / 2.0
        tight.append(("(i) tight", n,
                      rhs_resource(0.0, 1.0, x, xs) - lhs_resource(0.0, 1.0, x, xs)))
        # (ii) tight: y = 1.5 y*, mass spread evenly
        x = xs * 1.5
        tight.append(("(ii) tight", n,
                      rhs_resource(0.0, 1.0, x, xs) - lhs_resource(0.0, 1.0, x, xs)))
        # single player carrying everything, the asymmetric extreme
        x = np.zeros(n)
        x[0] = 1.5 * n
        tight.append(("(ii) tight, one player", n,
                      rhs_resource(0.0, 1.0, x, xs) - lhs_resource(0.0, 1.0, x, xs)))

    print(f"random resources checked: {n_checked}")
    print(f"worst (rhs - lhs): {worst_gap:.6e}   "
          f"{'OK' if worst_gap >= -1e-9 else 'VIOLATED'}")
    print("\ntight constructions (gap should be ~0 or positive, never negative)")
    for lab, n, g in tight:
        print(f"  {lab:<24s} n={n}  gap {g:+.3e}")
    worst_tight = min(g for _, _, g in tight)

    out = {"provenance": "Analytic check of the local-smoothness inequality "
                         "used in the proof of Theorem 1.  No grid data, no "
                         "workload: this is algebra, checked numerically.",
           "lambda": LAM, "mu": MU, "poa_bound": LAM / (1 - MU),
           "n_random_resources": n_checked,
           "worst_slack_random": worst_gap,
           "worst_slack_tight": worst_tight,
           "holds": bool(worst_gap >= -1e-9 and worst_tight >= -1e-9),
           "worst_case_random": worst_case}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n(lambda, mu) = ({LAM}, {MU:.4f})  ->  PoA <= {LAM/(1-MU):.4f}")
    print(f"holds: {out['holds']}")
    print(f"wrote {OUT}")
