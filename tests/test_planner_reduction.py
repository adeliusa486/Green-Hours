"""Regression test for the planner denominator.

Every ratio the paper reports is C(x)/C(x*), so C(x*) has to be the minimum of
C over the *product* set prod_i X_i.  An earlier version computed it by
replacing that product set with the single feasible set carrying the summed
parameters (E = sum E_i, u = sum u_i, R = sum R_i) and solving one QP in the
aggregate.  That set is a strict relaxation whenever operators differ in their
E/u ratio or their deadline staircase, so the "planner" cost came out too low
and every ratio measured against it came out too high.

This test pins three things:

  1. the exact planner value agrees with SciPy SLSQP on the full n*T problem;
  2. the aggregate relaxation is a genuine LOWER bound, and is strictly loose on
     instances with heterogeneous staircases -- so the bug cannot be
     reintroduced silently;
  3. the profile `_planner_fast` returns is feasible for every operator and
     attains the exact value.

Run:  python test_planner_reduction.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from gh.core import Feasible, Instance                      # noqa: E402
from gh.baselines import planner_value, planner_aggregate, _planner_fast  # noqa: E402


def random_instance(rng, homogeneous=False):
    """One random instance.  `homogeneous` gives every operator the same E/u
    ratio and the same staircase, which is the only regime in which the
    aggregate relaxation is tight."""
    T = int(rng.integers(6, 20))
    n = int(rng.integers(2, 9))
    a = rng.uniform(50.0, 400.0, T)
    m = a * rng.uniform(1.05, 3.0, T)
    b = rng.uniform(1e-4, 5e-2, T)
    slack0 = rng.uniform(1.5, T)
    cap0 = rng.uniform(0.15, 1.0)
    X = []
    for _ in range(n):
        E = float(rng.uniform(1.0, 20.0))
        cap = cap0 if homogeneous else rng.uniform(0.15, 1.0)
        u = np.full(T, E * cap)
        while u.sum() < E * 1.05:
            u = u * 1.5
        slack = slack0 if homogeneous else rng.uniform(1.5, T)
        done = np.clip(np.arange(T) / slack, 0.0, 1.0)
        R = np.maximum.accumulate(
            np.minimum(0.85 * E * done ** 1.7, np.cumsum(u) * 0.9))
        X.append(Feasible(E=E, u=u, R=R))
    return Instance(a=a, m=m, b=b, X=X, psi=np.zeros((n, T)), lam=np.zeros(n))


def slsqp_planner(inst):
    """C(x*) from SciPy, as an independent second opinion."""
    from scipy.optimize import minimize, LinearConstraint, Bounds
    n, T = inst.n, inst.T

    def f(z):
        y = z.reshape(n, T).sum(axis=0)
        return float(np.sum(inst.m * y + inst.b * y ** 2))

    def g(z):
        y = z.reshape(n, T).sum(axis=0)
        return np.tile(inst.m + 2.0 * inst.b * y, (n, 1)).ravel()

    rows, lo, hi = [], [], []
    tri = np.tril(np.ones((T, T)))
    for i, Xi in enumerate(inst.X):
        e = np.zeros((1, n * T))
        e[0, i * T:(i + 1) * T] = 1.0
        rows.append(e)
        lo.append(Xi.E)
        hi.append(Xi.E)
        M = np.zeros((T, n * T))
        M[:, i * T:(i + 1) * T] = tri
        rows.append(M)
        lo.extend(Xi.R.tolist())
        hi.extend([np.inf] * T)
    lc = LinearConstraint(np.vstack(rows), np.array(lo), np.array(hi))
    ub = np.concatenate([Xi.u for Xi in inst.X])
    res = minimize(f, inst.feasible_start().ravel(), jac=g, constraints=[lc],
                   bounds=Bounds(np.zeros(n * T), ub), method="SLSQP",
                   options=dict(maxiter=3000, ftol=1e-14))
    return f(res.x)


def aggregate_lower_bound(inst):
    y = planner_aggregate(inst)
    return float(np.sum(inst.m * y + inst.b * y ** 2))


def main():
    rng = np.random.default_rng(20260907)
    worst_vs_slsqp = 0.0
    worst_slack = 0.0
    n_loose = 0
    n_infeasible = 0
    n_below_lb = 0
    trials = 120

    for _ in range(trials):
        inst = random_instance(rng)
        exact = planner_value(inst)
        lb = aggregate_lower_bound(inst)

        # 1. agreement with an independent solver
        ref = slsqp_planner(inst)
        worst_vs_slsqp = max(worst_vs_slsqp, abs(exact - ref) / abs(ref))

        # 2. the relaxation is a lower bound, and is usually strictly loose
        if exact < lb - 1e-7 * abs(lb):
            n_below_lb += 1
        slack = (exact - lb) / abs(exact)
        worst_slack = max(worst_slack, slack)
        if slack > 1e-6:
            n_loose += 1

        # 3. the returned profile is feasible and attains the value
        xp = _planner_fast(inst)
        if not all(inst.X[i].check(xp[i], tol=1e-6) for i in range(inst.n)):
            n_infeasible += 1
        assert abs(inst.social(xp) - exact) <= 1e-6 * abs(exact)

    # the relaxation must be TIGHT when operators are homogeneous
    worst_hom = 0.0
    for _ in range(40):
        inst = random_instance(rng, homogeneous=True)
        worst_hom = max(worst_hom,
                        abs(planner_value(inst) - aggregate_lower_bound(inst))
                        / abs(planner_value(inst)))

    print("test_planner_reduction")
    print(f"  exact planner vs SLSQP, {trials} instances : "
          f"worst relative gap {worst_vs_slsqp:.3e}")
    print(f"  exact planner below the aggregate bound    : {n_below_lb}")
    print(f"  aggregate relaxation strictly loose        : {n_loose}/{trials} "
          f"(worst {100 * worst_slack:.2f}% -- this is the bug it replaced)")
    print(f"  relaxation tight on homogeneous operators  : worst "
          f"{worst_hom:.3e}")
    print(f"  infeasible planner profiles                : {n_infeasible}")

    ok = (worst_vs_slsqp < 1e-6 and n_below_lb == 0 and n_infeasible == 0
          and n_loose > trials // 2 and worst_hom < 1e-9)
    print("  ALL PASS" if ok else "  FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
