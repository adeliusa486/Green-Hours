"""Validate the fast separable-QP primitive against a general-purpose solver.

The whole experiment programme rests on solve_sep_qp being exact, so it is
checked against scipy SLSQP on random instances that exercise the deadline
staircase, saturated envelopes, and near-infeasible corners.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.optimize import minimize
from gh.core import Feasible, solve_sep_qp, solve_sep_lp


def brute(q, c, X: Feasible):
    T = X.T
    cons = [{"type": "eq", "fun": lambda x: x.sum() - X.E}]
    if X.R is not None and X.R.max() > 0:
        cons.append({"type": "ineq", "fun": lambda x: np.cumsum(x) - X.R})
    x0 = np.full(T, X.E / T)
    x0 = np.minimum(x0, X.u)
    x0 = x0 * X.E / max(x0.sum(), 1e-12)
    res = minimize(lambda x: float(q @ x + c @ (x ** 2)), x0,
                   jac=lambda x: q + 2 * c * x,
                   bounds=[(0.0, ui) for ui in X.u], constraints=cons,
                   method="SLSQP", options={"maxiter": 800, "ftol": 1e-14})
    return res.x, float(q @ res.x + c @ (res.x ** 2))


def random_instance(rng, T=12, with_stair=True):
    u = rng.uniform(0.4, 2.0, T)
    E = rng.uniform(0.25, 0.75) * u.sum()
    R = None
    if with_stair:
        # a feasible staircase: cumulative requirement below what caps allow
        frac = np.sort(rng.uniform(0, 1, T)) ** 2
        R = frac * E * rng.uniform(0.3, 0.9)
        R = np.minimum(R, np.cumsum(u) * 0.95)
        R = np.maximum.accumulate(R)
    q = rng.normal(0, 3, T)
    c = rng.uniform(0.05, 2.0, T)
    return q, c, Feasible(E=E, u=u, R=R)


def brute_trust(q, c, X: Feasible, x0):
    """A stronger reference than SLSQP.

    SLSQP was the only reference here for several drafts and it was not enough:
    it reports success=False on these sets and still returns whatever point it
    reached, so a solver that agreed with it agreed with a non-solution.
    trust-constr handles the linear inequality chain properly, and is started
    both from the candidate and from a flat profile so a shared local basin
    cannot hide a difference.
    """
    from scipy.optimize import Bounds, LinearConstraint
    T = X.T
    f = lambda z: float(q @ z + c @ (z * z))            # noqa: E731
    cons = [LinearConstraint(np.ones((1, T)), X.E, X.E)]
    if X.R is not None and X.R.max() > 0:
        cons.append(LinearConstraint(np.tril(np.ones((T, T))), X.R, np.inf))
    best = None
    for s in (x0, np.clip(np.full(T, X.E / T), 0, X.u)):
        r = minimize(f, np.clip(s, 0, X.u), jac=lambda z: q + 2 * c * z,
                     bounds=Bounds(np.zeros(T), X.u), constraints=cons,
                     method="trust-constr",
                     options=dict(maxiter=3000, gtol=1e-12, xtol=1e-14))
        z = np.clip(r.x, 0, X.u)
        ok = (abs(z.sum() - X.E) < 1e-6 * max(1.0, X.E)
              and (X.R is None
                   or np.all(np.cumsum(z) >= X.R - 1e-6 * max(1.0, X.E))))
        if ok and (best is None or f(z) < best):
            best = f(z)
    return best


def spiky_instance(rng, T=None):
    """The family that broke the old staircase decomposition.

    Two ingredients together, neither sufficient alone: a staircase with a
    large late jump -- what a short deferral horizon against a bursty arrival
    profile produces, and what E11's trace-derived sets have -- and curvature
    spanning three orders of magnitude across slots, which the measured beta
    does.  With both, the box relaxation puts its mass late, a prefix looks
    violated that is NOT active at the optimum, and forcing it to equality
    excludes the optimum.  `random_instance` above generates neither, which is
    why 400 passing cases said nothing about this one.
    """
    T = T or int(rng.integers(8, 25))
    E = float(rng.uniform(1, 10))
    u = E * rng.uniform(0.05, 0.9, T)
    if u.sum() < E * 1.05:
        u = u * (E * 1.2 / u.sum())
    R = np.zeros(T)
    j = int(rng.integers(1, T))
    R[j:] = rng.uniform(0.5, 0.98) * E
    for _ in range(int(rng.integers(0, 4))):
        i0 = int(rng.integers(0, T))
        R[i0:] = np.maximum(R[i0:], rng.uniform(0, 0.4) * E)
    R = np.minimum(np.maximum.accumulate(
        np.minimum(R, np.cumsum(u) * 0.95)), E)
    c = 10.0 ** rng.uniform(-2, 1, T)
    q = rng.normal(0, 100, T)
    return q, c, Feasible(E=E, u=u, R=R)


def test_staircase_regression():
    """The exact block E11 exposed, plus the family it came from.

    The failure was not a tolerance: the old routine returned a point 1.5%
    above the block optimum, which propagated into a planner cost ABOVE the
    cost of a feasible baseline -- a ratio below one, which is impossible, and
    is what made it visible.  Both halves of this test failed before the
    active-set repair in gh.core.solve_sep_qp and pass after it.
    """
    p = os.path.join(os.path.dirname(__file__),
                     "staircase_regression_block.npz")
    if os.path.exists(p):
        d = np.load(p)
        X = Feasible(E=float(d["E"]), u=d["u"], R=d["R"])
        q, c = d["q"], d["c"]
        x = solve_sep_qp(q, c, X)
        assert X.check(x, tol=1e-7), "regression block: infeasible"
        mine = float(q @ x + c @ (x * x))
        ref = brute_trust(q, c, X, x)
        assert ref is not None
        rel = (mine - ref) / max(1.0, abs(ref))
        assert rel < 1e-7, (f"regression block: solver is {rel:.3e} above the "
                            f"reference ({mine} vs {ref})")

    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(120):
        q, c, X = spiky_instance(rng)
        try:
            x = solve_sep_qp(q, c, X)
        except ValueError:
            continue
        assert X.check(x, tol=1e-7), "spiky family: infeasible point"
        ref = brute_trust(q, c, X, x)
        if ref is None:
            continue
        worst = max(worst, (float(q @ x + c @ (x * x)) - ref)
                    / max(1.0, abs(ref)))
    assert worst < 1e-7, f"spiky family: worst relative excess {worst:.3e}"
    return worst


def main():
    rng = np.random.default_rng(7)
    worst_obj, worst_feas = 0.0, 0.0
    n_better = 0
    N = 400
    for k in range(N):
        q, c, X = random_instance(rng, T=int(rng.integers(4, 20)),
                                  with_stair=(k % 3 != 0))
        x = solve_sep_qp(q, c, X)
        f = float(q @ x + c @ (x ** 2))
        xb, fb = brute(q, c, X)
        assert X.check(x, tol=1e-7), f"case {k}: solver returned infeasible point"
        # our objective must be <= SLSQP's (up to its own tolerance)
        rel = (f - fb) / max(1.0, abs(fb))
        worst_obj = max(worst_obj, rel)
        if f < fb - 1e-9:
            n_better += 1
        worst_feas = max(worst_feas, max(0.0, float(np.max(X.R - np.cumsum(x)))
                                         if X.R is not None else 0.0))
    print(f"solve_sep_qp vs SLSQP over {N} random instances")
    print(f"  worst relative excess over SLSQP : {worst_obj:.3e}")
    print(f"  cases where ours beat SLSQP      : {n_better}")
    print(f"  worst staircase violation        : {worst_feas:.3e}")
    assert worst_obj < 1e-7, "fast solver is not matching the reference"
    assert worst_feas < 1e-7

    # LP path
    rng = np.random.default_rng(11)
    for k in range(60):
        q, c, X = random_instance(rng, T=10, with_stair=(k % 2 == 0))
        x = solve_sep_lp(q, X)
        assert X.check(x, tol=1e-7)
        # compare against QP with vanishing curvature
        xq = solve_sep_qp(q, np.full(X.T, 1e-9), X)
        assert float(q @ x) <= float(q @ xq) + 1e-6
    print("  LP path feasible and optimal on 60 instances")
    w = test_staircase_regression()
    print(f"  staircase regression (spiky R, wide curvature): worst excess "
          f"{w:.3e} over trust-constr on 120 instances, plus the pinned block")
    print("PASS")


if __name__ == "__main__":
    main()
