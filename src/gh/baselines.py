"""Baselines.

Beyond the seven in the paper this adds the two that REVIEW.md X1/X2 argue are
essential:

  B0  naive       one-shot best response to the PUBLISHED forecast, with no
                  anticipation of aggregate response.  This is what a deployed
                  day-ahead scheduler actually does; Nash is a strictly more
                  sophisticated model of practice than practice deserves.
  B8a mef_static  agents charged the published MARGINAL factor m_t.
  B8b mef_resp    agents charged the responsive marginal factor m_t + 2 b_t y_t,
                  i.e. the fix the paper's own discussion recommends.

MARL (the paper's B4) is not here: it needs GPU training and is on the user's
side of the plan.
"""

from __future__ import annotations
import numpy as np
from .core import Instance, nash, planner, solve_sep_qp, solve_sep_lp, \
    mef_signal_equilibrium
from .mech import shade


def project(inst: Instance, target):
    """Euclidean projection of a target profile onto each X_i."""
    x = np.zeros((inst.n, inst.T))
    for i in range(inst.n):
        x[i] = solve_sep_qp(-2.0 * target[i], np.ones(inst.T), inst.X[i])
    return x


def carbon_agnostic(inst: Instance):
    """Run on arrival: the closest feasible profile to the raw arrival shape."""
    return project(inst, inst.arrival)


def naive(inst: Instance):
    """B0.  Minimise sum_t x_it * a_t against the published forecast, with no
    congestion term at all.  Linear, so every operator picks the same slots."""
    x = np.zeros((inst.n, inst.T))
    tie = 1e-9 * np.arange(inst.T)          # deterministic tie-break
    for i in range(inst.n):
        x[i] = solve_sep_lp(inst.a + tie + inst.lam[i] * inst.psi[i], inst.X[i])
    return x


def mef_static(inst: Instance):
    """B8a.  Same as naive but against a published marginal factor."""
    x = np.zeros((inst.n, inst.T))
    tie = 1e-9 * np.arange(inst.T)
    for i in range(inst.n):
        x[i] = solve_sep_lp(inst.m + tie + inst.lam[i] * inst.psi[i], inst.X[i])
    return x


def threshold(inst: Instance, pct=35.0):
    """B3.  Wait-for-a-cleaner-hour: place work only in slots whose published
    intensity is below the pct-th percentile, earliest such slot first."""
    tau = np.percentile(inst.a, pct)
    w = np.maximum(inst.a - tau, 0.0)
    tie = 1e-6 * np.arange(inst.T)
    x = np.zeros((inst.n, inst.T))
    for i in range(inst.n):
        x[i] = solve_sep_lp(w + tie + inst.lam[i] * inst.psi[i], inst.X[i])
    return x


def jitter(inst: Instance, sigma=0.10, seed=0, **kw):
    """B5.  Independent per-operator dispersion on the common signal -- the
    cheapest decorrelation heuristic, and the direct empirical probe of
    Theorem 2."""
    rng = np.random.default_rng(seed)
    S = inst.a[None, :] * (1.0 + sigma * rng.standard_normal((inst.n, inst.T)))
    S = np.maximum(S, 1e-6)
    return nash(inst, signal=S, **kw)[0]


def proportional_cap(inst: Instance, declare=None, slack=1.5, **kw):
    """B6.  A disclosure-based coordinator: compute the aggregate optimum, hand
    each operator a per-slot quota pro rata to its declared demand, then let
    operators optimise inside the quota.  `declare` lets an operator inflate its
    declaration, which is the manipulation the paper mentions.

    Returns None when the quota empties some operator's feasible set, which is
    what happens on every calibrated instance: the planner concentrates load
    into a few hours, a pro-rata slice of that concentration is narrower than
    the deadline staircase allows, and no feasible schedule exists inside it.
    An earlier version caught the exception and returned the planner profile
    instead -- so the baseline scored the planner's own cost and appeared to
    beat every mechanism.  Reporting the scheme as inapplicable is the finding;
    a number would have been fiction.
    """
    from .core import Feasible
    xp = _planner_fast(inst)
    ystar = xp.sum(axis=0)
    E = np.array([Xi.E for Xi in inst.X])
    d = E.copy() if declare is None else np.asarray(declare, float)
    share = d / d.sum()
    X2 = []
    for i in range(inst.n):
        u = np.minimum(inst.X[i].u,
                       np.maximum(slack * share[i] * ystar,
                                  inst.X[i].E / inst.T * 1e-3))
        if u.sum() < inst.X[i].E - 1e-9:
            return None                      # quota cannot carry E_i
        R = inst.X[i].R
        if R is not None and np.any(np.cumsum(u) < R - 1e-9):
            return None                      # quota cannot meet the deadlines
        X2.append(Feasible(E=inst.X[i].E, u=u, R=R))
    inst2 = Instance(a=inst.a, m=inst.m, b=inst.b, X=X2,
                     psi=inst.psi, lam=inst.lam, arrival=inst.arrival)
    try:
        return nash(inst2, **kw)[0]
    except ValueError:
        return None


def planner_aggregate(inst: Instance):
    """The planner's AGGREGATE profile under the *aggregate relaxation*.

    With no deferral penalty C depends on x only through y, so one is tempted
    to replace the product set prod_i X_i by the single set carrying the summed
    parameters (E = sum E_i, u = sum u_i, R = sum R_i) and solve one QP in y.

    THAT SET IS A STRICT RELAXATION IN GENERAL.  These feasible sets are base
    polytopes of laminar polymatroids (the constraints are a chain of suffixes
    plus a partition into singletons), and the Minkowski sum of base polytopes
    is the base polytope of the SUM OF THE RANK FUNCTIONS -- which is not the
    rank function built from the summed bounds, because
    min(a1,b1) + min(a2,b2) can be strictly less than min(a1+a2, b1+b2).
    Concretely, with u_1 = (5,5), E_1 = 1 and u_2 = (3,3), E_2 = 5, no profile
    puts more than 1 + 3 = 4 units in slot 1 while the relaxation allows 6.
    Equality holds when every operator has the same E/u ratio and the same
    deadline staircase; `tests/test_planner_reduction.py` pins both directions.

    So this returns a LOWER bound on the planner's cost.  Use `planner_value`
    for anything reported; this is kept because the cliff solver and the bound
    scans need a cheap aggregate scan range, and because `planner_value` uses
    it as a certificate.
    """
    from .core import Feasible
    agg = Feasible(E=sum(X.E for X in inst.X),
                   u=sum(X.u for X in inst.X),
                   R=(sum(X.R for X in inst.X)
                      if inst.X[0].R is not None else None))
    return solve_sep_qp(inst.m, inst.b, agg)


def _planner_bcd(inst: Instance, ftol=1e-13, max_sweeps=5000, patience=3):
    """Exact block minimisation of C over prod_i X_i, stopped on the OBJECTIVE.

    With lam = 0, C depends on x only through the aggregate y, so the minimiser
    is a face rather than a point and the ITERATE need not converge even though
    the value does: cyclic block descent slides along the face indefinitely.
    Stopping on ||x^(k) - x^(k-1)|| therefore never fires, which is how a
    tolerance of 1e-12 turned a three-sweep solve into an unbounded loop.  Each
    block subproblem is strictly convex (curvature b_t > 0) so its minimiser is
    unique, and cyclic block descent on a convex differentiable objective over a
    Cartesian product of compact convex sets converges in value to the global
    minimum; that value is what every ratio in the paper divides by.
    """
    x = inst.feasible_start()
    y = x.sum(axis=0)
    prev = inst.social(x)
    flat = 0
    for sweep in range(max_sweeps):
        for i in range(inst.n):
            s = y - x[i]
            xi = solve_sep_qp(inst.m + 2.0 * inst.b * s
                              + inst.lam[i] * inst.psi[i], inst.b, inst.X[i])
            y = s + xi
            x[i] = xi
        cur = inst.social(x)
        if prev - cur <= ftol * max(1.0, abs(prev)):
            flat += 1
            if flat >= patience:
                return x, sweep + 1
        else:
            flat = 0
        prev = cur
    return x, max_sweeps


def planner_value(inst: Instance, ftol=1e-13):
    """C(x*), the true social optimum over prod_i X_i.

    Computed by `_planner_bcd` above, which stops on the objective rather
    than on the iterate; `tests/test_planner_reduction.py` checks the value
    against SciPy SLSQP.

    History.  This used to return the aggregate relaxation of
    `planner_aggregate` whenever lam = 0.  That relaxation is not tight when
    operators have different deadline staircases, which they do in every
    experiment here, and it understated C(x*) by 0.5-0.9% on the calibrated
    instances -- inflating every ratio reported against it, including the
    equilibrium gap and the share of that gap the mechanism removes.  The
    relaxation is retained below only as a lower-bound certificate.
    """
    x = _planner_bcd(inst, ftol=ftol)[0]
    v = inst.social(x)
    if np.allclose(inst.lam, 0.0):
        lb = float(np.sum(inst.m * (y := planner_aggregate(inst)) + inst.b * y ** 2))
        if v < lb - 1e-7 * max(1.0, abs(lb)):
            raise RuntimeError(
                f"planner_value {v!r} below the aggregate relaxation {lb!r}")
    return float(v)


def _planner_fast(inst: Instance, ftol=1e-13):
    """A feasible planner PROFILE attaining C(x*).

    Only needed where a profile is required (the quota baseline, the
    operational metrics); ratios should use `planner_value`.  This used to
    decompose the aggregate relaxation greedily, which returned profiles that
    failed `Feasible.check` on 299 of 300 random instances; it now runs the
    same exact block descent as `planner_value`.
    """
    return _planner_bcd(inst, ftol=ftol)[0]


REGISTRY = {
    "carbon_agnostic": lambda inst, **kw: carbon_agnostic(inst),
    "naive":           lambda inst, **kw: naive(inst),
    "threshold":       lambda inst, **kw: threshold(inst),
    "nash":            lambda inst, **kw: nash(inst, **kw)[0],
    "jitter":          lambda inst, **kw: jitter(inst, **kw),
    "mef_static":      lambda inst, **kw: mef_static(inst),
    "mef_responsive":  lambda inst, **kw: mef_signal_equilibrium(inst)[0],
    "prop_cap":        lambda inst, **kw: proportional_cap(inst, **kw),
    "shade":           lambda inst, **kw: shade(inst, **kw)[0],
    "planner":         lambda inst, **kw: _planner_fast(inst),
}
