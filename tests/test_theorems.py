"""Independent numerical audit of every theorem the paper states.

This file deliberately does NOT reuse the paper's derivations.  Each check
re-derives the claim from the primitives and then tries to break it.

  T1  Proposition 1   Phi is an exact potential; argmin Phi is the unique NE.
  T2  Lemma 1         dC/dx_it - dJ_i/dx_it = (MEF-AEF) + beta*(y - x_i),
                      as an identity in x, not only at an optimum.
  T3  Theorem 1       PoA <= 3/(2(1-eta)) with eta = max over the ACTIVE set.
  T4  Proposition 2   PoA <= (3/2) * max(rho,1) / min(sigma,1) for a convex
                      response.  The clip is what makes the chaining valid when
                      a deferral penalty is present; the check runs both forms
                      and reports each, rather than asserting the clip is
                      empirically necessary.
  T5  Theorem 2       at an interior equilibrium, E[C] has no first-order term
                      in the signal dispersion and rises at order sigma^2.
  T6  Proposition 3   two-slot thresholds, and that the window is empty iff
                      n <= 3.
  T7  Theorem 3       every fixed point of SHADE minimises C, and the aggregate
                      is unique while the profile need not be.
  T8  Theorem 4       an eps-Nash profile of a game with potential C satisfies
                      C(x) - C(x*) <= n*eps.  (The paper asserts this; here it
                      is checked numerically as well as proved in the paper.)

Run:  python test_theorems.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from gh.core import (Feasible, Instance, nash, planner, solve_sep_qp,  # noqa: E402
                     nash_two_slot)
from gh.baselines import planner_value                                # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


def rand_instance(rng, n=None, T=None, lam=0.0, cap=None):
    T = T or int(rng.integers(5, 14))
    n = n or int(rng.integers(2, 7))
    a = rng.uniform(60.0, 380.0, T)
    m = a * rng.uniform(1.02, 3.2, T)
    b = rng.uniform(1e-3, 8e-2, T)
    X = []
    for _ in range(n):
        E = float(rng.uniform(2.0, 15.0))
        u = np.full(T, E * (cap if cap else rng.uniform(0.25, 1.0)))
        while u.sum() < E * 1.05:
            u = u * 1.4
        X.append(Feasible(E=E, u=u, R=None))
    psi = rng.uniform(0.0, 1.0, (n, T))
    return Instance(a=a, m=m, b=b, X=X, psi=psi, lam=np.full(n, float(lam)))


# ---------------------------------------------------------------- T1, T2
def t1_t2():
    rng = np.random.default_rng(11)
    worst_pot, worst_lem = 0.0, 0.0
    for _ in range(200):
        inst = rand_instance(rng, lam=float(rng.uniform(0, 40)))
        x = inst.feasible_start()
        i = int(rng.integers(0, inst.n))
        xi2 = solve_sep_qp(rng.uniform(-5, 5, inst.T), np.ones(inst.T), inst.X[i])
        x2 = x.copy(); x2[i] = xi2
        dJ = inst.agent_cost(x2, i) - inst.agent_cost(x, i)
        dP = inst.potential(x2) - inst.potential(x)
        worst_pot = max(worst_pot, abs(dJ - dP) / max(1.0, abs(dJ)))

        # Lemma 1 as an identity in x, by finite differences
        y = x.sum(0); h = 1e-5
        t = int(rng.integers(0, inst.T))
        e = np.zeros_like(x); e[i, t] = h
        dC = (inst.social(x + e) - inst.social(x - e)) / (2 * h)
        dJi = (inst.agent_cost(x + e, i) - inst.agent_cost(x - e, i)) / (2 * h)
        pred = (inst.m[t] - inst.a[t]) + inst.b[t] * (y[t] - x[i, t])
        worst_lem = max(worst_lem, abs((dC - dJi) - pred) / max(1.0, abs(pred)))
    check("T1 Phi is an exact potential", worst_pot < 1e-9, f"worst {worst_pot:.2e}")
    check("T2 Lemma 1 holds as an identity in x", worst_lem < 1e-5,
          f"worst {worst_lem:.2e}")


# ---------------------------------------------------------------- T3
def t3():
    rng = np.random.default_rng(12)
    worst = 0.0
    viol = 0
    for _ in range(250):
        inst = rand_instance(rng, lam=float(rng.choice([0.0, 20.0])))
        xn = nash(inst, tol=1e-11)[0]
        Cs = planner_value(inst)
        xp = planner(inst, tol=1e-11)[0]
        A = (xn.sum(0) > 1e-9) | (xp.sum(0) > 1e-9)
        eta = float(np.max(1.0 - inst.a[A] / inst.m[A]))
        poa = inst.social(xn) / Cs
        bound = 1.5 / (1.0 - eta)
        if poa > bound + 1e-9:
            viol += 1
        worst = max(worst, poa / bound)
    check("T3 PoA bound never violated", viol == 0,
          f"{viol}/250 violations, worst ratio to bound {100*worst:.1f}%")


# ---------------------------------------------------------------- T4
def t4():
    """A convex response that is NOT the affine one: a piecewise-quadratic with
    a kink, evaluated against the perceived affine cost."""
    rng = np.random.default_rng(13)
    viol_raw = viol_clip = 0
    for _ in range(150):
        lam = float(rng.choice([0.0, 25.0]))
        inst = rand_instance(rng, lam=lam)
        T = inst.T
        Etot = sum(Xi.E for Xi in inst.X)
        H = rng.uniform(0.1, 0.6, T) * Etot          # headroom per slot
        jump = float(rng.uniform(2.0, 8.0))

        def true_inc(y):                              # E_t(D+y) - E_t(D)
            lo = np.minimum(y, H)
            hi = np.maximum(y - H, 0.0)
            return (inst.m * lo + inst.b * lo ** 2
                    + jump * inst.m * hi + jump * inst.b * hi ** 2
                    + 2.0 * jump * inst.b * lo * hi)

        def C_true(x):
            return float(np.sum(true_inc(x.sum(0)))) + inst.pen(x)

        # rho / sigma over the reachable range, per slot
        ymax = np.minimum(sum(Xi.u for Xi in inst.X), Etot)
        rho = np.zeros(T); sig = np.zeros(T)
        for t in range(T):
            yy = np.linspace(max(ymax[t], 1e-9) / 2000, max(ymax[t], 1e-9), 2000)
            lo = np.minimum(yy, H[t]); hi = np.maximum(yy - H[t], 0.0)
            tr = (inst.m[t] * lo + inst.b[t] * lo ** 2
                  + jump * inst.m[t] * hi + jump * inst.b[t] * hi ** 2
                  + 2.0 * jump * inst.b[t] * lo * hi)
            per = yy * (inst.a[t] + inst.b[t] * yy)
            r = tr / per
            rho[t] = r.max(); sig[t] = r.min()

        xn = nash(inst, tol=1e-11)[0]
        # planner for the TRUE response, by block descent on C_true
        xp = xn.copy()
        for _ in range(300):
            moved = 0.0
            for i in range(inst.n):
                s = xp.sum(0) - xp[i]
                g = 1e-4
                # local quadratic surrogate: minimise C_true(s + x_i) exactly by
                # a fine 1-D search is too slow, so use a gradient step with the
                # exact derivative and project
                y = s + xp[i]
                lo = np.minimum(y, H); hi = np.maximum(y - H, 0.0)
                grad = np.where(y <= H, inst.m + 2 * inst.b * y,
                                jump * (inst.m + 2 * inst.b * (y - H))
                                + inst.m + 2 * inst.b * H)
                grad = grad + inst.lam[i] * inst.psi[i]
                new = solve_sep_qp(grad - 2.0 * g * xp[i], np.full(inst.T, g),
                                   inst.X[i])
                moved = max(moved, float(np.max(np.abs(new - xp[i]))))
                xp[i] = new
            if moved < 1e-10:
                break
        poa = C_true(xn) / C_true(xp)
        braw = 1.5 * rho.max() / sig.min()
        bclip = 1.5 * max(rho.max(), 1.0) / min(sig.min(), 1.0)
        if poa > braw + 1e-9:
            viol_raw += 1
        if poa > bclip + 1e-9:
            viol_clip += 1
    check("T4 generalised bound, with the clip at 1", viol_clip == 0,
          f"{viol_clip}/150 violations")
    print(f"       (unclipped form: {viol_raw}/150 violations. The clip is "
          f"required by the PROOF -- absorbing the common deferral penalty "
          f"needs rho>=1 one way and sigma<=1 the other -- not by anything "
          f"observed here.)")


# ---------------------------------------------------------------- T5
def t5():
    """Interior equilibrium: E[C(sigma)] - C(0) should have no first-order term
    in the signal dispersion and rise at order sigma^2.

    The construction has to earn the word "interior".  Best response has slope
    1/(2*beta) in the perceived signal, so a perturbation of size sigma*AEF
    moves a slot's load by about sigma*AEF/(2*beta); unless that is small
    against the equilibrium load itself, the non-negativity constraint binds and
    the instance is not interior at all -- which is what an earlier version of
    this check got wrong, reporting a failure of the theorem when it had simply
    tested the wrong regime.  We therefore fix a flat signal (so the sigma = 0
    equilibrium is the uniform profile, strictly inside every constraint) and
    take beta large enough that the active set is provably unchanged, and we
    ASSERT that it was unchanged rather than assuming it.

    This is also the honest reading of the theorem: the interior window is
    narrow.  At the beta measured on EIA-930 it corresponds to signal noise
    below roughly one percent of the mean intensity, so real forecasts sit
    outside it and the equilibrium is at a boundary -- which is why the paper
    reports the sign as indeterminate rather than leading with the interior
    case.
    """
    n, T, E = 5, 5, 10.0
    sigmas = [0.0005, 0.001, 0.002, 0.004, 0.008]
    ok_pos = ok_quad = ok_intact = 0
    trials = 0
    for beta in (50.0, 200.0, 500.0):
        a = np.full(T, 300.0)
        inst = Instance(a=a, m=a * 1.5, b=np.full(T, beta),
                        X=[Feasible(E=E, u=np.full(T, E), R=None)
                           for _ in range(n)],
                        psi=np.zeros((n, T)), lam=np.zeros(n))
        c0 = inst.social(nash(inst, ftol=1e-13)[0])
        d = []
        intact = True
        for sg in sigmas:
            acc = []
            for s in range(400):
                r = np.random.default_rng(9000 + s)
                S = a[None, :] + 300.0 * sg * r.standard_normal((n, T))
                xs = nash(inst, signal=S, ftol=1e-13)[0]
                if xs.min() < 1e-9 or xs.max() > E - 1e-9:
                    intact = False
                acc.append(inst.social(xs))
            d.append(np.mean(acc) - c0)
        trials += 1
        d = np.array(d)
        if intact:
            ok_intact += 1
        if np.all(d > 0):
            ok_pos += 1
        p = np.polyfit(np.log(sigmas), np.log(d), 1)[0]
        if 1.9 < p < 2.1:
            ok_quad += 1
    check("T5 the tested equilibria really are interior", ok_intact == trials,
          f"{ok_intact}/{trials} with the active set provably unchanged")
    check("T5 interior: cost rises with signal noise", ok_pos == trials,
          f"{ok_pos}/{trials}")
    check("T5 interior: the rise is quadratic, no first-order term",
          ok_quad == trials, f"{ok_quad}/{trials} with exponent in (1.9,2.1)")


# ---------------------------------------------------------------- T6
def t6():
    """Two-slot: Assumption 1 iff Delta >= beta*D*(1+1/n); spreading helps iff
    Delta < 2*beta*D*(1-1/n); compatible iff n >= 4."""
    beta, D = 0.5, 200.0
    bad = 0
    for n in (2, 3, 4, 8, 32):
        lo = beta * D * (1 + 1.0 / n)
        hi = 2 * beta * D * (1 - 1.0 / n)
        for delta in np.linspace(0.2 * lo, 1.6 * max(lo, hi), 40):
            inst = _two_slot(n, D, delta, beta)
            x, _ = nash_two_slot(inst)
            corner = x[:, 0].sum() > D - 1e-7
            if corner != (delta >= lo - 1e-9):
                bad += 1
        empty = lo >= hi - 1e-12
        if empty != (n <= 3):
            bad += 1
    check("T6 two-slot thresholds and the n>=4 window", bad == 0,
          f"{bad} mismatches over 5 values of n")


def _two_slot(n, D, delta, beta):
    Ei = D / n
    a = np.array([0.0, float(delta)])
    return Instance(a=a, m=a.copy(), b=np.array([beta, beta]),
                    X=[Feasible(E=Ei, u=np.full(2, Ei), R=None) for _ in range(n)],
                    psi=np.zeros((n, 2)), lam=np.zeros(n))


# ---------------------------------------------------------------- T7
def t7():
    """SHADE's fixed point minimises C, and its aggregate is unique."""
    rng = np.random.default_rng(15)
    worst_gap = 0.0
    worst_y = 0.0
    for _ in range(60):
        inst = rand_instance(rng, lam=0.0)
        Cs = planner_value(inst)
        n = inst.n
        x = inst.feasible_start(); yhat = x.sum(0).copy(); prev = yhat.copy()
        for _ in range(4000):
            for i in range(n):
                ymi = yhat - x[i]
                q = inst.a + inst.b * ymi + ((inst.m - inst.a) + inst.b * ymi)
                x[i] = solve_sep_qp(q, inst.b, inst.X[i])
            y = x.sum(0)
            yhat = np.maximum(prev + (1.8 / n) * (y - prev), 0.0)
            if np.max(np.abs(yhat - prev)) <= 1e-13 * max(1.0, float(np.max(yhat))):
                break
            prev = yhat
        worst_gap = max(worst_gap, inst.social(x) / Cs - 1.0)
        yp = planner(inst, tol=1e-12)[0].sum(0)
        worst_y = max(worst_y, float(np.max(np.abs(x.sum(0) - yp)))
                      / max(1.0, float(np.max(yp))))
    check("T7 SHADE fixed point attains the planner cost", worst_gap < 1e-7,
          f"worst excess {worst_gap:.2e}")
    check("T7 SHADE reaches the planner AGGREGATE", worst_y < 1e-5,
          f"worst relative |y - y*| {worst_y:.2e}")


# ---------------------------------------------------------------- T8
def t8():
    """If x is eps-Nash of a game whose potential is C, then
    C(x) - C(x*) <= n*eps.  Measure eps directly and check the implication."""
    rng = np.random.default_rng(16)
    bad = 0
    worst_slack = np.inf
    for _ in range(80):
        inst = rand_instance(rng, lam=0.0)
        Cs = planner_value(inst)
        # a deliberately under-converged SHADE run, so eps > 0
        n = inst.n
        x = inst.feasible_start(); yhat = x.sum(0).copy(); prev = yhat.copy()
        for _ in range(int(rng.integers(1, 8))):
            for i in range(n):
                ymi = yhat - x[i]
                q = inst.a + inst.b * ymi + ((inst.m - inst.a) + inst.b * ymi)
                x[i] = solve_sep_qp(q, inst.b, inst.X[i])
            y = x.sum(0)
            yhat = np.maximum(prev + (1.8 / n) * (y - prev), 0.0)
            prev = yhat
        # eps = max over i of the unilateral improvement in C
        eps = 0.0
        for i in range(n):
            s = x.sum(0) - x[i]
            bi = solve_sep_qp(inst.m + 2.0 * inst.b * s, inst.b, inst.X[i])
            xa = x.copy(); xa[i] = bi
            eps = max(eps, inst.social(x) - inst.social(xa))
        lhs = inst.social(x) - Cs
        if lhs > n * eps + 1e-6 * max(1.0, Cs):
            bad += 1
        if eps > 0:
            worst_slack = min(worst_slack, n * eps / max(lhs, 1e-12))
    check("T8 eps-Nash implies C(x) - C(x*) <= n*eps", bad == 0,
          f"{bad}/80 violations, tightest n*eps/gap = {worst_slack:.2f}x")


if __name__ == "__main__":
    print("test_theorems -- independent audit of the paper's claims\n")
    for f in (t1_t2, t3, t4, t5, t6, t7, t8):
        print(f.__name__ + ":")
        f()
    print("\n" + ("ALL PASS" if not FAIL else "FAILURES: " + ", ".join(FAIL)))
    sys.exit(1 if FAIL else 0)
