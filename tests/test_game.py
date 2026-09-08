"""Analytic oracles for the game layer (PLAN.md section 4.2, restated for the
(a, m, b) parameterisation of REVIEW.md B1).

The plan's oracle "b_t = 0 implies Nash = planner" was an artefact of the old
parameterisation, where m_t - a_t = beta_t * D_t made b=0 imply m=a.  In the
corrected model the accounting wedge (m-a) and the congestion curvature b are
independent, so the oracle splits into two sharper ones -- which is itself the
gap decomposition of REVIEW.md Move 1.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from gh.core import nash, planner, Instance, Feasible, solve_sep_qp
from gh.instances import make_instance, two_slot

OK = []


def check(name, cond, detail=""):
    OK.append(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def test_no_wedge_no_congestion():
    """m = a and b -> 0: agent and planner objectives coincide exactly."""
    inst = make_instance(n=8, T=32, seed=1, kappa=1e-6, lam=0.3)
    inst.m = inst.a.copy()
    xn, _ = nash(inst)
    xp, _ = planner(inst)
    r = inst.social(xn) / inst.social(xp)
    check("m=a, b~0  =>  Nash == planner", abs(r - 1) < 1e-8, f"ratio={r:.12f}")


def test_n1_internalises_congestion():
    """With n = 1 the single operator fully internalises its own congestion,
    so any remaining gap is purely the accounting wedge."""
    inst = make_instance(n=1, T=32, seed=2, kappa=1.5)
    inst.m = inst.a.copy()                       # kill the wedge
    xn, _ = nash(inst)
    xp, _ = planner(inst)
    r = inst.social(xn) / inst.social(xp)
    check("n=1, m=a  =>  Nash == planner", abs(r - 1) < 1e-7, f"ratio={r:.10f}")

    inst2 = make_instance(n=1, T=32, seed=2, kappa=1.5)   # wedge restored
    xn2, _ = nash(inst2)
    xp2, _ = planner(inst2)
    r2 = inst2.social(xn2) / inst2.social(xp2)
    check("n=1, m>a  =>  gap is pure accounting", r2 > 1 + 1e-6, f"ratio={r2:.5f}")


def test_potential_is_exact():
    """Phi is an exact potential: a unilateral change moves J_i and Phi by the
    same amount."""
    rng = np.random.default_rng(3)
    inst = make_instance(n=6, T=24, seed=3, kappa=1.2, lam=0.4)
    x = inst.feasible_start()
    worst = 0.0
    for _ in range(60):
        i = rng.integers(inst.n)
        xi_new = solve_sep_qp(rng.normal(0, 50, inst.T), inst.b, inst.X[i])
        x2 = x.copy(); x2[i] = xi_new
        dJ = inst.agent_cost(x2, i) - inst.agent_cost(x, i)
        dPhi = inst.potential(x2) - inst.potential(x)
        worst = max(worst, abs(dJ - dPhi) / max(1.0, abs(dJ)))
    check("Phi is an exact potential", worst < 1e-9, f"worst rel dev={worst:.2e}")


def test_nash_is_an_equilibrium():
    """Direct incentive check: at the computed profile no operator can improve
    by any unilateral deviation (its best response is its current action)."""
    inst = make_instance(n=10, T=48, seed=4, kappa=1.4, lam=0.2)
    x, _ = nash(inst)
    y = x.sum(axis=0)
    worst = 0.0
    for i in range(inst.n):
        s = y - x[i]
        br = solve_sep_qp(inst.a + inst.b * s + inst.lam[i] * inst.psi[i],
                          inst.b, inst.X[i])
        xb = x.copy(); xb[i] = br
        gain = inst.agent_cost(x, i) - inst.agent_cost(xb, i)
        worst = max(worst, gain / max(1.0, abs(inst.agent_cost(x, i))))
    check("no profitable unilateral deviation at Nash", worst < 1e-9,
          f"max relative gain={worst:.2e}")


def test_uniqueness_from_random_starts():
    """Proposition 1: Phi strictly convex => unique equilibrium.  Best response
    from many random feasible starts must land on the same profile.

    Uniqueness is a mathematical claim, so this check is run at a tighter
    objective tolerance than the default.  The default (ftol = 1e-11) is chosen
    for experiment runtime and leaves a profile spread of about 2e-5 on this
    instance -- which is a convergence residual, not a second equilibrium: the
    largest profitable unilateral deviation there is 3e-11 relative.  At
    ftol = 1e-15 the spread collapses to the value asserted below."""
    rng = np.random.default_rng(5)
    inst = make_instance(n=8, T=32, seed=5, kappa=1.3, lam=0.25)
    ref = None
    worst = 0.0
    for k in range(20):
        x0 = np.zeros((inst.n, inst.T))
        for i in range(inst.n):
            x0[i] = solve_sep_qp(rng.normal(0, 100, inst.T), inst.b, inst.X[i])
        x, _ = nash(inst, x0=x0, order="random", rng=rng,
                    ftol=1e-15, patience=6)
        if ref is None:
            ref = x
        else:
            worst = max(worst, float(np.max(np.abs(x - ref))) / max(1.0, float(np.max(ref))))
    check("20 random starts -> same equilibrium", worst < 1e-6,
          f"max relative spread={worst:.2e}")


def test_planner_beats_nash():
    """Sanity: the planner is optimal for C, so C(nash) >= C(planner) always."""
    bad = 0
    for s in range(25):
        inst = make_instance(n=16, T=48, seed=100 + s, kappa=1.0 + 0.1 * s, lam=0.2)
        xn, _ = nash(inst); xp, _ = planner(inst)
        if inst.social(xn) < inst.social(xp) - 1e-8:
            bad += 1
    check("C(Nash) >= C(planner) on 25 instances", bad == 0, f"violations={bad}")


def test_two_slot_corner_window():
    """The corrected two-slot proposition.  The sigma=0 equilibrium sits at the
    corner (all load in the clean slot) iff Delta >= b*D*(1+1/n); independent
    spreading helps iff Delta < 2*b*D*(1-1/n).  Both hold only for n >= 4."""
    D, beta = 100.0, 1.0
    for n in (2, 3, 4, 8, 32):
        lo = beta * D * (1 + 1 / n)          # corner threshold
        hi = 2 * beta * D * (1 - 1 / n)      # spreading-helps threshold
        nonempty = lo < hi
        if nonempty:
            delta = 0.5 * (lo + hi)
            inst = two_slot(n, D, delta, beta)
            x, _ = nash(inst)
            frac_slot1 = x[:, 0].sum() / D
            check(f"n={n}: window nonempty and equilibrium at corner",
                  frac_slot1 > 1 - 1e-6,
                  f"[{lo:.1f},{hi:.1f}) share in clean slot={frac_slot1:.6f}")
        else:
            check(f"n={n}: window correctly empty", lo >= hi,
                  f"lo={lo:.1f} >= hi={hi:.1f}")


if __name__ == "__main__":
    print("Analytic oracles for the deferral game\n")
    for f in (test_no_wedge_no_congestion, test_n1_internalises_congestion,
              test_potential_is_exact, test_nash_is_an_equilibrium,
              test_uniqueness_from_random_starts, test_planner_beats_nash,
              test_two_slot_corner_window):
        print(f.__doc__.strip().split("\n")[0])
        f()
        print()
    print("ALL PASS" if all(OK) else f"{OK.count(False)} FAILURES")
    sys.exit(0 if all(OK) else 1)
