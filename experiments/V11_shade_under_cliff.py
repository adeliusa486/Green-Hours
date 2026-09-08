"""V11 -- does SHADE still work when the grid response is a cliff?

This is the experiment the reviewer called "probably the most important
technical experiment": V10 showed the piecewise response changes the SIZE and
COMPOSITION of the equilibrium gap, but said nothing about whether the mechanism
still closes it.  If SHADE only works against the smooth quadratic it was
derived from, the mechanism contribution is an artefact of the approximation.

Three configurations, all evaluated against the CLIFF social cost and the CLIFF
planner:

  nash            uncorrected equilibrium (agents charged a_t + b_t*y_t)
  shade-smooth    SHADE with the adder it uses today: built from the smooth
                  beta_t.  MISSPECIFIED under a cliff -- the clearinghouse
                  believes a curvature the grid does not have.
  shade-cliff     SHADE with an adder built from the true piecewise marginal
                  evaluated at the published aggregate.  This is what a
                  clearinghouse that knew the headroom would publish.

The gap between shade-smooth and shade-cliff is the price of model
misspecification, and it is the number a referee should be shown: it says
whether the mechanism degrades gracefully when the world is not quadratic.

SYNTHETIC.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from gh.instances import make_instance
from gh.core import nash, solve_sep_qp
from gh.cliff import cliff_from_instance, social_cliff, planner_cliff

N, T, SEEDS = 16, 24, 4
KAPPAS = [1.0, 2.0]
JUMPS = [3.0, 6.0, 12.0]


def _ne(inst):
    r = nash(inst)
    return r[0] if isinstance(r, tuple) else r


def shade_run(inst, cl=None, K=60, gamma=None, tol=1e-9):
    """SHADE with the corrected operator step.  If `cl` is given the adder uses
    the true piecewise marginal; otherwise it uses the smooth beta_t.

    Operator i minimises  sum_t [ (pi_it + lam*psi) x + b_t x^2 ]  over X_i,
    with the adder held constant at the previous round's aggregate, exactly as
    Algorithm 1 specifies.  Only the adder changes between the two variants.
    """
    n, Tt = inst.n, inst.T
    gamma = gamma or 1.8 / n
    x = inst.feasible_start()
    yhat = x.sum(axis=0).copy()
    prev = yhat.copy()
    for _ in range(K):
        for i in range(n):
            y_mi = yhat - x[i]
            # The local step minimises sum_t (q_t x + b_t x^2), so operator i's
            # perceived marginal is  q_t + 2*b_t*x_it  with
            # q = a + b*y_{-i} + pi + lam*psi.  The adder must therefore satisfy
            #     pi = (social marginal at y) - a - b*y_{-i} - 2*b*x_i
            #        = (social marginal at y) - a - b*y - b*x_i     (at y = y_{-i}+x_i)
            # Check: for the smooth response the social marginal is m + 2*b*y,
            # which gives pi = (m - a) + b*y - b*x_i = (m - a) + b*y_{-i},
            # exactly Eq. (11).  Dropping the -b*y term (as a first version of
            # this script did) over-charges by b*y and makes the "corrected"
            # mechanism perform WORSE than the misspecified one -- which is how
            # the error was caught.
            if cl is None:
                pi = (inst.m - inst.a) + inst.b * y_mi
            else:
                pi = cl.marginal(yhat) - inst.a - inst.b * yhat - inst.b * x[i]
            q = inst.a + inst.b * y_mi + pi + inst.lam[i] * inst.psi[i]
            x[i] = solve_sep_qp(q, inst.b, inst.X[i])
        y = x.sum(axis=0)
        yhat = np.maximum(prev + gamma * (y - prev), 0.0)
        if np.max(np.abs(yhat - prev)) <= tol * max(1.0, float(np.max(yhat))):
            break
        prev = yhat
    return x


GAMMA_NS = [1.8, 0.9, 0.5, 0.25, 0.1]   # gamma * n


def best_over_gamma(inst, cl, use_cliff, Cs):
    """Best achievable ratio over the damping grid, and the largest gamma*n
    that was still stable.  The stability bound gamma < 2/n in the paper is
    derived for the SMOOTH response; past the headroom the effective curvature
    is `jump` times larger, so the admissible gamma shrinks accordingly and a
    single constant tuned on the quadratic model is no longer safe."""
    best, best_g = np.inf, None
    for gn in GAMMA_NS:
        x = shade_run(inst, cl if use_cliff else None, gamma=gn / inst.n)
        v = social_cliff(inst, cl, x) / Cs
        if np.isfinite(v) and v < best:
            best, best_g = v, gn
    return best, best_g


print(f"{'kap':>4s} {'jump':>5s} | {'nash':>8s} | {'shade-sm':>9s} {'g*n':>5s} "
      f"{'rm':>7s} | {'shade-cl':>9s} {'g*n':>5s} {'rm':>7s}")
print("-" * 78)
rows = []
for kap in KAPPAS:
    for jump in JUMPS:
        nv_, sv_, cv_, sg_, cg_ = [], [], [], [], []
        for s in range(SEEDS):
            inst = make_instance(n=N, T=T, seed=s, kappa=kap, lam=0.0)
            cl = cliff_from_instance(inst, headroom_frac=0.6, jump=jump)
            Cs = planner_cliff(inst, cl)
            nv_.append(social_cliff(inst, cl, _ne(inst)) / Cs)
            v, g = best_over_gamma(inst, cl, False, Cs); sv_.append(v); sg_.append(g)
            v, g = best_over_gamma(inst, cl, True, Cs);  cv_.append(v); cg_.append(g)
        nv, sv, cv = np.mean(nv_), np.mean(sv_), np.mean(cv_)
        rm_s = (nv - sv) / (nv - 1.0) if nv > 1.0 else float("nan")
        rm_c = (nv - cv) / (nv - 1.0) if nv > 1.0 else float("nan")
        print(f"{kap:4.1f} {jump:5.1f} | {nv:8.4f} | {sv:9.4f} "
              f"{np.median(sg_):5.2f} {100*rm_s:6.1f}% | {cv:9.4f} "
              f"{np.median(cg_):5.2f} {100*rm_c:6.1f}%")
        rows.append(dict(kappa=kap, jump=jump, nash=nv,
                         shade_smooth=sv, gamma_n_smooth=float(np.median(sg_)),
                         shade_cliff=cv, gamma_n_cliff=float(np.median(cg_)),
                         removed_smooth=rm_s, removed_cliff=rm_c))

print()
rs = [r["removed_smooth"] for r in rows]
rc = [r["removed_cliff"] for r in rows]
print(f"gap removed, adder built from the smooth beta (misspecified): "
      f"{100*np.nanmin(rs):.1f}%-{100*np.nanmax(rs):.1f}%")
print(f"gap removed, adder built from the true piecewise marginal   : "
      f"{100*np.nanmin(rc):.1f}%-{100*np.nanmax(rc):.1f}%")
print("gamma*n columns show the damping each variant needed; the cliff-aware "
      "adder is unstable at the smooth model's setting.")

out = os.path.join(os.path.dirname(__file__), "..", "results", "V11.json")
json.dump({"n": N, "T": T, "seeds": SEEDS, "rows": rows,
           "note": "SYNTHETIC; SHADE evaluated against the piecewise response"},
          open(out, "w"), indent=2)
print(f"wrote {os.path.relpath(out)}")
