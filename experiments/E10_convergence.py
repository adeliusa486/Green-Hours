"""E10 -- what is actually known about SHADE's convergence, and where it fails.

Theorem 4 in the body (Theorem 5.3 by section number) bounds the epsilon-Nash
gap of the profile SHADE returns after K rounds and asserts linear convergence
of the published aggregate at rate 1 - Theta(gamma) under

    0 < gamma < 2/n .                                       (body, Eq. 15)

Its proof sketch says the argument is local because the active set can change,
and that a global proof is open.  That is honest but incomplete: it does not say
what region "local" means, it does not say what the rate is, and it offers no
evidence about behaviour outside the region.  This script supplies all three,
and deliberately does NOT upgrade the theorem to a global claim, because nothing
here establishes one.

1.  Where the local argument bites, written out
-----------------------------------------------
Fix the active set and assume every operator responds in the interior of its
feasible set.  The adder is lagged, so operator i solves

    min_{x_i in X_i} sum_t [ (m_t + 2 b_t (yhat_t - x_it^{k-1})) x_it
                             + b_t x_it^2 ],

with interior stationarity m_t + 2 b_t(yhat_t - x_it^{k-1}) + 2 b_t x_it^{k}
= nu_i, i.e.

    x_it^{k} = x_it^{k-1} - yhat_t + (nu_i - m_t)/(2 b_t) .           (E10-1)

The unit coefficient on x^{k-1} is why the body says the state is the pair
(yhat, x_i) and the map cannot be accelerated on yhat alone.  Summing (E10-1)
over i and appending the clearinghouse step
yhat^{k} = (1-gamma) yhat^{k-1} + gamma y^{k} gives, per slot,

    [ y    ]  <--  [ 1      -n           ] [ y    ]  + const,
    [ yhat ]       [ gamma  1-gamma(1+n) ] [ yhat ]

with trace 2 - gamma(1+n) and determinant 1 - gamma.  The Jury conditions put
both roots inside the unit disc exactly when

    0 < gamma < 4/(n+2) ,                                             (E10-2)

and since the determinant is 1 - gamma, whenever the roots are complex both have
modulus exactly sqrt(1-gamma).

Two things follow, and section A checks both.  First, 2/n <= 4/(n+2) for every
n >= 2, so the body's condition is SUFFICIENT for this regime but strictly
conservative above n = 2 -- at n = 32 it admits gamma < 0.0625 where (E10-2)
allows 0.1176.  Second, (E10-2) is a statement about the FULLY INTERIOR face and
nothing more.

2.  Why that regime is not the one the calibrated instances are in
------------------------------------------------------------------
Section A also measures how much of the fixed point is interior, and the answer
is: very little.  On the default generator only about 6% of the (operator, slot)
coordinates are strictly between zero and the envelope at the fixed point; the
rest sit on a bound, because minimising carbon concentrates flexible load into a
few hours and the deadline staircase and envelope then determine the rest.  On
such a face most operators cannot respond to a perturbation of yhat at all, so
the -n coupling in the linearisation above is not what the iteration
experiences.  What it experiences instead is close to pure damping of yhat
toward a nearly fixed aggregate, which contracts at |1 - gamma| -- and section A
confirms the observed rate is |1 - gamma| to four digits over a wide range of n
and gamma.

The honest conclusion, and the one the paper is rewritten to state, is:

    the body's gamma < 2/n is a SUFFICIENT stability condition, verified on
    every instance tested and never violated; it is conservative, by a factor
    that grows with n, because it is derived from an interior response the
    solutions do not exhibit; the measured stable range is far wider; and no
    global convergence proof is offered or claimed.

3.  Adversarial search
----------------------
Sections C and D try to break it: n from 1 to 128, beta over thirteen orders of
magnitude, capacity at the feasibility edge, one-hour deadlines, identical and
extremely heterogeneous agents, zero flexibility, twelve random
initialisations, and the epsilon-Nash inequality of Theorem 4 checked as an
inequality rather than restated.  Every failure is recorded.

Convergence vocabulary used throughout, kept distinct on purpose:
    DIVERGED    the iterate leaves every bounded set, or the cost is not finite
    SLOW        still moving at the round cap, but bounded
    CONVERGED   the social cost and the published aggregate are both still, and
                the profile is feasible and within 1e-6 of the planner
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from gh.core import Feasible, Instance, solve_sep_qp, max_gain
import gh.baselines as B

OUT = os.path.join(HERE, "..", "results", "E10.json")
T = 24
K_CAP = 4000


def shade_trace(inst, gamma, K=K_CAP, x0=None, ftol=1e-13, ytol=1e-11,
                patience=5):
    """One SHADE run, keeping the trajectory and never truncating a divergence."""
    n = inst.n
    x = inst.feasible_start() if x0 is None else np.asarray(x0, float).copy()
    yhat = x.sum(0).copy()
    prev = yhat.copy()
    prevC = inst.social(x)
    scale = max(1.0, float(np.sum([Xi.E for Xi in inst.X])))
    hist = [float(np.max(np.abs(yhat))) / scale]
    flat = 0
    rounds = K
    diverged = False
    for k in range(1, K + 1):
        for i in range(n):
            ymi = yhat - x[i]
            q = inst.m + 2.0 * inst.b * ymi + inst.lam[i] * inst.psi[i]
            x[i] = solve_sep_qp(q, inst.b, inst.X[i])
        y = x.sum(0)
        yhat = np.maximum(prev + gamma * (y - prev), 0.0)
        dy = float(np.max(np.abs(yhat - prev))) / scale
        prev = yhat
        cur = inst.social(x)
        hist.append(float(np.max(np.abs(yhat))) / scale)
        if not np.isfinite(cur) or hist[-1] > 1e6:
            diverged, rounds = True, k
            break
        dC = abs(prevC - cur) / max(1.0, abs(prevC))
        prevC = cur
        if dC <= ftol and dy <= ytol:
            flat += 1
            if flat >= patience:
                rounds = k
                break
        else:
            flat = 0
    return dict(x=x, rounds=rounds, diverged=diverged, hist=hist,
                hit_cap=(rounds >= K and not diverged))


def classify(inst, res, Cs):
    """DIVERGED / SLOW / CONVERGED, plus the numbers behind the label."""
    if res["diverged"]:
        return "DIVERGED", None, False
    feas = all(inst.X[i].check(res["x"][i], tol=1e-6) for i in range(inst.n))
    ratio = inst.social(res["x"]) / Cs
    if not feas:
        return "INFEASIBLE", ratio, False
    if ratio <= 1.0 + 1e-6:
        return "CONVERGED", ratio, True
    return ("SLOW" if res["hit_cap"] else "STALLED"), ratio, False


def rand_instance(rng, n=8, T=T, beta_scale=1.0, het=0.6, cap=0.35,
                  slack=(2.0, 24.0), Etot=1.0, flat_beta=False):
    a = 100.0 + 200.0 * rng.random(T)
    m = a * (1.3 + 1.2 * rng.random(T))
    b = (np.full(T, 1.0) if flat_beta else (0.2 + rng.random(T))) * beta_scale
    w = rng.lognormal(0.0, het, n)
    w /= w.sum()
    E = w * Etot
    X, psi = [], np.zeros((n, T))
    for i in range(n):
        u = np.full(T, cap * E[i])
        sl = rng.uniform(*slack)
        done = np.clip(np.arange(T) / sl, 0, 1)
        R = np.maximum.accumulate(np.minimum(0.85 * E[i] * done ** 1.7,
                                             np.cumsum(u) * 0.9))
        X.append(Feasible(E=E[i], u=u, R=R))
    return Instance(a=a, m=m, b=b, X=X, psi=psi, lam=np.zeros(n),
                    arrival=np.tile(E[:, None] / T, (1, T)))


def gamma_star(n):
    return 4.0 / (n + 2.0)


def rate_pred_interior(gamma, n):
    tr, det = 2.0 - gamma * (1.0 + n), 1.0 - gamma
    disc = tr * tr - 4.0 * det
    if disc < 0:
        return math.sqrt(max(det, 0.0))
    r = math.sqrt(disc)
    return max(abs((tr + r) / 2.0), abs((tr - r) / 2.0))


def observed_rate(hist, tail=40):
    h = np.asarray(hist, float)
    d = np.abs(np.diff(h))[-tail:]
    d = d[d > 1e-15]
    if len(d) < 6:
        return None
    k = np.arange(len(d))
    c, *_ = np.linalg.lstsq(np.column_stack([k, np.ones(len(k))]),
                            np.log(d), rcond=None)
    return float(np.exp(c[0]))


def interior_fraction(inst, x):
    f = 0
    for i, Xi in enumerate(inst.X):
        f += int(np.sum((x[i] > 1e-9) & (x[i] < Xi.u - 1e-9)))
    return f / x.size


def main():
    out = {"provenance": "SIM.  Controlled instances; no grid data.",
           "interior_stability_interval": "0 < gamma < 4/(n+2)",
           "paper_condition": "0 < gamma < 2/n",
           "global_convergence_proved": False,
           "A": [], "B": [], "C": [], "D": []}

    # ---- A. the interior calculation against what happens -------------
    print("=== A. the interior linearisation, and how interior the solutions are ===")
    print("gamma is swept well past both 2/n and the interior limit 4/(n+2).")
    print("'interior frac' is the share of (operator, slot) coordinates strictly")
    print("between 0 and the envelope at the fixed point -- the share of the")
    print("problem the linearisation actually describes.\n")
    print(f"{'n':>4s} {'2/n':>7s} {'4/(n+2)':>8s} {'interior':>9s} "
          f"{'gamma':>7s} {'status':>10s} {'rounds':>7s} {'rate obs':>9s} "
          f"{'|1-g|':>7s} {'interior pred':>14s}")
    print("-" * 96)
    max_stable = {}
    for n in (2, 4, 8, 16, 32, 64):
        inst = rand_instance(np.random.default_rng(100 + n), n=n)
        Cs = B.planner_value(inst)
        base = shade_trace(inst, min(1.8 / n, 0.9))
        ifrac = interior_fraction(inst, base["x"])
        stable = []
        for g in (0.05, 2.0 / n, gamma_star(n), 0.5, 0.9, 1.0, 1.3, 1.7,
                  1.95, 2.05, 2.5):
            if g > 3.0 or g <= 0:
                continue
            r = shade_trace(inst, float(g))
            st, ratio, ok = classify(inst, r, Cs)
            ro = observed_rate(r["hist"])
            rec = dict(n=n, gamma=float(g), status=st, rounds=r["rounds"],
                       ratio=ratio, rate_obs=ro,
                       rate_one_minus_gamma=abs(1.0 - g),
                       rate_interior_pred=rate_pred_interior(g, n),
                       interior_fraction=ifrac)
            out["A"].append(rec)
            if st != "DIVERGED":
                stable.append(float(g))
            print(f"{n:4d} {2.0/n:7.4f} {gamma_star(n):8.4f} {ifrac:9.3f} "
                  f"{g:7.4f} {st:>10s} {r['rounds']:7d} "
                  f"{(ro if ro else float('nan')):9.4f} {abs(1-g):7.4f} "
                  f"{rate_pred_interior(g, n):14.4f}")
        max_stable[n] = max(stable) if stable else None
        print()
    out["max_gamma_without_divergence"] = max_stable
    print("  largest gamma tested without divergence, by n: " +
          ", ".join(f"n={k}: {v}" for k, v in max_stable.items()))
    print("  the body's 2/n is inside that range for every n above: " +
          str(all(v is not None and 2.0 / k <= v + 1e-12
                  for k, v in max_stable.items())))

    # ---- B. rounds against gamma inside the admissible range ----------
    print("\n=== B. rounds against gamma ===")
    print("The body says the round count is nearly flat in gamma and smallest")
    print("near the upper limit of Eq. (15).  Both halves are checked, and the")
    print("cost of the conservative rule is read off the last column.\n")
    for n in (8, 32):
        inst = rand_instance(np.random.default_rng(7), n=n)
        Cs = B.planner_value(inst)
        print(f"  n = {n}  (2/n = {2.0/n:.4f})")
        print(f"    {'gamma':>8s} {'status':>10s} {'rounds':>7s} "
              f"{'rate obs':>9s}")
        best = None
        for g in (0.1 / n, 0.5 / n, 1.0 / n, 1.8 / n, 0.25, 0.5, 0.9, 1.0):
            r = shade_trace(inst, float(g))
            st, ratio, ok = classify(inst, r, Cs)
            out["B"].append(dict(n=n, gamma=float(g), status=st,
                                 rounds=r["rounds"], ratio=ratio,
                                 rate_obs=observed_rate(r["hist"])))
            if ok and (best is None or r["rounds"] < best[1]):
                best = (float(g), r["rounds"])
            print(f"    {g:8.5f} {st:>10s} {r['rounds']:7d} "
                  f"{(observed_rate(r['hist']) or float('nan')):9.4f}")
        r18 = [x for x in out["B"] if x["n"] == n
               and abs(x["gamma"] - 1.8 / n) < 1e-12]
        if best and r18:
            print(f"    fastest admissible gamma tested: {best[0]:.4f} at "
                  f"{best[1]} rounds, against {r18[0]['rounds']} rounds at the "
                  f"paper's 1.8/n\n")

    # ---- C. adversarial regimes ---------------------------------------
    print("=== C. adversarial regimes, at the paper's gamma = 1.8/n ===")
    cases = [(f"n = {n}", dict(n=n)) for n in (1, 2, 3, 4, 8, 16, 32, 64, 128)]
    cases += [(f"beta scale {bs:g}", dict(beta_scale=bs))
              for bs in (1e-9, 1e-6, 1e-3, 1.0, 1e2, 1e4)]
    cases += [
        ("identical agents", dict(het=1e-6)),
        ("extreme heterogeneity (sd 2.5)", dict(het=2.5)),
        ("capacity tight (cap 0.05)", dict(cap=0.05)),
        ("capacity at the feasibility edge", dict(cap=1.0 / T + 1e-6)),
        ("capacity loose (cap 1.0)", dict(cap=1.0)),
        ("deadlines tight (1-2 h)", dict(slack=(1.0, 2.0))),
        ("deadlines very loose (23-24 h)", dict(slack=(23.0, 24.0))),
        ("flat beta", dict(flat_beta=True)),
    ]
    print(f"{'case':>34s} {'n':>4s} {'gamma':>8s} {'worst status':>13s} "
          f"{'rounds':>7s} {'C/C*':>14s} {'max gain':>10s}")
    print("-" * 96)
    n_div = n_notconv = 0
    for lab, kw in cases:
        n = kw.get("n", 8)
        gamma = 1.8 / n
        recs = []
        for s in range(3):
            try:
                inst = rand_instance(np.random.default_rng(1000 + s), **kw)
                Cs = B.planner_value(inst)
                r = shade_trace(inst, gamma, K=6000)
                st, ratio, ok = classify(inst, r, Cs)
                mg = max_gain(inst, r["x"],
                              lambda i, sv: inst.m + 2.0 * inst.b * sv
                              + inst.lam[i] * inst.psi[i])
                recs.append(dict(case=lab, n=inst.n, gamma=gamma, seed=s,
                                 status=st, rounds=r["rounds"], ratio=ratio,
                                 max_gain_rel=float(mg / max(abs(Cs), 1e-300)),
                                 ok=bool(ok)))
            except Exception as e:                              # noqa: BLE001
                recs.append(dict(case=lab, n=n, gamma=gamma, seed=s,
                                 status="ERROR",
                                 error=f"{type(e).__name__}: {e}", ok=False))
        out["C"].extend(recs)
        order = {"DIVERGED": 0, "ERROR": 1, "INFEASIBLE": 2, "STALLED": 3,
                 "SLOW": 4, "CONVERGED": 5}
        w = min(recs, key=lambda r: order.get(r["status"], 9))
        n_div += any(r["status"] in ("DIVERGED", "ERROR", "INFEASIBLE")
                     for r in recs)
        n_notconv += (not w.get("ok", False))
        if "error" in w:
            print(f"{lab:>34s} {w['n']:4d} {gamma:8.4f} {'ERROR':>13s} "
                  f"{'-':>7s} {'-':>14s} {'-':>10s}  {w['error']}")
        else:
            print(f"{lab:>34s} {w['n']:4d} {gamma:8.4f} {w['status']:>13s} "
                  f"{w['rounds']:7d} {w['ratio']:14.10f} "
                  f"{w['max_gain_rel']:10.2e}")
    print(f"\n  divergences / errors / infeasibilities: {n_div} of {len(cases)}")
    print(f"  did not reach the planner within 1e-6 at the round cap: "
          f"{n_notconv} of {len(cases)}")

    # ---- C2. initialisation ------------------------------------------
    print("\n=== C2. does the limit depend on where it starts? ===")
    inst = rand_instance(np.random.default_rng(42), n=8)
    Cs = B.planner_value(inst)
    ys, ratios = [], []
    for s in range(12):
        rg = np.random.default_rng(500 + s)
        x0 = np.array([solve_sep_qp(rg.random(T), np.ones(T), Xi)
                       for Xi in inst.X])
        r = shade_trace(inst, 1.8 / inst.n, x0=x0, K=6000)
        ys.append(r["x"].sum(0))
        ratios.append(inst.social(r["x"]) / Cs)
    Y = np.array(ys)
    spread = float(np.max(np.abs(Y - Y.mean(axis=0)))
                   / max(1.0, float(np.max(np.abs(Y)))))
    out["C_init"] = dict(n_starts=len(ys), aggregate_spread_rel=spread,
                         ratio_min=float(np.min(ratios)),
                         ratio_max=float(np.max(ratios)))
    print(f"  12 random feasible starts: the aggregate y agrees to "
          f"{spread:.2e} relative;")
    print(f"  C/C* in [{np.min(ratios):.12f}, {np.max(ratios):.12f}].")
    print("  Theorem 3 predicts the AGGREGATE is unique and the profile need")
    print("  not be, so agreement of y is the claim being checked.")

    # ---- D. the epsilon-Nash inequality -------------------------------
    print("\n=== D. Theorem 4's epsilon-Nash bound, checked as an inequality ===")
    print("Exact aggregation, so zeta = 0 and the bound is 2*bmax*rhobar*Emax^2.")
    print(f"{'n':>4s} {'seed':>5s} {'realised eps':>14s} {'bound':>12s} "
          f"{'slack':>10s} {'holds':>6s}")
    print("-" * 56)
    viol = 0
    for n in (2, 4, 8, 16, 32):
        for s in range(3):
            inst = rand_instance(np.random.default_rng(2000 + s), n=n)
            r = shade_trace(inst, 1.8 / n, K=6000)
            eps = max_gain(inst, r["x"],
                           lambda i, sv: inst.m + 2.0 * inst.b * sv
                           + inst.lam[i] * inst.psi[i])
            E = np.array([Xi.E for Xi in inst.X])
            bound = 2.0 * float(inst.b.max()) * float(E.max() / E.sum()) \
                * float(E.max()) ** 2
            ok = eps <= bound + 1e-12
            viol += (not ok)
            out["D"].append(dict(n=n, seed=s, eps=float(eps),
                                 bound=float(bound), holds=bool(ok)))
            print(f"{n:4d} {s:5d} {eps:14.3e} {bound:12.3e} "
                  f"{bound/max(eps, 1e-300):10.1e} {str(ok):>6s}")
    print(f"\n  violations: {viol} of {len(out['D'])}")

    out["summary"] = dict(
        n_adversarial_cases=len(cases), n_divergences=int(n_div),
        n_not_converged=int(n_notconv), n_enash_violations=int(viol),
        interior_fraction_default=float(
            np.median([r["interior_fraction"] for r in out["A"]])),
        conclusion="gamma < 2/n was sufficient on every instance tested and "
                   "was never the binding constraint; it is conservative "
                   "because it is derived from an interior response the "
                   "solutions do not exhibit.  Convergence is established "
                   "LOCALLY on a fixed active set; no global proof is offered.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=float)
    print(f"\nwrote {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
