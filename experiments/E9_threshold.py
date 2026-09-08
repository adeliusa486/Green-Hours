"""E9 -- at what scale of flexible datacenter demand does coordination start to matter?

E5 answered a version of this on six flexible shares and read the crossing off a
linear interpolation between two of them.  That is one number resting on one
segment, and it is the number this paper's practical claim depends on, so it
gets a proper treatment here:

  * a dense, log-spaced share grid rather than six points;
  * two thresholds with different meanings, both stated as definitions rather
    than discovered post hoc;
  * a threshold computed PER SEED, so what is reported is a distribution and an
    inter-seed range rather than a single crossing of a mean curve;
  * sensitivity to the seven things that could move it -- the number of
    operators, workload heterogeneity, deadline slack, the curvature beta, the
    accounting boundary, the estimator, and signal precision;
  * uncertainty propagated from E8: the thresholds are recomputed on the
    hourly factors each ALTERNATIVE specification of the marginal-emissions
    estimator produces, so the interval quoted in the paper is an interval over
    the identification uncertainty and not only over workload seeds.

Two thresholds
--------------
sigma_strat(theta)   the smallest flexible share at which the STRATEGIC share of
                     the equilibrium gap reaches theta.  The gap decomposes as

                       C(nash) -> C(wedge_fixed) -> C(planner)

                     where wedge_fixed charges agents the marginal factor level
                     but leaves them seeing only their own congestion, so the
                     residual from there to the planner is exactly the
                     congestion externality.  theta = 0.25 is the reported
                     default: below a quarter, correcting the published number
                     dominates anything a mechanism can add, and the honest
                     recommendation is to fix the signal.  This is the
                     THEORETICALLY meaningful threshold -- it is about the model,
                     not about any one remedy.

sigma_static(theta)  the smallest share at which publishing a level-correct but
                     LOAD-INDEPENDENT marginal factor stops closing 1 - theta of
                     the equilibrium gap.  theta = 0.05 reproduces E5's
                     definition; theta = 0.25 is reported beside it.  This is the
                     DEPLOYMENT threshold -- it is the point at which the cheap
                     intervention stops being enough and a load-responsive
                     mechanism such as SHADE earns its complexity.

Neither is a property of the grid alone: both depend on the workload, which is
why they are swept rather than quoted.

Cost note.  SHADE costs about ten seconds an instance against half a second for
the rest, and Theorem 3 (verified numerically in E3, and again here on a
subgrid) says it attains the planner exactly.  It is therefore run on a coarse
subgrid as a check rather than on every point of the dense one, and the log says
which points carried it.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from gh.core import Feasible, Instance, nash, wedge_fixed
import gh.baselines as B
import E3_main
from E3_main import conform
from E1_calibrate import BAS, load, hourly_profile
import E8_mef_identification as E8

OUT = os.path.join(HERE, "..", "results", "E9.json")
NICE = {"CISO": "CAISO", "ERCO": "ERCOT", "PJM": "PJM"}

TODAY = 0.0038
SHARES = np.unique(np.concatenate([
    np.geomspace(0.001, 0.20, 28), [TODAY], [0.01, 0.02, 0.04, 0.08, 0.16]]))
SEEDS = 6
SHADE_EVERY = 5                    # run SHADE on every 5th share point
THETA_STRAT = 0.25
THETA_STATIC = (0.05, 0.25)

# defaults, matching E3
N_DEFAULT = 32
SIGMA_DEFAULT = 0.6                # log-normal spread of operator sizes
SLACK_DEFAULT = (2.0, 24.0)
CAP_FRAC = 0.35
T = 24


def build(aef, mef, beta, daily_MWh, share, seed, n=N_DEFAULT,
          size_sigma=SIGMA_DEFAULT, slack=SLACK_DEFAULT, cap_frac=CAP_FRAC,
          beta_scale=1.0):
    """E3's generator with the axes E9 sweeps exposed as arguments."""
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0.0, size_sigma, n)
    w /= w.sum()
    E = w * share * daily_MWh
    X, psi, arrival = [], np.zeros((n, T)), np.zeros((n, T))
    for i in range(n):
        u = np.full(T, cap_frac * E[i])
        sl = rng.uniform(*slack)
        done = np.clip(np.arange(T) / sl, 0, 1)
        R = np.maximum.accumulate(np.minimum(0.85 * E[i] * done ** 1.7,
                                             np.cumsum(u) * 0.9))
        X.append(Feasible(E=E[i], u=u, R=R))
        h0 = rng.uniform(6, 18)
        psi[i] = np.abs(np.arange(T) - h0) / 24.0
        arr = np.exp(-0.5 * ((np.arange(T) - h0) / 2.0) ** 2)
        arrival[i] = E[i] * arr / arr.sum()
    return Instance(a=aef, m=mef, b=beta * beta_scale, X=X, psi=psi,
                    lam=np.zeros(n), arrival=arrival)


def one_point(prof, share, seed, want_shade=False, **kw):
    """Every quantity the thresholds need, for one (profile, share, seed)."""
    a, m, b, dm = prof
    inst = build(a, m, b, dm, share, seed, **kw)
    Cs = B.planner_value(inst)
    xn = nash(inst)[0]
    xw = wedge_fixed(inst)[0]
    ne = inst.social(xn) / Cs
    wf = inst.social(xw) / Cs
    ms = inst.social(B.mef_static(inst)) / Cs
    ag = inst.social(B.carbon_agnostic(inst)) / Cs
    y = xn.sum(0)
    A = y > 1e-9
    kappa = float(np.mean(b[A] * y[A] / np.maximum(m[A] - a[A], 1e-9))) \
        if A.any() else float("nan")
    gap = ne - 1.0
    # strategic share: what remains once the accounting wedge is repaired
    strat = (wf - 1.0) / gap if gap > 1e-12 else float("nan")
    # what a level-correct but load-independent published factor leaves
    static_leaves = (ms - 1.0) / gap if gap > 1e-12 else float("nan")
    row = dict(nash=ne, wedge_fixed=wf, mef_static=ms, agnostic=ag,
               kappa=kappa, gap=gap, strategic_share=strat,
               static_leaves=static_leaves)
    if want_shade:
        xs = E3_main.shade(inst)
        row["shade"] = inst.social(xs) / Cs
    return row


def crossing(shares, values, theta):
    """First share at which `values` reaches theta, by linear interpolation on
    log(share).  Returns None if it never does inside the grid.

    Interpolating in log(share) rather than in share matters: the grid is
    log-spaced, and interpolating linearly in the level between two points a
    factor of 1.3 apart biases the crossing toward the upper point.
    """
    v = np.asarray(values, float)
    s = np.asarray(shares, float)
    ok = np.isfinite(v)
    v, s = v[ok], s[ok]
    if v.size == 0:
        return None
    hit = np.flatnonzero(v >= theta)
    if hit.size == 0:
        return None
    j = int(hit[0])
    if j == 0:
        return float(s[0])
    if v[j] == v[j - 1]:
        return float(s[j])
    w = (theta - v[j - 1]) / (v[j] - v[j - 1])
    return float(math.exp(math.log(s[j - 1])
                          + w * (math.log(s[j]) - math.log(s[j - 1]))))


def sweep(prof, shares, seeds, want_shade_at=(), label="", **kw):
    """Per-seed curves and per-seed thresholds."""
    per_seed = []
    for sd in seeds:
        rows = []
        for k, sh in enumerate(shares):
            rows.append(one_point(prof, sh, sd,
                                  want_shade=(k in want_shade_at), **kw))
        per_seed.append(rows)
    out = dict(label=label, shares=[float(s) for s in shares],
               seeds=list(seeds), rows_by_seed=per_seed)
    th = {"strat": [], "static05": [], "static25": []}
    for rows in per_seed:
        th["strat"].append(crossing(shares, [r["strategic_share"] for r in rows],
                                    THETA_STRAT))
        th["static05"].append(crossing(shares, [r["static_leaves"] for r in rows],
                                       THETA_STATIC[0]))
        th["static25"].append(crossing(shares, [r["static_leaves"] for r in rows],
                                       THETA_STATIC[1]))
    out["thresholds_by_seed"] = th
    out["thresholds"] = {k: summar(v) for k, v in th.items()}
    return out


def summar(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return dict(n=0, median=None, lo=None, hi=None,
                    n_beyond_grid=len(vals))
    return dict(n=len(v), median=float(np.median(v)), lo=float(np.min(v)),
                hi=float(np.max(v)), n_beyond_grid=len(vals) - len(v))


def fmt(t):
    if t is None or t.get("median") is None:
        return "beyond grid"
    s = f"{100*t['median']:.2f}% [{100*t['lo']:.2f}-{100*t['hi']:.2f}]"
    if t.get("n_beyond_grid"):
        s += f" ({t['n_beyond_grid']}/{t['n']+t['n_beyond_grid']} beyond grid)"
    return s


def main():
    t_start = time.time()
    print("loading EIA-930 ...")
    store = load()
    prof = {}
    for ba in BAS:
        a, m, b, dm = hourly_profile(store, ba)
        m2, b2, _, _ = conform(a, m, b)
        prof[ba] = (a, m2, b2, dm)

    out = {
        "provenance": "REPLAY-GRID / SIM-WORKLOAD.  Grid factors measured "
                      "(EIA-930 Jul-Dec 2024, consumption boundary, net-load "
                      "regressor); workload synthetic.  Shares above roughly "
                      "4% EXTRAPOLATE the second-order expansion beyond the "
                      "load excursions it was fitted on, and describe the "
                      "model rather than the grid.",
        "shares": [float(s) for s in SHARES], "seeds": SEEDS,
        "today_share": TODAY,
        "theta_strategic": THETA_STRAT,
        "theta_static": list(THETA_STATIC),
        "definitions": {
            "strategic_share":
                "(C(wedge_fixed)-C*)/(C(nash)-C*): the part of the equilibrium "
                "gap that survives repairing the published factor's LEVEL, "
                "i.e. the congestion externality",
            "static_leaves":
                "(C(static MEF)-C*)/(C(nash)-C*): the part of the gap a "
                "level-correct load-INDEPENDENT published factor leaves"},
        "dense": {}, "sensitivity": {}, "propagation": {}}

    shade_at = tuple(range(0, len(SHARES), SHADE_EVERY))

    # ---- A. dense sweep, primary specification -------------------------
    print(f"\n=== A. dense sweep: {len(SHARES)} shares x {SEEDS} seeds "
          f"x {len(BAS)} regions, SHADE at {len(shade_at)} of them ===")
    for ba in BAS:
        t0 = time.time()
        r = sweep(prof[ba], SHARES, range(SEEDS), want_shade_at=shade_at,
                  label=NICE[ba])
        out["dense"][ba] = r
        print(f"\n  {NICE[ba]}  ({time.time()-t0:.0f}s)")
        print(f"   {'flex share':>10s} {'kappa':>8s} {'Nash':>8s} "
              f"{'wedgefix':>9s} {'staticMEF':>10s} {'SHADE':>8s} "
              f"{'strategic':>10s} {'staticleft':>11s}")
        mean = [{k: float(np.mean([s[i][k] for s in r["rows_by_seed"]
                                   if k in s[i]]))
                 for k in ("kappa", "nash", "wedge_fixed", "mef_static",
                           "strategic_share", "static_leaves")}
                for i in range(len(SHARES))]
        for i, sh in enumerate(SHARES):
            sh_v = [s[i]["shade"] for s in r["rows_by_seed"] if "shade" in s[i]]
            sv = f"{np.mean(sh_v):8.5f}" if sh_v else f"{'-':>8s}"
            mark = "  <-- today" if abs(sh - TODAY) < 1e-9 else ""
            print(f"   {100*sh:9.3f}% {mean[i]['kappa']:8.4f} "
                  f"{mean[i]['nash']:8.5f} {mean[i]['wedge_fixed']:9.5f} "
                  f"{mean[i]['mef_static']:10.5f} {sv} "
                  f"{100*mean[i]['strategic_share']:9.1f}% "
                  f"{100*mean[i]['static_leaves']:10.1f}%{mark}")
        out["dense"][ba]["mean_curve"] = mean
        t = r["thresholds"]
        print(f"    strategic share reaches {100*THETA_STRAT:.0f}% at "
              f"{fmt(t['strat'])}")
        print(f"    static signal leaves  >{100*THETA_STATIC[0]:.0f}% at "
              f"{fmt(t['static05'])}")
        print(f"    static signal leaves >{100*THETA_STATIC[1]:.0f}% at "
              f"{fmt(t['static25'])}")

    # ---- B. sensitivity of the thresholds ------------------------------
    print("\n\n=== B. is the threshold fragile? one axis at a time, CAISO ===")
    coarse = np.geomspace(0.002, 0.20, 11)
    coarse_seeds = range(3)
    axes = [
        ("baseline", {}),
        ("n = 4", dict(n=4)), ("n = 8", dict(n=8)), ("n = 16", dict(n=16)),
        ("n = 64", dict(n=64)),
        ("sizes homogeneous (sigma 0.1)", dict(size_sigma=0.1)),
        ("sizes very skewed (sigma 1.2)", dict(size_sigma=1.2)),
        ("deadlines tight (2-8 h)", dict(slack=(2.0, 8.0))),
        ("deadlines loose (12-24 h)", dict(slack=(12.0, 24.0))),
        ("envelope tight (cap 0.15)", dict(cap_frac=0.15)),
        ("envelope loose (cap 0.60)", dict(cap_frac=0.60)),
        ("beta halved", dict(beta_scale=0.5)),
        ("beta doubled", dict(beta_scale=2.0)),
        ("beta x5", dict(beta_scale=5.0)),
    ]
    print(f"   {'axis':>32s} {'strategic>=25%':>26s} {'static leaves>5%':>26s}")
    for lab, kw in axes:
        r = sweep(prof["CISO"], coarse, coarse_seeds, label=lab, **kw)
        out["sensitivity"][lab] = {k: v for k, v in r.items()
                                   if k != "rows_by_seed"}
        t = r["thresholds"]
        print(f"   {lab:>32s} {fmt(t['strat']):>26s} {fmt(t['static05']):>26s}")

    # ---- C. uncertainty propagated from the identification audit -------
    print("\n\n=== C. the same thresholds under E8's alternative estimators ===")
    print("Each row re-estimates the hourly AEF, MEF and beta under a different")
    print("defensible specification and recomputes the threshold on it, so the")
    print("range below is identification uncertainty rather than seed noise.")
    specs = [("consumption", "netload", "ols", "full"),
             ("production", "netload", "ols", "full"),
             ("consumption_gas", "netload", "ols", "full"),
             ("consumption", "decomposed", "ols", "full"),
             ("consumption", "netload", "binlocal", "full"),
             ("consumption", "netload", "ols", "summer")]
    print(f"   {'region':>6s} {'specification':>44s} {'eta':>6s} "
          f"{'strategic>=25%':>24s} {'static>5%':>24s}")
    for ba in BAS:
        for bnd, reg, est, win in specs:
            e = E8.estimate(store, ba, bnd, reg, est, win)
            a, m, b = e["aef"].copy(), e["mef"].copy(), e["beta"].copy()
            for v in (a, m, b):
                bad = ~np.isfinite(v)
                if bad.all():
                    break
                if bad.any():
                    v[bad] = np.interp(np.flatnonzero(bad),
                                       np.flatnonzero(~bad), v[~bad], period=T)
            if not (np.isfinite(a).all() and np.isfinite(m).all()):
                continue
            m2, b2, _, _ = conform(a, m, b)
            dm = float(np.nanmean(e["dem"]) * T)
            r = sweep((a, m2, b2, dm), coarse, coarse_seeds,
                      label=f"{ba}|{bnd}|{reg}|{est}|{win}")
            out["propagation"][f"{ba}|{bnd}|{reg}|{est}|{win}"] = {
                k: v for k, v in r.items() if k != "rows_by_seed"}
            eta = float(np.nanmax(1.0 - a / m2))
            t = r["thresholds"]
            tag = f"{bnd}/{reg}/{est}/{win}"
            print(f"   {NICE[ba]:>6s} {tag:>44s} {eta:6.3f} "
                  f"{fmt(t['strat']):>24s} {fmt(t['static05']):>24s}")
        print()

    # ---- D. headline intervals -----------------------------------------
    print("=== D. the threshold, over BOTH sources of uncertainty ===")
    head = {}
    for ba in BAS:
        vals_s, vals_t = [], []
        for k, v in out["propagation"].items():
            if not k.startswith(ba + "|"):
                continue
            for x in v["thresholds_by_seed"]["strat"]:
                if x is not None:
                    vals_s.append(x)
            for x in v["thresholds_by_seed"]["static05"]:
                if x is not None:
                    vals_t.append(x)
        for x in out["dense"][ba]["thresholds_by_seed"]["strat"]:
            if x is not None:
                vals_s.append(x)
        for x in out["dense"][ba]["thresholds_by_seed"]["static05"]:
            if x is not None:
                vals_t.append(x)
        head[ba] = dict(
            strategic=dict(median=float(np.median(vals_s)) if vals_s else None,
                           p10=float(np.percentile(vals_s, 10)) if vals_s else None,
                           p90=float(np.percentile(vals_s, 90)) if vals_s else None,
                           n=len(vals_s)),
            static=dict(median=float(np.median(vals_t)) if vals_t else None,
                        p10=float(np.percentile(vals_t, 10)) if vals_t else None,
                        p90=float(np.percentile(vals_t, 90)) if vals_t else None,
                        n=len(vals_t)))
        h = head[ba]
        def sh(d):
            return ("beyond grid" if d["median"] is None else
                    f"{100*d['median']:.2f}%  (10-90th pct "
                    f"{100*d['p10']:.2f}-{100*d['p90']:.2f}%, n={d['n']})")
        print(f"  {NICE[ba]:>6s}  strategic share >= {100*THETA_STRAT:.0f}%: "
              f"{sh(h['strategic'])}")
        print(f"  {'':>6s}  static signal leaves > {100*THETA_STATIC[0]:.0f}%: "
              f"{sh(h['static'])}")
    out["headline"] = head
    out["today_multiple"] = {
        ba: (None if head[ba]["static"]["median"] is None
             else head[ba]["static"]["median"] / TODAY) for ba in BAS}
    print(f"\n  today's measured flexible share is {100*TODAY:.2f}% of regional "
          f"demand; the deployment threshold sits at")
    for ba in BAS:
        mm = out["today_multiple"][ba]
        if mm:
            print(f"    {NICE[ba]:>6s}  {mm:.1f}x today")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=float)
    print(f"\nwrote {os.path.relpath(OUT)}  ({time.time()-t_start:.0f}s total)")


if __name__ == "__main__":
    main()
