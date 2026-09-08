"""E3 -- the main comparison.

Grid side is measured: per-hour-of-day AEF, MEF and beta for CAISO, ERCOT and
PJM from EIA-930 (Jul-Dec 2024), on the consumption boundary with the net-load
regressor, exactly as in E1.  Workload side is synthetic: operator sizes,
deadline staircases and power envelopes come from the generator, because the
production traces (Azure, Borg, Alibaba) are not redistributable.  So this is a
REAL-GRID / SYNTHETIC-WORKLOAD experiment and is labelled that way throughout.

Baselines run: carbon-agnostic, naive forecast-taking, static MEF signalling,
independent greedy (Nash), threshold deferral, randomized jitter, proportional
cap, SHADE, responsive MEF (oracle), planner.  Cooperative MARL is NOT run
(GPU training) and is reported as absent, not as a number.

Three things this script records that an earlier version silently did:

  * how often the model hypotheses MEF >= AEF and beta >= 0 have to be imposed
    on the measured hourly profile, and what share of the load-carrying hours
    they touch;
  * eta on the REALISED active set A (the slots carrying flexible load in the
    equilibrium or the planner solution), which is the quantity Theorem 1 needs,
    rather than eta on a decile chosen before the game is solved;
  * SHADE's residual at the round it stops, so its distance from the planner is
    attributable to convergence or to the mechanism rather than left ambiguous.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from gh.core import Feasible, Instance, nash, solve_sep_qp
import gh.baselines as B
from E1_calibrate import BAS, load, hourly_profile

T = 24
N = 32
SEEDS = 20
FLEX_SHARE = 0.0038
CAP_FRAC = 0.35
LAM = 0.0                    # pure carbon objective; LAM > 0 swept in E4
SHADE_K = 2000
NICE = {"CISO": "CAISO", "ERCO": "ERCOT", "PJM": "PJM"}


def conform(aef, mef, beta):
    """Impose the model's hypotheses on the measured profile, and say how much
    had to be imposed.  MEF >= AEF is Theorem 1's hypothesis; beta > 0 is
    Proposition 1's.  Clipping a negative beta to (almost) zero REMOVES
    congestion, so it is conservative for every claim in the paper."""
    n_mef = int(np.sum(mef < aef))
    n_beta = int(np.sum(beta < 0))
    mef = np.maximum(mef, aef)
    pos = beta[beta > 0]
    floor = 1e-3 * float(np.median(pos)) if pos.size else 1e-9
    beta = np.maximum(beta, floor)
    return mef, beta, n_mef, n_beta


def build(aef, mef, beta, daily_MWh, seed):
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0.0, 0.6, N)
    w /= w.sum()
    E = w * FLEX_SHARE * daily_MWh
    X, psi, arrival = [], np.zeros((N, T)), np.zeros((N, T))
    for i in range(N):
        u = np.full(T, CAP_FRAC * E[i])
        slack = rng.uniform(2.0, 24.0)
        done = np.clip(np.arange(T) / slack, 0, 1)
        R = np.maximum.accumulate(np.minimum(0.85 * E[i] * done ** 1.7,
                                             np.cumsum(u) * 0.9))
        X.append(Feasible(E=E[i], u=u, R=R))
        h0 = rng.uniform(6, 18)
        psi[i] = np.abs(np.arange(T) - h0) / 24.0
        arr = np.exp(-0.5 * ((np.arange(T) - h0) / 2.0) ** 2)
        arrival[i] = E[i] * arr / arr.sum()
    return Instance(a=aef, m=mef, b=beta, X=X, psi=psi,
                    lam=np.full(N, LAM), arrival=arrival)


def shade(inst, K=SHADE_K, gamma=None, ftol=1e-12, ytol=1e-9,
          patience=5, return_rounds=False):
    """SHADE, Eq. (10)-(11).  The local marginal is

        a + b*(yhat - x_i) + 2*b*x_i  +  [(m - a) + b*(yhat - x_i)]

    which at a fixed point with yhat = y equals m + 2*b*y = dC/dx_it exactly.

    Convergence is declared only when BOTH the social cost and the published
    aggregate have stopped moving, for `patience` consecutive rounds.  Either
    test alone is unsafe.  Stopping on the aggregate alone reports a round count
    several times too large, because with no deferral penalty the minimiser of C
    is a face and the aggregate keeps drifting inside it after the objective has
    settled.  Stopping on the objective alone fires on transient plateaus: at a
    4% flexible share the damped iteration sat within 1e-12 of its own cost for
    three rounds at 1.0049 of the planner and then went on to reach 1.0000.
    """
    n = inst.n
    gamma = gamma if gamma is not None else 1.8 / n
    x = inst.feasible_start()
    yhat = x.sum(0).copy()
    prev = yhat.copy()
    prevC = inst.social(x)
    flat = 0
    rounds = K
    for k in range(1, K + 1):
        for i in range(n):
            ymi = yhat - x[i]
            q = inst.a + inst.b * ymi + ((inst.m - inst.a) + inst.b * ymi)
            x[i] = solve_sep_qp(q + inst.lam[i] * inst.psi[i], inst.b, inst.X[i])
        y = x.sum(0)
        yhat = np.maximum(prev + gamma * (y - prev), 0.0)
        dy = float(np.max(np.abs(yhat - prev))) / max(1.0, float(np.max(yhat)))
        prev = yhat
        cur = inst.social(x)
        dC = abs(prevC - cur) / max(1.0, abs(prevC))
        prevC = cur
        if dC <= ftol and dy <= ytol:
            flat += 1
            if flat >= patience:
                rounds = k
                break
        else:
            flat = 0
    return (x, rounds) if return_rounds else x


def responsive_mef(inst, K=SHADE_K, gamma=None, ftol=1e-12, ytol=1e-9,
                   patience=5):
    """Oracle: agents charged the responsive factor m + 2*b*y, computed from the
    aggregate they would have to disclose.  Its fixed-point marginal is
    m + 2*b*y + 2*b*x_i, so it over-corrects by 2*b*x_i -- which is why it is an
    upper reference for SHADE rather than an unreachable one."""
    n = inst.n
    gamma = gamma if gamma is not None else 1.8 / n
    x = inst.feasible_start()
    yhat = x.sum(0).copy()
    prev = yhat.copy()
    prevC = inst.social(x)
    flat = 0
    for _ in range(K):
        for i in range(n):
            q = inst.m + 2.0 * inst.b * yhat + inst.lam[i] * inst.psi[i]
            x[i] = solve_sep_qp(q, inst.b, inst.X[i])
        y = x.sum(0)
        yhat = np.maximum(prev + gamma * (y - prev), 0.0)
        dy = float(np.max(np.abs(yhat - prev))) / max(1.0, float(np.max(yhat)))
        prev = yhat
        cur = inst.social(x)
        dC = abs(prevC - cur) / max(1.0, abs(prevC))
        prevC = cur
        if dC <= ftol and dy <= ytol:
            flat += 1
            if flat >= patience:
                break
        else:
            flat = 0
    return x


_last_rounds = [0]


def _shade_rec(inst):
    x, k = shade(inst, return_rounds=True)
    _last_rounds[0] = k
    return x


def _ne(inst):
    r = nash(inst)
    return r[0] if isinstance(r, tuple) else r


METHODS = [
    ("Carbon-agnostic",        lambda I, r: B.carbon_agnostic(I)),
    ("Naive forecast-taking",  lambda I, r: B.naive(I)),
    ("Static MEF signalling",  lambda I, r: B.mef_static(I)),
    ("Independent greedy",     lambda I, r: _ne(I)),
    ("Threshold deferral",     lambda I, r: B.threshold(I)),
    ("Randomized jitter",      lambda I, r: B.jitter(I, seed=r)),
    ("Proportional cap",       lambda I, r: B.proportional_cap(I)),
    ("SHADE (ours)",           lambda I, r: _shade_rec(I)),
    ("Responsive MEF",         lambda I, r: responsive_mef(I)),
]
BASE = "Independent greedy"


def ci95(v):
    """Bootstrap 95% interval of the mean."""
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), float(v.mean())
    rng = np.random.default_rng(0)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    return tuple(float(z) for z in np.percentile(bs, [2.5, 97.5]))


if __name__ == "__main__":
    print("loading EIA-930 ...")
    store = load()
    prof, conf = {}, {}
    for b in BAS:
        a, m, be, dm = hourly_profile(store, b)
        m2, b2, n_mef, n_beta = conform(a, m, be)
        prof[b] = (a, m2, b2, dm)
        conf[b] = dict(n_mef_imposed=n_mef, n_beta_clipped=n_beta)
        print(f"  {NICE[b]}: AEF {a.min():.0f}-{a.max():.0f}, "
              f"MEF {m2.min():.0f}-{m2.max():.0f}, daily {dm/1e3:.0f} GWh, "
              f"MEF<AEF imposed in {n_mef}/24 h, beta<0 clipped in {n_beta}/24 h")

    res = {name: {b: [] for b in BAS} for name, _ in METHODS}
    infeasible = {name: 0 for name, _ in METHODS}
    active_eta = {b: [] for b in BAS}
    active_beta_clip = {b: [] for b in BAS}
    shade_rounds = []

    for b in BAS:
        a, m, be, dm = prof[b]
        eta_h = 1.0 - a / m
        raw_beta_neg = np.array(hourly_profile(store, b)[2]) < 0
        for s in range(SEEDS):
            inst = build(a, m, be, dm, s)
            Cs = B.planner_value(inst)
            xne = _ne(inst)
            xpl = B._planner_fast(inst)
            A = (xne.sum(0) > 1e-9) | (xpl.sum(0) > 1e-9)
            active_eta[b].append(float(np.max(eta_h[A])))
            active_beta_clip[b].append(int(np.sum(raw_beta_neg & A)))
            for name, fn in METHODS:
                try:
                    x = fn(inst, s)
                except Exception as e:
                    print(f"    {name}/{b}/{s}: {type(e).__name__} {e}")
                    continue
                if x is None:
                    infeasible[name] += 1
                    continue
                bad = [i for i, Xi in enumerate(inst.X)
                       if not Xi.check(x[i], tol=1e-6)]
                if bad:
                    infeasible[name] += 1
                    continue
                res[name][b].append(inst.social(x) / Cs)
            shade_rounds.append(_last_rounds[0])

    base_per = [float(np.mean(res[BASE][b])) for b in BAS]
    base_mean = float(np.mean(base_per))

    print(f"\n{'Method':<24s} " + " ".join(f"{NICE[b]:>16s}" for b in BAS)
          + f" {'Mean':>8s} {'Gap rm':>8s}")
    print("-" * 96)
    table = {}
    for name, _ in METHODS:
        if infeasible[name]:
            print(f"{name:<24s} INFEASIBLE in {infeasible[name]} of "
                  f"{len(BAS)*SEEDS} runs -- not scored")
            table[name] = dict(infeasible=infeasible[name])
            continue
        per = [float(np.mean(res[name][b])) for b in BAS]
        sd = [float(np.std(res[name][b], ddof=1)) for b in BAS]
        cis = [ci95(res[name][b]) for b in BAS]
        mean = float(np.mean(per))
        gap = ((base_mean - mean) / (base_mean - 1.0)) if base_mean > 1 + 1e-12 else float("nan")
        # Paired bootstrap on the share of the gap removed.  Resample SEEDS
        # (the unit of randomness) and recompute BOTH the method's mean and the
        # equilibrium's mean on the same resample; holding the denominator
        # fixed while resampling only the numerator inflates the interval and
        # produced intervals reaching past 100% of a gap.
        M = np.array([res[name][b] for b in BAS])          # regions x seeds
        Q = np.array([res[BASE][b] for b in BAS])
        rng = np.random.default_rng(1)
        gbs = []
        for _ in range(4000):
            i = rng.integers(0, M.shape[1], M.shape[1])
            mm = M[:, i].mean(axis=1).mean()
            qq = Q[:, i].mean(axis=1).mean()
            if qq > 1.0 + 1e-12:
                gbs.append((qq - mm) / (qq - 1.0))
        glo, ghi = np.percentile(gbs, [2.5, 97.5]) if gbs else (np.nan, np.nan)
        table[name] = dict(per=per, sd=sd, ci=cis, mean=mean, gap=gap,
                           gap_ci=[float(glo), float(ghi)], n=len(res[name][BAS[0]]))
        g = "n/a" if name == "Carbon-agnostic" else f"{100*gap:6.1f}%"
        print(f"{name:<24s} " + " ".join(
            f"{v:8.4f}+-{d:6.4f}" for v, d in zip(per, sd))
            + f" {mean:8.4f} {g:>8s}  [{100*glo:5.1f},{100*ghi:5.1f}]")
    print(f"{'Planner (oracle)':<24s} " + " ".join(f"{1.0:8.4f}{'':8s}" for _ in BAS)
          + f" {1.0:8.4f} {100.0:6.1f}%")

    print(f"\nactive-set eta (max over slots carrying load): " + ", ".join(
        f"{NICE[b]} {np.mean(active_eta[b]):.3f}" for b in BAS))
    print("beta clipped inside the active set: " + ", ".join(
        f"{NICE[b]} {np.mean(active_beta_clip[b]):.1f}/24 h" for b in BAS))
    print(f"SHADE rounds to a 1e-12 residual: median {int(np.median(shade_rounds))}, "
          f"max {int(np.max(shade_rounds))}")

    out = os.path.join(os.path.dirname(__file__), "..", "results", "E3.json")
    json.dump({"provenance": "REPLAY-GRID / SIM-WORKLOAD. Grid: EIA-930 "
                             "Jul-Dec 2024, consumption boundary, net-load "
                             "regressor. Workload: synthetic. MARL omitted "
                             "(GPU training).",
               "n": N, "T": T, "seeds": SEEDS, "flex_share": FLEX_SHARE,
               "lam": LAM, "shade_K": SHADE_K,
               "regions": BAS, "conformance": conf,
               "active_eta": {b: float(np.mean(active_eta[b])) for b in BAS},
               "active_beta_clipped": {b: float(np.mean(active_beta_clip[b]))
                                       for b in BAS},
               "shade_rounds_median": int(np.median(shade_rounds)),
               "baseline_for_gap": BASE, "table": table},
              open(out, "w"), indent=2)
    print(f"\nwrote {os.path.relpath(out)}")
