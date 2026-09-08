"""E11 -- the main comparison on a trace-derived workload instead of a generated one.

Everything on the workload side of Table 1 was generated: operator sizes
log-normal with sigma = 0.6, deadline slack uniform on two to twenty-four hours,
a Gaussian arrival bump, an envelope of a flat 35% of daily energy.  This script
replaces the parts of that a real production trace can replace, keeps the parts
it cannot, and reports whether the conclusions move.

Source
------
Azure Public Dataset V2 (the 2019 VM trace), CC BY 4.0, reduced by
`derive_azure_workload.py` to `data/azure2019/derived_workload.json`.

Taken from the trace
--------------------
  operator count and sizes   the 5,248 subscriptions with delay-insensitive
                             load, and their shares of delay-insensitive
                             core-hours.  The top n by size become the n
                             operators; their relative sizes are the trace's,
                             not a log-normal's.  The largest holds 4.5% of
                             flexible core-hours, which is the market-share
                             parameter rho-bar of Theorem 4.
  arrival shape              each operator's own diurnal profile of flexible
                             work starts, weighted by core-hours.
  power envelope             each operator's per-hour peak concurrent
                             delay-insensitive core count over the thirty days,
                             normalised, in place of the flat 35% cap.

Still assumed, and swept
------------------------
  deferral horizon           no public trace records deadlines.  The staircase
                             is built from a horizon H: work arriving by hour h
                             must be complete by h + H.  H is swept over
                             {2, 4, 8, 12, 24} hours.
  diurnal phase              trace timestamps carry no timezone, so the arrival
                             curve is known only up to a rotation.  All 24
                             alignments against the grid clock are run and the
                             range reported.  This doubles as a question worth
                             asking anyway: how much of the result depends on
                             whether a datacenter's busy hours coincide with the
                             green hour?
  energy level               core-hours are a proxy; the aggregate is scaled to
                             the flexible share of regional demand exactly as in
                             the synthetic setting.

Reported
--------
A  Table 1's methods on the trace-derived workload, against the synthetic one.
B  the phase sweep: the equilibrium gap and the mechanism's benefit at each of
   the 24 possible alignments.
C  the deferral-horizon sweep.
D  the flexible-share thresholds of E9, recomputed on the trace workload.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from gh.core import Feasible, Instance, nash, wedge_fixed
import gh.baselines as B
import E3_main
from E3_main import conform
from E1_calibrate import BAS, load, hourly_profile
from E9_threshold import crossing, THETA_STRAT, THETA_STATIC

DERIVED = os.path.join(HERE, "..", "data", "azure2019", "derived_workload.json")
OUT = os.path.join(HERE, "..", "results", "E11.json")
NICE = {"CISO": "CAISO", "ERCO": "ERCOT", "PJM": "PJM"}

T = 24
N = 32
SHARE = 0.0038
HORIZONS = (2, 4, 8, 12, 24)
H_DEFAULT = 12


def build_trace(aef, mef, beta, daily_MWh, ops, share=SHARE, n=N,
                horizon=H_DEFAULT, phase=0, jitter_seed=None):
    """An Instance whose operator sizes, arrival shape and envelope come from
    the trace, and whose deadlines come from `horizon`.

    `phase` rotates every operator's arrival profile by the same number of hours
    against the grid clock, which is the unknown the trace cannot pin down.
    `jitter_seed`, when given, resamples which n of the top operators are used,
    so the result is not a property of one particular slice.
    """
    sel = ops[:n]
    if jitter_seed is not None:
        rng = np.random.default_rng(jitter_seed)
        sel = [ops[i] for i in rng.choice(len(ops), size=n, replace=False)]
    w = np.array([o["core_hours"] for o in sel], float)
    w = w / w.sum()
    E = w * share * daily_MWh
    X, psi, arrival = [], np.zeros((n, T)), np.zeros((n, T))
    for i, o in enumerate(sel):
        a_i = np.roll(np.asarray(o["arrival_hod"], float), phase)
        a_i = a_i / max(a_i.sum(), 1e-12)
        # envelope: the operator's own demonstrated per-hour peak concurrency,
        # scaled so its daily energy fits.  Floored so a slot the trace never
        # used is not made infeasible by an exactly-zero cap.
        pk = np.roll(np.asarray(o["peak_cores_hod"], float), phase)
        pk = np.maximum(pk, 0.05 * pk.max() if pk.max() > 0 else 1.0)
        u = E[i] * pk / pk.sum() * (T / max(horizon, 1)) * 1.6
        u = np.maximum(u, 1.05 * E[i] / T)
        # deadline staircase: work arriving by hour h is due by h + horizon
        cum_arr = np.cumsum(a_i) * E[i]
        R = np.zeros(T)
        for t in range(T):
            src = t - horizon
            R[t] = cum_arr[src] if src >= 0 else 0.0
        R = np.maximum.accumulate(np.minimum(R, np.cumsum(u) * 0.95))
        R[-1] = min(R[-1], E[i])
        if u.sum() < E[i]:
            u = u * (E[i] / u.sum()) * 1.05
        X.append(Feasible(E=E[i], u=u, R=R))
        h0 = int(np.argmax(a_i))
        psi[i] = np.abs((np.arange(T) - h0 + 12) % 24 - 12) / 24.0
        arrival[i] = E[i] * a_i
    return Instance(a=aef, m=mef, b=beta, X=X, psi=psi, lam=np.zeros(n),
                    arrival=arrival)


def evaluate(inst, want_shade=True):
    Cs = B.planner_value(inst)
    xn = nash(inst)[0]
    r = dict(nash=inst.social(xn) / Cs,
             agnostic=inst.social(B.carbon_agnostic(inst)) / Cs,
             naive=inst.social(B.naive(inst)) / Cs,
             threshold=inst.social(B.threshold(inst)) / Cs,
             jitter=inst.social(B.jitter(inst, seed=0)) / Cs,
             mef_static=inst.social(B.mef_static(inst)) / Cs,
             wedge_fixed=inst.social(wedge_fixed(inst)[0]) / Cs)
    if want_shade:
        r["shade"] = inst.social(E3_main.shade(inst)) / Cs
    y = xn.sum(0)
    A = y > 1e-9
    r["kappa"] = float(np.mean(inst.b[A] * y[A]
                               / np.maximum(inst.m[A] - inst.a[A], 1e-9))) \
        if A.any() else float("nan")
    gap = r["nash"] - 1.0
    r["gap"] = gap
    r["strategic_share"] = (r["wedge_fixed"] - 1.0) / gap if gap > 1e-12 else float("nan")
    r["static_leaves"] = (r["mef_static"] - 1.0) / gap if gap > 1e-12 else float("nan")
    r["peak_to_mean"] = float(y.max() / max(y.mean(), 1e-12))
    return r


def mean_over(rows, key):
    v = [r[key] for r in rows if key in r and np.isfinite(r[key])]
    return float(np.mean(v)) if v else float("nan")


def main():
    if not os.path.exists(DERIVED):
        sys.exit(f"missing {DERIVED}\n"
                 "  run experiments/derive_azure_workload.py first "
                 "(it needs data/azure2019/vmtable.csv.gz; see data/README.md)")
    d = json.load(open(DERIVED))
    ops = d["operators"]
    sd = d["size_distribution"]
    print(f"trace: {d['n_subscriptions_flexible']:,} flexible subscriptions, "
          f"top {len(ops)} retained")
    print(f"  largest holds {100*sd['largest_share_of_flexible_core_hours']:.1f}%"
          f" of flexible core-hours  (rho-bar)")
    print(f"  log-size sd: all {sd['log_size_sd_all']:.2f}, "
          f"top128 {sd['log_size_sd_top128']:.2f}, "
          f"top32 {sd['log_size_sd_top32']:.2f}  "
          f"(generator assumes {sd['generator_sigma_for_comparison']})")
    print(f"  {100*d['censoring']['frac_flexible_spanning_whole_window']:.1f}% "
          f"of flexible VMs span the whole 30-day window, so this is "
          f"long-running")
    print("  capacity rather than short batch work; deferral here means "
          "modulating")
    print("  the power of running capacity, which is what the model represents.")

    print("\nloading EIA-930 ...")
    store = load()
    prof = {}
    for ba in BAS:
        a, m, b, dm = hourly_profile(store, ba)
        m2, b2, _, _ = conform(a, m, b)
        prof[ba] = (a, m2, b2, dm)

    out = {"provenance": "REPLAY-GRID / TRACE-DERIVED WORKLOAD.  Grid: EIA-930 "
                         "Jul-Dec 2024, consumption boundary, net-load "
                         "regressor.  Workload sizes, arrival shape and "
                         "envelope from the Azure 2019 VM trace (CC BY 4.0); "
                         "deadlines and diurnal phase are assumed and swept.",
           "trace": {k: d[k] for k in
                     ("source", "licence", "n_subscriptions_flexible",
                      "size_distribution", "censoring", "parse_validation")},
           "n": N, "share": SHARE, "horizon_default": H_DEFAULT,
           "A_main": {}, "B_phase": {}, "C_horizon": {}, "D_threshold": {}}

    # ---- A. the main comparison, trace against synthetic ---------------
    print("\n=== A. Table 1's methods on the trace-derived workload ===")
    print("Mean over 24 diurnal alignments x 3 operator slices; the synthetic")
    print("column is E3's number on the same grid.\n")
    syn_ref = {"CISO": dict(nash=1.0138, agnostic=1.0313, naive=1.0142,
                            threshold=1.0460, jitter=1.0145,
                            mef_static=1.0001, shade=1.0000),
               "ERCO": dict(nash=1.0423, agnostic=1.0329, naive=1.0425,
                            threshold=1.0312, jitter=1.0394,
                            mef_static=1.0000, shade=1.0000),
               "PJM": dict(nash=1.0696, agnostic=1.0560, naive=1.0923,
                           threshold=1.1054, jitter=1.0665,
                           mef_static=1.0000, shade=1.0000)}
    keys = ["agnostic", "naive", "nash", "threshold", "jitter", "mef_static",
            "shade"]
    for ba in BAS:
        rows = []
        for ph in range(T):
            for js in (None, 11, 12):
                inst = build_trace(*prof[ba], ops, phase=ph, jitter_seed=js)
                rows.append(evaluate(inst,
                                     want_shade=(js is None and ph % 6 == 0)))
        out["A_main"][ba] = dict(
            rows=rows,
            mean={k: mean_over(rows, k) for k in keys + ["kappa", "gap",
                                                         "strategic_share",
                                                         "static_leaves",
                                                         "peak_to_mean"]})
        m = out["A_main"][ba]["mean"]
        print(f"  {NICE[ba]}")
        print(f"    {'method':>22s} {'trace':>9s} {'synthetic':>10s} "
              f"{'difference':>11s}")
        for k in keys:
            s = syn_ref[ba].get(k)
            print(f"    {k:>22s} {m[k]:9.4f} {s:10.4f} {m[k]-s:+11.4f}")
        print(f"    {'kappa':>22s} {m['kappa']:9.4f}")
        print(f"    {'strategic share of gap':>22s} "
              f"{100*m['strategic_share']:8.1f}%")
        print(f"    {'static signal leaves':>22s} "
              f"{100*m['static_leaves']:8.1f}%")
        print()

    # ---- B. the phase sweep --------------------------------------------
    print("=== B. how much depends on the unknown diurnal alignment? ===")
    print("The trace fixes the SHAPE of flexible-work arrival but not its phase")
    print("against the grid clock.  All 24 rotations, CAISO.\n")
    print(f"    {'phase (h)':>10s} {'Nash':>9s} {'agnostic':>9s} "
          f"{'staticMEF':>10s} {'kappa':>8s} {'strategic':>10s}")
    ph_rows = []
    for ph in range(T):
        inst = build_trace(*prof["CISO"], ops, phase=ph)
        r = evaluate(inst, want_shade=False)
        r["phase"] = ph
        ph_rows.append(r)
        print(f"    {ph:10d} {r['nash']:9.5f} {r['agnostic']:9.5f} "
              f"{r['mef_static']:10.5f} {r['kappa']:8.4f} "
              f"{100*r['strategic_share']:9.1f}%")
    out["B_phase"]["CISO"] = ph_rows
    ns = [r["nash"] for r in ph_rows]
    print(f"\n    equilibrium gap over the 24 alignments: "
          f"{100*(min(ns)-1):.2f}% to {100*(max(ns)-1):.2f}% "
          f"(a factor of {(max(ns)-1)/max(min(ns)-1, 1e-12):.1f})")
    ags = [r["agnostic"] for r in ph_rows]
    beats = sum(1 for r in ph_rows if r["nash"] < r["agnostic"])
    print(f"    carbon-aware beats carbon-agnostic in {beats} of 24 alignments")
    out["B_phase"]["summary"] = dict(
        nash_min=float(min(ns)), nash_max=float(max(ns)),
        agnostic_min=float(min(ags)), agnostic_max=float(max(ags)),
        n_alignments_carbon_aware_wins=int(beats))

    # ---- C. the deferral horizon ---------------------------------------
    print("\n=== C. the deferral horizon, the one thing the trace cannot fix ===")
    print(f"    {'horizon (h)':>12s} {'Nash':>9s} {'staticMEF':>10s} "
          f"{'SHADE':>9s} {'kappa':>8s} {'strategic':>10s}")
    for H in HORIZONS:
        rows = [evaluate(build_trace(*prof["CISO"], ops, phase=ph, horizon=H),
                         want_shade=(ph % 8 == 0)) for ph in range(0, T, 4)]
        mm = {k: mean_over(rows, k) for k in
              ("nash", "mef_static", "shade", "kappa", "strategic_share")}
        out["C_horizon"][str(H)] = mm
        print(f"    {H:12d} {mm['nash']:9.5f} {mm['mef_static']:10.5f} "
              f"{mm['shade']:9.5f} {mm['kappa']:8.4f} "
              f"{100*mm['strategic_share']:9.1f}%")

    # ---- D. the thresholds, on the trace workload ----------------------
    print("\n=== D. E9's thresholds recomputed on the trace-derived workload ===")
    shares = np.geomspace(0.002, 0.20, 11)
    print(f"    {'region':>7s} {'strategic >= 25%':>18s} "
          f"{'static leaves > 5%':>20s}")
    for ba in BAS:
        st, sl = [], []
        for ph in (0, 6, 12, 18):
            strat, stat = [], []
            for s in shares:
                r = evaluate(build_trace(*prof[ba], ops, share=float(s),
                                         phase=ph), want_shade=False)
                strat.append(r["strategic_share"])
                stat.append(r["static_leaves"])
            c1 = crossing(shares, strat, THETA_STRAT)
            c2 = crossing(shares, stat, THETA_STATIC[0])
            if c1 is not None:
                st.append(c1)
            if c2 is not None:
                sl.append(c2)
        out["D_threshold"][ba] = dict(
            strategic=[float(v) for v in st], static=[float(v) for v in sl],
            strategic_median=float(np.median(st)) if st else None,
            static_median=float(np.median(sl)) if sl else None)
        f1 = (f"{100*np.median(st):.2f}% [{100*min(st):.2f}-{100*max(st):.2f}]"
              if st else "beyond grid")
        f2 = (f"{100*np.median(sl):.2f}% [{100*min(sl):.2f}-{100*max(sl):.2f}]"
              if sl else "beyond grid")
        print(f"    {NICE[ba]:>7s} {f1:>18s} {f2:>20s}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=float)
    print(f"\nwrote {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
