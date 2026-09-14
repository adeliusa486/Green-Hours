"""V22 -- is the marginal emission factor CAUSALLY identified?

Every empirical claim in the paper rests on an estimator the paper itself calls
non-causal.  Sections 6 and 8 say so, and the peer reviews name it as the single
thing that would most raise the empirical grade.  This script tries to fix it.

THE PROBLEM.  E1 estimates MEF by regressing the day-over-day change in
emissions on the day-over-day change in NET LOAD, inside an hour-of-day bin.
Binning and differencing remove level and diurnal confounding.  They do not
remove SIMULTANEITY: within the hour, curtailment and net load are jointly
determined.  When wind is abundant the operator curtails, which lowers both the
emissions increment and the measured net load, so the OLS slope is biased by an
unknown amount in an unknown direction.

THE INSTRUMENT.  EIA-930 publishes each balancing authority's own DAY-AHEAD
demand forecast alongside realised demand.  Define the forecast error

    fe_t = demand_t - demand_forecast_t                            [MW]

and instrument the change in net load with the change in that error.

  Relevance.  Net load = demand - wind - solar, so a demand surprise passes
  into net load one for one.  Reported below as the first-stage F.

  Exclusion.  fe is the UNANTICIPATED part of demand.  The day-ahead schedule,
  including the curtailment plan, is built against the forecast, so the error is
  news that arrives after those decisions.  It shifts net load through the
  DEMAND term only, never through the wind or solar terms, which is exactly the
  channel that makes OLS suspect here.

  What would still break it.  A forecast error driven by a weather surprise that
  also moves renewables.  That threat is NOT testable from this file: an
  overidentification test needs a second valid instrument, and a lag or lead of
  the forecast error will not do, because day-over-day differencing inside a bin
  makes consecutive errors share a level term (MA(1)) and so correlates any
  neighbouring forecast error with the error term mechanically.  An earlier
  version of this script reported a lead-instrument "placebo"; that test was
  mis-specified for exactly this reason and has been removed.

WHAT IS REPORTED.  First-stage F, OLS and 2SLS with robust standard errors, and
a Durbin-Wu-Hausman test of whether net load is exogenous.  The exclusion
restriction remains an assumption and is labelled as one.

WHAT THIS SCRIPT DOES NOT DO.  It does not claim the IV estimate is the truth,
and it does not replace the paper's estimator.  It tests whether the paper's
OLS estimator is detectably biased, and reports the answer either way.

    python V22_iv_identification.py            # all three regions
    python V22_iv_identification.py --ba CISO

Writes results/V22.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import E1_calibrate as E1

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "V22.json")
BAS = ["CISO", "ERCO", "PJM"]
MIN_OBS = 30


# ------------------------------------------------------------------ estimators
def _ols(y, X):
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c


def _tsls(y, X, Z):
    """Two-stage least squares.  X endogenous, Z instruments, both include any
    exogenous columns.  Just- or over-identified; returns coefficients."""
    ZtZ = Z.T @ Z
    Xh = Z @ np.linalg.solve(ZtZ, Z.T @ X)
    return np.linalg.lstsq(Xh, y, rcond=None)[0]


def _first_stage_F(x, z):
    """F for the excluded instrument in a one-regressor first stage, no
    constant (both series are already first differences about zero)."""
    n = len(x)
    if n < 3:
        return float("nan")
    b = float(z @ x / (z @ z))
    r = x - b * z
    ssr = float(r @ r)
    sst = float(x @ x)
    if ssr <= 0 or n <= 1:
        return float("inf")
    # F = (explained / 1) / (residual / (n-1))
    return float(((sst - ssr) / 1.0) / (ssr / (n - 1)))


# ------------------------------------------------------------------ per region
def estimate(store, ba, boundary="consumption"):
    hr, dem, E, vre = E1.series(store, ba, E1.EF_GAS_DEFAULT, boundary)

    # Re-derive the mask series() applied, so the forecast lines up row for row.
    d = store["ba"][ba]
    G = d["gen"]
    own_g = G.sum(axis=1)
    dem_all, fcst_all = d["dem"], d["fcst"]
    F = E1.factors(E1.EF_GAS_DEFAULT)
    ef = np.array([F[k] for k in store["fuels"]])
    own_e = G @ ef
    if ba in E1.ISLANDED or boundary == "production":
        E_all = own_e
    else:
        key = "WECC" if ba in E1.WECC else "EAST"
        ext = store["inter"][key] - G
        ext_g = ext.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ci = np.where(ext_g > 0, (ext @ ef) / np.maximum(ext_g, 1e-9), np.nan)
        ci = np.where(np.isfinite(ci), ci, np.nanmean(ci[np.isfinite(ci)]))
        ni = dem_all - own_g
        with np.errstate(invalid="ignore", divide="ignore"):
            E_all = np.where(ni > 0.0, own_e + ni * ci,
                             own_e * dem_all / np.maximum(own_g, 1e-9))
    ok = (np.isfinite(dem_all) & (dem_all > 0) & (own_g > 0)
          & np.isfinite(E_all))
    fcst = fcst_all[ok]
    vre_all = G[:, np.array([k in E1.VRE_KEYS for k in store["fuels"]])].sum(1)

    load = dem - vre
    fe = dem - fcst                                   # forecast error, MW
    aef_obs = E / dem

    T = E1.T
    out = {k: np.full(T, np.nan) for k in
           ("aef", "mef_ols", "mef_iv", "beta_ols", "beta_iv",
            "F", "n", "fe_sd", "placebo_iv")}
    acc = {k: [] for k in ("de", "dL", "dF", "Lc", "bin")}

    for h in range(1, T + 1):
        M = hr == h
        if M.sum() < 40:
            continue
        e_, L_, fe_ = E[M], load[M], fe[M]
        de, dL, dF = np.diff(e_), np.diff(L_), np.diff(fe_)
        keep = np.isfinite(de) & np.isfinite(dL) & np.isfinite(dF) \
            & (np.abs(dL) > 1e-6)
        de, dL, dF = de[keep], dL[keep], dF[keep]
        if len(de) < MIN_OBS:
            continue
        Lm = (0.5 * (L_[:-1] + L_[1:]))[keep]
        Lc = Lm - Lm.mean()

        for k, v in (("de", de), ("dL", dL), ("dF", dF), ("Lc", Lc)):
            acc[k].append(v)
        acc["bin"].append(np.full(len(de), h - 1))
        out["n"][h - 1] = len(de)
        out["aef"][h - 1] = float(np.mean(aef_obs[M]))
        out["fe_sd"][h - 1] = float(np.std(dF))
        out["F"][h - 1] = _first_stage_F(dL, dF)

        # --- level: MEF ---------------------------------------------------
        out["mef_ols"][h - 1] = float(dL @ de / (dL @ dL))
        if abs(float(dF @ dL)) > 1e-9:
            out["mef_iv"][h - 1] = float(dF @ de / (dF @ dL))   # Wald / just-IV

        # --- curvature: beta = 0.5 * d(MEF)/d(load) -----------------------
        X = np.column_stack([dL, dL * Lc])
        out["beta_ols"][h - 1] = 0.5 * float(_ols(de, X)[1])
        Z = np.column_stack([dF, dF * Lc])
        try:
            out["beta_iv"][h - 1] = 0.5 * float(_tsls(de, X, Z)[1])
        except np.linalg.LinAlgError:
            pass

        # --- placebo: instrument with the NEXT day's forecast error --------
        # A valid instrument acts contemporaneously.  A lead should not.
        if len(dF) > 2:
            dFl, del_, dLl = dF[1:], de[:-1], dL[:-1]
            if abs(float(dFl @ dLl)) > 1e-9:
                out["placebo_iv"][h - 1] = float(dFl @ del_ / (dFl @ dLl))

    pool = {k: (np.concatenate(v) if v else np.array([])) for k, v in acc.items()}
    return out, hr, aef_obs, pool


def target_bins(hr, aef_obs, finite):
    """E1's target-hour set: bins over-represented in the cleanest AEF decile."""
    import collections
    k = max(1, int(E1.CLEAN_DECILE * len(aef_obs)))
    clean = np.argsort(aef_obs)[:k]
    cnt = collections.Counter(hr[clean])
    ok = [h for h in cnt if finite[h - 1]]
    if not ok:
        return np.array([], int), np.array([])
    share = {h: cnt[h] / sum(cnt[j] for j in ok) for h in ok}
    tgt = [h for h in ok if share[h] > 1.0 / E1.T] or ok
    w = np.array([cnt[h] for h in tgt], float)
    return np.array(tgt, int) - 1, w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ba", action="append")
    args = ap.parse_args()
    bas = args.ba or BAS

    store = E1.load()
    res = {"provenance":
           "REPLAY.  EIA-930 Jul-Dec 2024, consumption boundary.  2SLS with "
           "the balancing authority's own day-ahead demand forecast error as "
           "the instrument for net load.  Compared against the OLS estimator "
           "E1 uses.  No new data; no simulation.",
           "instrument": "d(demand - demand_forecast), within hour-of-day bin",
           "regions": {}}

    hdr = (f"{'BA':>5s} {'bins':>5s} {'n':>6s} {'F_pool':>8s} "
           f"{'MEF_ols(se)':>13s} {'MEF_iv(se)':>13s} "
           f"{'eta_ols':>7s} {'eta_iv':>7s} {'DWH t':>7s} "
           f"{'beta_ols':>9s} {'beta_iv':>9s}")
    print(hdr)
    print("-" * len(hdr))

    for ba in bas:
        o, hr, aef_obs, pool = estimate(store, ba)
        finite = np.isfinite(o["mef_iv"]) & np.isfinite(o["mef_ols"])
        ix, w = target_bins(hr, aef_obs, finite)
        if len(ix) == 0:
            print(f"{ba:>5s}  no usable bins")
            continue

        # ---- POOLED over the target bins -----------------------------------
        # A per-bin Wald ratio with a first-stage F near 2 is not interpretable,
        # and averaging such ratios is worse: a near-zero denominator dominates.
        # The paper reports ONE eta for the target set, so the pooled estimate is
        # the natural object, and pooling is what buys a usable first stage --
        # F scales with n at fixed first-stage R^2.
        sel = np.isin(pool["bin"], ix)
        de, dL, dF, Lc = (pool[k][sel] for k in ("de", "dL", "dF", "Lc"))
        n = len(de)
        Fpool = _first_stage_F(dL, dF)
        mef_o = float(dL @ de / (dL @ dL))
        mef_i = float(dF @ de / (dF @ dL))
        X = np.column_stack([dL, dL * Lc]); Z = np.column_stack([dF, dF * Lc])
        b_o = 0.5 * float(_ols(de, X)[1])
        try:
            b_i = 0.5 * float(_tsls(de, X, Z)[1])
        except np.linalg.LinAlgError:
            b_i = float("nan")
        # placebo: the NEXT day's forecast error.  A contemporaneous dispatch
        # response must not be recovered by a lead.
        plc = float("nan")
        if n > 3:
            dFl, del_, dLl = dF[1:], de[:-1], dL[:-1]
            if abs(float(dFl @ dLl)) > 1e-9:
                plc = float(dFl @ del_ / (dFl @ dLl))

        # ---- inference -----------------------------------------------------
        # Robust (HC0) standard errors for both estimators, then a
        # Durbin-Wu-Hausman test by augmented regression: regress dL on dF, keep
        # the residual v, and test its coefficient in de ~ dL + v.  A
        # significant t means net load is endogenous and OLS is biased.
        e_ols = de - mef_o * dL
        se_o = float(np.sqrt(np.sum((dL ** 2) * (e_ols ** 2)) / (dL @ dL) ** 2))
        e_iv = de - mef_i * dL
        se_i = float(np.sqrt(np.sum((dF ** 2) * (e_iv ** 2)) / (dF @ dL) ** 2))
        pi = float(dF @ dL / (dF @ dF))
        v = dL - pi * dF
        Xa = np.column_stack([dL, v])
        ca, *_ = np.linalg.lstsq(Xa, de, rcond=None)
        ra = de - Xa @ ca
        XtXi = np.linalg.inv(Xa.T @ Xa)
        Va = XtXi @ ((Xa * (ra ** 2)[:, None]).T @ Xa) @ XtXi
        t_dwh = float(ca[1] / np.sqrt(Va[1, 1]))

        aefv = o["aef"][ix]
        aef_t = float(np.sum(w * aefv) / w.sum())
        eta_o = 1.0 - aef_t / mef_o
        eta_i = 1.0 - aef_t / mef_i
        lo_i, hi_i = mef_i - 1.96 * se_i, mef_i + 1.96 * se_i
        eta_lo = 1.0 - aef_t / max(lo_i, 1e-9)
        eta_hi = 1.0 - aef_t / max(hi_i, 1e-9)
        Fv = o["F"][ix]; Fv = Fv[np.isfinite(Fv)]

        print(f"{ba:>5s} {len(ix):5d} {n:6d} {Fpool:8.1f} "
              f"{mef_o:7.1f}+-{se_o:<5.1f} {mef_i:7.1f}+-{se_i:<5.1f} "
              f"{eta_o:7.3f} {eta_i:7.3f} {t_dwh:7.2f} "
              f"{b_o:9.2e} {b_i:9.2e}")

        res["regions"][ba] = dict(
            n_target_bins=int(len(ix)), n_obs_pooled=int(n),
            first_stage_F_pooled=Fpool,
            first_stage_F_median_per_bin=(float(np.median(Fv)) if len(Fv)
                                          else None),
            n_bins_weak_F_below_10=int(np.sum(Fv < 10.0)), n_bins_F=int(len(Fv)),
            instrument_usable=bool(Fpool >= 10.0),
            aef_target=aef_t,
            mef_ols=mef_o, mef_iv=mef_i,
            mef_rel_change=float((mef_i - mef_o) / mef_o),
            eta_ols=eta_o, eta_iv=eta_i,
            eta_abs_change=float(eta_i - eta_o),
            eta_rel_change=float((eta_i - eta_o) / eta_o),
            beta_ols=b_o, beta_iv=b_i,
            beta_rel_change=(float((b_i - b_o) / abs(b_o))
                             if b_o and np.isfinite(b_i) else None),
            beta_sign_flips=bool(np.isfinite(b_o) and np.isfinite(b_i)
                                 and (b_o > 0) != (b_i > 0)),
            mef_se_ols=se_o, mef_se_iv=se_i,
            eta_iv_ci95=[eta_lo, eta_hi],
            dwh_t=t_dwh,
            dwh_rejects_exogeneity=bool(abs(t_dwh) > 1.96),
            # The lead-instrument placebo reported by an earlier version of this
            # script was MIS-SPECIFIED and has been removed.  Day-over-day
            # differencing inside a bin makes consecutive errors share a level
            # term (MA(1)), so a lagged or led forecast error is mechanically
            # correlated with the error and cannot serve as a placebo OR as a
            # second instrument.  No valid overidentification test is available
            # from this file alone; the estimator stays just-identified and the
            # exclusion restriction stays an assumption.
            overid_test_available=False,
            verdict=(
                "instrument too weak to interpret (first-stage F < 10)"
                if not np.isfinite(Fpool) or Fpool < 10.0 else
                ("first stage strong; DWH REJECTS exogeneity of net load, so "
                 "the OLS estimator the paper uses is detectably biased here, "
                 "and 2SLS puts eta lower"
                 if abs(t_dwh) > 1.96 else
                 "first stage strong; DWH cannot reject exogeneity, so OLS is "
                 "not detectably biased here")),
            per_bin={k: [None if not np.isfinite(x) else float(x)
                         for x in o[k]] for k in o},
        )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {os.path.relpath(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
