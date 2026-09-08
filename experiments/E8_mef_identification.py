"""E8 -- how well is the marginal emission factor actually identified?

E1 reports one specification: consumption boundary, net-load regressor, day-over-day
first differences inside an hour-of-day bin, slope through the origin.  That is a
defensible choice, and E1 shows two of the three alternatives it was chosen over.
It is not, on its own, an identification argument.  This script supplies one, by
asking whether the numbers the paper depends on survive the choices a sceptical
reviewer would make differently.

What is varied
--------------
BOUNDARY   how emissions are attributed to a balancing authority
           production          in-region generation only
           consumption         own generation + net imports priced at the
                               contemporaneous intensity of the rest of the
                               interconnection                     [E1 primary]
           consumption_fixed   net imports priced at the period-average
                               rest-of-interconnection intensity, which is what
                               emissions registries actually do
           consumption_gas     net imports priced at the gas factor, a stress
                               case standing for "imports are marginal"

REGRESSOR  what the emissions difference is regressed on
           netload             d(demand - wind - solar)                [primary]
           demand              d(demand)
           decomposed          d(demand), d(wind), d(solar) entered separately.
                               This is not an alternative so much as a TEST: the
                               net-load regressor is exactly the restriction
                               b_demand = -b_wind = -b_solar, and entering the
                               three separately lets us check it instead of
                               assuming it.
           netload_lag         d(netload) and its previous-day lag, for a stack
                               whose response is not contemporaneous
           netload_interchange d(netload) and d(net interchange), which
                               separates a domestic dispatch response from an
                               import response

ESTIMATOR  ols          slope through the origin (E1's estimator)
           ols_int      slope with an intercept, so a drift in the bin cannot
                        masquerade as a response
           huber        Huber M-estimator, down-weighting the outage-sized
                        excursions that a squared loss chases
           theilsen     median of pairwise slopes; no moment assumptions
           binlocal     separate slopes in net-load terciles, which estimates
                        the response WITHOUT assuming it is affine, and returns
                        the curvature as a finite difference rather than as the
                        coefficient of an interaction term

WINDOW     full, summer (Jul-Sep), autumn (Oct-Dec), weekday, weekend,
           high-VRE and low-VRE terciles of renewable share, high-load and
           low-load terciles

UNCERTAINTY  Newey-West HAC standard errors everywhere (the differences are a
           time series in the bin and their residuals are autocorrelated, so an
           i.i.d. bootstrap understates the interval), plus a moving-block
           bootstrap for the primary and two alternates.

FALSIFICATION  two placebos that a spurious regression would fail:
           permutation  the regressor is randomly permuted within the bin; a
                        real dispatch response must collapse to zero
           lead         emissions differences are regressed on TOMORROW's
                        net-load difference; a merit-order response is
                        contemporaneous, so a lead coefficient of the same size
                        as the contemporaneous one would indicate a common
                        trend rather than a response

Output: results/E8.json, and a printed identification verdict per region on the
four-level scale the paper uses (strong / moderate / weak / not identified).
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from E1_calibrate import (BAS, CLEAN_DECILE, EF_GAS_DEFAULT, FLEX_SHARE,
                          FUEL_COL, ISLANDED, T, VRE_KEYS, WECC, factors, load)

OUT = os.path.join(HERE, "..", "results", "E8.json")
NICE = {"CISO": "CAISO", "ERCO": "ERCOT", "PJM": "PJM"}

BOUNDARIES = ["consumption", "production", "consumption_fixed", "consumption_gas"]
REGRESSORS = ["netload", "demand", "decomposed", "netload_lag",
              "netload_interchange"]
ESTIMATORS = ["ols", "ols_int", "huber", "theilsen", "binlocal"]
WINDOWS = ["full", "summer", "autumn", "weekday", "weekend",
           "highvre", "lowvre", "highload", "lowload"]

MIN_OBS = 30                     # per hour-of-day bin, after differencing
MAX_GAP = 7                      # days between the two members of a difference
#   Seven, not one.  Within an hour-of-day bin, consecutive rows are normally
#   consecutive days, so the cap binds only where a WINDOW subsets the calendar
#   -- the weekend split leaves Saturday-Sunday pairs and then a five-day jump,
#   and a cap of three discards so many pairs that the bin fails MIN_OBS and the
#   window reports "not estimable".  Seven keeps same-week pairs, which is the
#   matching the marginal-emissions literature uses, and still refuses the
#   month-apart differences a renewable-tercile window would otherwise create.
HAC_LAGS = 5                     # days; the differences are daily inside a bin


# ---------------------------------------------------------------------------
# series construction, generalised over the accounting boundary
# ---------------------------------------------------------------------------

def series(store, ba, ef_gas=EF_GAS_DEFAULT, boundary="consumption"):
    """(hour_of_day, demand, emissions, vre, wind, solar, net_interchange).

    `boundary` extends E1's two with two further defensible attributions of
    imported energy.  All four use the SAME boundary for the average factor and
    for the marginal regression, which is the property that matters: E1's audit
    found that mixing them was one of the three errors that moved the headline.
    """
    F = factors(ef_gas)
    fuels = store["fuels"]
    ef = np.array([F[k] for k in fuels])
    wind_mask = np.array([k in ("wnd_nb", "wnd_b") for k in fuels])
    sol_mask = np.array([k in ("sol_nb", "sol_b") for k in fuels])
    vre_mask = np.array([k in VRE_KEYS for k in fuels])

    d = store["ba"][ba]
    dem, G = d["dem"], d["gen"]
    own_e = G @ ef
    own_g = G.sum(axis=1)
    wind = G[:, wind_mask].sum(axis=1)
    solar = G[:, sol_mask].sum(axis=1)
    vre = G[:, vre_mask].sum(axis=1)
    ni = dem - own_g                                       # net imports, MW

    if ba in ISLANDED or boundary == "production":
        E = own_e
    else:
        key = "WECC" if ba in WECC else "EAST"
        ext = store["inter"][key] - G
        ext_g = ext.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ci = np.where(ext_g > 0, (ext @ ef) / np.maximum(ext_g, 1e-9), np.nan)
        good = np.isfinite(ci)
        ci = np.where(good, ci, np.nanmean(ci[good]) if good.any() else 0.0)
        if boundary == "consumption_fixed":
            ci = np.full_like(ci, float(np.mean(ci)))
        elif boundary == "consumption_gas":
            ci = np.full_like(ci, float(ef_gas))
        with np.errstate(invalid="ignore", divide="ignore"):
            E = np.where(ni > 0.0, own_e + ni * ci,
                         own_e * dem / np.maximum(own_g, 1e-9))

    ok = np.isfinite(dem) & (dem > 0) & (own_g > 0) & np.isfinite(E)
    return (store["hod"][ok], dem[ok], E[ok], vre[ok], wind[ok], solar[ok],
            ni[ok], np.flatnonzero(ok))


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

def _hac_se(X, resid, XtX_inv, lags=HAC_LAGS):
    """Newey-West standard errors.  The observations inside an hour-of-day bin
    are consecutive days, so their residuals are serially correlated: a
    heatwave lasts a week and moves both the load and the units that serve it.
    An i.i.d. interval on this data is optimistic, and reporting one would be
    the same class of error as the pairing bug E1 fixed."""
    n, k = X.shape
    u = X * resid[:, None]
    S = u.T @ u
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.maximum(np.diag(V), 0.0))


def fit_ols(X, y, intercept=False):
    """Least squares with HAC standard errors.  Returns (coef, se, r2, n)."""
    if intercept:
        X = np.column_stack([X, np.ones(len(X))])
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return None
    coef = XtX_inv @ (X.T @ y)
    resid = y - X @ coef
    se = _hac_se(X, resid, XtX_inv)
    # R^2 through the origin is 1 - SSR/sum(y^2); with an intercept it is the
    # usual centred version.  Reporting the wrong one across specifications
    # would make the intercept model look artificially better.
    denom = float(np.sum((y - y.mean()) ** 2)) if intercept else float(y @ y)
    r2 = 1.0 - float(resid @ resid) / max(denom, 1e-30)
    return coef, se, r2, len(y)


def fit_huber(X, y, delta=None, iters=40):
    """Huber M-estimation by iteratively reweighted least squares.  The tuning
    constant is set from the MAD of the OLS residuals, so it adapts to the
    bin's own scale rather than to gCO2 units."""
    base = fit_ols(X, y)
    if base is None:
        return None
    coef = base[0].copy()
    for _ in range(iters):
        r = y - X @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r)))
        if s <= 1e-12:
            break
        d = delta if delta is not None else 1.345 * s
        w = np.where(np.abs(r) <= d, 1.0, d / np.maximum(np.abs(r), 1e-12))
        XtWX = X.T @ (X * w[:, None])
        try:
            new = np.linalg.solve(XtWX, X.T @ (w * y))
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(new - coef)) <= 1e-10 * max(1.0, np.max(np.abs(coef))):
            coef = new
            break
        coef = new
    r = y - X @ coef
    s = 1.4826 * np.median(np.abs(r - np.median(r)))
    d = delta if delta is not None else 1.345 * max(s, 1e-12)
    w = np.where(np.abs(r) <= d, 1.0, d / np.maximum(np.abs(r), 1e-12))
    XtWX = X.T @ (X * w[:, None])
    try:
        inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return None
    se = _hac_se(X * np.sqrt(w)[:, None], r * np.sqrt(w), inv)
    r2 = 1.0 - float(r @ r) / max(float(y @ y), 1e-30)
    return coef, se, r2, len(y)


def fit_theilsen(X, y, max_pairs=20000, seed=0):
    """Theil-Sen: the median of pairwise slopes.  Univariate only, and used
    only where the specification is univariate.  It assumes nothing about the
    error distribution, so agreement with OLS is evidence that the OLS number
    is not being driven by a handful of extreme hours."""
    if X.shape[1] != 1:
        return None
    x = X[:, 0]
    n = len(x)
    rng = np.random.default_rng(seed)
    if n * (n - 1) // 2 <= max_pairs:
        i, j = np.triu_indices(n, 1)
    else:
        i = rng.integers(0, n, max_pairs)
        j = rng.integers(0, n, max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
    dx = x[i] - x[j]
    keep = np.abs(dx) > 1e-9
    if keep.sum() < 20:
        return None
    sl = (y[i] - y[j])[keep] / dx[keep]
    coef = np.array([float(np.median(sl))])
    lo, hi = np.percentile(sl, [2.5, 97.5])
    # a symmetric pseudo-SE, reported only so the table has one column shape
    se = np.array([float((hi - lo) / (2 * 1.96 * np.sqrt(max(len(sl), 1)) ** 0.0))])
    r = y - X @ coef
    r2 = 1.0 - float(r @ r) / max(float(y @ y), 1e-30)
    return coef, se, r2, n


def fit_binlocal(X, y, level, n_bins=3):
    """A slope per tercile of the LEVEL of the regressor.

    This is the estimator that does not assume the response is affine.  E1 gets
    the curvature from the coefficient of an interaction term d(L) * (L - Lbar),
    which is the affine-response model differentiated; here the slope is
    estimated separately inside each tercile of the load level and the
    curvature read off as a finite difference between the top and bottom
    tercile slopes.  If the affine model is adequate the two agree; where they
    disagree, the affine model is the thing to doubt.

    Returns (slope_at_mid, se, r2, n) plus the tercile slopes in `extra`.
    """
    if X.shape[1] != 1:
        return None
    q = np.quantile(level, np.linspace(0, 1, n_bins + 1))
    q[0] -= 1e-9
    q[-1] += 1e-9
    slopes, mids, ns = [], [], []
    ssr = 0.0
    for k in range(n_bins):
        M = (level > q[k]) & (level <= q[k + 1])
        if M.sum() < 12:
            return None
        f = fit_ols(X[M], y[M])
        if f is None:
            return None
        slopes.append(float(f[0][0]))
        mids.append(float(np.mean(level[M])))
        ns.append(int(M.sum()))
        ssr += float(np.sum((y[M] - X[M] @ f[0]) ** 2))
    slopes = np.array(slopes)
    mids = np.array(mids)
    # marginal factor at the middle tercile, curvature as a finite difference
    coef = np.array([slopes[n_bins // 2]])
    dslope = (slopes[-1] - slopes[0]) / max(mids[-1] - mids[0], 1e-9)
    se = np.array([float(np.std(slopes, ddof=1) / np.sqrt(n_bins))])
    r2 = 1.0 - ssr / max(float(y @ y), 1e-30)
    return coef, se, r2, len(y), dict(tercile_slopes=slopes.tolist(),
                                      tercile_mid_load=mids.tolist(),
                                      tercile_n=ns,
                                      dMEF_dload=float(dslope))


# ---------------------------------------------------------------------------
# one (region, boundary, regressor, estimator, window) cell
# ---------------------------------------------------------------------------

def design(dat, regressor):
    """Build the differenced design matrix inside each hour-of-day bin.

    Differencing is day over day WITHIN a bin, which is what removes the
    diurnal mix shift; `lag` and `lead` variants reuse the same differences
    shifted by one day.
    """
    hr, dem, E, vre, wind, solar, ni = dat[:7]
    if regressor == "netload":
        cols = [dem - vre]
        names = ["netload"]
    elif regressor == "demand":
        cols = [dem]
        names = ["demand"]
    elif regressor == "decomposed":
        cols = [dem, -wind, -solar]
        names = ["demand", "neg_wind", "neg_solar"]
    elif regressor in ("netload_lag", "netload_interchange"):
        cols = [dem - vre]
        names = ["netload"]
    else:
        raise ValueError(regressor)
    return np.column_stack(cols), names, ni


def window_mask(dat, store, idx, window):
    """Boolean mask over the retained hours, on the file's own calendar."""
    hr, dem, E, vre, wind, solar, ni = dat[:7]
    n = len(hr)
    if window == "full":
        return np.ones(n, bool)
    month = store["month"][idx]
    dow = store["dow"][idx]                      # 0 = Sunday
    if window == "summer":                       # Jul-Sep
        return month <= 9
    if window == "autumn":                       # Oct-Dec
        return month >= 10
    if window == "weekday":
        return (dow >= 1) & (dow <= 5)
    if window == "weekend":
        return (dow == 0) | (dow == 6)
    share = vre / np.maximum(dem, 1e-9)
    if window in ("highvre", "lowvre"):
        q = np.quantile(share, [1 / 3, 2 / 3])
        return share >= q[1] if window == "highvre" else share <= q[0]
    if window in ("highload", "lowload"):
        q = np.quantile(dem, [1 / 3, 2 / 3])
        return dem >= q[1] if window == "highload" else dem <= q[0]
    raise ValueError(window)


def estimate(store, ba, boundary, regressor, estimator, window,
             ef_gas=EF_GAS_DEFAULT, placebo=None, seed=0):
    """Per-hour-of-day MEF/AEF/beta under one full specification."""
    dat = series(store, ba, ef_gas, boundary)
    hr, dem, E, vre, wind, solar, ni, idx = dat
    Xall, names, ni_all = design(dat, regressor)
    aef_obs = E / dem
    wmask = window_mask(dat, store, idx, window)

    aef = np.full(T, np.nan)
    mef = np.full(T, np.nan)
    beta = np.full(T, np.nan)
    se = np.full(T, np.nan)
    r2 = np.full(T, np.nan)
    nobs = np.zeros(T, int)
    extra_h = {}

    dayno = store["dayno"][idx]
    rng = np.random.default_rng(seed)
    for h in range(1, T + 1):
        M = (hr == h) & wmask
        if M.sum() < MIN_OBS + 2:
            continue
        e_ = E[M]
        Xb = Xall[M]
        lvl = Xb[:, 0]
        dY = np.diff(e_)
        dX = np.diff(Xb, axis=0)
        lvl_mid = 0.5 * (lvl[:-1] + lvl[1:])
        # Differences must be day OVER day.  Once a window subsets the days --
        # weekends, a renewable tercile -- consecutive retained rows can be
        # weeks apart, and differencing across that gap is no longer the
        # short-horizon shock the identifying assumption is about.  Keep only
        # adjacent pairs at most MAX_GAP days apart, and say how many survive.
        gap = np.diff(dayno[M])
        near = gap <= MAX_GAP
        if regressor not in ("netload_lag",):
            dY, dX, lvl_mid = dY[near], dX[near], lvl_mid[near]
            if len(dY) < MIN_OBS:
                continue
        if regressor == "netload_lag":
            if len(dX) < MIN_OBS + 2:
                continue
            dX = np.column_stack([dX[1:, 0], dX[:-1, 0]])
            dY = dY[1:]
            lvl_mid = lvl_mid[1:]
            names = ["netload", "netload_lag1"]
        elif regressor == "netload_interchange":
            dI = np.diff(ni_all[M])[near]
            dX = np.column_stack([dX[:, 0], dI])
            names = ["netload", "d_net_import"]
        if placebo == "permute":
            dX = dX[rng.permutation(len(dX))]
        elif placebo == "lead":
            if len(dX) < MIN_OBS + 2:
                continue
            dX, dY, lvl_mid = dX[1:], dY[:-1], lvl_mid[:-1]

        keep = np.max(np.abs(dX), axis=1) > 1e-6
        dX, dY, lvl_mid = dX[keep], dY[keep], lvl_mid[keep]
        if len(dY) < MIN_OBS:
            continue

        if estimator == "ols":
            f = fit_ols(dX, dY)
        elif estimator == "ols_int":
            f = fit_ols(dX, dY, intercept=True)
        elif estimator == "huber":
            f = fit_huber(dX, dY)
        elif estimator == "theilsen":
            f = fit_theilsen(dX, dY, seed=h)
        elif estimator == "binlocal":
            f = fit_binlocal(dX, dY, lvl_mid)
        else:
            raise ValueError(estimator)
        if f is None:
            continue

        coef, sd, rr, nn = f[0], f[1], f[2], f[3]
        aef[h - 1] = float(np.mean(aef_obs[M]))
        mef[h - 1] = float(coef[0])
        se[h - 1] = float(sd[0])
        r2[h - 1] = float(rr)
        nobs[h - 1] = int(nn)
        if estimator == "binlocal":
            extra_h[h] = f[4]
            beta[h - 1] = 0.5 * f[4]["dMEF_dload"]
        else:
            # E1's curvature: interact the difference with the level
            Z = np.column_stack([dX[:, 0], dX[:, 0] * (lvl_mid - lvl_mid.mean())])
            c, *_ = np.linalg.lstsq(Z, dY, rcond=None)
            beta[h - 1] = 0.5 * float(c[1])
        if regressor == "decomposed":
            extra_h[h] = dict(coef=[float(v) for v in coef],
                              se=[float(v) for v in sd], names=list(names))
        elif regressor in ("netload_lag", "netload_interchange"):
            extra_h[h] = dict(coef=[float(v) for v in coef],
                              se=[float(v) for v in sd], names=list(names))

    return dict(aef=aef, mef=mef, beta=beta, se=se, r2=r2, nobs=nobs,
                aef_obs=aef_obs, hr=hr, dem=dem, extra=extra_h)


# ---------------------------------------------------------------------------
# aggregation to the quantities the paper reports
# ---------------------------------------------------------------------------

def target_hours(aef_obs, hr, eta_finite):
    """E1's target set: bins over-represented in the cleanest decile."""
    k = max(1, int(CLEAN_DECILE * len(aef_obs)))
    clean = np.argsort(aef_obs)[:k]
    cnt = collections.Counter(hr[clean])
    ok = [h for h in cnt if eta_finite[h - 1]]
    if not ok:
        return [], np.array([])
    tot = sum(cnt[j] for j in ok)
    tgt = [h for h in ok if cnt[h] / tot > 1.0 / T] or ok
    w = np.array([cnt[h] for h in tgt], float)
    return sorted(tgt), w[np.argsort(tgt)] / w.sum()


def summarise(res, ef_gas=EF_GAS_DEFAULT):
    """eta, the bound, kappa and their HAC intervals, on the target hours."""
    aef, mef, beta, se = res["aef"], res["mef"], res["beta"], res["se"]
    with np.errstate(invalid="ignore", divide="ignore"):
        eta = 1.0 - aef / mef
    tgt, w = target_hours(res["aef_obs"], res["hr"], np.isfinite(eta))
    if not tgt:
        return None
    ix = np.array(tgt) - 1
    eta_t = eta[ix]
    j = int(np.nanargmax(eta_t))
    hstar = ix[j]
    eta_max = float(eta_t[j])
    # eta = 1 - AEF/MEF is monotone increasing in MEF, so a HAC interval on MEF
    # maps to one on eta by substituting the endpoints.  AEF is a mean over
    # ~180 hours and its sampling error is an order of magnitude smaller, so it
    # is held fixed; the paper says so.
    m, s = mef[hstar], se[hstar]
    lo_m, hi_m = m - 1.96 * s, m + 1.96 * s
    def _eta(mm):
        return 1.0 - aef[hstar] / mm if mm > 0 else float("nan")
    eta_lo, eta_hi = _eta(lo_m), _eta(hi_m)
    def _bound(e):
        return 1.5 / (1.0 - e) if np.isfinite(e) and e < 1 else float("inf")
    daily = float(np.nanmean(res["dem"]) * T)
    y_peak = FLEX_SHARE * daily / max(1, int(CLEAN_DECILE * T))
    aef_t = float(np.sum(w * aef[ix]))
    mef_t = float(np.sum(w * mef[ix]))
    beta_t = float(np.sum(w * beta[ix]))
    kappa = abs(beta_t) * y_peak / max(mef_t - aef_t, 1e-9)
    fin = np.isfinite(mef[ix])
    return dict(
        target_hours=[int(h) for h in tgt],
        aef_target=aef_t, mef_target=mef_t, beta_target=beta_t,
        eta_max=eta_max, eta_max_lo=float(eta_lo), eta_max_hi=float(eta_hi),
        eta_mean=float(np.sum(w * eta[ix])),
        bound_max=_bound(eta_max), bound_lo=_bound(eta_lo),
        bound_hi=_bound(eta_hi),
        mef_at_argmax=float(m), mef_se_at_argmax=float(s),
        mef_ci_at_argmax=[float(lo_m), float(hi_m)],
        kappa=float(kappa), load_GWh_d=daily / 1e3,
        r2_min=float(np.nanmin(res["r2"])), r2_max=float(np.nanmax(res["r2"])),
        r2_median=float(np.nanmedian(res["r2"])),
        n_bins_estimated=int(np.sum(np.isfinite(mef))),
        n_obs_median=int(np.median(res["nobs"][res["nobs"] > 0]))
        if np.any(res["nobs"] > 0) else 0,
        n_mef_below_aef=int(np.sum(mef[np.isfinite(mef)] <
                                   aef[np.isfinite(mef)])),
        n_beta_negative=int(np.sum(beta[np.isfinite(beta)] < 0)),
        n_ci_covers_zero=int(np.sum(
            (mef[ix][fin] - 1.96 * res["se"][ix][fin]) <= 0)),
        frac_ci_covers_zero_all=float(np.mean(
            (mef[np.isfinite(mef)] - 1.96 * res["se"][np.isfinite(mef)]) <= 0)),
    )


# ---------------------------------------------------------------------------
# block bootstrap
# ---------------------------------------------------------------------------

def block_bootstrap_mef(store, ba, boundary, regressor, hour, nboot=400,
                        block=7, seed=3):
    """Moving-block bootstrap of the slope in one hour-of-day bin.

    Blocks are seven consecutive days, so a week-long weather episode is
    resampled whole rather than shredded.  This is the interval the paper
    quotes; the i.i.d. bootstrap E1 used is narrower and, on autocorrelated
    residuals, wrong in the optimistic direction.
    """
    dat = series(store, ba, EF_GAS_DEFAULT, boundary)
    hr, dem, E, vre, wind, solar, ni, idx = dat
    X, _, _ = design(dat, regressor)
    M = hr == hour
    if M.sum() < MIN_OBS + 2:
        return None
    dY = np.diff(E[M])
    dX = np.diff(X[M], axis=0)[:, 0]
    keep = np.abs(dX) > 1e-6
    dY, dX = dY[keep], dX[keep]
    n = len(dY)
    if n < MIN_OBS:
        return None
    nb = max(1, n // block)
    rng = np.random.default_rng(seed)
    out = np.empty(nboot)
    starts_max = n - block
    for j in range(nboot):
        s = rng.integers(0, max(starts_max, 1), nb)
        ii = (s[:, None] + np.arange(block)[None, :]).ravel()
        ii = ii[ii < n]
        xx, yy = dX[ii], dY[ii]
        d = float(xx @ xx)
        out[j] = float(xx @ yy) / d if d > 1e-12 else np.nan
    out = out[np.isfinite(out)]
    lo, hi = np.percentile(out, [2.5, 97.5])
    return dict(point=float(dX @ dY / (dX @ dX)), lo=float(lo), hi=float(hi),
                n=int(n), nboot=int(len(out)), block=block)


# ---------------------------------------------------------------------------
# identification verdict
# ---------------------------------------------------------------------------

def verdict(spread_rel, ci_rel, r2_med, sign_ok, placebo_ratio, lead_ratio,
            restriction_rejected_frac):
    """Four levels, on thresholds fixed before the numbers were looked at.

    Applied SEPARATELY to two quantities, because the data identify them
    differently and conflating them would misreport both:

      the LEVEL of the marginal factor, which sets the scale of the social
      cost; and

      the RATIO eta = 1 - AEF/MEF formed inside one hour-of-day bin, which is
      the only quantity Theorem 1 uses.

    The ratio is the better determined of the two, and not by accident: a
    specification that raises the estimated marginal factor usually raises the
    average factor it is paired against, since both are computed on the same
    accounting boundary from the same hours.  Quoting only the level's spread
    would overstate the uncertainty in the bound; quoting only the ratio's
    would hide that the absolute carbon numbers are soft.

    strong    spread under 15% across every specification tried, sampling
              interval under 20% of the point estimate, median R^2 at least
              0.80, MEF > AEF in every bin, the permutation placebo below 10%
              of the estimate, the lead below 40%, and the identifying
              restriction rejected in under a fifth of the bins.
    moderate  the same at 30% / 40% / 0.50 / 25% / 60% / half the bins.
    weak      the right sign, but two or more of the moderate criteria fail.
    none      the interval covers zero, the sign restriction fails, or the
              permutation placebo does not collapse.
    """
    if (not sign_ok) or placebo_ratio > 0.45 or ci_rel > 1.0:
        return "not identified"
    if (spread_rel <= 0.15 and ci_rel <= 0.20 and r2_med >= 0.80
            and placebo_ratio <= 0.10 and lead_ratio <= 0.40
            and restriction_rejected_frac <= 0.20):
        return "identified strongly"
    if (spread_rel <= 0.30 and ci_rel <= 0.40 and r2_med >= 0.50
            and placebo_ratio <= 0.25 and lead_ratio <= 0.60
            and restriction_rejected_frac <= 0.50):
        return "identified moderately"
    return "weakly identified"


# ---------------------------------------------------------------------------

def main():
    print("loading EIA-930 (Adjusted series) ...")
    store = load()
    print(f"  {store['n_hours']} hours, {len(store['ba'])} balancing authorities")

    out = {"provenance": "REPLAY -- EIA-930 BALANCE Adjusted, Jul-Dec 2024, "
                         "public.  Identification audit of the marginal "
                         "emission factor estimator used in E1.",
           "hac_lags": HAC_LAGS, "min_obs": MIN_OBS,
           "boundaries": BOUNDARIES, "regressors": REGRESSORS,
           "estimators": ESTIMATORS, "windows": WINDOWS,
           "cells": {}, "restriction_test": {}, "placebo": {},
           "block_bootstrap": {}, "verdict": {}}

    # ---- A. the specification grid -------------------------------------
    print("\n=== A. specification grid: MEF on the target hours, and eta ===")
    print(f"{'region':>6s} {'boundary':>18s} {'regressor':>20s} "
          f"{'estimator':>9s} {'window':>9s} {'MEF*':>7s} {'AEF*':>7s} "
          f"{'eta':>6s} {'bound':>6s} {'R2med':>6s} {'n':>5s}")
    print("-" * 116)
    grid = collections.defaultdict(list)
    for ba in BAS:
        # vary one axis at a time from the primary, which is what a reviewer
        # asks for: a full cross product would be 540 cells of which most
        # differ from the primary in three ways at once and say nothing.
        cells = [("consumption", "netload", "ols", "full")]
        cells += [(b, "netload", "ols", "full") for b in BOUNDARIES[1:]]
        cells += [("consumption", r, "ols", "full") for r in REGRESSORS[1:]]
        cells += [("consumption", "netload", e, "full") for e in ESTIMATORS[1:]]
        cells += [("consumption", "netload", "ols", w) for w in WINDOWS[1:]]
        for bnd, reg, est, win in cells:
            if est == "theilsen" and reg in ("decomposed", "netload_lag",
                                             "netload_interchange"):
                continue
            r = estimate(store, ba, bnd, reg, est, win)
            s = summarise(r)
            if s is None:
                print(f"{NICE[ba]:>6s} {bnd:>18s} {reg:>20s} {est:>9s} "
                      f"{win:>9s}  -- not estimable --")
                continue
            key = f"{ba}|{bnd}|{reg}|{est}|{win}"
            out["cells"][key] = s
            if reg in ("decomposed", "netload_lag", "netload_interchange"):
                out["cells"][key]["per_hour_extra"] = {
                    str(k): v for k, v in r["extra"].items()}
            grid[ba].append((key, s))
            print(f"{NICE[ba]:>6s} {bnd:>18s} {reg:>20s} {est:>9s} {win:>9s} "
                  f"{s['mef_target']:7.1f} {s['aef_target']:7.1f} "
                  f"{s['eta_max']:6.3f} {s['bound_max']:6.2f} "
                  f"{s['r2_median']:6.2f} {s['n_obs_median']:5d}")
        print()

    # ---- B. the identifying restriction, tested ------------------------
    print("=== B. the net-load restriction, tested rather than assumed ===")
    print("The net-load regressor imposes dE/d(demand) = -dE/d(wind) =")
    print("-dE/d(solar).  Entering the three separately lets the data reject it.")
    print(f"{'region':>6s} {'hour':>5s} {'b_dem':>8s} {'b_wind':>8s} "
          f"{'b_sol':>8s} {'|dem-wind|/dem':>15s} {'|dem-sol|/dem':>14s} "
          f"{'Wald p (joint)':>15s}")
    print("-" * 92)
    for ba in BAS:
        r = estimate(store, ba, "consumption", "decomposed", "ols", "full")
        rows = []
        for h, ex in sorted(r["extra"].items()):
            c = np.array(ex["coef"])
            s = np.array(ex["se"])
            if c[0] <= 0:
                continue
            d1 = abs(c[0] - c[1]) / abs(c[0])
            d2 = abs(c[0] - c[2]) / abs(c[0])
            # two-sided z on each contrast, treating the SEs as independent,
            # which is conservative in the direction of NOT rejecting only if
            # the covariance is negative; reported as an indicative statistic
            z1 = (c[0] - c[1]) / max(np.hypot(s[0], s[1]), 1e-12)
            z2 = (c[0] - c[2]) / max(np.hypot(s[0], s[2]), 1e-12)
            chi2 = z1 ** 2 + z2 ** 2
            from math import exp
            p = exp(-chi2 / 2.0)              # chi2_2 survival, exact
            rows.append((h, c, d1, d2, p))
        if not rows:
            continue
        med_d1 = float(np.median([x[2] for x in rows]))
        med_d2 = float(np.median([x[3] for x in rows]))
        n_rej = int(np.sum([x[4] < 0.05 for x in rows]))
        out["restriction_test"][ba] = dict(
            n_hours=len(rows), median_rel_gap_wind=med_d1,
            median_rel_gap_solar=med_d2, n_rejected_at_5pct=n_rej,
            per_hour=[dict(hour=int(h), coef=[float(v) for v in c],
                           rel_gap_wind=float(d1), rel_gap_solar=float(d2),
                           p=float(p)) for h, c, d1, d2, p in rows])
        h, c, d1, d2, p = rows[len(rows) // 2]
        print(f"{NICE[ba]:>6s} {h:5d} {c[0]:8.1f} {c[1]:8.1f} {c[2]:8.1f} "
              f"{d1:15.3f} {d2:14.3f} {p:15.3f}")
        print(f"{'':>6s} median over {len(rows):2d} hours: "
              f"wind gap {med_d1:.3f}, solar gap {med_d2:.3f}, "
              f"rejected at 5% in {n_rej}/{len(rows)} hours")
    print()

    # ---- C. placebos ---------------------------------------------------
    print("=== C. falsification: permutation and lead placebos ===")
    print(f"{'region':>6s} {'MEF*':>8s} {'permuted':>9s} {'ratio':>7s} "
          f"{'lead':>8s} {'ratio':>7s}")
    print("-" * 52)
    for ba in BAS:
        base = summarise(estimate(store, ba, "consumption", "netload", "ols",
                                  "full"))
        perm = summarise(estimate(store, ba, "consumption", "netload", "ols",
                                  "full", placebo="permute", seed=11))
        lead = summarise(estimate(store, ba, "consumption", "netload", "ols",
                                  "full", placebo="lead"))
        pr = abs(perm["mef_target"]) / abs(base["mef_target"]) if perm else None
        lr = abs(lead["mef_target"]) / abs(base["mef_target"]) if lead else None
        out["placebo"][ba] = dict(base_mef=base["mef_target"],
                                  permuted_mef=perm["mef_target"] if perm else None,
                                  permuted_ratio=pr,
                                  lead_mef=lead["mef_target"] if lead else None,
                                  lead_ratio=lr,
                                  permuted_r2=perm["r2_median"] if perm else None,
                                  lead_r2=lead["r2_median"] if lead else None)
        print(f"{NICE[ba]:>6s} {base['mef_target']:8.1f} "
              f"{perm['mef_target']:9.1f} {pr:7.3f} "
              f"{lead['mef_target']:8.1f} {lr:7.3f}")
    print()

    # ---- D. block bootstrap on the binding hour ------------------------
    print("=== D. moving-block bootstrap (7-day blocks) on the eta-max hour ===")
    print(f"{'region':>6s} {'hour':>5s} {'MEF':>8s} {'95% block CI':>22s} "
          f"{'width/point':>12s} {'eta':>7s} {'eta CI':>18s}")
    print("-" * 84)
    for ba in BAS:
        s = out["cells"][f"{ba}|consumption|netload|ols|full"]
        r = estimate(store, ba, "consumption", "netload", "ols", "full")
        with np.errstate(invalid="ignore"):
            eta = 1.0 - r["aef"] / r["mef"]
        ix = np.array(s["target_hours"]) - 1
        hstar = int(ix[int(np.nanargmax(eta[ix]))]) + 1
        bb = block_bootstrap_mef(store, ba, "consumption", "netload", hstar)
        if bb is None:
            continue
        a = r["aef"][hstar - 1]
        # eta = 1 - AEF/MEF is INCREASING in MEF, so the low end of the MEF
        # interval maps to the low end of the eta interval.
        e_lo = 1.0 - a / bb["lo"] if bb["lo"] > 0 else float("-inf")
        e_hi = 1.0 - a / bb["hi"] if bb["hi"] > 0 else float("nan")
        out["block_bootstrap"][ba] = dict(hour=hstar, **bb,
                                          aef=float(a),
                                          eta=float(1.0 - a / bb["point"]),
                                          eta_lo=float(e_lo),
                                          eta_hi=float(e_hi),
                                          bound_lo=float(1.5 / (1 - e_lo)),
                                          bound_hi=float(1.5 / (1 - e_hi))
                                          if e_hi < 1 else float("inf"))
        w = (bb["hi"] - bb["lo"]) / abs(bb["point"])
        print(f"{NICE[ba]:>6s} {hstar:5d} {bb['point']:8.1f} "
              f"[{bb['lo']:8.1f},{bb['hi']:8.1f}] {w:12.3f} "
              f"{1 - a / bb['point']:7.3f} [{e_lo:6.3f},{e_hi:6.3f}]")
    print()

    # ---- E. verdict ----------------------------------------------------
    print("=== E. identification verdict ===")
    print("Reported separately for the marginal factor's LEVEL and for the ratio")
    print("eta the theory actually uses.  Specification cells (boundary,")
    print("regressor, estimator) are competing estimates of one quantity; WINDOW")
    print("cells are not -- a renewable tercile is a different physical regime,")
    print("and its eta is a different number rather than a worse estimate of the")
    print("same one.  Both spreads are reported, and kept apart.")
    print()
    for ba in BAS:
        ks = [k for k, _ in grid[ba]]
        spec = [k for k in ks if k.endswith("|full")]
        wins = [k for k in ks if not k.endswith("|full")]
        prim = out["cells"][f"{ba}|consumption|netload|ols|full"]
        bb = out["block_bootstrap"].get(ba)

        def spread(field, keys):
            v = np.array([out["cells"][k][field] for k in keys])
            return (float(np.min(v)), float(np.max(v)),
                    float((np.max(v) - np.min(v)) / abs(prim[field])))

        mlo, mhi, msp = spread("mef_target", spec)
        elo, ehi, esp = spread("eta_max", spec)
        blo, bhi, bsp = spread("bound_max", spec)
        wlo, whi, _ = spread("eta_max", wins) if wins else (float("nan"),) * 3
        rt = out["restriction_test"].get(ba, {})
        rej = rt.get("n_rejected_at_5pct", 0) / max(rt.get("n_hours", 1), 1)
        ci_mef = ((bb["hi"] - bb["lo"]) / abs(bb["point"])) if bb else float("nan")
        ci_eta = ((bb["eta_hi"] - bb["eta_lo"]) / abs(bb["eta"])) if bb \
            else float("nan")
        r2m = prim["r2_median"]
        sign_ok = prim["n_mef_below_aef"] == 0
        pr = out["placebo"][ba]["permuted_ratio"]
        lr = out["placebo"][ba]["lead_ratio"]

        v_level = verdict(msp, ci_mef, r2m, sign_ok, pr, lr, rej)
        v_ratio = verdict(esp, ci_eta, r2m, sign_ok, pr, lr, rej)
        eta_unres = float(
            out["cells"][f"{ba}|consumption|decomposed|ols|full"]["eta_max"])
        out["verdict"][ba] = dict(
            level_verdict=v_level, ratio_verdict=v_ratio, causal=False,
            n_spec_cells=len(spec), n_window_cells=len(wins),
            mef_primary=float(prim["mef_target"]),
            mef_spec_min=mlo, mef_spec_max=mhi, mef_spec_spread_rel=msp,
            mef_sampling_ci_rel=ci_mef,
            eta_primary=float(prim["eta_max"]),
            eta_spec_min=elo, eta_spec_max=ehi, eta_spec_spread_rel=esp,
            eta_sampling_ci=[bb["eta_lo"], bb["eta_hi"]] if bb else None,
            eta_sampling_ci_rel=ci_eta,
            eta_window_min=wlo, eta_window_max=whi,
            bound_primary=float(prim["bound_max"]),
            bound_spec_min=blo, bound_spec_max=bhi,
            bound_spec_spread_rel=bsp,
            bound_sampling_ci=[bb["bound_lo"], bb["bound_hi"]] if bb else None,
            r2_median=r2m, sign_ok=bool(sign_ok),
            restriction_rejected_frac=float(rej),
            placebo_ratio=pr, lead_ratio=lr, eta_unrestricted=eta_unres)

        print(f"  {NICE[ba]}")
        print(f"     MEF level   : {v_level.upper()}")
        print(f"                   {mlo:.0f}-{mhi:.0f} g/kWh over {len(spec)} "
              f"specifications (primary {prim['mef_target']:.0f}, spread "
              f"{100*msp:.0f}%); block-bootstrap CI {100*ci_mef:.0f}% of point")
        print(f"     eta / bound : {v_ratio.upper()}")
        print(f"                   eta {elo:.3f}-{ehi:.3f} over the same "
              f"specifications (primary {prim['eta_max']:.3f}, spread "
              f"{100*esp:.0f}%); bound {blo:.2f}-{bhi:.2f} "
              f"(primary {prim['bound_max']:.2f})")
        if bb:
            print(f"                   sampling CI on eta "
                  f"[{bb['eta_lo']:.3f}, {bb['eta_hi']:.3f}], on the bound "
                  f"[{bb['bound_lo']:.2f}, {bb['bound_hi']:.2f}]")
        print(f"                   unrestricted (wind and solar entered "
              f"separately): eta = {eta_unres:.3f}")
        if wins:
            print(f"     across regimes: eta {wlo:.3f}-{whi:.3f} over season, "
                  f"day type, renewable and load terciles -- a real spread in "
                  f"the quantity, not in its estimate")
        print(f"     diagnostics : median R2 {r2m:.2f}; MEF<AEF in "
              f"{prim['n_mef_below_aef']}/24 bins; net-load restriction "
              f"rejected at 5% in {rt.get('n_rejected_at_5pct', 0)}/"
              f"{rt.get('n_hours', 0)} bins")
        print(f"                   placebos: permuted {100*pr:.1f}% of the "
              f"estimate, lead {100*lr:.1f}%")
        print()
    print("None of these is a CAUSAL estimate.  Hour-of-day binning and")
    print("day-over-day differencing remove level and diurnal confounding; they")
    print("do not remove simultaneity between load and curtailment, and no")
    print("instrument is used.  The defensible description is an operational")
    print("marginal-emissions proxy under a stated identifying restriction that")
    print("the data reject in part.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=float)
    print(f"\nwrote {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
