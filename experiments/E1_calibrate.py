"""E1 -- dispatch-response calibration from EIA-930.

Input
-----
EIA-930 Hourly Grid Monitor, BALANCE file, July-December 2024 (public, no API
key).  `data/README.md` gives the exact download.  Hourly demand, net
generation by fuel, and interchange for every US balancing authority.

Output, per balancing authority and per hour-of-day bin
-------------------------------------------------------
    AEF_h    average emission factor at the operating point   [gCO2/kWh]
    MEF_h    marginal emission factor                         [gCO2/kWh]
    beta_h   HALF the rate at which MEF rises with load, so that
             E(D+y) - E(D) = MEF*y + beta*y^2                 [gCO2/kWh/MW]
    eta_h    = 1 - AEF_h/MEF_h, and the bound 3/(2(1-eta))
    kappa    congestion strength beta*y_peak / (MEF - AEF) at a reference
             flexible share

Three methodological points, each of which changed the numbers
---------------------------------------------------------------
1.  ACCOUNTING BOUNDARY.  CAISO imports 17.5% of its energy on average and up
    to 33% in an hour.  Attributing only in-region generation understates its
    average factor badly, and the carbon-intensity feeds operators actually
    consume are consumption-based.  We therefore compute consumption-based
    emissions -- own generation plus net imports priced at the contemporaneous
    average intensity of the rest of the interconnection, computed from the
    same file -- and use the SAME boundary for AEF and for MEF.  The
    production-based variant is reported as a sensitivity.  ERCOT is
    electrically islanded (|interchange| < 1% of demand), so the two coincide.

2.  REGRESSOR.  MEF is the response of emissions to an increment of load.  A
    merit-order stack responds to NET load (demand minus variable renewable
    output): an extra MWh of demand and a lost MWh of wind are the same shock
    to the dispatchable fleet.  Regressing on demand alone discards the
    dominant source of variation at a fixed hour, and where renewables are
    large the estimate collapses -- in ERCOT (VRE = 32% of demand) the
    demand-only regression has R^2 = 0.02-0.18, returns MEF < AEF in 11 of 24
    hours, and its bootstrap intervals cover zero.  On net load the same bins
    give R^2 = 0.94-0.98, MEF = 480-560 gCO2/kWh (a gas turbine, which is what
    sets ERCOT's margin), and MEF > AEF everywhere.  Net load is the default;
    demand-only is reported as a sensitivity.  The identifying assumption is
    stated in the paper: dispatchable output responds to net load, so
    dE/d(load) = dE/d(net load) at fixed renewable output.

3.  PAIRING.  eta_h is a per-slot quantity, so AEF and MEF must come from the
    SAME hours.  An earlier version divided an AEF averaged over the cleanest
    decile of individual hours by an MEF taken as the median over all 24
    hour-of-day bins; that mixes an extreme subset with a central tendency and
    inflated CAISO's eta from 0.67 to 0.80.  Here every quantity is computed
    within an hour-of-day bin and eta_h is formed inside the bin.

What this is not
----------------
These are estimated marginal emission *associations* under the stated
identifying assumption, not the output of a dispatch model or a causal
experiment.  Hour-of-day binning and first differencing remove level and
diurnal confounding; they do not remove simultaneity between demand and
renewable curtailment, and we say so in the paper.

Emission factors, gCO2/kWh of generation (direct combustion, IPCC/EPA ranges):
coal 1000, natural gas 430, petroleum 700, other/unknown 500; nuclear, hydro,
wind, solar, geothermal, storage 0.  A sensitivity over gas in [370, 550] is
reported, since gas sets the margin in all three regions.
"""
import csv
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSVP = os.path.join(HERE, "..", "..", "data", "EIA930_BALANCE_2024_Jul_Dec.csv")
if not os.path.exists(CSVP):
    CSVP = os.path.join(HERE, "..", "data", "EIA930_BALANCE_2024_Jul_Dec.csv")
OUT = os.path.join(HERE, "..", "results", "E1.json")

BAS = ["CISO", "ERCO", "PJM"]
EF_GAS_DEFAULT = 430.0
T = 24
FLEX_SHARE = 0.0038          # aggregate flexible load as a share of demand
CLEAN_DECILE = 0.10          # the hours carbon-aware agents target

# Adjusted-series column indices in the EIA-930 BALANCE file.  The Adjusted
# series is used throughout: mixing adjusted demand with raw generation, as an
# earlier version did, compares two different vintages of the same hour.
COL_DEMAND_ADJ = 13
FUEL_COL = {
    "coal": 48, "gas": 49, "nuc": 50, "oil": 51, "hyd": 52, "pump": 53,
    "sol_nb": 54, "sol_b": 55, "wnd_nb": 56, "wnd_b": 57, "bat": 58,
    "oes": 59, "ues": 60, "geo": 61, "oth": 62, "unk": 63,
}
VRE_KEYS = ["sol_nb", "sol_b", "wnd_nb", "wnd_b"]

# Interconnection membership, for pricing imports.  ERCOT is islanded.
WECC = set("""CISO BPAT PACE PACW PSCO NEVP AZPS SRP WACM PSEI LDWP PGE IPCO
BANC TEPC PNM AVA NWMT EPE SCL WALC WAUW GCPD TIDC IID DOPD CHPD GRID TPWR GWA
WWA DEAA HGMA GRIF AVRN""".split())
ISLANDED = {"ERCO"}


def factors(ef_gas):
    z = {k: 0.0 for k in FUEL_COL}
    z.update({"coal": 1000.0, "gas": float(ef_gas), "oil": 700.0,
              "oth": 500.0, "unk": 500.0})
    return z


def _weekday(y, m, d):
    """Sakamoto's algorithm: 0 = Sunday.  Used only to label the calendar axis
    for E8's weekday/weekend split; kept here so the loader owns every column
    it derives from the file."""
    t = (0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4)
    if m < 3:
        y -= 1
    return (y + y // 4 - y // 100 + y // 400 + t[m - 1] + d) % 7


def _num(s):
    s = (s or "").replace(",", "")
    if not s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load(path=None):
    """Read the file once into arrays.

    Returns a dict with, for every balancing authority, the hour-of-day index,
    demand, and generation by fuel; plus per-hour totals of generation by fuel
    for each interconnection.  Keeping the fuel breakdown (rather than an
    emissions number) means the import intensity for any gas emission factor is
    a linear combination of arrays already in memory, so the sensitivity sweeps
    cost nothing -- the earlier version rescanned 61 balancing authorities for
    every hour on every call and took minutes.
    """
    path = path or CSVP
    fuels = list(FUEL_COL)
    rows = collections.defaultdict(list)
    hours = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        head = next(r)
        for key, j in FUEL_COL.items():          # fail loudly on layout change
            if not head[j].startswith("Net Generation (MW)"):
                raise RuntimeError(f"column {j} is {head[j]!r}, not a fuel column")
        if head[COL_DEMAND_ADJ] != "Demand (MW) (Adjusted)":
            raise RuntimeError(f"column {COL_DEMAND_ADJ} is "
                               f"{head[COL_DEMAND_ADJ]!r}")
        for row in r:
            key = (row[1], int(row[2]))
            if key not in hours:
                hours[key] = len(hours)
            rows[row[0]].append(
                (hours[key], _num(row[COL_DEMAND_ADJ]),
                 [_num(row[FUEL_COL[k]]) for k in fuels]))
    H = len(hours)
    ba_data = {}
    for ba, rs in rows.items():
        idx = np.array([x[0] for x in rs], int)
        dem = np.full(H, np.nan); gen = np.zeros((H, len(fuels)))
        dem[idx] = [x[1] for x in rs]
        gen[idx] = np.nan_to_num(np.array([x[2] for x in rs], float))
        ba_data[ba] = dict(dem=dem, gen=gen)
    order = sorted(hours, key=lambda k: hours[k])
    hod = np.array([k[1] for k in order], int)
    # Calendar labels for the same hour axis.  E1 itself never needs them --
    # every quantity it reports is formed inside an hour-of-day bin -- but the
    # identification audit (E8) splits those bins by season and by day of week,
    # and deriving a weekday from row position rather than from the date is the
    # kind of shortcut that silently mislabels a series once a BA misses an
    # hour.  Parsed here, once, from the file's own "Data Date" column.
    _mm = np.array([int(k[0][0:2]) for k in order], int)
    _dd = np.array([int(k[0][3:5]) for k in order], int)
    _yy = np.array([int(k[0][6:10]) for k in order], int)
    _dow = np.array([_weekday(y, m, d) for y, m, d in zip(_yy, _mm, _dd)], int)
    _cum = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    _leap = ((_yy % 4 == 0) & (_yy % 100 != 0)) | (_yy % 400 == 0)
    _doy = (_yy - _yy.min()) * 366 + _cum[_mm - 1] + _dd \
        + (_leap & (_mm > 2)).astype(int)
    inter = {}
    for name, members in (("WECC", WECC), ("EAST", set(rows) - WECC - ISLANDED)):
        tot = np.zeros((H, len(fuels)))
        for ba in members:
            if ba in ba_data:
                tot += ba_data[ba]["gen"]
        inter[name] = tot
    return dict(fuels=fuels, hod=hod, ba=ba_data, inter=inter, n_hours=H,
                month=_mm, day=_dd, year=_yy, dow=_dow, dayno=_doy)


def series(store, ba, ef_gas=EF_GAS_DEFAULT, boundary="consumption"):
    """(hour_of_day, demand, emissions, vre) for one balancing authority.

    `boundary` is 'consumption' (own generation plus net imports priced at the
    contemporaneous average intensity of the rest of the interconnection) or
    'production' (own generation only).
    """
    F = factors(ef_gas)
    fuels = store["fuels"]
    ef = np.array([F[k] for k in fuels])
    vre_mask = np.array([k in VRE_KEYS for k in fuels])
    d = store["ba"][ba]
    dem, G = d["dem"], d["gen"]
    own_e = G @ ef
    own_g = G.sum(axis=1)
    vre = G[:, vre_mask].sum(axis=1)

    if ba in ISLANDED or boundary == "production":
        E = own_e
    else:
        key = "WECC" if ba in WECC else "EAST"
        ext = store["inter"][key] - G
        ext_g = ext.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ci = np.where(ext_g > 0, (ext @ ef) / np.maximum(ext_g, 1e-9), np.nan)
        ci = np.where(np.isfinite(ci), ci, np.nanmean(ci[np.isfinite(ci)]))
        ni = dem - own_g                                   # net imports
        with np.errstate(invalid="ignore", divide="ignore"):
            E = np.where(ni > 0.0, own_e + ni * ci,
                         own_e * dem / np.maximum(own_g, 1e-9))

    ok = np.isfinite(dem) & (dem > 0) & (own_g > 0) & np.isfinite(E)
    return store["hod"][ok], dem[ok], E[ok], vre[ok]


def _slope_ci(y, x, nboot=400, seed=1):
    """Slope through the origin, its bootstrap 95% interval, and R^2."""
    s = float(x @ y / (x @ x))
    rng = np.random.default_rng(seed)
    bs = np.empty(nboot)
    n = len(x)
    for j in range(nboot):
        i = rng.integers(0, n, n)
        bs[j] = x[i] @ y[i] / (x[i] @ x[i])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    r2 = 1.0 - float(np.sum((y - s * x) ** 2) / np.sum(y ** 2))
    return s, float(lo), float(hi), r2


def calibrate(store, ba, ef_gas=EF_GAS_DEFAULT, boundary="consumption",
              regressor="netload", nboot=400):
    """Per-hour-of-day AEF, MEF, beta, eta for one balancing authority."""
    hr, dem, E, vre = series(store, ba, ef_gas, boundary)
    aef_obs = E / dem
    load = dem - vre if regressor == "netload" else dem

    aef = np.full(T, np.nan); mef = np.full(T, np.nan)
    beta = np.full(T, np.nan); r2 = np.full(T, np.nan)
    lo = np.full(T, np.nan); hi = np.full(T, np.nan)
    for h in range(1, T + 1):
        M = hr == h
        if M.sum() < 40:
            continue
        e_, L_ = E[M], load[M]
        # first differences DAY OVER DAY within the hour-of-day bin, so no
        # diurnal mix shift is read as a marginal response
        de, dL = np.diff(e_), np.diff(L_)
        keep = np.abs(dL) > 1e-6
        de, dL = de[keep], dL[keep]
        if len(de) < 30:
            continue
        Lm = (0.5 * (L_[:-1] + L_[1:]))[keep]
        s, l, u, rr = _slope_ci(de, dL, nboot=nboot, seed=h)
        # curvature: let the slope vary with the load level.  The model writes
        # E(D+y) - E(D) = MEF*y + beta*y^2, so beta is HALF d(MEF)/d(load).
        X = np.column_stack([dL, dL * (Lm - Lm.mean())])
        c, *_ = np.linalg.lstsq(X, de, rcond=None)
        aef[h - 1] = float(np.mean(aef_obs[M]))
        mef[h - 1] = s
        lo[h - 1], hi[h - 1], r2[h - 1] = l, u, rr
        beta[h - 1] = 0.5 * float(c[1])

    eta = 1.0 - aef / mef
    # The hours agents target.  Take the cleanest decile of individual hours by
    # AEF, then keep the hour-of-day bins that are OVER-REPRESENTED in it
    # (share above the uniform 1/24).  Without that filter the "target" set
    # degenerates to all 24 bins in ERCOT and PJM, whose cleanest hours are
    # wind-driven and therefore scattered across the clock rather than
    # concentrated in a solar window as CAISO's are -- a difference between the
    # regions that is itself worth reporting.
    k = max(1, int(CLEAN_DECILE * len(aef_obs)))
    clean = np.argsort(aef_obs)[:k]
    cnt = collections.Counter(hr[clean])
    ok = [h for h in cnt if np.isfinite(eta[h - 1])]
    share = {h: cnt[h] / sum(cnt[j] for j in ok) for h in ok}
    tgt = [h for h in ok if share[h] > 1.0 / T] or ok
    w = np.array([cnt[h] for h in tgt], float); w /= w.sum()
    ix = np.array(tgt) - 1
    eta_max = float(np.max(eta[ix]))
    eta_mean = float(np.sum(w * eta[ix]))
    aef_t = float(np.sum(w * aef[ix])); mef_t = float(np.sum(w * mef[ix]))
    beta_t = float(np.sum(w * beta[ix]))

    daily = float(np.mean(dem) * T)
    y_peak = FLEX_SHARE * daily / max(1, int(CLEAN_DECILE * T))
    kappa = abs(beta_t) * y_peak / max(mef_t - aef_t, 1e-9)

    return dict(
        ba=ba, boundary=boundary, regressor=regressor, ef_gas=ef_gas,
        n_hours=int(len(dem)),
        aef_h=aef.tolist(), mef_h=mef.tolist(), beta_h=beta.tolist(),
        mef_lo_h=lo.tolist(), mef_hi_h=hi.tolist(), r2_h=r2.tolist(),
        eta_h=eta.tolist(),
        target_hours=[int(h) for h in sorted(tgt)],
        target_mass=float(sum(cnt[h] for h in tgt) / sum(cnt[j] for j in ok)),
        target_weights=[float(v) for v in w[np.argsort(tgt)]],
        n_beta_negative_target=int(np.sum(beta[ix] < 0)),
        beta_target_hours=[float(v) for v in beta[ix]],
        aef_target=aef_t, mef_target=mef_t, beta_target=beta_t,
        eta_max=eta_max, eta_mean=eta_mean,
        bound_max=1.5 / (1.0 - eta_max), bound_mean=1.5 / (1.0 - eta_mean),
        kappa=float(kappa), load_GWh_d=daily / 1e3,
        aef_all=float(np.nanmean(aef)),
        n_eta_negative=int(np.sum(eta[np.isfinite(eta)] < 0)),
        n_beta_negative=int(np.sum(beta[np.isfinite(beta)] < 0)),
        r2_min=float(np.nanmin(r2)), r2_max=float(np.nanmax(r2)),
    )


def hourly_profile(store, ba, ef_gas=EF_GAS_DEFAULT, boundary="consumption",
                   regressor="netload"):
    """(aef, mef, beta, daily_MWh) as T-vectors, for building instances.

    Any hour-of-day bin that fails to estimate is filled from the neighbouring
    bins.  Whether MEF >= AEF and beta >= 0 hold in the data is REPORTED by
    `calibrate`, not enforced here; `E3_main` records how often the model's
    hypotheses have to be imposed.
    """
    c = calibrate(store, ba, ef_gas, boundary, regressor, nboot=1)
    a = np.array(c["aef_h"]); m = np.array(c["mef_h"]); b = np.array(c["beta_h"])
    for v in (a, m, b):
        bad = ~np.isfinite(v)
        if bad.any():
            v[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad),
                               v[~bad], period=T)
    return a, m, b, c["load_GWh_d"] * 1e3


def _fmt(c):
    return (f"{c['ba']:>6s} {c['load_GWh_d']:8.0f} {c['aef_target']:7.1f} "
            f"{c['mef_target']:7.1f} {c['eta_mean']:7.3f} {c['eta_max']:7.3f} "
            f"{c['bound_max']:7.2f} {c['beta_target']:10.2e} {c['kappa']:8.4f} "
            f"{c['r2_min']:5.2f}-{c['r2_max']:.2f} {c['n_eta_negative']:4d} "
            f"{c['n_beta_negative']:4d}")


if __name__ == "__main__":
    if not os.path.exists(CSVP):
        sys.exit(f"missing {CSVP} -- see data/README.md for the download")
    print("loading EIA-930 (Adjusted series) ...")
    store = load()
    print(f"  {store['n_hours']} hours, {len(store['ba'])} balancing "
          f"authorities")

    hdr = (f"\n{'BA':>6s} {'load':>8s} {'AEF':>7s} {'MEF':>7s} {'etaMn':>7s} "
           f"{'etaMx':>7s} {'bound':>7s} {'beta':>10s} {'kappa':>8s} {'R2':>9s} "
           f"{'eta<0':>4s} {'b<0':>4s}")
    print("\n=== PRIMARY: consumption boundary, net-load regressor ===")
    print(hdr); print("-" * 96)
    main = {}
    for b in BAS:
        c = calibrate(store, b)
        main[b] = c
        print(_fmt(c))

    print("\n=== SENSITIVITY: accounting boundary and regressor ===")
    print(hdr); print("-" * 96)
    sens = {}
    for bound in ("consumption", "production"):
        for reg in ("netload", "demand"):
            if bound == "consumption" and reg == "netload":
                continue
            for b in BAS:
                c = calibrate(store, b, boundary=bound, regressor=reg, nboot=1)
                sens[f"{b}|{bound}|{reg}"] = {k: c[k] for k in
                                              ("aef_target", "mef_target",
                                               "eta_mean", "eta_max",
                                               "bound_max", "r2_min", "r2_max",
                                               "n_eta_negative")}
                print(f"{bound[:4]}/{reg[:4]} " + _fmt(c)[6:])

    print("\n=== SENSITIVITY: natural-gas emission factor (gCO2/kWh) ===")
    print(f"{'BA':>6s} " + " ".join(f"{g:>9d}" for g in (370, 430, 490, 550)))
    print("-" * 50)
    gas = {}
    for b in BAS:
        row = [calibrate(store, b, ef_gas=float(g), nboot=1)["eta_max"]
               for g in (370, 430, 490, 550)]
        gas[b] = row
        print(f"{b:>6s} " + " ".join(f"{v:9.3f}" for v in row))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"provenance": "REPLAY -- EIA-930 BALANCE Adjusted, "
                             "Jul-Dec 2024, public",
               "boundary": "consumption", "regressor": "netload",
               "ef_gas": EF_GAS_DEFAULT, "flex_share": FLEX_SHARE,
               "regions": main, "sensitivity_boundary_regressor": sens,
               "eta_max_vs_gas_factor": gas},
              open(OUT, "w"), indent=2)
    print(f"\nwrote {os.path.relpath(OUT)}")
