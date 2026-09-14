"""V18 -- does the calibration survive winter, and a year it was not fitted on?

Every number in the paper comes from EIA-930 July-December 2024.  That window is
the solar-heavy half of the year, which is the regime most favourable to a large
average-to-marginal divergence, and all three reviewers of the draft asked the
same question: is eta a seasonal artifact?

The honest test needs hours the paper has never seen.  This script reads the
2025 January-June BALANCE file -- a different half-year AND a different year --
and re-runs the preferred specification on three windows:

  winter   Jan-Feb 2025   the low-solar months the 2024 H2 file cannot reach
  spring   Apr-Jun 2025   rising solar, the closest 2025 analogue to Jul-Sep
  full     Jan-Jun 2025   the whole out-of-sample half-year

and reports eta, the bound, beta and kappa beside the 2024 H2 values.  Nothing
is re-fitted or tuned: same boundary (consumption), same regressor (net load),
same estimator (OLS), same hour-of-day binning, same target-hour rule.

Two questions, kept apart:

  SEASON        does eta move between a low-solar and a high-solar window?
  OUT-OF-SAMPLE does the 2024 H2 calibration reproduce on 2025 H1 at all?

Writes results/V18.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from E8_mef_identification import estimate, summarise

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "V18.json")
E8J = os.path.join(HERE, "..", "results", "E8.json")
CSV_2025 = os.path.join(HERE, "..", "data", "EIA930_BALANCE_2025_Jan_Jun.csv")

BAS = ["CISO", "ERCO", "PJM"]
NICE = {"CISO": "CAISO", "ERCO": "ERCOT", "PJM": "PJM"}

# the preferred specification, unchanged
BOUNDARY, REGRESSOR, ESTIMATOR = "consumption", "netload", "ols"
WINDOWS_2025 = ["full", "winter", "spring"]
# the 2024 H2 windows to line up against, read from E8.json
WINDOWS_2024 = ["full", "summer", "autumn"]

KEEP = ("aef_target", "mef_target", "beta_target", "eta_max", "eta_max_lo",
        "eta_max_hi", "bound_max", "kappa", "r2_median", "n_bins_estimated",
        "n_mef_below_aef", "n_beta_negative", "target_hours")


if __name__ == "__main__":
    if not os.path.exists(CSV_2025):
        sys.exit(f"missing {CSV_2025}\n"
                 "  curl -o data/EIA930_BALANCE_2025_Jan_Jun.csv \\\n"
                 "    https://www.eia.gov/electricity/gridmonitor/"
                 "sixMonthFiles/EIA930_BALANCE_2025_Jan_Jun.csv")

    # E1.load is imported by E8 at module scope against the 2024 file; load the
    # 2025 one explicitly and hand it to estimate(), which takes the store.
    from E1_calibrate import load
    print(f"loading {os.path.basename(CSV_2025)} ...", flush=True)
    store = load(CSV_2025)
    print(f"  {len(store['month'])} rows, months "
          f"{store['month'].min()}-{store['month'].max()}, "
          f"year {sorted(set(store['year'].tolist()))}", flush=True)

    with open(E8J) as f:
        e8 = json.load(f)

    out = {"provenance": "REPLAY.  EIA-930 BALANCE Jan-Jun 2025, public, "
                         "downloaded separately from the Jul-Dec 2024 file "
                         "every other experiment uses.  Preferred "
                         "specification only (consumption boundary, net-load "
                         "regressor, OLS); nothing re-fitted.",
           "source": "EIA930_BALANCE_2025_Jan_Jun.csv",
           "spec": f"{BOUNDARY}|{REGRESSOR}|{ESTIMATOR}",
           "y2025": {}, "y2024": {}, "comparison": {}}

    print(f"\n{'region':<7s} {'window':<8s} {'AEF*':>7s} {'MEF*':>7s} "
          f"{'eta':>7s} {'bound':>7s} {'beta':>9s} {'kappa':>7s} {'R2':>6s}")
    print("-" * 72)
    for ba in BAS:
        out["y2025"][ba] = {}
        for w in WINDOWS_2025:
            res = estimate(store, ba, BOUNDARY, REGRESSOR, ESTIMATOR, w)
            s = summarise(res)
            out["y2025"][ba][w] = {k: s[k] for k in KEEP if k in s}
            print(f"{NICE[ba]:<7s} {w:<8s} {s['aef_target']:>7.1f} "
                  f"{s['mef_target']:>7.1f} {s['eta_max']:>7.4f} "
                  f"{s['bound_max']:>7.2f} {s['beta_target']:>9.5f} "
                  f"{s['kappa']:>7.4f} {s['r2_median']:>6.3f}", flush=True)
        # the 2024 H2 counterparts, straight from E8
        out["y2024"][ba] = {}
        for w in WINDOWS_2024:
            c = e8["cells"].get(f"{ba}|{BOUNDARY}|{REGRESSOR}|{ESTIMATOR}|{w}")
            if c:
                out["y2024"][ba][w] = {k: c[k] for k in KEEP if k in c}
        print("-" * 72)

    # ---- the two questions -----------------------------------------------
    comp = {}
    for ba in BAS:
        a = out["y2025"][ba]
        b = out["y2024"][ba]
        eta25_full = a["full"]["eta_max"]
        eta24_full = b["full"]["eta_max"]
        # season: low-solar vs high-solar window, within each half-year
        seasons = [a["winter"]["eta_max"], a["spring"]["eta_max"],
                   b["summer"]["eta_max"], b["autumn"]["eta_max"]]
        comp[ba] = {
            "eta_2024H2": eta24_full,
            "eta_2025H1": eta25_full,
            "eta_out_of_sample_rel_change": (eta25_full - eta24_full) / eta24_full,
            "eta_winter_2025": a["winter"]["eta_max"],
            "eta_summer_2024": b["summer"]["eta_max"],
            "eta_winter_vs_summer_rel": (a["winter"]["eta_max"]
                                         - b["summer"]["eta_max"])
                                        / b["summer"]["eta_max"],
            "eta_season_min": float(min(seasons)),
            "eta_season_max": float(max(seasons)),
            "eta_season_rel_range": float((max(seasons) - min(seasons))
                                          / np.median(seasons)),
            "bound_2024H2": b["full"]["bound_max"],
            "bound_2025H1": a["full"]["bound_max"],
            "beta_2024H2": b["full"]["beta_target"],
            "beta_2025H1": a["full"]["beta_target"],
            "kappa_2024H2": b["full"]["kappa"],
            "kappa_2025H1": a["full"]["kappa"],
        }
    out["comparison"] = comp

    print("\nout-of-sample and seasonal stability of eta")
    print(f"{'region':<7s} {'2024H2':>8s} {'2025H1':>8s} {'change':>8s}   "
          f"{'winter25':>9s} {'summer24':>9s} {'w-vs-s':>8s}   "
          f"{'season range':>12s}")
    for ba in BAS:
        c = comp[ba]
        print(f"{NICE[ba]:<7s} {c['eta_2024H2']:>8.4f} {c['eta_2025H1']:>8.4f} "
              f"{c['eta_out_of_sample_rel_change']:>+7.1%}   "
              f"{c['eta_winter_2025']:>9.4f} {c['eta_summer_2024']:>9.4f} "
              f"{c['eta_winter_vs_summer_rel']:>+7.1%}   "
              f"{c['eta_season_rel_range']:>11.1%}")

    print("\nbeta and kappa, the quantities the negative result rests on")
    print(f"{'region':<7s} {'beta 24':>9s} {'beta 25':>9s}   "
          f"{'kappa 24':>9s} {'kappa 25':>9s}")
    for ba in BAS:
        c = comp[ba]
        print(f"{NICE[ba]:<7s} {c['beta_2024H2']:>9.5f} {c['beta_2025H1']:>9.5f}   "
              f"{c['kappa_2024H2']:>9.4f} {c['kappa_2025H1']:>9.4f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT}")
