"""V15 -- paired significance tests for Table 1, from E3's stored per-seed data.

Table 1 reports a mean, a standard deviation and a paired bootstrap interval,
and a reviewer can reasonably object that the comparison the paper's framing
leans on hardest -- carbon-agnostic at 1.0401 against the equilibrium at 1.0419
-- is a difference of 0.0018 against standard deviations as large as 0.0046.
Overlapping error bars are not a test, and the seeds are paired: every method
sees the same twenty generated workloads on the same three grids, so the
comparison should use that pairing rather than throw it away.

This script re-runs E3's methods on E3's seeds, keeps the per-seed ratios
instead of only their summary, and reports for each method against the
equilibrium:

  * the mean paired difference and its 95% interval, over seeds x regions;
  * a Wilcoxon signed-rank statistic, which assumes neither normality nor equal
    variance and is the appropriate test for 60 paired observations;
  * Cohen's d_z for paired samples, because a p-value on 60 pairs says only that
    a difference exists and the paper needs to say whether it matters.

Writes results/V15.json.  No new modelling: same instances, same solver, same
seeds as E3.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import stats

import gh.baselines as B
from E1_calibrate import BAS, load, hourly_profile
from E3_main import (METHODS, NICE, SEEDS, build, conform, _ne)

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "V15.json")
REF = "Independent greedy"          # the equilibrium: Table 1's baseline


def paired(diff):
    """Summarise a vector of paired differences (method - equilibrium)."""
    d = np.asarray(diff, float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    try:
        w, p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        w, p = float(w), float(p)
    except ValueError:                       # all differences identically zero
        w, p = float("nan"), 1.0
    return {
        "n_pairs": n,
        "mean_diff": mean,
        "ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "wilcoxon_W": w,
        "p_value": p,
        "cohens_dz": (mean / sd) if sd > 0 else float("inf"),
        "significant_at_05": bool(p < 0.05),
    }


if __name__ == "__main__":
    print("loading EIA-930 ...")
    store = load()
    prof = {}
    for b in BAS:
        a, m, be, dm = hourly_profile(store, b)
        m2, b2, _, _ = conform(a, m, be)
        prof[b] = (a, m2, b2, dm)

    # per-seed, per-region ratio for every method: exactly E3's inner loop
    per = {name: [] for name, _ in METHODS}
    for b in BAS:
        a, m, be, dm = prof[b]
        for s in range(SEEDS):
            inst = build(a, m, be, dm, s)
            Cs = B.planner_value(inst)
            for name, fn in METHODS:
                try:
                    x = fn(inst, s)
                except Exception:
                    per[name].append(np.nan)
                    continue
                if x is None or any(not Xi.check(x[i], tol=1e-6)
                                    for i, Xi in enumerate(inst.X)):
                    per[name].append(np.nan)
                    continue
                per[name].append(float(inst.social(x) / Cs))
        print(f"  {NICE[b]} done")

    ref = np.asarray(per[REF], float)
    out = {"provenance": {"seeds": SEEDS, "regions": list(BAS),
                          "reference": REF,
                          "what": "paired Wilcoxon of each method against the "
                                  "equilibrium, pooled and per region"},
           "tests": {}, "by_region": {}, "per_seed": {}}

    # Pooling seeds across regions treats the region as a nuisance dimension,
    # which sits badly with this paper's own argument that the three regions
    # differ qualitatively -- CAISO's average factor inverts, PJM's marginal
    # factor swings.  So report each region on its own as well.  The ratios are
    # stored region-major, SEEDS at a time.
    for k, b in enumerate(BAS):
        sl = slice(k * SEEDS, (k + 1) * SEEDS)
        rb = np.asarray(per[REF], float)[sl]
        out["by_region"][b] = {}
        for name, _ in METHODS:
            if name == REF:
                continue
            v = np.asarray(per[name], float)[sl]
            ok = np.isfinite(v) & np.isfinite(rb)
            if ok.sum() < 5:
                continue
            out["by_region"][b][name] = paired(v[ok] - rb[ok])
    # keep the raw ratios so a later question needs no rerun
    out["per_seed"] = {n: [None if not np.isfinite(x) else float(x)
                           for x in per[n]] for n, _ in METHODS}
    print(f"\npaired against '{REF}' ({len(ref)} pairs = "
          f"{SEEDS} seeds x {len(BAS)} regions)\n")
    print(f"{'method':28s} {'mean diff':>11s} {'p':>10s} {'d_z':>8s}  verdict")
    for name, _ in METHODS:
        v = np.asarray(per[name], float)
        ok = np.isfinite(v) & np.isfinite(ref)
        if name == REF or ok.sum() < 5:
            continue
        r = paired(v[ok] - ref[ok])
        out["tests"][name] = r
        verdict = ("worse than equilibrium" if r["mean_diff"] > 0
                   else "better than equilibrium")
        if not r["significant_at_05"]:
            verdict = "NOT DISTINGUISHABLE from the equilibrium"
        print(f"{name:28s} {r['mean_diff']:+11.5f} {r['p_value']:10.2e} "
              f"{r['cohens_dz']:8.2f}  {verdict}")

    print("\nper region (20 pairs each)")
    for b in BAS:
        row = out["by_region"].get(b, {})
        ca = row.get("Carbon-agnostic")
        sh = row.get("SHADE (ours)")
        if ca and sh:
            print(f"  {NICE[b]:6s} carbon-agnostic p={ca['p_value']:.3f} "
                  f"(d_z {ca['cohens_dz']:+.2f})   "
                  f"SHADE p={sh['p_value']:.2e} (d_z {sh['cohens_dz']:+.2f})")

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
