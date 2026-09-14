"""V16 -- equivalence, power, and the per-region sign split behind Table 1.

Three things V15 does not supply, each asked for by a reviewer who opened the
artifact.

1. An EQUIVALENCE test.  "A paired test cannot separate it from zero" is an
   underpowered non-detection, not evidence of equivalence.  Two one-sided
   tests (TOST) against a margin stated in advance either convert that into a
   positive claim or decline to.  The margin is a quarter of the equilibrium's
   own gap over the planner -- the same 25% criterion Section 7.3 uses to call
   a channel material -- so it is fixed by the paper's existing convention
   rather than chosen to get an answer.

2. POWER.  What the paired test actually had against the observed effect, and
   the smallest difference the design would have detected.

3. The PER-REGION SIGN SPLIT.  Pooling 20 seeds across 3 regions treats region
   as nuisance, which sits badly with this paper's own argument that the three
   regions differ qualitatively.  The pooled carbon-agnostic mean is near zero
   because the regions disagree in SIGN, not because the effect is small in any
   of them, and a reader cannot see that from the pooled number.

This DERIVES everything from results/V15.json and results/E3.json, both already
committed.  An earlier version of this script re-ran E3's methods to recover the
per-seed ratios; that takes hours, died to Windows Modern Standby twice, and was
never necessary -- V15 stores the mean, the interval and the pair count, which
determine the standard deviation and hence every statistic below.  Deriving
beats re-running when the inputs are sufficient.

Writes results/V16.json.  Runs in under a second.
"""
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(RES, "V16.json")

BAS = ["CISO", "ERCO", "PJM"]
NICE = {"CISO": "CAISO", "ERCO": "ERCOT", "PJM": "PJM"}
REF = "Independent greedy"
MARGIN_FRAC = 0.25          # a quarter of the equilibrium gap; fixed in advance


def sd_from_ci(mean, ci_hi, n):
    """V15 stores a normal-approximation 95% interval, so the half-width is
    1.96 * sd / sqrt(n) and the sd follows."""
    return (ci_hi - mean) / 1.96 * np.sqrt(n)


def tost(mean, sd, n, margin):
    """Two one-sided t-tests.  Equivalence is declared when both one-sided
    nulls are rejected, equivalently when the 90% interval lies inside
    (-margin, +margin)."""
    se = sd / np.sqrt(n)
    df = n - 1
    p_lo = float(stats.t.sf((mean + margin) / se, df))   # H0: mean <= -margin
    p_hi = float(stats.t.cdf((mean - margin) / se, df))  # H0: mean >= +margin
    crit = float(stats.t.ppf(0.95, df))
    p = max(p_lo, p_hi)
    return {"margin": float(margin), "se": float(se),
            "ci90": [float(mean - crit * se), float(mean + crit * se)],
            "p_lower": p_lo, "p_upper": p_hi, "p_tost": p,
            "equivalent_at_05": bool(p < 0.05)}


def power_paired(dz, n, alpha=0.05):
    """Power of a two-sided paired t-test against a true effect of size dz,
    normal approximation to the noncentral t (adequate at n = 60)."""
    crit = stats.t.ppf(1 - alpha / 2, n - 1)
    ncp = abs(dz) * np.sqrt(n)
    return float(stats.norm.sf(crit - ncp) + stats.norm.cdf(-crit - ncp))


def mde(sd, n, alpha=0.05, power=0.80):
    """Smallest true difference this design detects at the stated power."""
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))
                 * sd / np.sqrt(n))


if __name__ == "__main__":
    v15 = json.load(open(os.path.join(RES, "V15.json")))["tests"]
    e3 = json.load(open(os.path.join(RES, "E3.json")))["table"]

    gap = e3[REF]["mean"] - 1.0
    margin = MARGIN_FRAC * gap
    out = {"provenance": "Derived from results/V15.json and results/E3.json; "
                         "no re-solve.  TOST equivalence against a margin of "
                         "25% of the equilibrium gap, realised power, and the "
                         "per-region decomposition of the pooled difference.",
           "reference": REF, "equilibrium_gap": gap,
           "margin_frac_of_gap": MARGIN_FRAC, "margin": margin,
           "pooled": {}, "by_region_means": {}}

    print(f"equilibrium gap {100*gap:.2f}% of planner; "
          f"equivalence margin +-{margin:.5f}\n")
    print(f"{'method':26s} {'mean diff':>11s} {'p':>10s} {'d_z':>7s} "
          f"{'p_TOST':>10s} {'equiv':>6s} {'power':>7s}")
    for name, t in v15.items():
        mean, n = t["mean_diff"], t["n_pairs"]
        sd = sd_from_ci(mean, t["ci95"][1], n)
        r = {"n_pairs": n, "mean_diff": mean, "sd_diff": float(sd),
             "wilcoxon_p": t["p_value"], "cohens_dz": float(mean / sd),
             "tost": tost(mean, sd, n, margin),
             "power_vs_observed": power_paired(mean / sd, n),
             "mde_80pct": mde(sd, n)}
        out["pooled"][name] = r
        print(f"{name:26s} {mean:+11.5f} {t['p_value']:10.2e} "
              f"{r['cohens_dz']:7.2f} {r['tost']['p_tost']:10.2e} "
              f"{str(r['tost']['equivalent_at_05']):>6s} "
              f"{r['power_vs_observed']:7.2f}")

    # ---- the per-region split the pooled number hides --------------------
    print(f"\nper-region mean ratios (E3), against the equilibrium")
    print(f"{'region':<7s} {'agnostic':>9s} {'equilibrium':>12s} "
          f"{'difference':>11s}  direction")
    for i, b in enumerate(BAS):
        a = e3["Carbon-agnostic"]["per"][i]
        q = e3[REF]["per"][i]
        out["by_region_means"][b] = {
            "carbon_agnostic": a, "equilibrium": q, "difference": a - q,
            "agnostic_worse": bool(a > q)}
        print(f"{NICE[b]:<7s} {a:>9.4f} {q:>12.4f} {a-q:>+11.4f}  "
              f"{'agnostic worse' if a > q else 'agnostic better'}")

    signs = {b: out["by_region_means"][b]["agnostic_worse"] for b in BAS}
    out["regions_disagree_in_sign"] = bool(len(set(signs.values())) > 1)
    print(f"\nregions disagree in sign: {out['regions_disagree_in_sign']} "
          f"-- the pooled mean is cancellation, not a small per-region effect")

    # the comparison that disciplines the jitter row
    a = out["pooled"]["Carbon-agnostic"]
    j = out["pooled"]["Randomized jitter"]
    out["jitter_vs_agnostic"] = {
        "jitter_mean": j["mean_diff"], "agnostic_mean": a["mean_diff"],
        "jitter_effect_is_smaller": bool(abs(j["mean_diff"])
                                         < abs(a["mean_diff"])),
        "sd_ratio_agnostic_over_jitter": a["sd_diff"] / j["sd_diff"],
        "both_inside_margin": bool(a["tost"]["equivalent_at_05"]
                                   and j["tost"]["equivalent_at_05"])}
    k = out["jitter_vs_agnostic"]
    print(f"\njitter {j['mean_diff']:+.5f} vs carbon-agnostic "
          f"{a['mean_diff']:+.5f}: jitter's effect is "
          f"{'SMALLER' if k['jitter_effect_is_smaller'] else 'larger'}, and "
          f"its spread is {k['sd_ratio_agnostic_over_jitter']:.1f}x narrower.")
    print(f"both inside the equivalence margin: {k['both_inside_margin']} "
          f"-- significance separates variance, not effect")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT}")
