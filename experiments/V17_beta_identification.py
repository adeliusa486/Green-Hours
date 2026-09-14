"""V17 -- how well is BETA identified?  The half of the audit E8 never reported.

Section 6 audits the identification of eta, the ratio Theorem 1 uses, and finds
it better identified than the marginal factor's level: eta moves 13-24% across
specifications while the level spans 43%.  That is true and it is the wrong
reassurance, because eta is not what the paper's negative result rests on.

The negative result -- that the congestion channel is small, that a
level-correct static factor suffices, that SHADE is redundant at today's share
-- is a statement about BETA, the rate at which the marginal factor rises with
load, and about kappa, the congestion term's size relative to the accounting
wedge.  Neither is reported anywhere in the manuscript as a function of
specification.  This script reads E8's cells and reports them.

It re-estimates nothing.  Every number is already in results/E8.json; what is
new is reading the beta and kappa columns instead of only the eta column.

Two groupings, because they answer different questions:

  SPEC   -- the 12 (boundary x regressor x estimator) cells at the full window.
            This is the apples-to-apples set: same sample, different modelling
            choice.  It is the uncertainty that should propagate into the
            threshold.
  WINDOW -- the 8 sampling windows at the preferred specification.  This is
            sample variation, not specification variation, and it is reported
            separately because a season or a renewable tercile is a different
            population, not a different estimate of the same one.

Writes results/V17.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E8 = os.path.join(HERE, "..", "results", "E8.json")
OUT = os.path.join(HERE, "..", "results", "V17.json")

BAS = ["CISO", "ERCO", "PJM"]
NICE = {"CISO": "CAISO", "ERCO": "ERCOT", "PJM": "PJM"}
PREFERRED = "consumption|netload|ols|full"


def spread(vals):
    """Summarise a set of estimates of one quantity across specifications."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return None
    med = float(np.median(v))
    lo, hi = float(v.min()), float(v.max())
    return {
        "n": int(v.size), "min": lo, "max": hi, "median": med,
        # relative range about the median: the scale-free measure Section 6
        # already uses for eta and the MEF level, so the three are comparable
        "rel_range": float((hi - lo) / abs(med)) if med != 0 else None,
        # ratio of largest to smallest, defined only when the sign is constant
        "ratio": float(hi / lo) if lo > 0 else None,
        "sign_changes": bool(lo < 0 < hi) or bool(lo < 0),
    }


if __name__ == "__main__":
    with open(E8) as f:
        e8 = json.load(f)
    cells = e8["cells"]

    out = {"provenance": "Derived from results/E8.json; no re-estimation.  "
                         "Specification sensitivity of beta and kappa, the "
                         "quantities the negative result depends on, reported "
                         "beside eta, the quantity Section 6 already audits.",
           "preferred": PREFERRED, "regions": BAS, "by_region": {}}

    print(f"{'region':<7s} {'group':<8s} {'quantity':<7s} {'n':>3s} "
          f"{'min':>10s} {'max':>10s} {'median':>10s} {'rel range':>10s} "
          f"{'ratio':>7s}")
    print("-" * 80)
    for ba in BAS:
        full = [k for k in cells if k.startswith(ba + "|") and k.endswith("|full")]
        wins = [k for k in cells if k.startswith(ba + "|") and not k.endswith("|full")]
        row = {"preferred": {q: cells[f"{ba}|{PREFERRED}"][f"{q}_target"
                                                           if q == "beta" else q]
                             for q in ("beta", "kappa")}}
        row["preferred"]["eta"] = cells[f"{ba}|{PREFERRED}"]["eta_max"]
        for grp, keys in (("spec", full), ("window", wins)):
            row[grp] = {
                "cells": sorted(k.split("|", 1)[1] for k in keys),
                "eta": spread([cells[k]["eta_max"] for k in keys]),
                "beta": spread([cells[k]["beta_target"] for k in keys]),
                "kappa": spread([cells[k]["kappa"] for k in keys]),
            }
            for q in ("eta", "beta", "kappa"):
                s = row[grp][q]
                if s is None:
                    continue
                ratio = "--" if s["ratio"] is None else f"{s['ratio']:.1f}"
                print(f"{NICE[ba]:<7s} {grp:<8s} {q:<7s} {s['n']:>3d} "
                      f"{s['min']:>10.5f} {s['max']:>10.5f} {s['median']:>10.5f} "
                      f"{s['rel_range']:>9.1%} {ratio:>7s}")
        out["by_region"][ba] = row
        print("-" * 80)

    # The headline comparison: beta and kappa against eta, on the SPEC group.
    head = {}
    for ba in BAS:
        s = out["by_region"][ba]["spec"]
        head[ba] = {
            "eta_rel_range": s["eta"]["rel_range"],
            "beta_rel_range": s["beta"]["rel_range"],
            "kappa_rel_range": s["kappa"]["rel_range"],
            "beta_ratio": s["beta"]["ratio"],
            "kappa_ratio": s["kappa"]["ratio"],
        }
    out["headline"] = head
    print("\nspecification sensitivity, full window, relative range about the median")
    print(f"{'region':<7s} {'eta':>8s} {'beta':>8s} {'kappa':>8s}   "
          f"{'beta x':>7s} {'kappa x':>8s}")
    for ba in BAS:
        h = head[ba]
        print(f"{NICE[ba]:<7s} {h['eta_rel_range']:>7.1%} "
              f"{h['beta_rel_range']:>7.1%} {h['kappa_rel_range']:>7.1%}   "
              f"{h['beta_ratio']:>7.1f} {h['kappa_ratio']:>8.1f}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT}")
