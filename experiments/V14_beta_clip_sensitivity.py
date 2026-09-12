"""V14 -- does clipping negative curvature change anything the paper claims?

Theorem 1 assumes an affine dispatch response with beta >= 0.  The measurement
does not always oblige: curvature comes out negative in 14 of CAISO's 24
hour-of-day bins, 3 of ERCOT's and none of PJM's, and `E3_main.conform` clips
those to a small positive floor.  E3 already records how many clipped hours fall
inside the realised active set -- 8 in CAISO, 3 in ERCOT, 0 in PJM -- but the
paper reports the clipping without testing it, which is the gap this script
closes.

Three arms, run on the same seeds and the same grid profiles:

  clip     the status quo: beta <- max(beta, floor) everywhere.
  exclude  identical dynamics, but hours with raw beta < 0 are removed from the
           active set A before eta is maximised over it.  This isolates the
           REPORTING question: is the headline eta being set by an hour whose
           curvature we had to invent?
  drop     the hours with raw beta < 0 are removed from the horizon entirely --
           no operator may schedule in them.  This is the conservative
           robustness arm: if the model cannot be trusted in those hours, do not
           use them at all, and see what the equilibrium gap does.

Two things are worth predicting before reading the output, because they are
consequences of the definitions rather than of the data:

  * eta = 1 - AEF/MEF does not contain beta.  Clipping curvature therefore
    cannot move eta or the PoA bound; it can only move which hours are active,
    and hence which hours the maximum ranges over.
  * clipping a negative beta UP to ~0 removes congestion in that hour.  Since
    the paper's central claim is that the congestion channel is small, the clip
    pushes the measurement toward the claim, which is the wrong direction for
    comfort.  The `drop` arm is the one that can embarrass us.

Writes results/V14.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import gh.baselines as B
from gh.core import Feasible, Instance
from E1_calibrate import BAS, load, hourly_profile
from E3_main import CAP_FRAC, FLEX_SHARE, LAM, N, NICE, T, _ne, conform

SEEDS = 20
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "V14.json")


def build_on(aef, mef, beta, daily_MWh, seed, keep):
    """E3's generator, restricted to the hours in `keep`.

    The workload is built on the full 24-hour clock and then projected: an
    operator's envelope is zeroed outside `keep`, and its deadline staircase is
    re-accumulated over the surviving hours so the feasible set stays non-empty.
    Total energy is preserved, so the arms are comparable.
    """
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0.0, 0.6, N)
    w /= w.sum()
    E = w * FLEX_SHARE * daily_MWh
    nkeep = int(keep.sum())
    X, psi = [], np.zeros((N, T))
    for i in range(N):
        # spread the same energy over the surviving hours
        u = np.where(keep, CAP_FRAC * E[i] * (T / max(nkeep, 1)), 0.0)
        if u.sum() < E[i]:                      # keep the box feasible
            u = np.where(keep, E[i] / max(nkeep, 1) * 1.05, 0.0)
        slack = rng.uniform(2.0, 24.0)
        done = np.clip(np.arange(T) / slack, 0, 1)
        R = np.maximum.accumulate(np.minimum(0.85 * E[i] * done ** 1.7,
                                             np.cumsum(u) * 0.9))
        X.append(Feasible(E=E[i], u=u, R=R))
        h0 = rng.uniform(6, 18)
        psi[i] = np.abs(np.arange(T) - h0) / 24.0
    lam = np.full(N, LAM)
    return Instance(a=aef, m=mef, b=beta, X=X, psi=psi, lam=lam)


def arm(store, b, mode):
    a, m_raw, be_raw, dm = hourly_profile(store, b)
    m, be, _, _ = conform(a, m_raw, be_raw)
    neg = np.asarray(be_raw) < 0
    keep = np.ones(T, bool) if mode != "drop" else ~neg
    eta_h = 1.0 - a / m

    etas, ratios, clipped_active = [], [], []
    for s in range(SEEDS):
        inst = build_on(a, m, be, dm, s, keep)
        xne = _ne(inst)
        xpl = B._planner_fast(inst)
        A = (xne.sum(0) > 1e-9) | (xpl.sum(0) > 1e-9)
        if mode == "exclude":
            A = A & ~neg
        if not A.any():
            continue
        etas.append(float(np.max(eta_h[A])))
        clipped_active.append(int(np.sum(neg & A)))
        cpl = B.planner_value(inst)
        ratios.append(float(inst.social(xne) / cpl))
    return {
        "eta_active": float(np.mean(etas)),
        "eta_active_sd": float(np.std(etas)),
        "poa_bound": float(1.5 / (1.0 - float(np.mean(etas)))),
        "nash_over_planner": float(np.mean(ratios)),
        "nash_over_planner_sd": float(np.std(ratios)),
        "clipped_hours_in_A": float(np.mean(clipped_active)),
        "hours_kept": int(keep.sum()),
        "n_beta_negative": int(neg.sum()),
    }


if __name__ == "__main__":
    print("loading EIA-930 ...")
    store = load()
    out = {"provenance": {"seeds": SEEDS, "n": N, "T": T,
                          "flex_share": FLEX_SHARE,
                          "what": "sensitivity of eta and the equilibrium gap "
                                  "to clipping negative measured curvature"},
           "regions": {}}
    for b in BAS:
        out["regions"][b] = {}
        print(f"\n{NICE[b]}")
        for mode in ("clip", "exclude", "drop"):
            r = arm(store, b, mode)
            out["regions"][b][mode] = r
            print(f"  {mode:8s} eta {r['eta_active']:.4f}  "
                  f"bound {r['poa_bound']:.2f}  "
                  f"Nash/planner {r['nash_over_planner']:.4f} "
                  f"+-{r['nash_over_planner_sd']:.4f}  "
                  f"(beta<0 in A: {r['clipped_hours_in_A']:.1f}, "
                  f"hours kept {r['hours_kept']})")

    # the headline sentence the paper needs
    print("\nsummary for the manuscript")
    for b in BAS:
        c, e, d = (out["regions"][b][k] for k in ("clip", "exclude", "drop"))
        print(f"  {NICE[b]:6s} eta {c['eta_active']:.3f} -> "
              f"{e['eta_active']:.3f} (exclude) -> {d['eta_active']:.3f} (drop);"
              f"  gap {100*(c['nash_over_planner']-1):.2f}% -> "
              f"{100*(d['nash_over_planner']-1):.2f}% (drop)")
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
