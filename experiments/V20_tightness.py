"""V20 -- how loose is Theorem 1, and which factor is doing the losing?

Theorem 1 gives PoA <= (3/2)(1-eta)^-1, which is 4.56 in CAISO.  The measured
equilibrium ratio is 1.0138.  A bound that overstates the quantity it bounds by
a factor of a hundred invites the question of what it is for, and the paper has
not answered it.

The bound is a product of two worst cases:

    3/2            attained only when the equilibrium over-concentrates so that
                   y_t = 1.5 y*_t on the slots carrying curvature;
    (1-eta)^-1     attained only when ALL load sits in the single slot that
                   maximises eta over the active set.

They pull in opposite directions.  Concentrating everything into the
max-eta slot is what makes the second factor tight, and it is exactly what
leaves no room for the first: with load in one slot there is nothing to
misallocate.  So the product is not attained by any instance, and the interesting
question is how close an adversary can get.

This script answers it three ways.

  A. RANDOM SEARCH over instances satisfying Theorem 1's hypotheses
     (beta_t >= 0, mef_t >= aef_t), reporting realised PoA as a fraction of
     the bound.
  B. ADVERSARIAL CONSTRUCTION: two slots, a wedge on one, curvature tuned to
     make the equilibrium over-concentrate, caps forcing a split.
  C. DECOMPOSITION on the calibrated instances: the realised PoA against the
     accounting-only PoA (the same instance with beta = 0) and against the
     bound, to say how much of the looseness is the max-over-slots in eta and
     how much is the 3/2.

Writes results/V20.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import gh.baselines as B
from gh.core import Feasible, Instance
from E3_main import _ne

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "V20.json")


def poa_of(inst):
    """Realised PoA and the Theorem 1 bound on the realised active set."""
    xn = _ne(inst)
    Cs = B.planner_value(inst)
    xs, _sweeps = B._planner_bcd(inst)     # returns (profile, sweeps)
    yn, ys = xn.sum(axis=0), xs.sum(axis=0)
    active = (yn > 1e-12) | (ys > 1e-12)
    if not active.any():
        return None
    return {"poa": float(inst.social(xn) / Cs),
            "bound": float(inst.poa_bound(active)),
            "eta_active": float(np.max(inst.eta[active])),
            "n_active": int(active.sum())}


def random_instance(rng):
    """An instance satisfying Theorem 1's hypotheses, drawn wide."""
    T = int(rng.integers(2, 6))
    n = int(rng.integers(2, 9))
    a = rng.uniform(10, 400, T)
    eta = rng.uniform(0.0, 0.85, T)
    m = a / (1.0 - eta)
    b = rng.uniform(0, 1, T) * 10 ** rng.uniform(-4, 1)
    # energy and caps: keep the instance feasible, vary how binding the caps are
    E = rng.uniform(1, 100, n)
    slack = rng.uniform(1.05, 6.0)
    X = [Feasible(E=float(E[i]), u=np.full(T, float(E[i]) * slack / T), R=None)
         for i in range(n)]
    return Instance(a=a, m=m, b=b, X=X, psi=np.zeros((n, T)),
                    lam=np.zeros(n), label="random")


def adversarial(delta_eta, beta, n, cap_frac):
    """Two slots: slot 0 cheap on the published factor and carrying the wedge,
    slot 1 honest.  Caps force the equilibrium to spill into slot 1."""
    a = np.array([100.0, 100.0 * (1.0 + delta_eta)])
    m = np.array([a[0] / (1.0 - 0.85), a[1]])       # wedge on the cheap slot
    b = np.array([beta, beta])
    Ei = 10.0
    X = [Feasible(E=Ei, u=np.full(2, Ei * cap_frac), R=None) for _ in range(n)]
    return Instance(a=a, m=m, b=b, X=X, psi=np.zeros((n, 2)),
                    lam=np.zeros(n), label="adversarial")


if __name__ == "__main__":
    out = {"provenance": "SYNTHETIC.  Tightness of Theorem 1: how much of the "
                         "bound an adversary can realise, and how the two "
                         "factors trade off.  No grid data.",
           "random": {}, "adversarial": {}, "calibrated": {}}

    # ---- A. random search ------------------------------------------------
    rng = np.random.default_rng(0)
    rows, best = [], None
    for k in range(400):
        inst = random_instance(rng)
        try:
            r = poa_of(inst)
        except Exception:
            continue
        if r is None or not np.isfinite(r["bound"]):
            continue
        r["frac"] = r["poa"] / r["bound"]
        rows.append(r)
        if best is None or r["frac"] > best["frac"]:
            best = dict(r)
    fr = np.array([r["frac"] for r in rows])
    pa = np.array([r["poa"] for r in rows])
    out["random"] = {"n": len(rows),
                     "frac_of_bound_max": float(fr.max()),
                     "frac_of_bound_median": float(np.median(fr)),
                     "poa_max": float(pa.max()),
                     "violations": int(np.sum(pa > np.array(
                         [r["bound"] for r in rows]) + 1e-9)),
                     "worst_case": best}
    print(f"A. random instances: {len(rows)}")
    print(f"   worst realised PoA {pa.max():.4f}, "
          f"largest fraction of its own bound {fr.max():.3f}, "
          f"median {np.median(fr):.3f}, violations {out['random']['violations']}")

    # ---- B. adversarial construction -------------------------------------
    adv, badv = [], None
    for delta_eta in (0.0, 0.02, 0.05, 0.1, 0.2):
        for beta in (0.01, 0.1, 1.0, 5.0, 20.0, 100.0):
            for n in (2, 4, 8, 32):
                for cap_frac in (0.55, 0.75, 1.0):
                    inst = adversarial(delta_eta, beta, n, cap_frac)
                    try:
                        r = poa_of(inst)
                    except Exception:
                        continue
                    if r is None or not np.isfinite(r["bound"]):
                        continue
                    r.update(delta_eta=delta_eta, beta=beta, n=n,
                             cap_frac=cap_frac,
                             frac=r["poa"] / r["bound"])
                    adv.append(r)
                    if badv is None or r["frac"] > badv["frac"]:
                        badv = dict(r)
    out["adversarial"] = {"n": len(adv), "best": badv,
                          "frac_max": max(r["frac"] for r in adv),
                          "poa_max": max(r["poa"] for r in adv)}
    print(f"B. adversarial: {len(adv)} constructions, "
          f"best PoA {out['adversarial']['poa_max']:.4f} at "
          f"{out['adversarial']['frac_max']:.3f} of its bound")
    print(f"   {badv}")

    # ---- C. decomposition on the calibrated instances --------------------
    # Averaged over E3's OWN twenty seeds, not a single one.  An earlier
    # version used seed 0 and the manuscript then printed 1.0136 in Section 4.1
    # beside Table 1's twenty-seed 1.0138 -- two values for one quantity, which
    # reads as an inconsistency even though neither is wrong.
    from E1_calibrate import BAS, load, hourly_profile
    from E3_main import build, conform, SEEDS
    print(f"C. calibrated instances, mean over {SEEDS} seeds ...", flush=True)
    store = load()
    for ba in BAS:
        a, m, be, dm = hourly_profile(store, ba)
        m2, b2, _, _ = conform(a, m, be)
        poas, poas0, bounds, etas = [], [], [], []
        for s in range(SEEDS):
            inst = build(a, m2, b2, dm, s)
            r = poa_of(inst)
            # The same instance with the congestion channel switched off: what
            # the accounting wedge alone costs at equilibrium.  The solver needs
            # strictly positive curvature, so use 1e-6 of the measured level
            # rather than exactly zero -- the congestion term is then six orders
            # of magnitude below the wedge and the equilibrium is the
            # accounting one.
            flat = Instance(a=inst.a, m=inst.m, b=inst.b * 1e-6, X=inst.X,
                            psi=inst.psi, lam=inst.lam, arrival=inst.arrival)
            r0 = poa_of(flat)
            poas.append(r["poa"])
            poas0.append(r0["poa"])
            bounds.append(r["bound"])
            etas.append(r["eta_active"])
        poa = float(np.mean(poas))
        poa0 = float(np.mean(poas0))
        bound = float(np.mean(bounds))
        eta = float(np.mean(etas))
        out["calibrated"][ba] = {
            "seeds": SEEDS,
            "poa": poa, "poa_sd": float(np.std(poas, ddof=1)),
            "bound": bound, "eta_active": eta,
            "frac_of_bound": poa / bound,
            "poa_accounting_only": poa0,
            "poa_accounting_only_sd": float(np.std(poas0, ddof=1)),
            "accounting_factor_bound": 1.0 / (1.0 - eta),
            "strategic_factor_realised": poa / poa0,
            # sign matters: < 1 means removing congestion makes things WORSE,
            # i.e. the externality is partly corrective on that grid
            "congestion_is_corrective": bool(poa < poa0),
        }
        c = out["calibrated"][ba]
        print(f"   {ba}: PoA {c['poa']:.4f}  bound {c['bound']:.2f}  "
              f"({100*c['frac_of_bound']:.1f}% of it); accounting-only PoA "
              f"{c['poa_accounting_only']:.4f}, accounting factor alone "
              f"{c['accounting_factor_bound']:.2f}, strategic factor realised "
              f"{c['strategic_factor_realised']:.5f}"
              f"{'  [congestion CORRECTIVE]' if c['congestion_is_corrective'] else ''}",
              flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT}")
