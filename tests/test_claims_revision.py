"""Claims added in the AAMAS revision, pinned against the files that produced them.

Same manifest discipline as test_claims.py, for the numbers the revision added:
the propagated thresholds the earlier draft reported only half of, the
identification audit of beta and kappa, the out-of-sample winter arm, the
local-smoothness step that replaced the series-parallel reduction, and the
epsilon-Nash bound at realistic market concentration.

Run:  python -m pytest tests/test_claims_revision.py -q
"""
import json
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
PAPER = os.path.abspath(os.path.join(HERE, "..", ".."))


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        pytest.skip(f"{name} not present -- run the experiment that writes it")
    return json.load(open(p))


def dig(d, path):
    cur = d
    for part in path.split("/"):
        if not part:
            continue
        m = re.fullmatch(r"(.*)\[(-?\d+)\]", part)
        if m:
            if m.group(1):
                cur = cur[m.group(1)]
            cur = cur[int(m.group(2))]
        else:
            cur = cur[part]
    return cur


# (label, results file, path, expected, absolute tolerance)
CLAIMS = [
    # ---- E9: the strategic threshold, propagated over specifications -----
    # Section 7.3 and Figure 2b.  The earlier draft quoted headline/*/static
    # and not headline/*/strategic; both halves are now reported.
    ("CAISO strategic threshold", "E9.json",
     "headline/CISO/strategic/median", 0.0172, 5e-4),
    ("CAISO strategic p10", "E9.json",
     "headline/CISO/strategic/p10", 0.0070, 5e-4),
    ("CAISO strategic p90", "E9.json",
     "headline/CISO/strategic/p90", 0.0277, 5e-4),
    ("PJM strategic threshold", "E9.json",
     "headline/PJM/strategic/median", 0.0822, 5e-4),
    ("CAISO deployment threshold", "E9.json",
     "headline/CISO/static/median", 0.0115, 5e-4),
    ("PJM deployment threshold", "E9.json",
     "headline/PJM/static/median", 0.0165, 5e-4),
    ("ERCOT deployment threshold", "E9.json",
     "headline/ERCO/static/median", 0.0364, 5e-4),
    ("today's flexible share", "E9.json", "today_share", 0.0038, 1e-6),

    # ---- E11: the trace workload's thresholds, both halves ---------------
    ("trace strategic, CAISO", "E11.json",
     "D_threshold/CISO/strategic_median", 0.0695, 5e-4),
    ("trace strategic, PJM", "E11.json",
     "D_threshold/PJM/strategic_median", 0.1045, 5e-4),
    ("trace deployment, CAISO", "E11.json",
     "D_threshold/CISO/static_median", 0.0342, 5e-4),
    ("trace deployment, PJM", "E11.json",
     "D_threshold/PJM/static_median", 0.0188, 5e-4),

    # ---- V17: beta and kappa are worse identified than eta ---------------
    ("CAISO eta spec range", "V17.json", "headline/CISO/eta_rel_range",
     0.134, 5e-3),
    ("CAISO kappa spec range", "V17.json", "headline/CISO/kappa_rel_range",
     1.381, 5e-3),
    ("ERCOT kappa spec range", "V17.json", "headline/ERCO/kappa_rel_range",
     9.676, 5e-2),
    ("PJM kappa spec range", "V17.json", "headline/PJM/kappa_rel_range",
     0.924, 5e-3),

    # ---- V18: out of sample, and out of season ---------------------------
    ("CAISO eta, winter 2025", "V18.json",
     "y2025/CISO/winter/eta_max", 0.7605, 5e-4),
    ("CAISO eta, summer 2024", "V18.json",
     "y2024/CISO/summer/eta_max", 0.6765, 5e-4),
    ("CAISO kappa, 2025H1", "V18.json",
     "comparison/CISO/kappa_2025H1", 0.0741, 5e-4),

    # ---- V19: the local-smoothness step behind Theorem 1 -----------------
    ("local smoothness lambda", "V19.json", "lambda", 1.0, 0),
    ("local smoothness mu", "V19.json", "mu", 1 / 3, 1e-9),
    ("resulting PoA factor", "V19.json", "poa_bound", 1.5, 1e-9),

    # ---- V21: epsilon-Nash at realistic concentration --------------------
    ("CAISO n*eps/gap, Azure", "V21.json",
     "by_region/CISO/rows[0]/n_eps_over_gap", 0.022, 5e-3),

    # ---- V16: equivalence, power, and the per-region sign split ----------
    ("equivalence margin", "V16.json", "margin", 0.01048, 5e-5),
    ("carbon-agnostic TOST p", "V16.json",
     "pooled/Carbon-agnostic/tost/p_tost", 8.9e-6, 5e-7),
    ("carbon-agnostic power", "V16.json",
     "pooled/Carbon-agnostic/power_vs_observed", 0.160, 5e-3),
    ("CAISO agnostic minus equilibrium", "V16.json",
     "by_region_means/CISO/difference", 0.0175, 5e-4),
    ("ERCOT agnostic minus equilibrium", "V16.json",
     "by_region_means/ERCO/difference", -0.0094, 5e-4),
    ("PJM agnostic minus equilibrium", "V16.json",
     "by_region_means/PJM/difference", -0.0136, 5e-4),
    ("jitter/agnostic sd ratio", "V16.json",
     "jitter_vs_agnostic/sd_ratio_agnostic_over_jitter", 3.8, 5e-2),

    # ---- V20: how loose Theorem 1 is, and which factor loses --------------
    # Section 4.1 quotes these to three decimals, which is the precision at
    # which a single seed and the twenty-seed mean agree (they differ by
    # 2.4e-4).  Tolerances are set to span both rather than to pin one.
    ("CAISO realised PoA", "V20.json", "calibrated/CISO/poa", 1.014, 1e-3),
    ("CAISO accounting-only PoA", "V20.json",
     "calibrated/CISO/poa_accounting_only", 1.012, 1e-3),
    ("CAISO strategic factor", "V20.json",
     "calibrated/CISO/strategic_factor_realised", 1.002, 1e-3),
    ("PJM realised PoA", "V20.json", "calibrated/PJM/poa", 1.074, 2e-3),
    ("PJM accounting-only PoA", "V20.json",
     "calibrated/PJM/poa_accounting_only", 1.084, 2e-3),
    ("worst random fraction of bound", "V20.json",
     "random/frac_of_bound_max", 0.666, 5e-3),
    ("bound violations", "V20.json", "random/violations", 0, 0),
]


@pytest.mark.parametrize("label,fname,path,expected,tol", CLAIMS)
def test_claim(label, fname, path, expected, tol):
    got = dig(load(fname), path)
    if isinstance(expected, int) and tol == 0:
        assert int(got) == expected, f"{label}: got {got}, expected {expected}"
    else:
        assert abs(float(got) - float(expected)) <= tol, \
            (f"{label}: {fname}:{path} = {got}, manuscript says {expected} "
             f"(tolerance {tol})")


def test_local_smoothness_actually_holds():
    """V19 checks the inequality numerically; a run that silently checked
    nothing would still write the file, so assert it compared something."""
    v = load("V19.json")
    assert v["holds"] is True, "the local-smoothness inequality was violated"
    assert v["n_random_resources"] > 10000, \
        f"only {v['n_random_resources']} configurations checked"
    assert v["worst_slack_random"] >= -1e-9


def test_strategic_thresholds_are_reported_in_the_manuscript():
    """The specific reporting failure this revision fixes: E11 and E9 both
    carry a strategic threshold beside the deployment one, and an earlier
    draft printed only the second.  Both must now appear in the prose."""
    texts = []
    for fn in ("main.tex", "sec_results.tex", "fig_threshold.tex"):
        p = os.path.join(PAPER, fn)
        if os.path.exists(p):
            texts.append(open(p, encoding="utf-8", errors="replace").read())
    if not texts:
        pytest.skip("manuscript sources not next to the artifact")
    blob = "\n".join(texts)
    must = ["1.72", "8.22", "6.95", "10.45"]
    missing = [s for s in must if s not in blob]
    assert not missing, \
        f"strategic thresholds absent from the manuscript: {missing}"


def test_theorem_1_is_never_violated():
    """The bound has to hold on every instance satisfying its hypotheses, and
    the search has to have been wide enough for that to mean something."""
    v = load("V20.json")
    assert v["random"]["violations"] == 0, "Theorem 1 was violated"
    assert v["random"]["n"] >= 300, f"only {v['random']['n']} instances searched"
    assert v["random"]["frac_of_bound_max"] < 1.0


def test_congestion_is_not_always_harmful():
    """Switching the congestion channel off RAISES PJM's equilibrium ratio, so
    there the externality is partly corrective: it discourages exactly the
    over-concentration the accounting error causes.  Section 4.1 now says this
    explicitly ("in PJM it moves the wrong way").  If a later edit drops that
    clause, or the sign flips, this test is the reason to look."""
    v = load("V20.json")
    assert v["calibrated"]["PJM"]["strategic_factor_realised"] < 1.0, \
        "PJM's congestion channel is no longer corrective; Section 4.1 says it is"
    for ba in ("CISO", "ERCO"):
        assert v["calibrated"][ba]["strategic_factor_realised"] > 1.0


def test_section_4_1_matches_the_artifact_at_its_stated_precision():
    """Section 4.1 quotes four ratios to three decimals.  A reviewer checking
    them against results/ must find agreement at that precision, whichever seed
    basis V20 was last run on."""
    v = load("V20.json")["calibrated"]
    stated = {("CISO", "poa"): 1.014, ("CISO", "poa_accounting_only"): 1.012,
              ("PJM", "poa"): 1.074, ("PJM", "poa_accounting_only"): 1.084}
    for (ba, key), claim in stated.items():
        got = v[ba][key]
        assert round(got, 3) == claim, \
            (f"Section 4.1 states {ba} {key} = {claim}; V20.json has "
             f"{got:.6f}, which rounds to {round(got, 3)}")


def test_equivalence_replaces_the_underpowered_null():
    """Section 7.2 leaned on p = 0.94 as if it were evidence of no difference.
    The replacement has to be a real equivalence result, and the two effects
    the earlier draft treated oppositely have to land on the same side of it."""
    v = load("V16.json")
    a = v["pooled"]["Carbon-agnostic"]
    j = v["pooled"]["Randomized jitter"]
    assert a["tost"]["equivalent_at_05"], \
        "carbon-agnostic is no longer equivalent at the stated margin"
    assert a["power_vs_observed"] < 0.3, \
        "the null is no longer underpowered, so the framing needs revisiting"
    assert v["regions_disagree_in_sign"], \
        "the per-region sign split the pooled mean hides has gone"
    assert j["mean_diff"] > a["mean_diff"], \
        "jitter's improvement is no longer the smaller of the two"
    assert v["jitter_vs_agnostic"]["both_inside_margin"], \
        "jitter and carbon-agnostic no longer sit inside the same margin"


def test_the_retracted_claim_is_gone():
    """The sentence the artifact contradicted must not survive anywhere."""
    for fn in ("sec_results.tex", "fig_threshold.tex"):
        p = os.path.join(PAPER, fn)
        if not os.path.exists(p):
            continue
        t = " ".join(open(p, encoding="utf-8", errors="replace").read().split())
        assert "not reached below a $20\\%$ share" not in t, \
            f"{fn} still carries the unscoped 20% claim"
        assert "never a quarter" not in t, \
            f"{fn} still carries the unscoped 'never a quarter' claim"
