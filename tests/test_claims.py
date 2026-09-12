"""Every headline number in the paper, checked against the file that produced it.

This test exists because of a specific failure.  The planner denominator was
computed over a relaxation of the feasible set for several drafts; the numbers
in the manuscript, the tables, the README and the notes all agreed with each
other, and all of them were wrong together, because they had been copied from
one another rather than each read from the result files.  Agreement between
documents is not evidence.  Agreement between a document and the JSON a script
wrote is.

So this file pins a manifest: for each claim, the file, the path inside it, and
the value the manuscript states.  A number that moves for a good reason will
fail here, and the fix is to update the manifest in the same commit as the text
-- which is the point.  A number that moves for a bad reason fails here too,
which is more the point.

Run:  python -m pytest tests/test_claims.py -q
      python tests/test_claims.py          (prints a table instead)
"""
import json
import os
import re
import sys

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
    """Walk a slash-separated path, with [i] for list indices."""
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
    # ---- E1: the calibration -------------------------------------------
    ("CAISO eta_max", "E1.json", "regions/CISO/eta_max", 0.671, 5e-4),
    ("CAISO PoA bound", "E1.json", "regions/CISO/bound_max", 4.56, 5e-3),
    ("CAISO AEF", "E1.json", "regions/CISO/aef_target", 130.0, 0.1),
    ("CAISO MEF", "E1.json", "regions/CISO/mef_target", 367.9, 0.1),
    ("CAISO kappa", "E1.json", "regions/CISO/kappa", 0.032, 1e-3),
    ("ERCOT eta_max", "E1.json", "regions/ERCO/eta_max", 0.492, 5e-4),
    ("ERCOT PoA bound", "E1.json", "regions/ERCO/bound_max", 2.95, 5e-3),
    ("ERCOT kappa", "E1.json", "regions/ERCO/kappa", 0.007, 1e-3),
    ("PJM eta_max", "E1.json", "regions/PJM/eta_max", 0.429, 5e-4),
    ("PJM PoA bound", "E1.json", "regions/PJM/bound_max", 2.63, 5e-3),
    ("PJM kappa", "E1.json", "regions/PJM/kappa", 0.052, 1e-3),
    ("CAISO daily load GWh", "E1.json", "regions/CISO/load_GWh_d", 667.0, 1.0),
    ("ERCOT daily load GWh", "E1.json", "regions/ERCO/load_GWh_d", 1324.0, 1.0),
    ("PJM daily load GWh", "E1.json", "regions/PJM/load_GWh_d", 2265.0, 1.0),
    ("MEF>=AEF never imposed (CAISO)", "E1.json",
     "regions/CISO/n_eta_negative", 0, 0),

    # ---- E3: Table 1 ----------------------------------------------------
    ("Table 1 carbon-agnostic mean", "E3.json",
     "table/Carbon-agnostic/mean", 1.0401, 5e-4),
    ("Table 1 naive mean", "E3.json",
     "table/Naive forecast-taking/mean", 1.0496, 5e-4),
    ("Table 1 Nash mean", "E3.json",
     "table/Independent greedy/mean", 1.0419, 5e-4),
    ("Table 1 threshold mean", "E3.json",
     "table/Threshold deferral/mean", 1.0609, 5e-4),
    ("Table 1 jitter mean", "E3.json",
     "table/Randomized jitter/mean", 1.0402, 5e-4),
    ("Table 1 static MEF mean", "E3.json",
     "table/Static MEF signalling/mean", 1.0000, 5e-4),
    ("Table 1 SHADE mean", "E3.json", "table/SHADE (ours)/mean", 1.0000, 5e-4),
    ("Table 1 SHADE gap removed", "E3.json",
     "table/SHADE (ours)/gap", 1.000, 5e-4),
    ("Table 1 static MEF gap removed", "E3.json",
     "table/Static MEF signalling/gap", 0.999, 1e-3),
    ("Table 1 Nash CAISO", "E3.json",
     "table/Independent greedy/per[0]", 1.0138, 5e-4),
    ("Table 1 Nash ERCOT", "E3.json",
     "table/Independent greedy/per[1]", 1.0423, 5e-4),
    ("Table 1 Nash PJM", "E3.json",
     "table/Independent greedy/per[2]", 1.0696, 5e-4),
    ("proportional cap infeasible in 60 runs", "E3.json",
     "table/Proportional cap/infeasible", 60, 0),

    # ---- E4/E7: operations and ablation ---------------------------------
    ("SHADE peak/mean", "E4_E7.json", "E4/SHADE (ours)/p2m",
     5.61, 5e-3),
    ("equilibrium peak/mean", "E4_E7.json",
     "E4/Independent greedy/p2m", 5.21, 5e-3),
    ("SHADE congestion term", "E4_E7.json",
     "E4/SHADE (ours)/uplift", 1.307, 5e-3),
    ("equilibrium congestion term", "E4_E7.json",
     "E4/Independent greedy/uplift", 1.773, 5e-3),
    ("ablation: no adder recovers the equilibrium", "E4_E7.json",
     "E7/No externality adder", 1.0419, 5e-4),
    ("ablation: adder coarsened to 3 h", "E4_E7.json",
     "E7/Adder coarsened to 3\\,h", 1.0212, 5e-4),

    # ---- E8: the identification audit -----------------------------------
    ("E8 CAISO eta primary", "E8.json", "verdict/CISO/eta_primary", 0.671, 5e-4),
    ("E8 CAISO MEF>AEF holds", "E8.json", "verdict/CISO/sign_ok", True, 0),
    ("E8 CAISO permutation placebo collapses", "E8.json",
     "verdict/CISO/placebo_ratio", 0.051, 0.02),
    ("E8 ERCOT permutation placebo collapses", "E8.json",
     "verdict/ERCO/placebo_ratio", 0.032, 0.02),
    ("E8 PJM permutation placebo collapses", "E8.json",
     "verdict/PJM/placebo_ratio", 0.030, 0.02),
    ("E8 no estimate is claimed causal", "E8.json", "verdict/CISO/causal",
     False, 0),

    # ---- E10: convergence ------------------------------------------------
    ("E10 no divergence in the adversarial suite", "E10.json",
     "summary/n_divergences", 0, 0),
    ("E10 no epsilon-Nash violation", "E10.json",
     "summary/n_enash_violations", 0, 0),
    ("E10 global convergence is NOT claimed", "E10.json",
     "global_convergence_proved", False, 0),
    # ---- V14: the beta-clipping sensitivity Section 6 now reports ---------
    # eta contains no beta, so all three arms must agree to four decimals.
    # If a future change makes these differ, the sentence in Section 6 is
    # wrong and should fail here before a reviewer finds it.
    ("V14 CAISO eta, clipped", "V14.json",
     "regions/CISO/clip/eta_active", 0.6711, 5e-4),
    ("V14 CAISO eta, clipped hours struck from the horizon", "V14.json",
     "regions/CISO/drop/eta_active", 0.6711, 5e-4),
    ("V14 CAISO clipped hours inside the active set", "V14.json",
     "regions/CISO/clip/clipped_hours_in_A", 8.0, 1e-9),
    ("V14 ERCOT clipped hours inside the active set", "V14.json",
     "regions/ERCO/clip/clipped_hours_in_A", 3.0, 1e-9),
    ("V14 PJM clipped hours inside the active set", "V14.json",
     "regions/PJM/clip/clipped_hours_in_A", 0.0, 1e-9),
    # clipping must OVERSTATE the gap, never understate it: dropping those
    # hours has to leave the equilibrium no further from the planner.
    ("V14 CAISO gap, clipped", "V14.json",
     "regions/CISO/clip/nash_over_planner", 1.0138, 5e-4),
    ("V14 CAISO gap, hours struck", "V14.json",
     "regions/CISO/drop/nash_over_planner", 1.0063, 5e-4),
    ("V14 ERCOT gap, hours struck", "V14.json",
     "regions/ERCO/drop/nash_over_planner", 1.0395, 5e-4),
    # ---- V15: the paired tests Table 1's caption now quotes --------------
    # The one comparison the paper declines to call a difference.  If a later
    # change makes this significant, the caption and Section 7.1 both become
    # wrong, and they should fail here first.
    ("V15 carbon-agnostic is NOT separable from the equilibrium",
     "V15.json", "tests/Carbon-agnostic/significant_at_05", False, 0),
    ("V15 carbon-agnostic p-value", "V15.json",
     "tests/Carbon-agnostic/p_value", 0.94, 0.02),
    ("V15 pairs compared", "V15.json",
     "tests/Carbon-agnostic/n_pairs", 60, 0),
    ("V15 naive forecast-taking IS worse", "V15.json",
     "tests/Naive forecast-taking/significant_at_05", True, 0),
    ("V15 SHADE IS better", "V15.json",
     "tests/SHADE (ours)/significant_at_05", True, 0),
    ("V15 jitter IS better, as Table 1 says", "V15.json",
     "tests/Randomized jitter/significant_at_05", True, 0),
]


@pytest.mark.parametrize("label,fname,path,expected,tol",
                         CLAIMS, ids=[c[0] for c in CLAIMS])
def test_claim(label, fname, path, expected, tol):
    d = load(fname)
    try:
        got = dig(d, path)
    except (KeyError, IndexError, TypeError) as e:
        pytest.fail(f"{label}: {fname}:{path} not found ({e})")
    if isinstance(expected, bool):
        assert bool(got) is expected, f"{label}: got {got}, expected {expected}"
    elif isinstance(expected, int) and tol == 0:
        assert int(got) == expected, f"{label}: got {got}, expected {expected}"
    else:
        assert abs(float(got) - float(expected)) <= tol, \
            (f"{label}: {fname}:{path} = {got}, manuscript says {expected} "
             f"(tolerance {tol})")


def test_manuscript_mentions_the_headline_numbers():
    """A weaker, textual check: the numbers the abstract and results state must
    literally appear in the .tex sources.  This catches a table regenerated
    from new results while the prose still quotes the old ones."""
    texts = []
    for fn in ("main.tex", "sec_results.tex", "tab_main.tex",
               "supplementary.tex", "supp_tables.tex"):
        p = os.path.join(PAPER, fn)
        if os.path.exists(p):
            texts.append(open(p, encoding="utf-8", errors="replace").read())
    if not texts:
        pytest.skip("manuscript sources not next to the artifact")
    blob = "\n".join(texts)
    must = ["0.671", "4.56", "0.492", "2.95", "0.429", "2.63",
            "1.0419", "1.0401", "1.0000"]
    missing = [s for s in must if s not in blob]
    assert not missing, f"headline numbers absent from the manuscript: {missing}"


def test_no_superseded_numbers_remain():
    """The three numbers the 7 September audit overturned must not reappear in
    a claim position.  They are allowed in the sentence that says they were
    wrong, so this checks the tables and the abstract, not the whole file."""
    dead = {"0.804": "the pre-audit CAISO eta",
            "7.64": "the pre-audit CAISO bound",
            "90.8": "the pre-audit share of the gap SHADE removed"}
    p = os.path.join(PAPER, "tab_main.tex")
    if not os.path.exists(p):
        pytest.skip("tab_main.tex not next to the artifact")
    t = open(p, encoding="utf-8", errors="replace").read()
    bad = [f"{k} ({v})" for k, v in dead.items() if k in t]
    assert not bad, f"superseded numbers in Table 1: {bad}"


if __name__ == "__main__":
    ok = bad = skipped = 0
    print(f"{'claim':<46s} {'file':<12s} {'found':>14s} {'stated':>10s}  ")
    print("-" * 92)
    for label, fname, path, expected, tol in CLAIMS:
        p = os.path.join(RES, fname)
        if not os.path.exists(p):
            print(f"{label:<46s} {fname:<12s} {'(missing)':>14s}")
            skipped += 1
            continue
        d = json.load(open(p))
        try:
            got = dig(d, path)
        except Exception as e:                                  # noqa: BLE001
            print(f"{label:<46s} {fname:<12s} {'ERR':>14s}  {e}")
            bad += 1
            continue
        if isinstance(expected, bool):
            good = bool(got) is expected
        elif isinstance(expected, int) and tol == 0:
            good = int(got) == expected
        else:
            good = abs(float(got) - float(expected)) <= tol
        ok += good
        bad += not good
        mark = "" if good else "   <-- MISMATCH"
        gs = f"{got:.5g}" if isinstance(got, float) else str(got)
        print(f"{label:<46s} {fname:<12s} {gs:>14s} {str(expected):>10s}{mark}")
    print("-" * 92)
    print(f"{ok} agree, {bad} disagree, {skipped} skipped")
    sys.exit(1 if bad else 0)
