"""Guards for the three Critical findings of PEER_REVIEW_2026-09-13.md.

Each one was a claim the manuscript made that the artifact did not support.
These tests pin the repair so the specific failure cannot silently return.

  C1  Theorem 5.2's epsilon-Nash bound carried a market-power term
      2*beta*rho*E^2.  Under exact aggregation the Shade fixed point minimises
      C, which is an exact potential for the corrected game, so it is an EXACT
      Nash equilibrium of that game and epsilon = 0.  E10 part D measures
      exactly this and reports 0 or ~1e-11.  The market-share term belongs to
      the misreporting proposition, which is a statement about incentives.

  C2  Proposition S1 (the two-slot n>=4 window) and Theorem 5.2 were stated
      with no proof anywhere in the submission, and the body pointed at a
      supplement section that did not exist.

  C3  Section 7.3 and Figure 2 quoted 1.72% / 8.22% as "the" strategic
      threshold.  Those are medians over ONLY the 6 of 18 cells that cross,
      drawn from two disjoint clusters, and they land in the gap between the
      clusters.  The censored fact is what the paper may state.
"""
import json
import os

PAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def _read(fn):
    p = os.path.join(PAPER, fn)
    with open(p, encoding="utf-8", errors="replace") as f:
        return " ".join(f.read().split())


def _json(fn):
    with open(os.path.join(RES, fn), encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ C1
def test_epsilon_is_zero_under_exact_aggregation():
    """E10 part D runs Shade to convergence with exact aggregation and measures
    the largest unilateral improvement.  Theorem 5.2(i) says it is exactly 0;
    anything above solver residual would falsify it."""
    d = _json("E10.json")
    eps = [r["eps"] for r in d["D"]]
    assert eps, "E10.json part D is empty; the exactness claim is unmeasured"
    assert max(eps) < 1e-9, (
        "Theorem 5.2(i) claims the fixed point is an EXACT Nash equilibrium of "
        f"the corrected game, but E10 measures eps up to {max(eps):.3e}")


def test_enash_bound_carries_no_market_power_term():
    """The market-share term bounds misreporting, not the equilibrium gap."""
    t = _read("main.tex")
    i = t.find("label{eq:enash}")
    assert i > 0, "eq:enash has gone"
    stmt = t[max(0, i - 1400):i]
    assert "market power" not in stmt.lower(), (
        "Theorem 5.2's epsilon bound has regained a market-power term; "
        "under exact aggregation epsilon is 0, so no such term belongs in it")
    assert "bar\\rho" not in stmt, (
        "market share (rho) is back inside the epsilon bound")


# ------------------------------------------------------------------ C2
def test_both_missing_proofs_exist():
    supp = _read("supp_proofs.tex")
    assert "Proof of Theorem~\\ref{M-thm:enash}" in supp, (
        "the body promises a full argument for Theorem 5.2 in the supplement; "
        "it must have a destination")
    for frag in ("Exact aggregation", "Aggregation error", "Before convergence"):
        assert frag in supp, f"Theorem 5.2's proof is missing part '{frag}'"

    tex = _read("supplementary.tex")
    j = tex.find("Exact two-slot case, full form")
    assert j > 0, "the two-slot proposition has gone"
    assert "\\begin{proof}" in tex[j:j + 4000], (
        "Proposition S1 is stated without a proof; it is a listed contribution")


def test_body_promises_have_destinations():
    """Every 'in the supplement' pointer in the body must resolve to something
    the supplement actually contains."""
    supp = _read("supp_proofs.tex") + _read("supplementary.tex")
    for thm in ("M-thm:poa", "M-thm:info", "M-thm:eff", "M-thm:enash"):
        assert f"Proof of Theorem~\\ref{{{thm}}}" in supp, \
            f"no proof section for {thm}"


# ------------------------------------------------------------------ C3
def test_strategic_threshold_is_censored_not_averaged():
    """Only 4 of 18 cells cross, in two disjoint clusters per region, and the
    median over the crossers falls in the gap between them."""
    prop = _json("E9.json")["propagation"]
    for reg, lo, hi in (("CISO", 0.0095, 0.0249), ("PJM", 0.0545, 0.1099)):
        vals = sorted(v for k, c in prop.items() if k.startswith(reg + "|")
                      for v in c["thresholds_by_seed"]["strat"] if v is not None)
        beyond = sum(1 for k, c in prop.items() if k.startswith(reg + "|")
                     for v in c["thresholds_by_seed"]["strat"] if v is None)
        assert len(vals) == 6 and beyond == 12, (
            f"{reg}: crossing structure changed ({len(vals)} cross, {beyond} do not)")
        mid = (vals[2] + vals[3]) / 2.0
        assert lo < mid < hi, (
            f"{reg}: the median of the crossers, {mid:.4f}, no longer falls in "
            "the empty gap between the two clusters")


def test_the_median_strategic_threshold_is_not_quoted():
    """1.72% and 8.22% are medians over the crossers only.  The manuscript must
    not present them as the strategic threshold."""
    for fn in ("sec_results.tex", "fig_threshold.tex", "main.tex", "sec_disc.tex"):
        t = _read(fn)
        for bad in ("$1.72\\%$", "$8.22\\%$"):
            assert bad not in t, f"{fn} quotes {bad} as a strategic threshold"
        assert "1.5$ times" not in t and "$1.5$ times it" not in t, \
            f"{fn} still compares the two thresholds as a ratio of medians"


def test_the_crossings_themselves_are_reported():
    """What replaced the median must actually be present."""
    t = _read("sec_results.tex") + _read("fig_threshold.tex")
    for frag in ("0.70$--$0.95", "2.49$--$3.02", "4.36$--$5.45", "11.0$--$15.0"):
        assert frag in t, f"the crossing range {frag} is no longer reported"


# ------------------------------------------------------------------ misc
def test_tost_margin_width_is_disclosed():
    """The equivalence margin also admits 'naive', which the paper calls worse."""
    t = _read("sec_results.tex")
    assert "0.0105" in t, "the TOST margin's absolute width is no longer stated"
    v = _json("V16.json")["pooled"]
    assert v["Naive forecast-taking"]["tost"]["equivalent_at_05"], \
        "naive no longer passes the same TOST; the caveat can be dropped"


def test_section_4_1_ratios_are_labelled_as_single_seed():
    """V20's calibrated arm is seed 0; Table 1 is a twenty-seed mean.  PJM
    differs at three decimals (1.074 vs 1.070), so wherever the four ratios are
    quoted, the basis must be stated beside them.

    They moved to the supplement when the architecture figure went into the body
    and the eight-page budget had to be repaid; the invariant is unchanged."""
    t = _read("supp_proofs.tex")
    i = t.find("1.084")
    assert i > 0, "the four decomposition ratios have gone"
    window = t[max(0, i - 400):i + 300]
    assert "seed-$0$" in window or "seed $0$" in window, \
        "the ratios are quoted without saying they are single-seed"
    assert "1.070" in window, \
        "Table 1's twenty-seed PJM value must be stated beside the seed-0 one"
    # and the body must not quietly reacquire them without the same caveat
    b = _read("main.tex")
    assert "1.084" not in b, \
        "Section 4.1 has reacquired the seed-0 ratios; keep them in one place"
