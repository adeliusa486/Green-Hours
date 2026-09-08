# Paper → repository traceability

Every quantitative claim in the paper, and the path from the input file to the
sentence. If a row has no script, the paper does not make the claim as a number.

`tests/test_claims.py` mechanically checks the values in the last column
against the JSON in the third, so this table cannot drift silently: it is a
pytest file, not a document.

## Theory

| Paper claim | Where | Establishes it | Independent test | Result |
|---|---|---|---|---|
| Exact potential Φ (Prop. 4.1) | §4 | derivation in paper; `src/gh/core.py:Instance.potential` | `tests/test_theorems.py` T1 — re-derives Φ from the primitives and differences it against ∂J/∂x | residual 3.3e-13 |
| Unique pure Nash (Prop. 4.1) | §4 | strict convexity of Φ | `tests/test_game.py` — 20 random starts | spread 8.3e-9 at ftol 1e-15 |
| Internalisation gap (Lemma 4.2) | §4 | algebra | `tests/test_theorems.py` T2 — checks it as an identity in x, not only at an optimum | 1.5e-8 |
| PoA ≤ 3/(2(1−η)) (Thm 4.3) | §4 | proof in `supp_proofs.tex` | `tests/test_theorems.py` T3 — 250 random instances | 0 violations, worst 55.1% of the bound |
| Generalised bound (Prop. 4.4) | §4 | proof in `supp_proofs.tex` | `tests/test_theorems.py` T4 — 150 piecewise instances | 0 violations |
| Signal precision, interior case (Thm 4.5) | §4 | proof in `supp_proofs.tex` | `tests/test_theorems.py` T5 — fits the exponent of E[C(σ)]−C(0) | 2.000 |
| Two-slot thresholds, n ≥ 4 (Prop. 4.6) | §4 | closed form in supplement | `tests/test_theorems.py` T6 | 0 mismatches |
| SHADE fixed point = planner (Thm 5.2) | §5 | proof in paper | `tests/test_theorems.py` T7 | excess 1.8e-13 |
| ε-Nash ⟹ C−C\* ≤ nε (Thm 5.3) | §5 | proof in `supp_proofs.tex` | `tests/test_theorems.py` T8, and `E10_convergence.py` §D checks the ε bound itself | 0/80 and 0/15 violations |
| Convergence region, γ < 2/n sufficient (Thm 5.3) | §5 | linearisation in §5.3 | `E10_convergence.py` §A–C | no divergence over n ∈ [1,128], β over 13 orders |
| Misreporting bounded by market share (Prop. 5.1) | §5 | proof in supplement | `V8_misreport.py` | sign as predicted; gain only at n = 2 |

## Empirical

| Paper claim | Where | Script | Input | Output | Test |
|---|---|---|---|---|---|
| η = 0.671 / 0.492 / 0.429, bound 4.56 / 2.95 / 2.63 | §6, Tab. S3 | `E1_calibrate.py` | `data/EIA930_BALANCE_2024_Jul_Dec.csv` | `results/E1.json` | `test_claims.py` |
| MEF weakly identified in CAISO/ERCOT, moderately in PJM; not causal | §6 | `E8_mef_identification.py` | same | `results/E8.json` | `test_claims.py` |
| η robust to the accounting boundary (0.630–0.671 in CAISO) | §6, Tab. S1 | `E8_mef_identification.py` | same | `results/E8.json` | — |
| Placebos collapse (permuted 3–5%, lead 2–19%) | §6, Tab. S2 | `E8_mef_identification.py` | same | `results/E8.json` | `test_claims.py` |
| Net-load restriction rejected in 17/24 CAISO bins | §6 | `E8_mef_identification.py` | same | `results/E8.json` | — |
| Table 1 (all methods, 3 regions, 20 seeds) | §7 | `E3_main.py` | E1 profiles + generator | `results/E3.json` | `test_claims.py` |
| Operational metrics, peak/mean, congestion term | Tab. S5 | `E4_E7_ablation.py` | same | `results/E4_E7.json` | `test_claims.py` |
| Ablation: no adder → 1.0419; 3 h coarsening → 1.0212 | Tab. S4 | `E4_E7_ablation.py` | same | `results/E4_E7.json` | `test_claims.py` |
| Deployment threshold (static signal stops sufficing) | §7 | `E9_threshold.py` (dense) and `E5_when_mechanism_matters.py` (coarse, superseded) | same | `results/E9.json` | — |
| Threshold sensitivity to n, heterogeneity, deadlines, envelope, β | §7, Tab. S7 | `E9_threshold.py` §B | same | `results/E9.json` | — |
| Threshold under alternative MEF specifications | §7 | `E9_threshold.py` §C | E8 profiles | `results/E9.json` | — |
| Trace-derived workload: sizes, ρ̄ = 4.5%, envelopes, arrival shape | §6 | `derive_azure_workload.py` | `data/azure2019/vmtable.csv.gz` | `data/azure2019/derived_workload.json` | parse check vs Microsoft's published CDF, in the JSON |
| Table 1 methods on the trace workload; 24 diurnal alignments | §7, Tab. S6 | `E11_trace_workload.py` | above + E1 profiles | `results/E11.json` | — |
| Convergence rate \|1−γ\|, interior fraction ≈ 6–16% | §5.3, Tab. S8 | `E10_convergence.py` | generated | `results/E10.json` | `test_claims.py` |
| DP is unaffordable: 1.11–1.14 of the planner at every ε | §7, Tab. S2 | `V7_privacy.py` | generated | `results/V7.json` | — |
| Curtailment cliff: strategic share 4.7% → 38.8% | §7, Tab. S9 | `V10_cliff.py`, `V11_shade_under_cliff.py` | generated | `results/V10.json`, `V11.json` | `test_cliff_solver.py` |
| 28-configuration robustness, SHADE removes 95.2–100% | Tab. S1 | `V12_robustness.py` | generated | `results/V12.json` | — |

## Not claimed as a number

| Thing | Status |
|---|---|
| Cooperative MARL baseline | Not run (GPU training). Reported absent. |
| Great Britain, Germany | Not calibrated: the NESO feed gives no absolute demand. `data/gb_2024H2.json` is kept and unused. |
| A tight PoA instance | Open. Best found reaches 55.1% of the bound. |
| Global convergence of the SHADE iteration | Open. Local, with the region stated. |
| Causal marginal emission rates | Not claimed. No instrument is used. |
| Deadlines from a production trace | Not available in any public trace. Swept. |
| The diurnal phase of the Azure arrival curve | Not recoverable from the trace. All 24 rotations reported. |
