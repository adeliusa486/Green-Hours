# Grid data

The calibration (`experiments/E1_calibrate.py`) and the main comparison
(`experiments/E3_main.py`) read one public file. It is not committed — it is
48 MB and the licence is EIA's, not ours — so fetch it before running either.

## Download

```bash
curl -o data/EIA930_BALANCE_2024_Jul_Dec.csv \
  https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2024_Jul_Dec.csv
```

Verify you have the same bytes we did:

```bash
sha256sum data/EIA930_BALANCE_2024_Jul_Dec.csv
# a602a8e577cfdd2fa3083413eaa4dd8ba7f0d2b26a2986f8f6d22917465ea49f
wc -c < data/EIA930_BALANCE_2024_Jul_Dec.csv
# 47928993
```

Retrieved 2 September 2026. EIA revises historical hours, so a later download
may differ; if the hash does not match, the numbers below will move slightly and
`results/E1.json` records what your copy produced.

## What it is

EIA-930, the Hourly Electric Grid Monitor, BALANCE file for July–December 2024:
hourly demand, net generation by fuel, and interchange for all 61 US balancing
authorities. Public, no API key, no registration.
<https://www.eia.gov/electricity/gridmonitor/about>

## Which columns are used, and why those

`E1_calibrate.py` reads the **Adjusted** series throughout — demand
(column 13) and the sixteen `Net Generation (MW) from … (Adjusted)` fuel columns
(48–63). The Adjusted series is EIA's imputed and balanced vintage. An earlier
version of this script paired *adjusted* demand with *raw* generation, and
omitted geothermal, pumped storage, battery storage and the two "other storage"
categories from the generation total, so its average emission factor had a
denominator that was neither the demand nor the generation of the hour.

`Hour Number` (column 2) is the hour ending in **local** time, 1–24, which is
what the hour-of-day bins use. Days on which a clock change makes 23 or 25
hours are kept as they are; the bin counts absorb it.

## Emission factors

Direct combustion only, gCO₂ per kWh of generation:

| fuel | factor | note |
|---|---|---|
| coal | 1000 | |
| natural gas | 430 | swept over 370–550 as a sensitivity; gas sets the margin in all three regions |
| petroleum | 700 | |
| other / unknown | 500 | |
| nuclear, hydro, wind, solar, geothermal, all storage | 0 | biomass is inside "other"; the intensity feeds treat it as neutral and so do we |

These are combustion factors, not life-cycle factors. Upstream methane and
construction emissions are excluded, which is the convention the carbon-intensity
feeds operators actually consume use, and is the convention the paper's average
factor has to match for the comparison with the marginal factor to be
like-for-like.

## Accounting boundary

CAISO imports 17.5% of its energy on average and up to 33% in an hour, so
attributing only in-region generation understates its average factor. The
default boundary is therefore **consumption**: own generation plus net imports
priced at the contemporaneous average intensity of the rest of the
interconnection (WECC for CAISO, the Eastern Interconnection for PJM), computed
from the same file. ERCOT is islanded — |interchange| stays under 1% of demand —
so its two boundaries coincide. `E1_calibrate.py` reports the production-based
variant as a sensitivity.

## What is estimated, and under what assumption

MEF is the slope of a regression of the hour-to-hour change in emissions on the
change in **net load** (demand minus wind and solar), taken day over day within
an hour-of-day bin. The identifying assumption is that the dispatchable fleet
responds to net load, so an added MWh of demand and a lost MWh of wind are the
same shock and dE/d(load) = dE/d(net load).

These are estimated marginal emission *associations* under that assumption, not
the output of a dispatch model and not a causal experiment. Hour-of-day binning
and first differencing remove level and diurnal confounding; they do not remove
simultaneity between demand and renewable curtailment.

## Files that are not used

`gb_2024H2.json` is a National Energy System Operator feed for Great Britain,
kept because it was collected. It is **not** used in any reported number: the
feed publishes carbon intensity and fuel mix but no absolute demand, so no
marginal regression is possible from it. Great Britain and Germany are excluded
from every calibrated claim in the paper for that reason.

---

# Workload data

`experiments/E11_trace_workload.py` pairs the grid above with a production
workload trace instead of the generator. The trace is the **Azure Public Dataset
V2** (the 2019 VM trace), released by Microsoft under **CC BY 4.0**, so unlike
Borg or Alibaba it *is* redistributable — but the source file is 437 MB, so it
is fetched rather than committed. What **is** committed is the small aggregate
`experiments/derive_azure_workload.py` reduces it to,
`data/azure2019/derived_workload.json` (168 kB), together with the published
distribution files Microsoft ships beside the trace.

## Download

```bash
bash scripts/fetch_azure.sh          # or the curl line it contains
python experiments/derive_azure_workload.py
```

Verify you have the same bytes we did:

```
data/azure2019/vmtable.csv.gz
  sha256 e8c9a0ab0e06b4322747147b49f8b44c07cf017327a150a92b6fb95aab8de3e5
  bytes  437594496
```

Retrieved 7 September 2026.

## Schema

No header row. Eleven columns, per Microsoft's `schema.csv`:

| # | field | used for |
|---|---|---|
| 1 | vm id (hashed) | — |
| 2 | subscription id (hashed) | **the unit of independent decision-making** |
| 3 | deployment id (hashed) | — |
| 4 | timestamp vm created (s) | arrival hour-of-day, duration |
| 5 | timestamp vm deleted (s) | duration, censoring |
| 6–8 | max / avg / p95 cpu (%) | avg cpu weights the energy proxy |
| 9 | vm category | `Delay-insensitive` selects the flexible population |
| 10 | vm virtual core count bucket | size |
| 11 | vm memory (GB) bucket | — |

## Parse validation

A silent column misalignment in a 2.7-million-row file would be invisible, so
the derivation checks itself against Microsoft's own published aggregate:
`azure2019_data_lifetime.txt` says 63.6% of VMs live at most one hour, and our
parse gives 64.0%. That agreement is recorded in
`derived_workload.json["parse_validation"]` and the script warns if it drifts
more than two points.

## What the trace fixes, and what it does not

**Measured, and used:**

* 5,248 subscriptions carry delay-insensitive load; the largest holds **4.5%**
  of delay-insensitive core-hours, which is the market-share parameter `ρ̄` of
  Theorem 4, and the top 128 hold 73%.
* Log-size spread of those shares: 1.89 over all, 0.80 over the top 128, 0.39
  over the top 32 — which brackets the generator's assumed σ = 0.6.
* Each operator's diurnal arrival shape, and its demonstrated per-hour peak
  concurrent delay-insensitive core count, which becomes the power envelope.

**Not in the trace, therefore assumed and swept:**

* **Deadlines.** A VM's deletion time is when the tenant stopped paying, not
  when the work was due. No public cloud trace records deadlines. The deferral
  horizon is swept over 2–24 h.
* **Diurnal phase.** Trace timestamps carry no timezone or start date, so the
  arrival curve is recovered only up to an unknown rotation against the grid
  clock. All 24 alignments are run and the range reported.
* **Watts per core.** Not invented: core-hours × mean utilisation is a
  *relative* weighting, and the aggregate level is set by the flexible-share
  parameter exactly as in the synthetic setting.

**Two things the parse found that are worth stating:**

1. The `Delay-insensitive` class is **not short batch work**. 75.4% of such VMs
   span the entire 30-day window; among the 8,399 that do not, the median
   lifetime is 267 hours. The label means tolerant of scheduling latency, not
   "finishes this afternoon". Deferral in this population is power modulation of
   long-running capacity — which is what the splittable-energy model represents,
   and is worth saying rather than implying a batch queue.
2. Microsoft's published 51/43/6 category split is **not** by VM count. By count
   the trace is 5.9% delay-insensitive, 2.9% interactive and 90.8% unlabelled;
   by core-hours it is 58.8 / 32.2 / 9.0. We use the core-hour weighting, which
   is the one that matters for energy, and report both.

## Attribution

Azure Public Dataset is CC BY 4.0. Cite Microsoft Azure and Hadary et al.,
*Protean: VM Allocation Service at Scale*, OSDI 2020.
