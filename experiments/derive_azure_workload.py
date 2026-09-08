"""Derive a workload description from the Azure 2019 VM trace.

Input   data/azure2019/vmtable.csv.gz   (Azure Public Dataset V2, CC BY 4.0)
Output  data/azure2019/derived_workload.json   (170 kB, committed to the repo)

Why this exists
---------------
Every workload number in the paper up to now came from a generator: log-normal
operator sizes with sigma = 0.6, uniform deadline slack, a Gaussian arrival
bump.  Nothing is wrong with that as a stress test, but it means the workload
side of Table 1 was assumed rather than measured, which is the most obvious
thing for a reviewer to push on.  This script replaces what the trace can
replace and says plainly what it cannot.

What the trace supplies, and what it does not
---------------------------------------------
MEASURED, and used:
  * how flexible demand is spread across independent decision-makers.  A
    subscription is the trace's unit of separate tenancy -- separate billing,
    separate quota, no shared scheduler -- so subscriptions stand in for
    operators.  There are 5,248 with delay-insensitive load; the largest holds
    4.5% of delay-insensitive core-hours and the top 128 hold 73%.  That
    largest share is the market-power parameter rho-bar of Theorem 4, measured
    rather than posited.
  * the heterogeneity of those sizes.  The log-size standard deviation is 1.89
    across all subscriptions, 0.80 across the top 128 and 0.39 across the top
    32, which brackets the generator's sigma = 0.6.
  * which work is deferrable.  The trace labels every VM Delay-insensitive,
    Interactive or Unknown -- a production operator's own classification.
  * the diurnal shape of flexible-work arrival, per subscription.
  * per-subscription peak concurrent delay-insensitive cores, which becomes the
    power envelope xbar, previously a flat 35% of daily energy.

NOT in the trace, and therefore still assumed, and labelled as such:
  * DEADLINES.  No public cloud trace records them: a VM's deletion time is when
    the tenant stopped paying, not when the work was due.  The deferral horizon
    remains a swept parameter.
  * the PHASE of the diurnal shape against the grid clock.  Trace timestamps are
    seconds from an unpublished start in an unnamed region, so the arrival curve
    is recovered only up to an unknown rotation.  E11 sweeps all 24 alignments
    and reports the range, which also answers a question worth asking on its
    own: how much does the result depend on whether a datacenter's busy hours
    coincide with the green hour?
  * the conversion from core-hours to megawatt-hours.  Rather than invent a
    watts-per-core constant, the aggregate is scaled to the flexible share of
    regional demand exactly as in the synthetic setting: the trace fixes
    relative sizes and shapes, the share parameter fixes the level.

Two findings that came out of the parse and that the paper reports
------------------------------------------------------------------
1.  The delay-insensitive class is NOT short batch work.  76.6% of
    delay-insensitive VMs span the entire 30-day window, and among the 8,394
    that both start and end inside it the median lifetime is 267 hours.  The
    label means "tolerant of scheduling latency", not "a job that finishes this
    afternoon".  Deferral in this population is therefore power modulation of
    long-running capacity rather than start-time deferral of jobs -- which is
    what the paper's splittable-energy model actually represents, and is worth
    saying out loud.
2.  The published category split (51 / 43 / 6 per cent in Microsoft's own
    aggregate file) is not a split by VM count.  By count the trace is 5.9%
    delay-insensitive, 2.9% interactive and 90.8% unlabelled; by core-hours it
    is 58.8 / 32.2 / 9.0.  We report the core-hour weighting, which is the one
    that matters for energy, and note that it is close to but not identical
    with the published figure.

Parse validation.  Over all VMs this parse reproduces Microsoft's own published
lifetime CDF: 64.0% of VMs live at most one hour against the 63.6% in
azure2019_data_lifetime.txt.  That agreement is checked here and recorded in the
output, because a silent column misalignment in a 2.7-million-row file would
otherwise be invisible.

Licence.  Azure Public Dataset is CC BY 4.0, so this derived aggregate is
redistributable with attribution.  The 437 MB source is not committed;
data/README.md carries its URL and SHA-256, and scripts/fetch_azure.sh
retrieves it.
"""
import collections
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "..", "data", "azure2019")
SRC = os.path.join(DDIR, "vmtable.csv.gz")
OUT = os.path.join(DDIR, "derived_workload.json")
LIFETIME_REF = os.path.join(DDIR, "azure2019_data_lifetime.txt")

HOUR = 3600.0
TRACE_DAYS = 30
TRACE_END = TRACE_DAYS * 24 * HOUR - 600.0     # 30 d, less a slack of ten min
TOP_K = 128
FLEX_LABEL = "Delay-insensitive"


def main():
    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC}\n"
                 "  retrieve it with scripts/fetch_azure.sh -- see "
                 "data/README.md")

    print(f"streaming {os.path.relpath(SRC)} ...")
    ch = collections.defaultdict(float)              # core-hours x utilisation
    nvm = collections.Counter()
    arr = collections.defaultdict(lambda: np.zeros(24))
    conc = collections.defaultdict(lambda: np.zeros(TRACE_DAYS * 24 + 2))
    cat_n = collections.Counter()
    cat_ch = collections.Counter()
    core_n = collections.Counter()
    dur_all, dur_flex, dur_flex_unc = [], [], []
    n_rows = n_bad = 0

    with gzip.open(SRC, "rt") as f:
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 11:
                n_bad += 1
                continue
            try:
                t0, t1 = float(p[3]), float(p[4])
                cores, util = float(p[9]), float(p[6]) / 100.0
            except ValueError:
                n_bad += 1
                continue
            n_rows += 1
            cat = p[8]
            d = (t1 - t0) / HOUR
            cat_n[cat] += 1
            cat_ch[cat] += cores * max(d, 0.0)
            dur_all.append(d)
            if cat != FLEX_LABEL or t1 <= t0:
                continue
            sub = p[1]
            dur_flex.append(d)
            if t0 > 0.0 and t1 < TRACE_END:
                dur_flex_unc.append(d)
            core_n[int(cores)] += 1
            e = cores * d * max(util, 0.01)
            ch[sub] += e
            nvm[sub] += 1
            arr[sub][int((t0 // HOUR) % 24)] += e
            i0 = int(t0 // HOUR)
            i1 = min(int(t1 // HOUR) + 1, TRACE_DAYS * 24 + 1)
            if 0 <= i0 < i1 < len(conc[sub]):
                conc[sub][i0] += cores
                conc[sub][i1] -= cores
            if n_rows % 500000 == 0:
                print(f"  {n_rows:,} rows, {len(ch):,} flexible subscriptions")

    da = np.array(dur_all)
    df = np.array(dur_flex)
    du = np.array(dur_flex_unc)
    tot_ch = sum(cat_ch.values())
    print(f"  {n_rows:,} VMs parsed ({n_bad} unparseable)")
    print("  category by count  : " +
          ", ".join(f"{k} {100*v/n_rows:.1f}%" for k, v in cat_n.most_common()))
    print("  category by core-h : " +
          ", ".join(f"{k} {100*v/tot_ch:.1f}%" for k, v in cat_ch.most_common()))

    # --- parse validation against Microsoft's own published lifetime CDF ---
    ref = None
    if os.path.exists(LIFETIME_REF):
        rows = [l.split("\t") for l in
                open(LIFETIME_REF).read().strip().split("\n")[1:]]
        tab = {float(a): float(b) for a, b in rows}
        ref = tab.get(1.0)
    ours = 100.0 * float((da <= 1.0).mean())
    print(f"  parse check: {ours:.1f}% of VMs live <= 1 h; Microsoft's "
          f"published CDF says {ref}%" if ref else
          f"  parse check: {ours:.1f}% of VMs live <= 1 h (no reference file)")
    if ref is not None and abs(ours - ref) > 2.0:
        print("  WARNING: lifetime CDF disagrees with the published aggregate "
              "by more than two points -- check the column layout")

    v = np.array(sorted(ch.values(), reverse=True))
    vs = v / v.sum()
    top = sorted(ch, key=ch.get, reverse=True)[:TOP_K]

    ops = []
    for s in top:
        a = arr[s].copy()
        a = a / a.sum() if a.sum() > 0 else np.full(24, 1.0 / 24)
        cc = np.cumsum(conc[s])[:TRACE_DAYS * 24].reshape(TRACE_DAYS, 24)
        peak = cc.max(axis=0)
        mean = cc.mean(axis=0)
        ops.append(dict(core_hours=float(ch[s]), n_vm=int(nvm[s]),
                        arrival_hod=[float(x) for x in a],
                        peak_cores_hod=[float(x) for x in peak],
                        mean_cores_hod=[float(x) for x in mean],
                        peak_to_mean=float(peak.max() / max(mean.mean(), 1e-9))))

    q = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    out = {
        "source": "Azure Public Dataset V2 (azure2019), vmtable.csv.gz",
        "source_url": "https://github.com/Azure/AzurePublicDataset",
        "licence": "CC BY 4.0",
        "citation": "Hadary et al., Protean: VM Allocation Service at Scale, "
                    "OSDI 2020.  Dataset released by Microsoft Azure under "
                    "CC BY 4.0.",
        "trace_days": TRACE_DAYS,
        "n_vm_total": n_rows, "n_unparseable": n_bad,
        "category_share_by_count_pct": {k: 100.0 * v / n_rows
                                        for k, v in cat_n.items()},
        "category_share_by_core_hours_pct": {k: 100.0 * v / tot_ch
                                             for k, v in cat_ch.items()},
        "parse_validation": {
            "our_frac_vms_le_1h_pct": ours,
            "published_frac_vms_le_1h_pct": ref,
            "agrees_within_2_points": (ref is None or abs(ours - ref) <= 2.0)},
        "n_vm_flexible": int(len(df)),
        "n_subscriptions_flexible": int(len(ch)),
        "censoring": {
            "frac_flexible_spanning_whole_window":
                float((df >= (TRACE_END / HOUR)).mean()),
            "n_flexible_uncensored": int(len(du)),
            "uncensored_median_hours": float(np.median(du)) if len(du) else None,
            "uncensored_p25_hours": float(np.percentile(du, 25)) if len(du) else None,
            "uncensored_p75_hours": float(np.percentile(du, 75)) if len(du) else None},
        "size_distribution": {
            "top_k": TOP_K,
            "largest_share_of_flexible_core_hours": float(vs[0]),
            "cumulative_share": {str(k): float(vs[:k].sum())
                                 for k in (1, 5, 10, 32, 64, 128, 256)
                                 if k <= len(vs)},
            "log_size_sd_all": float(np.log(vs[vs > 0]).std()),
            "log_size_sd_top128": float(np.log(vs[:128]).std())
            if len(vs) >= 128 else None,
            "log_size_sd_top32": float(np.log(vs[:32]).std())
            if len(vs) >= 32 else None,
            "generator_sigma_for_comparison": 0.6},
        "core_count_share_pct": {str(k): 100.0 * v / max(len(df), 1)
                                 for k, v in sorted(core_n.items())},
        "duration_hours_quantiles_all_vms": {str(p): float(np.percentile(da, p))
                                             for p in q},
        "duration_hours_quantiles_flexible_uncensored":
            {str(p): float(np.percentile(du, p)) for p in q} if len(du) else {},
        "operators": ops,
        "caveats": [
            "Deadlines are NOT in the trace; the deferral horizon is a swept "
            "assumption, not a measurement.",
            "Trace timestamps carry no timezone or start date, so the diurnal "
            "arrival shape is recovered only up to an unknown rotation; E11 "
            "sweeps all 24 alignments.",
            "core-hours x mean utilisation is an energy PROXY.  The aggregate "
            "level is set by the flexible-share parameter, not by it.",
            "A subscription is used as the unit of independent decision-making. "
            "It is the trace's unit of separate tenancy, but one cloud provider "
            "owns the physical fleet, so this understates how much a single "
            "provider could coordinate internally.",
            "The delay-insensitive population is long-running reserved "
            "capacity, not short batch jobs: 76.6% of such VMs span the whole "
            "window.  Deferral here means modulating the power of running "
            "capacity, which is what the splittable-energy model represents.",
            "90.8% of VMs by count carry no category label, so the flexible "
            "share by count is not recoverable from this trace; the core-hour "
            "weighting is used instead.",
        ],
    }
    os.makedirs(DDIR, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    sd = out["size_distribution"]
    print(f"\n  {len(ch):,} flexible subscriptions; largest holds "
          f"{100*sd['largest_share_of_flexible_core_hours']:.1f}% of flexible "
          f"core-hours (this is rho-bar); top {TOP_K} hold "
          f"{100*sd['cumulative_share'][str(TOP_K)]:.1f}%")
    print(f"  log-size sd: all {sd['log_size_sd_all']:.2f}, "
          f"top128 {sd['log_size_sd_top128']:.2f}, "
          f"top32 {sd['log_size_sd_top32']:.2f} "
          f"(generator assumes {sd['generator_sigma_for_comparison']})")
    c = out["censoring"]
    print(f"  {100*c['frac_flexible_spanning_whole_window']:.1f}% of flexible "
          f"VMs span the whole window; of the {c['n_flexible_uncensored']:,} "
          f"that do not, the median lifetime is "
          f"{c['uncensored_median_hours']:.0f} h")
    print(f"\nwrote {os.path.relpath(OUT)} "
          f"({os.path.getsize(OUT)/1024:.0f} kB)")


if __name__ == "__main__":
    main()
