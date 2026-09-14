"""Emit the pgfplots coordinate blocks for the paper's threshold figure.

The figure (fig_threshold.tex in the manuscript directory) is the paper's
central empirical result, so its data must be regenerable rather than
transcribed -- the same rule every table in this repository follows.  This
script reads results/E9.json and prints the three coordinate lists per panel,
wrapped to fit, ready to paste into the two `\\addplot` groups.

    python scripts/make_threshold_figure.py            # both panels
    python scripts/make_threshold_figure.py --check    # verify the .tex matches

`--check` re-derives the coordinates and diffs them against the committed
fig_threshold.tex, so a stale figure fails loudly instead of quietly disagreeing
with results/E9.json.  It looks for the manuscript one directory above the
repository root and skips if it is not there (the artifact ships without it).
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
E9 = os.path.join(ROOT, "results", "E9.json")
TEX = os.path.join(ROOT, "..", "fig_threshold.tex")

REGIONS = [("CISO", "CAISO"), ("ERCO", "ERCOT"), ("PJM", "PJM")]
PANELS = [("static_leaves", "(a) what a level-correct STATIC signal leaves"),
          ("strategic_share", "(b) the strategic share of the gap")]
FLOOR = {"static_leaves": 1e-3, "strategic_share": 0.0}


def coords(d, ba, key):
    """(share in %, value in %) pairs for one region and one panel.

    Panel (a) is drawn on a log axis, so exact zeros are floored to 1e-3 --
    below the axis minimum, therefore invisible, and pgfplots will not take the
    logarithm of zero.
    """
    out = []
    for s, r in zip(d["dense"][ba]["shares"], d["dense"][ba]["mean_curve"]):
        v = 100.0 * r[key]
        if v != v:                                   # NaN
            continue
        out.append((100.0 * s, max(v, FLOOR[key])))
    return out


def fmt(pts, width=88):
    """Wrap the coordinate list the way the .tex file has it."""
    lines, line = [], " "
    for x, y in pts:
        tok = f"({x:.4g},{y:.3f})"
        if len(line) + len(tok) > width:
            lines.append(line)
            line = " "
        line += tok
    lines.append(line)
    return "\n".join(lines)


def emit(d):
    for key, title in PANELS:
        print(f"%% ---- {title} ----")
        for ba, nice in REGIONS:
            print(f"%% {nice}")
            print(fmt(coords(d, ba, key)))
        print()
    print("%% thresholds, per region, as a percentage of regional demand")
    for ba, nice in REGIONS:
        t = d["dense"][ba]["thresholds"]["static05"]
        h = d["headline"][ba]["static"]
        print(f"%%   {nice:6s} marker at {100*t['median']:.2f}; "
              f"with estimator uncertainty "
              f"[{100*h['p10']:.2f}, {100*h['p90']:.2f}]")
    print(f"%%   today = {100*d['today_share']:.2f}%")


def check(d):
    """Every coordinate in the .tex must appear in the regenerated data."""
    if not os.path.exists(TEX):
        print("fig_threshold.tex not next to the artifact -- nothing to check")
        return 0
    tex = open(TEX, encoding="utf-8").read()
    # strip style coordinates -- legend anchors and the like -- so only plotted
    # points are compared
    tex = re.sub(r"at=\{\([^)]*\)\}", "", tex)
    want = set()
    for key, _ in PANELS:
        for ba, _n in REGIONS:
            for x, y in coords(d, ba, key):
                want.add(f"({x:.4g},{y:.3f})")
    have = set(re.findall(r"\((\d[\d.]*,[\d.]+)\)", tex))
    have = {f"({h})" for h in have}
    missing = sorted(w for w in want if w not in have)
    print(f"coordinates in results/E9.json : {len(want)}")
    print(f"coordinates in fig_threshold.tex: {len(have)}")
    if missing:
        print(f"MISSING from the figure ({len(missing)}): {missing[:6]} ...")

    # ---- panel (a): deployment markers, drawn as median +- (p10, p90) -------
    # Whitelist them by DERIVING them from E9 rather than tolerating a count.
    marker_ok, allowed = True, set()
    for mm in re.finditer(
            r"coordinates \{\(([\d.]+),5\)\s*\+-\s*\(([\d.]+),([\d.]+)\)\}", tex):
        x, lo, hi = (float(g) for g in mm.groups())
        allowed |= {f"({mm.group(1)},5)", f"({mm.group(2)},{mm.group(3)})"}
        band = (round(x - lo, 2), round(x + hi, 2))
        hit = [ba for ba, _ in REGIONS
               if (round(100 * d["headline"][ba]["static"]["p10"], 2),
                   round(100 * d["headline"][ba]["static"]["p90"], 2)) == band]
        if not hit:
            marker_ok = False
            print(f"  panel (a) marker at {x}% spans {band}, which is no "
                  "region's static p10-p90 in E9.json")

    # ---- panel (b): the crossing bars --------------------------------------
    # Each bar endpoint must be an ACTUAL crossing in E9.json, and the pair must
    # span one cluster.  Before this check the bars were verified by nothing,
    # which is how the withdrawn 1.72%/8.22% medians survived review.
    prop = d["propagation"]
    cross = {}
    for ba, _ in REGIONS:
        cross[ba] = sorted(round(100 * v, 2)
                           for k, c in prop.items() if k.startswith(ba + "|")
                           for v in c["thresholds_by_seed"]["strat"] if v is not None)
    bars = [(float(a), float(b)) for a, b in re.findall(
        r"axis cs:([\d.]+),25\)\s*--\s*\(axis cs:([\d.]+),25\)", tex)]
    bar_ok = True
    if not bars:
        bar_ok = False
        print("  panel (b) has no crossing bars; the strategic crossings are "
              "unverified against E9.json")
    for lo, hi in bars:
        owner = [ba for ba, _ in REGIONS
                 if round(lo, 2) in cross[ba] and round(hi, 2) in cross[ba]]
        if not owner:
            bar_ok = False
            print(f"  panel (b) bar {lo}-{hi}% is not a pair of crossings in "
                  "E9.json")
    n_cross = sum(len(v) for v in cross.values())
    if bars and len(bars) * 2 > n_cross:
        bar_ok = False
        print(f"  panel (b) draws {len(bars)} bars but E9.json has only "
              f"{n_cross} crossings")

    extra = sorted(h for h in have if h not in want and h not in allowed)
    if extra:
        print(f"in the figure, not in the data and not a derived marker "
              f"({len(extra)}): {extra[:6]} ...")
    ok = not missing and not extra and marker_ok and bar_ok
    print(f"panel (a) markers: {'ok' if marker_ok else 'WRONG'}; "
          f"panel (b) bars: {'ok' if bar_ok else 'WRONG'} ({len(bars)} drawn)")
    print("MATCH" if ok else "MISMATCH -- regenerate the figure")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify fig_threshold.tex against results/E9.json")
    a = ap.parse_args()
    if not os.path.exists(E9):
        sys.exit("missing results/E9.json -- run experiments/E9_threshold.py")
    d = json.load(open(E9))
    sys.exit(check(d) if a.check else emit(d) or 0)


if __name__ == "__main__":
    main()
