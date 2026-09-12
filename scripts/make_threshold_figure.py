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
    # the .tex also carries the three threshold markers, which are not curve
    # points; ignore anything the data does not claim to contain
    missing = sorted(w for w in want if w not in have)
    extra = sorted(h for h in have if h not in want)
    print(f"coordinates in results/E9.json : {len(want)}")
    print(f"coordinates in fig_threshold.tex: {len(have)}")
    if missing:
        print(f"MISSING from the figure ({len(missing)}): {missing[:6]} ...")
    if extra:
        print(f"in the figure but not in the data ({len(extra)}): "
              f"{extra[:6]} ...")
        print("  (three of these are the threshold markers and are expected)")
    ok = not missing and len(extra) <= 3
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
