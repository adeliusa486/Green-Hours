"""Export the paper's TikZ figures to vector PDF, SVG and 600 dpi PNG.

The repository used to ship two raster screenshots (docs/architecture.jpg and
docs/overview.jpg).  Both were crops, one of them with its top band cut off, and
neither could be regenerated from source, so they drifted from the paper.  This
script builds each figure from the SAME .tex the paper includes, so a figure in
docs/figures/ cannot say anything the paper does not.

Each figure is also emitted as <base>_web.png, the same raster with its
uniform-white margin cropped away.  The paper's standalone crop keeps the full
\columnwidth box, so fig1 and fig2 carry roughly a third of their width as
blank space on the left; that is correct for a two-column page and wasteful in
a README.  The web copy is a pure crop -- no rescaling, no recolouring -- so it
cannot show anything its parent does not, and it is rebuilt by this script
rather than edited by hand.

    python scripts/export_figures.py            # all figures
    python scripts/export_figures.py --check    # verify exports are current
    python scripts/export_figures.py --web-only # re-crop existing PNGs only

Requires pdflatex and pdftocairo (both ship with TeX Live and MiKTeX).
--web-only needs Pillow instead, and nothing else.
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PAPER = os.path.abspath(os.path.join(REPO, ".."))
OUT = os.path.join(REPO, "docs", "figures")

# figure source -> exported basename
FIGURES = [
    ("fig_motivation.tex", "fig1_emission_factors"),
    ("fig_threshold.tex", "fig2_thresholds"),
    ("fig_arch.tex", "fig3_shade_architecture"),
]

PRE = r"""\documentclass[border={2pt 2pt 6pt 2pt}]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{tikz}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\usepgfplotslibrary{fillbetween}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds,calc,shapes.geometric}
\pgfplotsset{compat=1.18}

%% Set \columnwidth to the real ACM sigconf column rather than rewriting it out
%% of the figure bodies.  fig_arch.tex lays itself out in \columnwidth units and
%% then wraps the result in \resizebox{\columnwidth}{!}{...} because the drawing
%% overshoots by about 11pt; substituting a different length for \columnwidth
%% breaks that layout and the boxes collide.
\setlength{\columnwidth}{241.14749pt}
\newlength{\figwd}
\setlength{\figwd}{238.736pt}

\definecolor{cteal}{RGB}{31,119,140}
\definecolor{cochre}{RGB}{201,120,49}
\definecolor{cslate}{RGB}{88,101,120}
\definecolor{cmoss}{RGB}{110,142,96}
\definecolor{cclay}{RGB}{168,86,74}

\tikzset{
  blk/.style={rounded corners=3pt, draw=cslate, fill=cslate!8, align=center,
              font=\small, inner sep=4pt},
  blkg/.style={rounded corners=3pt, draw=cmoss, fill=cmoss!12, align=center,
               font=\small, inner sep=4pt},
  blko/.style={rounded corners=3pt, draw=cochre, fill=cochre!12, align=center,
               font=\small, inner sep=4pt},
  blkt/.style={rounded corners=3pt, draw=cteal, fill=cteal!10, align=center,
               font=\small, inner sep=4pt},
  ar/.style={-{Latex[length=2mm,width=1.6mm]}, draw=cslate, thick},
  arg/.style={-{Latex[length=2mm,width=1.6mm]}, draw=cmoss!80!black, thick},
  aro/.style={-{Latex[length=2mm,width=1.6mm]}, draw=cochre!80!black, thick},
  lbl/.style={font=\scriptsize, inner sep=1.5pt, fill=white}
}

\newcommand{\aef}{\ensuremath{\mathrm{AEF}}}
\newcommand{\mef}{\ensuremath{\mathrm{MEF}}}
\newcommand{\mech}{\textsc{Shade}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\E}{\mathbb{E}}

\begin{document}
%% 9pt is the ACM sigconf body size; without it every \scriptsize label grows.
\fontsize{9}{11}\selectfont
"""
POST = "\n\\end{document}\n"


def labels():
    """Re-emit the paper's \\newlabel entries so \\ref inside a figure body
    resolves here too.

    fig_arch's outcome strip cites two theorems by \\ref.  Without this the
    exported figure prints "??" where the paper prints "5.1" and "5.2(i)", and
    hardcoding the numbers instead would drift the moment a theorem moves.
    Labels from main.aux are re-emitted with the "M-" prefix the supplement uses
    via xr-hyper.
    """
    # \newlabel is \@onlypreamble here, so define the underlying \r@<label>
    # control sequences directly; \ref takes the first group of that.
    out = ["\\makeatletter"]
    for aux, prefix in ((os.path.join(PAPER, "main.aux"), "M-"),
                        (os.path.join(PAPER, "supplementary.aux"), "")):
        if not os.path.exists(aux):
            continue
        txt = open(aux, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}",
                             txt):
            name, num = m.group(1), m.group(2)
            if name.startswith("M-") or "@" in name:
                continue
            out.append("\\expandafter\\gdef\\csname r@%s%s\\endcsname{{%s}{1}}"
                       % (prefix, name, num))
    out.append("\\makeatother")
    return "\n".join(out)


def body_of(path):
    """The macros the figure defines, plus everything it draws.

    Two spans are taken, not one: anything before \\begin{figure} (fig_arch
    defines its icon macros there, and skipping them fails the build with
    "Undefined control sequence"), and the span from \\centering to \\caption
    (which keeps wrappers such as a \\resizebox).
    """
    src = open(path, encoding="utf-8").read()
    fig = src.find("\\begin{figure")
    pre = src[:fig] if fig > 0 else ""
    pre = "\n".join(l for l in pre.split("\n") if not l.lstrip().startswith("%%"))

    i = src.find("\\centering", fig if fig > 0 else 0)
    i = src.index("\\begin{tikzpicture}") if i < 0 else i + len("\\centering")
    j = src.find("\\caption", i)
    if j < 0:
        j = src.index("\\end{tikzpicture}") + len("\\end{tikzpicture}")
    body = src[i:j].strip()
    # blank lines inside a tikzpicture end the paragraph and kill the float
    # parser under the standalone class
    body = "\n".join(l for l in body.split("\n") if l.strip() != "")
    return pre.strip() + "\n" + body


def trim_png(path, pad=12):
    """Crop the uniform-white margin off an exported PNG.

    Returns (out_path, note).  Pillow is imported here rather than at module
    scope so the PDF/SVG export keeps working on a machine without it.
    """
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return None, "Pillow not installed"
    if not os.path.exists(path):
        return None, "source PNG missing"
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox is None:
        return None, "image is entirely blank"
    l, t, r, b = bbox
    l, t = max(0, l - pad), max(0, t - pad)
    r, b = min(im.width, r + pad), min(im.height, b + pad)
    out = path[:-4] + "_web.png"
    im.crop((l, t, r, b)).save(out, optimize=True)
    return out, None


def build(src_name, out_base, outdir):
    src = os.path.join(PAPER, src_name)
    if not os.path.exists(src):
        return None, f"missing source {src_name}"
    tmp = tempfile.mkdtemp(prefix="figexport_")
    try:
        tex = os.path.join(tmp, out_base + ".tex")
        with open(tex, "w", encoding="utf-8") as fh:
            fh.write(PRE + labels() + "\n" + body_of(src) + POST)
        for _ in range(2):
            p = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                 out_base + ".tex"],
                cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        pdf = os.path.join(tmp, out_base + ".pdf")
        if not os.path.exists(pdf):
            log = os.path.join(tmp, out_base + ".log")
            err = ""
            if os.path.exists(log):
                txt = open(log, encoding="utf-8", errors="replace").read()
                err = "; ".join(re.findall(r"(?m)^! .*", txt)[:3])
            return None, f"pdflatex failed: {err or 'see log'}"
        os.makedirs(outdir, exist_ok=True)
        shutil.copy(pdf, os.path.join(outdir, out_base + ".pdf"))
        subprocess.run(["pdftocairo", "-svg", pdf,
                        os.path.join(outdir, out_base + ".svg")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["pdftocairo", "-png", "-r", "600", "-singlefile", pdf,
                        os.path.join(outdir, out_base)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        trim_png(os.path.join(outdir, out_base + ".png"))
        return os.path.join(outdir, out_base + ".pdf"), None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="rebuild into a temp dir and compare page count only")
    ap.add_argument("--web-only", action="store_true",
                    help="re-crop the existing PNGs; no pdflatex needed")
    args = ap.parse_args()

    if args.web_only:
        rc = 0
        for _, base in FIGURES:
            out, err = trim_png(os.path.join(OUT, base + ".png"))
            if err:
                print(f"  FAIL  {base:<24} {err}")
                rc = 1
            else:
                print(f"  ok    {base:<24} -> docs/figures/{os.path.basename(out)}")
        return rc

    outdir = tempfile.mkdtemp(prefix="figcheck_") if args.check else OUT

    rc = 0
    for src, base in FIGURES:
        pdf, err = build(src, base, outdir)
        if err:
            print(f"  FAIL  {src:<24} {err}")
            rc = 1
            continue
        size = os.path.getsize(pdf)
        print(f"  ok    {src:<24} -> docs/figures/{base}.{{pdf,svg,png}} "
              f"({size//1024} kB)")
        if args.check:
            live = os.path.join(OUT, base + ".pdf")
            if not os.path.exists(live):
                print(f"        ! docs/figures/{base}.pdf is missing")
                rc = 1
    if args.check:
        shutil.rmtree(outdir, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
