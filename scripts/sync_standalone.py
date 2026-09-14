"""Rebuild fig_threshold_standalone.tex from the in-paper fig_threshold.tex so
the exported PDF/SVG/PNG can never drift from what the paper prints.

Takes the tikzpicture body verbatim, swaps \columnwidth for an explicit length
(the standalone class does not set it up like an article class), and strips
blank lines inside the picture -- they end the paragraph and pgfplots' float
parser dies on that under standalone.
"""
import os
import re

os.chdir(r"C:\Users\adeel\OneDrive\Desktop\AAMAS CONFERNCE PAPER")
src = open("fig_threshold.tex", encoding="utf8").read()

i = src.index("\\begin{tikzpicture}")
j = src.index("\\end{tikzpicture}") + len("\\end{tikzpicture}")
body = src[i:j]
body = body.replace("0.99\\columnwidth", "\\figwd")
body = "\n".join(l for l in body.split("\n") if l.strip() != "")

PRE = r"""%% Standalone build of the paper's Figure 2, for export to PDF / SVG / PNG.
%% GENERATED from fig_threshold.tex -- do not edit by hand; re-run
%% scripts/sync_standalone.py (or paste the tikzpicture body again) instead.
%%
%%   pdflatex fig_threshold_standalone            -> .pdf  (vector, cropped)
%%   pdftocairo -svg fig_threshold_standalone.pdf -> .svg  (vector)
%%   pdftoppm -r 600 -png ...                     -> .png  (600 dpi)
%% 9pt matches the ACM sigconf body size, so \scriptsize and 	iny inside
%% the picture come out the same physical size as they do in the paper.
%% The asymmetric border leaves room for the (a)/(b) tags, which sit just
%% inside the right-hand axis line and would otherwise be cropped.
\documentclass[border={2pt 2pt 12pt 2pt}]{standalone}
\usepackage[T1]{fontenc}
%% lmodern is scalable, so the 9pt base size below is set exactly rather than
%% substituted to the nearest available bitmap size.
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{tikz}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
%% arrows.meta: panel (b)'s crossing bars use {Bar[width=...]} end caps.  The
%% paper's preamble loads this for the architecture figure, so the figure body
%% may assume it; without it here the standalone export fails with eight
%% "I do not know the key 'width'" errors while main.pdf builds clean.
\usetikzlibrary{arrows.meta}
\pgfplotsset{compat=1.18}

%% The ACM sigconf column is 241.14749pt; 0.99 of it is 238.736pt.
\newlength{\figwd}
\setlength{\figwd}{238.736pt}

\definecolor{cteal}{RGB}{31,119,140}
\definecolor{cochre}{RGB}{201,120,49}
\definecolor{cclay}{RGB}{168,86,74}

\begin{document}
%% 9pt is the ACM sigconf body size.  Without this the picture inherits
%% article's 10pt, every \scriptsize label grows by a tenth, and panel (b)'s
%% y label -- which only just fits its 2.15cm panel in the paper -- overflows.
\fontsize{9}{11}\selectfont
"""
POST = "\n\\end{document}\n"
open("fig_threshold_standalone.tex", "w", encoding="utf8").write(PRE + body + POST)
print("regenerated fig_threshold_standalone.tex,",
      len(body.split("\n")), "body lines")
