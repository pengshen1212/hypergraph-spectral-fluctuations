"""Assemble this revision from an immutable original and reviewed new sections.

This script intentionally uses the archived original. It is a one-revision
builder, not a general manuscript editor. Preserve manual main.tex edits before
rerunning it. The original theorem, proof and table blocks remain byte-exact
after newline normalization.
"""
from pathlib import Path
import re

PAPER = Path(__file__).resolve().parents[2]
ARCHIVE = PAPER / "archive/revisions/20261021-before-normalized-calibration"
old = (ARCHIVE / "main.tex").read_text(encoding="utf-8")

title = "Calibrating Normalized Hypergraph Spectral Risk under Node-Induced Dependence"
abstract = r"""Prediction-constructed hypergraphs couple incident relations through
shared node predictions, making edge marginals insufficient for spectral-risk
assessment. We connect normalized conditional influence, calibration under a
declared prediction law, and complete-run diagnostics. Conditional
expectations decompose the normalized operator into orthogonal node
responses and a remainder; a three-operator comparison separates fixed
degree weighting from dynamic normalization. On Cora-CC and DBLP-CA,
computed centered-norm ratios between node-induced and independent-edge
models decreased from 2.15--3.47 before normalization to 1.03--1.26
afterward, while a coherent controlled family reversed the comparison.
Under specified node laws at a 5\% target, node-calibrated native
exceedance means were 3.9\%--5.9\%, compared with 47.5\%--100\% for
independent-edge thresholds. With trained predictors, fitted node-product
thresholds instead yielded zero observed exceedances in the tested
settings. This conservative direction extended to NTU2012 and ModelNet40,
with zero exceedances in 200 evaluation runs per visual dataset;
independent-edge calibration yielded 26.5\% on ModelNet40. Complete-run
and split-sensitivity comparisons distinguish marginal rank validity
from the conditional behavior of a frozen threshold. A standard rank
argument supplies finite-sample validity under the declared exchangeable
score law. Together, the analysis and cross-domain comparisons identify
which joint dependence a calibration model must preserve and separate
the probability targets needed for upstream operator monitoring."""

front = old[:old.index(r"\section{Introduction}")]
front = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}",
               lambda _: "\\begin{abstract}\n"+abstract+"\n\\end{abstract}",
               front, count=1, flags=re.S)
front = front.replace("Node-Induced Dependence and Spectral Fluctuations in Estimated Hypergraph Relations", title)
front = front.replace("Node-Induced Dependence and Spectral Fluctuations\nin Estimated Hypergraph Relations", title)
front = front.replace(r"\usetikzlibrary{arrows.meta}",
                      r"\usetikzlibrary{arrows.meta,positioning}")
front = front.replace(r"\usepackage{tikz}",
                      r"\usepackage{tikz}"+"\n"+r"\usepackage{float}"+"\n"+r"\usepackage{graphicx}")

raw = old[old.index(r"\section{Model and Matched Controls}"):
          old.index(r"\section{Experiments}")]
raw_figure = re.search(r"\\begin\{figure\}.*?\\end\{figure\}", raw, re.S).group(0)
raw = raw.replace(raw_figure, "")
raw = raw.replace(r"\subsection{", r"\subsubsection{")
raw = raw.replace(r"\section{", r"\subsection{")
raw = raw.replace(r"\subsection{Model and Matched Controls}\label{sec:model}",
                  r"\section{Raw-Operator Model and Mechanism}\label{sec:raw-mechanism}\label{sec:model}"+
                  "\nThe following raw-operator theory supplies the mechanism foundation. "
                  "Its matched expectations and separation claims concern the unnormalized operator; "
                  "the normalized analysis in Section~\\ref{sec:normalized-influence} does not assume that these conclusions transfer.\n")
raw = raw.replace("Fix a candidate hypergraph", raw_figure+"\n\nFix a candidate hypergraph", 1)

raw_exp = old[old.index(r"\subsection{Controlled structure families}"):
              old.index(r"\section{Discussion and Limitations}")]
raw_exp = (r"\section{Raw-Operator Validation and Predictor Comparisons}\label{sec:raw-validation}"+
           "\nThese retained measurements concern the raw relation operator and the original per-edge calibration protocol. "
           "They are distinct from the normalized calibration experiments in Section~\\ref{sec:exp}. "
           "The Walmart inputs use the categorical benchmark associated with "
           "\\cite{amburg2020categorical}. All original result tables are retained below.\n\n" + raw_exp)

declarations = old[old.index(r"\section*{CRediT Authorship"):
                   old.index(r"\appendix")]
availability = r"""\section*{Data and Code Availability}
The complete source code, frozen configurations, certificates, saved
measurements, and reproducibility instructions for both the normalized
calibration experiments and the raw-operator mechanism checks are publicly
available at\\
\url{https://github.com/pengshen1212/hypergraph-spectral-fluctuations}.
The normalized calibration materials are also provided in the accompanying
submission package.
Frozen visual prediction caches are available separately as a
\href{https://github.com/pengshen1212/hypergraph-spectral-fluctuations/releases/tag/paper1-visual-cache-20261022}{GitHub Release attachment},
with per-file SHA-256 values in the repository's cache manifest.
Cache-based reanalysis also requires the original third-party inputs.
Cora-CC and DBLP-CA inputs are available through
\href{https://github.com/jianhao2016/AllSet}{AllSet}; preprocessed
NTU2012 and ModelNet40 visual hypergraphs are obtained from the public
\href{https://github.com/Graph-COM/ED-HNN}{ED-HNN benchmark repository}~\cite{wang2023edhnn}.
The retained raw
Walmart experiments use
\href{https://www.cs.cornell.edu/~arb/data/walmart-trips/}{Austin R. Benson's dataset collection}.
Third-party inputs are not redistributed in the revision package.

"""
appendix = old[old.index(r"\appendix"):]
marker = r"\section{Proofs of the Influence Theorems}"
appendix = appendix.replace(marker, raw+"\n"+raw_exp+"\n\\clearpage\n"+marker, 1)
appendix = appendix.replace(r"\end{document}",
                           "\\input{sections/calibration-appendix}\n\\end{document}")
result = front + "\\input{sections/calibration-main}\n\n" + availability + declarations + "\\clearpage\n" + appendix
(PAPER / "main.tex").write_text(result, encoding="utf-8")

additions = r"""

@inproceedings{wang2023edhnn,
  author = {Wang, Peihao and Yang, Shenghao and Liu, Yunyu and Wang, Zhangyang and Li, Pan},
  title = {Equivariant Hypergraph Diffusion Neural Operators},
  booktitle = {International Conference on Learning Representations},
  year = {2023},
  url = {https://arxiv.org/abs/2207.06680}
}

@inproceedings{vovk2012conditional,
  author = {Vovk, Vladimir},
  title = {Conditional Validity of Inductive Conformal Predictors},
  booktitle = {Proceedings of the Asian Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {25},
  pages = {475--490},
  year = {2012},
  url = {https://proceedings.mlr.press/v25/vovk12.html}
}

@inproceedings{angelopoulos2024crc,
  author = {Angelopoulos, Anastasios N. and Bates, Stephen and Fisch, Adam and Lei, Lihua and Schuster, Tal},
  title = {Conformal Risk Control},
  booktitle = {International Conference on Learning Representations},
  year = {2024},
  url = {https://research.google/pubs/conformal-risk-control/}
}
"""
(PAPER / "references.bib").write_text(
    (ARCHIVE / "references.bib").read_text(encoding="utf-8").rstrip()+additions,
    encoding="utf-8")
print("Integrated mainline; original raw theorem/proof/table blocks moved to appendices")


if __name__ == "__main__":
    pass
