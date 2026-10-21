# Hypergraph Spectral Fluctuations

Reproducibility artifacts for **Node-Induced Dependence and Spectral Fluctuations in Estimated Hypergraph Relations** by Shen Peng, Rui Wang, and Yunchao He.

The experiments compare node-driven and independent-edge relation errors with matched per-edge marginals and the same expected operator. The material supports coherent-influence amplification, the flat-influence cancellation boundary, and independent-edge spectral alarm miscalibration under the stated models.

## Install and check

The artifact set was checked with Python 3.12.3, NumPy 1.26.4, and SciPy 1.15.2. Python 3.11 or newer is required by the release runner.

```bash
python -m pip install -r requirements.txt
python run.py verify
python run.py test
```

`verify` checks every released file hash, the frozen native-risk source hashes, all 220 exact conditional-state bounds, and 48,000 saved sampling records. It reconstructs the eight validation envelopes and retains the 49 ambiguous norm intervals and original N4 failed gates. This check uses saved artifacts and does not require third-party data or repeat expensive sampling.

`test` runs the numerical and exhaustive synthetic checks for the influence decomposition and native-risk calculations. Some experiment commands require the original benchmark inputs described below.

## Artifact layout

| Path | Content |
| --- | --- |
| `paper/paper1/experiments/prob2-relation-error/upgrade/` | Influence decomposition, controlled families, frozen protocol, and formal validation results |
| `paper/paper1/experiments/prob2-relation-error/native_risk/` | Exact conditional moments, finite calibration, norm enclosures, native protocols, and N4-R execution |
| `paper/paper1/notes/native-risk/` | Frozen directions and certificates, calibration and evaluation summaries, and all saved N4-R draw intervals |
| `paper/paper1/experiments/prob2-relation-error/results/` | Saved real-predictor and legacy spectral verification results |
| `paper_tables.json` | The seven manuscript tables as preserved numerical LaTeX rows |
| `artifact_manifest.json` | SHA-256 and byte counts for the public artifact snapshot |
| `input_hashes.json` | Exact identity of benchmark input files used in the source workspace |

Source and result files are copied byte-for-byte. The release runner uses Python's AST to relocate only literal paths into the current checkout; the files themselves and their frozen hashes remain unchanged. Historical and pilot records retain their original status and are not additional evidence for the manuscript's claims.

## Run an experiment

Run a preserved script through the release runner so local path literals are relocated:

```bash
python run.py script paper/paper1/experiments/prob2-relation-error/upgrade/run_influence_validation.py --help
python run.py script paper/paper1/experiments/prob2-relation-error/native_risk/n4_repair.py --help
```

The runner sets the working directory to the selected script's directory. Full experiments can be costly and can write output files. Use a separate checkout for a new run; keep the released evidence snapshot for comparison. The 48,000-draw experiment has not been rerun as part of publishing this repository.

## Benchmark data

The original third-party datasets are not redistributed. Cora-CC and DBLP-CA benchmark distributions are available through [AllSet](https://github.com/jianhao2016/AllSet). The Walmart-trips data with **node department labels** are available from [Austin R. Benson's dataset collection](https://www.cs.cornell.edu/~arb/data/). This is distinct from the categorical-edge Walmart dataset.

Place the original files in `data/cora-cc/`, `data/dblp-ca/`, and `data/web-hypergraphs/walmart-trips/` relative to the repository root. Cora and DBLP use `features.pickle`, `hypergraph.pickle`, and `labels.pickle`. Match their SHA-256 values against `input_hashes.json` before replaying a frozen native experiment. If a different upstream version does not match, do not interpret it as reproducing the frozen result.

The manuscript's Walmart experiment uses an induced subgraph on the first 20,000 original node IDs and label propagation; it does not use node features. Dataset permissions and terms remain those of their original providers.

## Scope and reuse

The native certificates concern specified independent-node error laws on two fixed hypergraphs. Real-predictor results are empirical comparisons under per-edge calibration. The material does not establish a universal classification ranking or a replacement alarm-threshold algorithm.

Original code, configurations, and saved result artifacts in this repository are released under the MIT license. Third-party data remain subject to their own terms. The original raw-operator snapshot excludes the manuscript and author photographs. The revised manuscript is included in the normalized-calibration release below; manuscript copyright remains with the authors.

## Normalized calibration release (2026-10-22)

The revised manuscript and the complete normalized-calibration reproducibility
package are in [`normalized-calibration/`](normalized-calibration/).
Read the [manuscript PDF](normalized-calibration/paper/paper1/main.pdf) and
[reproduction instructions](normalized-calibration/README.md).
The root-level commands and evidence directories above retain the original
raw-operator mechanism checks and certificates.

For the normalized release, use Python with NumPy, SciPy, scikit-learn,
matplotlib and pytest, then run:

```bash
cd normalized-calibration
python run.py verify
python run.py test
python run.py figures
python run.py tables
python run.py compile
```

PDF compilation requires a LaTeX installation. The package records 658 formal
visual runs on NTU2012 and ModelNet40, without additional training for this
revision. It preserves the NTU L3 14% clean exceedance result and reports internal
threshold sensitivity separately from independent validation. It makes no
classification-improvement claim.

Third-party datasets are excluded. The 459 MB prediction cache is provided as
a [separate release attachment](https://github.com/pengshen1212/hypergraph-spectral-fluctuations/releases/tag/paper1-visual-cache-20261022)
with archive and per-file SHA-256 values. It is excluded from Git history and
the small reproduction package. Saved measurements rebuild tables, figures
and the PDF; rerunning s15 requires these caches and the original third-party
inputs, or retraining under the frozen protocol. Publishing these artifacts
does not submit the manuscript to a journal.
