# Paper1: normalized spectral risk calibration and diagnosis

One question: **how does node-induced dependence change normalized spectral
risk, and which randomness law must a calibration procedure preserve?**

The argument is normalized influence → law-specific rank calibration →
real-run mismatch diagnosis. Directions, amplitude scans, analytic tail-bound
prototypes and downstream weighting are supporting diagnostics or future work,
not additional methods.

## Read first

- [Claims and evidence](docs/claims_evidence.md): three main claims M1–M3.
- [Status](docs/status.md): measured results, repaired code and unexecuted work.
- [Gate report](docs/gate_ab_report.md): regenerated historical measurements
  with corrected interpretation.
- [Evidence audit](results/evidence_audit.json): reconstructed counts,
  conditional confidence intervals and raw-result hashes.
- [Repair record](docs/repair_record.md): changes and fresh validation.
- [Manuscript integration](docs/manuscript_integration.md): a single argument
  and the placement of supporting analyses.
- [Operator contract](docs/operator_contract.md) and
  [theory notes](docs/theorem_notes.md): definitions, proofs and limits.

## Available evidence

S1–S4 support normalized influence and model-law calibration. S6 studies a
fitted synthetic law. S7/S8 compare LP, prototype and cross-fit GCN run pools.
S11 already includes fixed-split initialization-only GCN on Cora and DBLP,
with 60 and 40 evaluation runs. S12 is a separate-seed rerun of selected
synthetic-law experiments; it does not add independent L3 training runs.

Rank validity averages over calibration and a new exchangeable observation,
conditional on the independent reference stage. It does not require the
conditional exceedance rate of every realized threshold to equal alpha.
Model-law node calibration does not transfer automatically to trained
predictors. Run-pool is the practical baseline for the mechanisms tested.

## Visual alarm validation (completed 2026-10-21)

Two released ED-HNN preprocessed object hypergraphs, NTU2012 and ModelNet40,
were used for an external calibration/detection/application check. The
frozen protocol is [docs/visual_alarm_protocol.md](docs/visual_alarm_protocol.md);
the formal resource budget is 30 reference / 99 calibration / 200 evaluation
complete runs per dataset (658 training runs, one batch). Results are in
`results/visual_alarm_v1/analysis/` and `results/visual_alarm_v1/report/`.

Headline findings, retained as mixed/negative evidence:

- At alpha=5%, the full-run L3 clean FPR was 14.0% on NTU (28/200) and 5.5%
  on ModelNet (11/200); L2 node-product was 0/200 on both; edge-iid was 5.0%
  on NTU and 26.5% on ModelNet. The L2 conservative direction replicates, but
  L3 is not uniformly nominal at 99 calibration runs and the edge failure
  direction is domain-dependent.
- The NTU 14% is one realized threshold: 10,000 resplits of the same 299
  scores give a 97.5th percentile of 11.0%; uniform rounding leaves 13.5%,
  so near-ties explain only 0.5 pp. A post-hoc tolerance rule (k=98) gives
  9.5%; it is exploratory, not a repair. These are internal split-sensitivity
  diagnostics, not new runs.
- At sigma=1, nominal T alarm rates are 70.5%/62.5%, but at a matched 10%
  realized FPR the retrospective ROC gives T detection 65.5% vs Frobenius
  92.5% on NTU and 66.5% vs 100% on ModelNet. T is not superior at a fair
  operating point; the ROC is in-sample, not a deployment rule.
- Only NTU at sigma=1 has >=3pp task drops (6/200). T catches all six among
  141 alarms (4.3% precision); at sigma<=0.5 there are no task events despite
  14--34% alarm rates.
- The fixed refinement is structurally class-preserving: active edges are
  homogeneous in Y, S(Y) is nonnegative with a positive diagonal, so all
  lambda in [0,1] preserve the predicted class (Proposition H.2). The five
  policies are identical by construction; this is a design limitation, not
  evidence of operational value or insufficient training.

Commands (no data download required if the four vision files are present):

```powershell
python scripts/s14_visual_run_cache.py --config configs/visual_alarm_v1.json --stage formal --dataset ntu2012 --batch 1 --jobs 6
python scripts/s14_visual_run_cache.py --config configs/visual_alarm_v1.json --stage formal --dataset modelnet40 --batch 1 --jobs 6
python scripts/s15_visual_alarm_analysis.py --protocol results/visual_alarm_v1/protocol_frozen.json
python scripts/s15b_threshold_stability.py
python scripts/s15c_equal_fpr.py
python scripts/s15d_cache_manifest.py
python scripts/s16_visual_alarm_report.py --results results/visual_alarm_v1
```

The 459 MB run cache is not redistributed with the analysis bundle;
`results/visual_alarm_v1/cache_manifest.json` lists all 660 cache-file
SHA-256 values. The analysis/report CSV/JSON files are the audit trail for
the reported tables, but rerunning s15 itself requires the cache or retraining
under the frozen protocol.

## Reproduce the repair (no training)

Run from this directory:

```powershell
python -m pytest tests -q
python scripts/audit_evidence.py
python scripts/build_gate_report.py
```

The audit leaves the historical result CSV/JSON files unchanged. It replaces
the pooled-binomial inference for S12-C1 with a conditional Hoeffding interval,
and supplies the missing S12-C3 intervals. C2 is descriptive, not a confirmed
directional advantage.

## Optional scientific reruns

Historical inputs remain in results/*.csv. The scripts changed by this repair
write separate v2 files so old and new direction pipelines are not mixed:

| Script | Future output | Status |
|---|---|---|
| s9b_directional_detection.py | directional_detection_v2.csv | not rerun |
| s9c_dependence_dial.py | dependence_dial_v2.csv | not rerun |
| s10a_amplitude_scan.py | amplitude_scan_v2.csv | actual matrix remainder; not fully rerun |
| s10b_two_sided.py | two_sided_detection_v2.csv | not rerun |
| s12_new_seed_confirm.py | confirm_results_v2.csv | corrected inference; not rerun |

The Cora L1 calibration component was rerun separately with three new seeds;
`s12_cora_batch_detail.py` preserves each batch's count, threshold, interval,
and seed in `cora_calibration_batches_20271021.csv` plus a JSON summary. This
is an unregistered descriptive follow-up; it does not claim prospective
confirmation or replace the historical aggregate in the gate report.

The historical gate report deliberately reads the original CSVs and labels
their limitations. A future v2 report must explicitly select and validate its
new inputs. The historical amplitude-scan residual column was a scalar
norm-gap lower bound, not a matrix-remainder norm; the claimed effective
amplitude range is withdrawn.

S2/S3/S4 can still be rerun using their existing scripts, but they can append
to historical result files. Use an isolated copy of the experiment for a full
reproduction. No expensive scientific rerun is needed for the three repair
commands above.

Existing raw evidence, the manuscript and submission packages are preserved.
The pre-repair docs/scripts and result hashes are stored under
archive/20261021-before-followup-repair/. The first manuscript is now integrated in paper/paper1/main.tex and main.pdf
(27 pages); the current local source/code/result bundle is available as
release/paper1-visual-alarm-20261021.zip. The source-to-table audit,
41 tests, and independent unpacked PDF compilation passed. The visual
resource-budget validation adds 658 complete training runs; its L3 FPR,
baseline comparison and inert-action null are reported without an
operational-benefit claim. Stronger numerical
certification, Cora/DBLP prospective L3 confirmation, the visual full
three-batch budget and full novelty screening remain
separate scientific work described in docs/followups.md.
