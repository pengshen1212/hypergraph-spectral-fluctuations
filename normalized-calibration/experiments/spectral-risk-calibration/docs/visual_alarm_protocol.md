# Visual object-hypergraph alarm validation: frozen protocol

Snapshot: 2026-10-21, post-freeze diagnostics 2026-10-22.  Status: protocol frozen;
both formal resource caches complete (30/99/200 per dataset, 658 formal
training runs, zero failures); analysis/report and the no-training threshold
and equal-FPR diagnostics generated.  This document records the executed
protocol and its deviations.  Numeric conclusions are in
`results/visual_alarm_v1/analysis/` and `results/visual_alarm_v1/report/` and
must be quoted from those files, not from this narrative.

Frozen artifact: `results/visual_alarm_v1/protocol_frozen.json`
(SHA-256 recorded in `analysis_summary.json`).  Formal runners refuse to start
without it and refuse caches whose embedded configuration hash differs.

## 1. Data assets

| Dataset | Objects | Classes | Features | Hyperedges | Released slot size |
|---|---:|---:|---:|---:|---:|
| NTU2012 | 2,012 | 67 | 100 | 2,012 | 5 |
| ModelNet40 | 12,311 | 40 | 100 | 12,311 | 5 |

Input SHA-256 (verified at every load and before freeze):

| file | SHA-256 |
|---|---|
| NTU2012.content | `f59d6f62c94d1ef4861a95ddcd76c24411d7deb6fd4f2777e40ea83563630d2c` |
| NTU2012.edges | `3de34493d0e37b2d1b7d5df35e0d9246e4d3dff0de39ec4fa7f16abdfacee8b8` |
| ModelNet40.content | `51565c050db09532ea16457bb04958aa5dec8acaab1f9009df47771ce76b8604` |
| ModelNet40.edges | `e6e0ae37a39369b4a55a230554148e0187c58ee47f6e650b62f5436c6eac4048` |

`vision_data.py` verifies, not assumes, the released orientation:

- `.content` field 1 is the ID; the first `n_objects` IDs are classification
  objects.  Remaining rows are auxiliary hyperedge-node representations and are
  never classified.
- `.edges` columns are `object_id  hyperedge_id`.  Hyperedge IDs occupy
  `[n_objects, 2*n_objects)`.  Every hyperedge must contain exactly five
  distinct object IDs.
- Object rows are selected by ID, so shuffled content rows are parsed
  correctly; missing IDs, duplicate IDs, duplicate incidences and illegal
  references fail loudly.
- Published edge slots and their order are preserved; members inside a slot
  are sorted only to canonicalise an unordered set.
- ModelNet labels 1--40 are remapped to 0--39; NTU labels already 0--66.

The setting is the released preprocessed transductive object hypergraph.  It is
not raw point-cloud end-to-end training.  The 100-dimensional features are the
distributed ED-HNN preprocessing, not untreated MVCNN/GVCNN features.  Because
the graph and features were released for transductive use, the protocol cannot
claim that the publisher's preprocessing used only training objects.

## 2. Fixed split and predictor

- Stratified 60/20/20 split, seed `202610210`; per-class quotas
  `floor(0.6 n_c) / floor(0.2 n_c) / remainder`.
- NTU2012: 1,182 train / 377 validation / 453 test; fingerprint
  `53b03d02cfb3a5965a506c99baf0241954bda94b00bb7672ef07ecc403292577`.
- ModelNet40: 7,372 train / 2,446 validation / 2,493 test; fingerprint
  `6c41c51eb545fb528acca7e533a058e5bcb1e776f837afed2b6c8b2cb085ddea`.
- Features standardised with TRAIN-node mean/std only; zero-variance
  dimensions get scale 1.
- Predictor: `logits = S_all ReLU(S_all X W1) W2`, hidden 64, no bias,
  no dropout, full batch, Adam, learning rate 0.01, weight decay 5e-4 as L2,
  cross-entropy on train nodes only; checkpoint by validation loss.
- Randomness per run is the weight-initialisation seed.  Node split, graph,
  preprocessing and training protocol are fixed.
- Predictor epoch limit and learning rate selected on 10 development runs per
  dataset (batch 0, role `dev`), before any formal run:
  - NTU2012: lr 0.01, 100 epochs (checkpoint at 35--45).
  - ModelNet40: lr 0.01, 400 epochs (validation loss still lowest at 400;
    validation accuracy is marginally higher at 200, but the protocol selects
    by validation loss).
  - No development run diverged, so no learning-rate fallback was used.

Development runs are not part of the confirmation pool and are not used for
confirmation claims.

## 3. Monitored operator and scores

The monitored operator is built from the model's prediction vector, not from
the fixed predictor matrix:

```
a_e(Y) = 1{all predictions on edge e agree}
S(Y)   = D(Y)^(-1/2) (sum_e a_e H_e + I) D(Y)^(-1/2),   kappa = 1
T      = || S(Y) - M_hat ||_2
```

`M_hat` is the elementwise mean of `S(Y)` over the reference runs of the same
batch and is frozen.  `S_all` (all candidate edges active) is only the fixed
predictor propagation matrix; it is never described as varying across runs.

Scores analysed together:

1. `T`: primary two-sided spectral norm.
2. `Frobenius`: `||S(Y) - M_hat||_F`, a diffuse-change baseline.
3. `entropy`: mean predictive entropy over non-train objects (no labels).
4. `active_dev`: `|active-edge fraction - reference median|`.
5. `U_two_sided`: `|T - m_ref|`, `m_ref = median` reference `T`; frozen
   secondary sensitivity score, not a replacement for `T`.

Spectral norms are two-sided.  Dense `eigvalsh` is used only for `n <= 600`;
both datasets use `eigsh` with a fixed start vector.  Ends, residuals, iteration
settings and fallback status are recorded.  No solver fallback occurred in the
two pilots; formal residuals are written to every cache file.

## 4. Calibration arms and runs

Three calibrators share the same frozen center and the same evaluation runs:

- `L3_full_run`: 99 independent calibration training runs (resource budget).
  Each contributes the score of its complete prediction vector.
- `L2_node_product`: node class frequencies fitted from the 30 reference runs
  with uniform Dirichlet mass 1; 99 independent complete node vectors sampled.
  This is a fitted-law guarantee only.
- `edge_iid`: edge probabilities from the same fitted node law; 99 independent
  edge-flag vectors.  This is a counterfactual misspecification control.

Runs are partitioned into disjoint reference / calibration / evaluation seed
streams.  Multiple inferences, nodes, resampling draws or repeated predictions
of one model never count as independent training runs.  The resource budget
(30/99/200, one batch) was selected by the pre-registered rule because the
projected full budget (3 batches, 50/199/400 per dataset, 3,894 training runs)
was 31.8 projected hours, above the 8-hour limit; projected full cache was
2.47 GB, below the 6 GB limit.  Resource-budget results are preliminary
cross-domain evidence, not equal in confirmation strength to the full budget.

Seeds:

```
seed = 2026102100 + dataset_index*100000 + batch*10000 + role*1000 + run_index
role: ref=1, cal=2, eval=3, dev=8, pilot=9
dataset_index: NTU=1, ModelNet=2
```

Noise uses a separate `SeedSequence` namespace (`noise_seed_base=20261023`).
Bootstrap seed is `20261024`.

## 5. Feature stress

The candidate graph, labels, train mask, trained weights, center and thresholds
are frozen.  For every evaluation run one noise field `Z` is drawn once and
shared across strengths:

```
X_std(sigma) = X_std + sigma * Z,   Z ~ N(0,1)
Z[u, :] = 0 on train objects and on zero-variance feature dimensions
sigma in {0, 0.1, 0.25, 0.5, 1.0}
```

Only non-train object features are perturbed.  Standardisation is not
recomputed.  Reference and calibration predictions stay clean, so nominal FPR
control is not claimed under shift.  This is a fixed-graph objective-feature
perturbation, not raw point-cloud geometry and not a marginal-preserving pure
dependence intervention.

## 6. Task association and fixed action

Task events are defined offline relative to the same run's clean prediction:
test accuracy drop `>= 3` percentage points.  Alarms are run-level; this is not
node-level error localisation.  If no event occurs, detection/precision are
reported as not estimable.

Fixed action (selected on development validation nodes, then frozen):

```
P_refine = row_normalize((1-lambda) P + lambda S(Y) P)
lambda in {0, 0.25, 0.5, 0.75, 1.0}
```

With records:

```
if alarm:  output P, flag the relation-refinement step for review
else:      output P_refine
```

Development selection chose `lambda = 0` for BOTH datasets because all
candidate lambdas produced exactly the same validation accuracy (NTU
0.864721; ModelNet 0.958913, mean over 10 runs).  At `lambda=0` the refinement
is the identity, so the five application policies (base, always refine,
spectral fallback, entropy fallback, equal-count random fallback) are
mathematically identical.  This is retained as a negative/inert result; the
protocol forbids changing lambda to a more dangerous value to manufacture a
fallback gain.  A non-registered development diagnostic additionally found that
even `lambda=1` did not change any validation prediction at any of the five
strengths on NTU.  Consequently this action cannot demonstrate operational
value; the application test is degenerate and will be reported as such.

## 7. Statistics

- Each batch keeps its own center and threshold.  Per-batch counts and
  Clopper--Pearson intervals are reported; batch rates are combined by equal
  weight.  The endpoint envelope is not a confidence interval for the mean.
- Alpha is 0.05 primary, 0.10 sensitivity.  Detection cost curves use
  `0.01/0.025/0.05/0.10/0.20`; all thresholds come from the clean calibration
  pool.
- Rank threshold: `k = ceil((M+1)(1-alpha))`, `q = T_(k)` ascending, alarm iff
  strictly greater.  Ties are conservative and reported.
- Paired comparisons use run IDs, resampling within batch (10,000 bootstrap
  draws), never nodes.  The primary application comparison is spectral
  fallback minus always-refine at `sigma=0.5`, with Holm correction across the
  two datasets.  Sign-flip permutation p-values accompany bootstrap intervals.
- The intervals describe evaluation uncertainty given the frozen calibration
  artifacts; they do not cover re-drawing the reference/calibration stage or
  new graphs.

## 8. Files and commands

Implementation (all under `experiments/spectral-risk-calibration/`):

| file | role |
|---|---|
| `src/vision_data.py` | ID-explicit loading, hashes, split, noise |
| `src/vision_predictor.py` | two-layer predictor and checkpointing |
| `src/monitoring_workflow.py` | scores, calibrators, refinement/fallback |
| `src/vision_runs.py` | run cache, seeds, atomic resume |
| `configs/visual_alarm_v1.json` | base protocol |
| `scripts/s13_visual_asset_audit.py` | audit / development / pilot / budget decision |
| `scripts/s13b_lambda_inertia.py` | development-only lambda inertness diagnostic (post-freeze) |
| `scripts/s14_visual_run_cache.py` | freeze / pilot / formal cache |
| `scripts/s15_visual_alarm_analysis.py` | E1--E4 analysis from cache |
| `scripts/s15b_threshold_stability.py` | threshold resplit and PAC diagnostics |
| `scripts/s15c_equal_fpr.py` | retrospective equal-FPR ROC comparison |
| `scripts/s15d_cache_manifest.py` | SHA-256 manifest for the external cache |
| `scripts/s16_visual_alarm_report.py` | table, figure, machine verification |
| `tests/test_visual_alarm_protocol.py` | ID, split, rank, isolation, cache tests |

```powershell
python -m pytest experiments/spectral-risk-calibration/tests/test_visual_alarm_protocol.py -q
python experiments/spectral-risk-calibration/scripts/s14_visual_run_cache.py --config experiments/spectral-risk-calibration/configs/visual_alarm_v1.json --stage formal --dataset modelnet40 --batch 1 --jobs 6
python experiments/spectral-risk-calibration/scripts/s15_visual_alarm_analysis.py --protocol experiments/spectral-risk-calibration/results/visual_alarm_v1/protocol_frozen.json
python experiments/spectral-risk-calibration/scripts/s16_visual_alarm_report.py --results experiments/spectral-risk-calibration/results/visual_alarm_v1
```

Resume skips only complete cache files whose embedded config hash matches.
Failures are written as timestamped JSON and are not silently dropped.

## 9. Deviations and implementation notes

1. During the pilot, fresh runs were repeated after a metadata-only RSS field
   was added; pilot results are timing/degeneracy diagnostics only and are not
   used for confirmation.
2. The reference summary is stored at the batch root
   (`batchNN/reference_summary.npz`); an earlier pilot copy under `ref/` is
   superseded.  Formal caches use the batch-root path consistently.
3. The development selection for ModelNet40 uses validation loss, which picks
   400 epochs although mean validation accuracy is 0.00025 higher at 200; no
   post-hoc switch was made.
4. `lambda=0` is a genuine selection outcome, not a placeholder.  The
   application experiment is therefore an inert-action negative result.
5. The pilot's NTU calibration tie mode fraction was 2/19; formal ties are
   reported per score and threshold.
6. After the protocol freeze, the report script (`s16`) gained a compact
   main-text table/figure and a non-estimable-event check.  These are
   presentation/verification changes only; no cache, threshold, score or
   analysis CSV value was altered.  The frozen code hash predates that
   formatting change.
7. A development-only lambda-inertia diagnostic
   (`scripts/s13b_lambda_inertia.py`) was added after the freeze.  It reads
   the ten saved development runs per dataset and applies all lambdas and
   strengths; it does not read or alter formal caches.  It found that even
   `lambda=1` changed no validation prediction at any strength on either
   dataset.  Outputs are `development/lambda_inertia_<dataset>.json`; this is
   explanatory and not a formal confirmation result.
8. After the freeze, `s15b`/`s15c` add no-training diagnostics (threshold
   resplitting, post-hoc tolerance rule, retrospective equal-FPR ROC) and
   `s15d` hashes the external cache.  None changes a formal cache, threshold
   or result file.  The paper records the class-preservation proposition and
   the diagnostic numbers, with their exploratory/in-sample status.
9. The local analysis bundle excludes the 459 MB prediction cache by design;
   `cache_manifest.json` is the verification bridge for an external archive.

## 10. Formal results snapshot (resource budget)

These are the values in the generated CSVs at the snapshot time; see
`analysis/` and `report/` for the authoritative tables and intervals.

Calibration at alpha=5%, 200 clean evaluation runs per dataset:

| dataset | L3 full run | L2 node product | edge independent |
|---|---:|---:|---:|
| NTU2012 | 28/200 = 14.0% (CP 9.5–19.6) | 0/200 (upper CP 1.83) | 10/200 = 5.0% |
| ModelNet40 | 11/200 = 5.5% (CP 2.8–9.6) | 0/200 (upper CP 1.83) | 53/200 = 26.5% |

The L2 conservative direction replicates on both visual datasets.  The L3
arm is not uniformly nominal at 99 calibration runs, and the direction of the
edge-iid failure changes with the dataset.  The NTU L3 over-exceedance is a
realized conditional-threshold effect: the 95th percentile sits just below a
score atom that is more frequent in the evaluation stream.  This does not
violate the joint rank guarantee, but it does mean 99 runs are insufficient
for a stable conditional 5% threshold on that score distribution.

Detection and task association:

- At sigma=1, `T` alarm rates are 70.5% (NTU) and 62.5% (ModelNet);
  Frobenius reaches 87.5%/100%, entropy 26%/100%, and active-edge deviation
  17.5%/7.5%.  `T` is not the best detector, and its NTU advantage is
  accompanied by a 14.0% clean rate.
- A test-accuracy drop of at least 3 percentage points occurs in 6/200 NTU
  runs at sigma=1 and in none of the ModelNet runs.  At sigma=1 NTU, `T`
  alarms on 141 runs and catches all six events, so precision is 4.3%; the
  other 135 alarms have no material drop.  At sigma<=0.5, alarm rates are
  14–32% (NTU) and 5.5–34% (ModelNet) with zero task events.
- The fixed action is inert: development selected lambda=0, so all five
  policies coincide; the paired spectral-fallback minus always-refine
  difference at sigma=0.5 is 0.00 pp [0.00, 0.00] with Holm p=1 on both
  datasets.  No operational benefit is claimed.

Conclusion boundary: the visual check supports the product-law mismatch
direction (L2 conservative on both datasets) but does not establish a
uniformly calibrated spectral alarm, a task-level detector, or a useful
post-alarm correction.  The engineering-value question remains only partially
addressed.

## 11. Post-freeze no-training diagnostics (2026-10-22)

No new training was started.  These analyses read the existing frozen cache
and are retained separately from the formal result.

- **Class preservation.** Active edges are homogeneous in `Y`, `S(Y)` is
  nonnegative with a positive diagonal, and `S(Y)[u,v]>0` only when
  `Y_u=Y_v`.  For every `lambda in [0,1]`, the gated refinement therefore
  preserves the predicted class; a unique base argmax remains unique.  This
  is a structural limitation of the action, not a sample-size effect.  The
  paper records it as Proposition H.2; the empirical five-policy identity is
  retained in the appendix, not emphasized in the main text.
- **Threshold stability.** 10,000 random 99/200 splits of the pooled NTU 299
  scores give mean clean exceedance 5.0%, median 4.5%, 97.5th percentile
  11.0%; the frozen threshold is at the 0.4th percentile of resampled
  thresholds and 4.4% of splits exceed 10%.  Uniform rounding to 6–14
  decimals changes 28/200 to 27/200 (13.5%), so approximate ties explain only
  0.5 pp.  ModelNet is central (mean 5.0%, 97.5th 11.0%).  This is internal
  split sensitivity, not new independent runs; it does not identify a unique
  cause.  Outputs: `analysis/threshold_stability_<dataset>.json`.
- **Conditional-risk tolerance rule.** At alpha=delta=.05 the smallest
  admissible k is 98 (M=99, tail 0.039), 195 (199, 0.027), 483 (499, 0.035)
  and 961 (999, 0.044).  Applied post hoc to the old data, M=99/k=98 gives
  19/200=9.5% on NTU and 3/200=1.5% on ModelNet.  This is exploratory and not
  a pre-registered repair; a future 499/200 NTU protocol would freeze k=483
  before generation.  See `analysis/threshold_stability_*.json`.
- **Equal-FPR detection.** Retrospective in-sample ROC at sigma=1: at 10%
  realized clean FPR, T detects 65.5% vs Frobenius 92.5% on NTU (AUC 0.879
  vs 0.976) and 66.5% vs 100% on ModelNet (AUC 0.925 vs 0.998), with entropy
  100% on ModelNet.  Observed paired differences are -27.0 pp and -33.5 pp,
  with paired-run percentile bootstrap 95% intervals [-36.5,-18.5] and
  [-41.0,-24.0]. The JSON `estimate` is the original-sample contrast;
  `mean` is the bootstrap replicate mean (-27.75697 pp for NTU) and is
  not substituted for the observed estimate. These are score-ranking
  diagnostics, not deployment thresholds.  Outputs:
  `analysis/equal_fpr_roc.csv`, `analysis/equal_fpr_bootstrap.json` and
  `report/figure_visual_alarm_roc.pdf`.
- **Cache boundary.** The 459 MB formal cache is not in the analysis/report
  package.  `cache_manifest.json` lists all 660 cache-file SHA-256 values;
  a recipient can verify a separately transferred cache or rerun training
  under the frozen protocol.  Table/PDF regeneration from stored analysis
  files is not the same as rerunning s15 from predictions.




## Manuscript focus revision (2026-10-22)

The manuscript now focuses on cross-domain calibration and the distinction
between marginal and frozen-threshold conditional risk. Table 5 separates
L3, L2 and edge-iid rates with CP intervals; the two-panel main figure
compares those rates and internal split variability. The class-preservation
proposition, task-association narrative, action panels and duplicate detailed
figure have been removed from the manuscript, while their original results
and development diagnostics remain in this protocol and the audit outputs.
Appendix H.2 contains calibration diagnostics; H.3 retains the secondary ROC
and the different spectral functionals measured by the two operator norms.
No score, threshold, count, input cache or formal analysis CSV was changed.
Repeated partitions of the same 299 scores are not new independent
validation; tolerance-rule evaluation reuses the observed evaluation set.
The original time-budget decision remains in the frozen protocol rather
than in the manuscript narrative.

Figures can be rebuilt without prediction caches: the main figure uses
stored calibration and split summaries; ROC redraw uses
`report/plot_data/equal_fpr_roc_points.csv`. Use `s15c_equal_fpr.py --plot-only`
for the latter. Recomputing s15 from prediction vectors still requires the
separately transferred cache or a full training reproduction.
