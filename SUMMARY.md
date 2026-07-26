# SkyFinder Experiment Summary

Last updated: 2026-06-20 14:07 PDT

All numbers below were independently reproduced from the raw per-fold
predictions on 2026-06-20 with no discrepancies. This revision adds a transfer
comparison to the DIR paper, sharpened tail and per-camera findings,
artifact/reproducibility notes, and a paper-grade ablation TODO.

## Executive Summary

The completed five-fold leave-one-camera-out (LOCO) experiments show that
unseen-camera generalization, rather than in-camera fitting, is the central
problem. The strongest completed image-only method is ResNet-50 with FDS at
8.055 C test MAE, a small 0.161 C (1.97%) improvement over the ResNet-50
baseline. LDS does not improve overall LOCO performance, and combining LDS
with FDS is worse than FDS alone.

The metadata-only model is much stronger than every image-only result: 4.475 C
test MAE. Its inputs are camera ID, hour, month, latitude, and longitude. For
held-out cameras, the one-hot camera-ID feature is unknown and ignored, so the
LOCO gain must principally come from time and location information. An
exploratory post-hoc blend of 90% metadata prediction and 10% FDS prediction
reaches 4.426 C, but the blend weight was selected on the test folds and is not
an unbiased headline result without validation-only selection or nested CV.

More images from the same cameras are unlikely to close the main gap. The
highest-value additional data are new cameras and coverage across locations,
seasons, times of day, weather, hardware, and rare temperature ranges.

## Dataset And Evaluation Protocol

- Dataset: 81,453 labeled images from 47 cameras.
- Target: temperature, spanning -27.2 C to 50.0 C (mean 14.005 C, SD 10.192 C).
- Camera imbalance is substantial: 175 to 4,091 images per camera (median 1,601).
- Headline protocol: five-fold LOCO. Each test fold holds out 9-10 entire
  cameras; each image belongs to the held-out-camera test set exactly once.
- Validation images come from training cameras. Therefore validation MAE measures
  an easier, in-camera setting and must not be used as a proxy for LOCO test MAE.
- Reported `+/-` values below are sample SDs across folds unless otherwise noted.
  Lower MAE is better.

The completed main runs are in [results](/gscratch/stf/thuang27/skyfinder-dir/results),
and the experimental protocol is documented in
[AGENT_PLAN.md](/gscratch/stf/thuang27/skyfinder-dir/AGENT_PLAN.md).

## Main Five-Fold Image Results

| Backbone | Method | Validation MAE | LOCO test MAE | Test change vs backbone baseline |
|---|---|---:|---:|---:|
| ResNet-50 | Baseline | 2.644 +/- 0.130 | 8.216 +/- 0.814 | - |
| ResNet-50 | LDS | 2.770 +/- 0.118 | 8.504 +/- 1.109 | +0.288 |
| ResNet-50 | FDS | 2.643 +/- 0.115 | **8.055 +/- 0.965** | **-0.161** |
| ResNet-50 | LDS + FDS | 2.823 +/- 0.160 | 8.408 +/- 1.161 | +0.192 |
| ViT-B/16 | Baseline | 5.235 +/- 0.485 | 8.597 +/- 0.729 | - |
| ViT-B/16 | LDS | 5.423 +/- 0.431 | 9.096 +/- 0.996 | +0.499 |
| ViT-B/16 | FDS | 4.809 +/- 0.533 | 8.403 +/- 0.803 | -0.194 |
| ViT-B/16 | LDS + FDS | 6.378 +/- 0.362 | 9.405 +/- 0.801 | +0.808 |

Interpretation:

1. ResNet-50 is the better image-only backbone in this setup. Its FDS result
   is 0.348 C better than the best ViT result.
2. FDS is the only DIR variant that improves both backbone families overall.
3. LDS improves some validation tail measurements, but this does not transfer
   to LOCO test performance. Using validation tail metrics alone would have
   selected a misleading method.
4. The baseline ResNet validation-to-test gap is 5.572 C; even FDS retains a
   5.412 C gap. This is direct evidence that camera/domain shift dominates the
   final error.

## Label-Frequency Analysis

The DIR partitions use 1 C target bins defined from the training fold:
`many` has at least 100 training examples, `medium` has 20-99, and `few` has
fewer than 20. Values are mean LOCO test MAE across folds.

| ResNet-50 method | Overall | Many | Medium | Few |
|---|---:|---:|---:|---:|
| Baseline | 8.216 | 7.936 | 22.370 | 28.169 |
| LDS | 8.504 | 8.232 | 22.293 | 28.769 |
| FDS | **8.055** | **7.790** | **21.164** | **26.694** |
| LDS + FDS | 8.408 | 8.150 | 22.076 | 27.551 |

The `few` column uses four folds because fold 2 has no test examples that are
`few` under its training-bin frequencies. It should be treated as directional,
not as a high-precision estimate.

FDS changes relative to the ResNet baseline:

- Overall: +1.97% improvement (0.161 C lower MAE).
- Many: +1.84% improvement.
- Medium: +5.39% improvement (1.206 C lower MAE).
- Few: +5.24% improvement (1.475 C lower MAE across the four available folds).

The paired fold bootstrap for FDS minus ResNet baseline is -0.161 C overall,
with 95% CI [-0.407, +0.127]. This does not exclude zero with only five folds.
For the medium regime, the paired difference is -1.206 C with 95% CI
[-1.862, -0.577], which is the clearest evidence of an FDS benefit. The few-bin
difference is -1.475 C but its 95% CI [-3.119, +0.169] also includes zero, so
among all bins only the medium effect is statistically distinguishable from
zero. The overall conclusion should therefore be: FDS is the preferred image-only
baseline, but its aggregate gain is modest and needs confirmation with more
independent camera groups.

## Transfer Comparison To The DIR Paper

A stated goal was to check whether DIR's per-bin improvements on SkyFinder match
those reported in the original paper (IMDB-WIKI and AgeDB age regression). They
do not. The table below is percentage improvement in MAE versus each backbone's
own baseline (positive is better); the paper columns are from the DIR paper and
the SkyFinder columns are LOCO test.

| Variant | IMDB-WIKI overall | AgeDB overall | SkyFinder RN-50 overall | SkyFinder ViT overall |
|---|---:|---:|---:|---:|
| LDS | +2.9% | +1.3% | -3.5% | -5.8% |
| FDS | +2.6% | +2.8% | +2.0% | +2.3% |
| LDS + FDS | +3.5% | +2.8% | -2.3% | -9.4% |

The key result: FDS is the only DIR component whose paper behavior transfers to
the unseen-camera setting. Its overall SkyFinder gain (+2.0% ResNet, +2.3% ViT)
is close to the paper's (+2.6% / +2.8%). LDS, and even the paper's strongest
variant LDS + FDS, flip sign and become net-negative under LOCO: the combination
that is best in the paper is the worst here.

The tail behavior is directionally paper-like for FDS but weaker. SkyFinder FDS
improves medium/few by +5.4% / +5.2% (ResNet) and +5.5% / +4.1% (ViT), versus
roughly +6% to +18% for medium and +5% to +21% for few in the paper. Only the
ResNet FDS medium effect has a bootstrap CI that excludes zero.

A plausible mechanism for the LDS failure is that LDS reweights the loss toward
the training label (temperature) distribution, which is miscalibrated when
held-out cameras span different temperature ranges; FDS smooths features and is
less tied to the training label prior, so it is more robust to camera shift.
This is a hypothesis, not a tested claim. Source data:
`analysis_outputs/transfer_rn.txt` and `transfer_vit.txt`.

## Per-Camera Domain-Shift Diagnosis

Per-camera test diagnostics use the saved ResNet predictions and compare each
held-out camera's MAE with two descriptive quantities: distance of its mean
temperature from the training-fold mean, and geographic distance to the nearest
training camera. These are post-hoc diagnostics, not deployment features.

| Model | Cameras | Mean unweighted camera MAE | Climate-distance correlation | Geographic-distance correlation |
|---|---:|---:|---:|---:|
| ResNet-50 baseline | 47 | 8.095 | +0.549 | -0.031 |
| ResNet-50 + FDS | 47 | 7.888 | +0.508 | +0.016 |

The correlation pattern is clear: cameras whose temperature distributions are
far from the training mean are harder, while geographic distance alone has
almost no linear association with error. FDS helps this problem slightly but
does not remove it. The most difficult camera is 4232: baseline MAE 23.72 C,
FDS MAE 21.43 C, climate distance 15.31 C, and only 175 images. The LDS + FDS
combination, by contrast, makes this camera worse than the untouched baseline
(25.78 C), a further sign that adding LDS degrades rather than helps the hardest
domains. Other difficult cameras include 75, 65, 4181, and 9730. This
supports collecting diverse climate and camera conditions, not merely more
samples from already-common cameras.

## Non-Image Baselines And Blending

| Method | Validation MAE | LOCO test MAE | Status |
|---|---:|---:|---|
| C1 global mean | 8.349 | 8.452 | Completed baseline |
| C2 metadata-only | 3.415 | **4.475** | Completed, fold-aligned |
| FDS image-only | 2.643 | 8.055 | Completed, fold-aligned |
| Post-hoc C2/FDS blend, alpha=0.9 | - | 4.426 | Exploratory only |

C2 uses `CamId`, `Hour`, `Month`, `Latitude`, and `Longitude` in a one-hot
plus random-forest model. In the LOCO test split, unseen camera IDs are
ignored by the encoder. The 3.580 C advantage over FDS therefore demonstrates
that time and location metadata provide much stronger transferable signal than
the current image model.

The metadata advantage is larger in the tail than overall. C2's mean LOCO test
MAE is about 10.2 C in the medium bin and 15.5 C in the few bin, versus 21.2 C
and 26.7 C for FDS - roughly halving the rare-temperature error that DIR is
specifically designed to reduce. Rare temperatures are largely predictable from
season, time of day, and location, which the metadata encodes directly, whereas
the image models are most catastrophic exactly in that regime.

The post-hoc blend uses `alpha * C2 + (1 - alpha) * FDS`, where alpha=1 is
pure metadata. The test-fold alpha sweep yields 4.506 C at alpha=0.8, 4.426 C
at alpha=0.9, and 4.475 C for metadata alone. This small apparent 0.049 C gain
may be real, but choosing alpha on test data biases the estimate. Select alpha
from validation data within each fold, or use nested CV, before reporting it as
a final model.

The C2 result should also be accepted only if hour, month, latitude, and
longitude are reliably available at inference time and are not proxies for
unavailable future information. Their availability and provenance should be
documented before deployment claims.

## Additional Ablations

### Camera-Conditioned ResNet

All five folds are complete. The model adds a 64-dimensional camera embedding
and uses 5% camera dropout, so held-out cameras are mapped to an unknown token
at test time.

| Method | Validation MAE | LOCO test MAE | Delta vs ResNet baseline |
|---|---:|---:|---:|
| Camera-conditioned ResNet | 2.695 +/- 0.129 | 8.221 +/- 0.951 | +0.005 |

The paired 95% CI for camera-conditioned minus baseline is [-0.193, +0.300].
There is no evidence that this approach improves unseen-camera performance.
It may improve known-camera calibration, but that is a different deployment
setting and was not the LOCO headline metric.

### DINOv2 Frozen Probe

DINOv2 ViT-S/14 features followed by Ridge regression completed with mean
validation MAE 5.143 C and mean LOCO test MAE 8.553 C. It is comparable to the
ViT baseline but below ResNet-50 + FDS. A frozen generic representation alone
does not resolve the camera shift.

### Random-Split Control

The ten random-split training jobs completed. Their validation MAEs are 2.730 C
for ResNet-50 and 5.862 C for ViT-B/16. However, test inference has not been
run, so this control cannot yet quantify the random-split validation/test gap
or be compared with LOCO results. The run JSONs were also written under
`results/*_rand` despite `configs/main_random.yaml` specifying `results_random`;
this output-path inconsistency should be fixed before the control is finalized.

### Learned Image-Metadata Fusion

No learned-fusion result is available. Job 36212250 failed while loading a
ResNet checkpoint because the head was stripped before `load_state_dict`, which
made `fc.weight` and `fc.bias` unexpected keys. Load the checkpoint into the
full model first, then replace the head with `Identity`, and rerun all five
folds. Compare the repaired model against C2 and a validation-selected blend.

## Artifact And Reproducibility Notes

- The per-camera and blend figures in this report use `fds_resnet50` (the
  headline image model). The analysis runbook saved the `lds_fds_resnet50`
  versions (`percam_ldsfds.txt`, `ensemble.txt`), which report different numbers
  (camera 4232 at 25.78 C; blend 4.432 C). The FDS versions are now saved as
  `analysis_outputs/percam_fds.txt` and `ensemble_fds.txt`. The unweighted
  per-camera mean MAE (8.095 baseline, 7.888 FDS) is the mean of the `mae`
  column from `per_camera.per_camera_stats`, which the CLI does not print.
- The medium-bin bootstrap is now saved as `analysis_outputs/boot_rn_medium.txt`
  (FDS minus baseline -1.206 C, 95% CI [-1.862, -0.577]); previously only the
  overall and few bins were saved.
- The default aggregate glob (`results/**/*.json`, grouped only by model and the
  LDS/FDS flags) now over-collects. For `--split test`, the `resnet50` baseline
  group pools `baseline_resnet50` and `cam_cond_resnet50` (both have
  `use_lds=use_fds=False` and test predictions), giving 10 fold rows instead of
  5, and `bootstrap` then fails with a shape (5,) vs (10,) error. The saved
  overall and few bootstraps predate `cam_cond_resnet50` and so were unaffected
  when written. Until the grouping disambiguates by run-name prefix or an
  explicit experiment filter is added, run the test bootstrap against a
  directory that contains only the eight main configs (the medium bootstrap
  above was produced that way); moving `cam_cond_*` and `*_rand` out of
  `results/` would also fix it.

## Completion State

Completed:

- Main 40-cell LOCO sweep and test inference.
- Constant and metadata baselines.
- Five-fold camera-conditioned ablation.
- Random-split training sweep, without random-split test inference.
- DINOv2 frozen probe.
- Fold-level, per-bin, per-camera, and blend analyses in this report.

Not yet complete:

- Random-split test inference and a corrected random-split aggregate table.
- Repaired learned image-metadata fusion.
- Validation-selected or nested-CV evaluation of blend weights.
- Re-running aggregate scripts with an explicit main-results filter. The default
  glob mixes the camera-conditioned (and, for validation, the random-split) runs
  into the ResNet baseline row, and for test it breaks the bootstrap outright
  (see Artifact And Reproducibility Notes). The numerical tables in this report
  were produced from explicit, fold-aligned run directories and are not affected.

At the time of this summary, no SkyFinder experiment jobs are queued or running.

## Recommended Next Steps

1. Fix the fusion checkpoint-load order and rerun the five-fold learned fusion
   experiment on an H200, A100, L40S, or A40. This is the most direct test of
   whether images add complementary information beyond metadata.
2. Run random-split test inference after fixing the results directory, then
   report random validation versus random test alongside LOCO validation versus
   LOCO test. This isolates camera shift from ordinary sample variance.
3. Use ResNet-50 + FDS as the image-only reference and stop allocating compute
   to LDS variants under the current setup.
4. Select all blend hyperparameters from validation data only. Preserve the
   untouched LOCO test folds for the final comparison.
5. Expand data by adding cameras and balancing climate, season, time-of-day,
   hardware, and rare-temperature coverage. Validate the impact with a
   camera-grouped learning curve that holds test cameras fixed while separately
   varying image count and number of distinct cameras.
6. Report future results with per-camera MAE, medium/few-bin MAE, and fold-level
   uncertainty. Overall sample-weighted MAE alone hides the worst-camera
   failures that matter for deployment.

## Planned Ablations (TODO)

The next round targets paper-grade evidence on the central question: do images
add signal beyond cheap metadata (which is essentially climatology), and if so
where? Items are ordered by how directly they secure that claim. Compute is not
the limiting factor for this round; statistical rigor is. This extends
Recommended Next Steps with a prioritized program.

### P0 - rigor on existing data and the causal control

- Multiple seeds. Every config is currently a single seed (seed 0), but fold
  SDs are 0.8-1.2 C, so the 0.161 C FDS gain is not separable from seed noise.
  Run 5 seeds x 5 folds for all eight main configs and report method effects
  with seed-by-fold variance. Requires adding a seed dimension to `run_sweep.py`.
- Per-camera bootstrap for every headline comparison. The 5-fold CIs are
  underpowered; the 47 held-out cameras give a much stronger paired test (the
  `bootstrap` module anticipates this). The ResNet FDS-minus-baseline per-camera
  bootstrap already reaches significance (-0.207 C, 95% CI [-0.424, -0.006],
  helps 30/47 cameras), versus the non-significant 5-fold result. Promote this
  into the analysis layer and report 47-camera CIs for overall and per bin.
- Climatology bar and metadata decomposition (CPU). Add a climate-normal
  predictor keyed on (latitude/longitude, month, hour), and run C2 with feature
  subsets (time only, location only, month only). This establishes that the
  metadata model is essentially climatology and whether the signal is seasonal
  or geographic; it is the bar every image experiment must clear.
- Random-split test inference. Run test inference on the random-split
  checkpoints and report random (val, test) beside LOCO (val, test). The clean
  proof that the gap is camera shift, not merely hard examples.

### P1 - the decisive test and image-side upper bounds

- Residual fusion. Train the image model to predict temperature minus the
  climatology/metadata prediction, then add the normal back; include a
  concat-fusion variant. Run all five folds across seeds with a
  validation-selected blend and nested CV. This is the direct test of whether
  the sky image reads the weather deviation on top of the seasonal/locational
  normal; a negative result is still a strong paper result.
- Stronger backbones, fine-tuned. Fine-tune (not frozen-probe) DINOv2-L and
  CLIP ViT-L across folds, to preempt the objection that the image model was too
  weak.
- Domain generalization and test-time adaptation. Domain-adversarial training
  (gradient reversal on CamId), CORAL or GroupDRO over camera groups, and
  test-time BN/TENT adaptation on each held-out camera's unlabeled frames. The
  camera-conditioned ablation tested adding camera identity; these test removing
  or adapting to it, which is the correct direction for LOCO.

### P2 - mechanism and deployment

- Data-ROI learning curve. Hold the test cameras fixed and vary the number of
  training cameras and the number of images per camera independently, to answer
  whether more cameras or more images per camera is the better investment.
- Few-shot per-camera adaptation curve. Test MAE versus K labeled images from a
  new camera; the practical deployment metric.
- Saliency and feature attribution. Whether the image model attends to sky
  versus ground, and whether FDS changes that.
