# SUMMARY — DIR on SkyFinder: results

**Status: TEMPLATE + preliminary.** The server agent fills every `[FILL …]` from `analysis_outputs/`
after the aligned run (see `AGENT_PLAN.md`). Numbers tagged **(prelim)** are from the *archived*
sweep (partial / misaligned folds, bin_w varies) — keep as narrative scaffolding but **REPLACE** with
the clean aligned numbers. Framing: lead Part I (DIR transfer eval), land Part II (distribution shift).

---

## 0. TL;DR (3 sentences)
`[FILL: (1) DIR's effect transfers only partially — LDS improves the temperature tail but, unlike the`
`paper, regresses overall/many; FDS does nothing. (2) On LOCO test, image-only regression fails and a`
`5-feature metadata GBM beats every CNN. (3) The real bottleneck is distribution shift across cameras,`
`not label imbalance.]`

## 1. Setup & protocol
- **Data:** 47 cameras, ~81,447 images, target `TempM` (°C). **5-fold LOCO** (`loco_5fold.json`).
- **Models:** ResNet-50 / ViT-B/16, ImageNet-pretrained, 1-output head, L1 loss; LDS / FDS toggles.
- **Metric:** per-bin MAE on **All / Many / Medium / Few**, shots by TRAIN frequency (≥100 / 20–99 / <20),
  bin width **1.0 °C**. NOTE: the **few bin is underpowered** (test n≈220 total, ~10–17/fold; some folds 0).
- **Baselines:** C1 (constant per-cam-month), C2 (HistGBR on CamId/Hour/Month/Lat/Long).
- **Repro:** `skyfinder-dir` @ `[FILL git rev]`; commands in `AGENT_PLAN.md`.

---

# PART I — DIR transfer evaluation (the assignment)

## T1. Per-bin MAE, mean ± std over 5 folds   `[FILL agg_val.txt / agg_test.txt]`

**Validation (in-camera)** — bin_w=1.0
| model | method | All | Many | Medium | Few | folds |
|---|---|---|---|---|---|---|
| resnet50 | baseline | `[FILL]` | | | | |
| resnet50 | lds | | | | | |
| resnet50 | fds | | | | | |
| resnet50 | lds_fds | | | | | |
| vit_b_16 | baseline … | | | | | |

**LOCO test** — same shape `[FILL agg_test.txt]`. (Add C1 / C2 rows from `run_baselines` for test.)

> (prelim, resnet50 val) baseline 2.84 / lds 3.15 / fds 2.89 / lds_fds 3.45 overall; lds few 4.78 vs baseline 7.90.

## T2. Transfer rate vs the paper (method − baseline, %-improvement)   `[FILL transfer_rn.txt / transfer_vit.txt]`
| dataset | method | All | Many | Medium | Few |
|---|---|---|---|---|---|
| IMDB-WIKI | lds_fds | +3.5% | +0.4% | +16.6% | +15.7% |
| AgeDB | lds_fds | +2.8% | −5.9% | +13.7% | +21.1% |
| **SkyFinder** (val) | lds_fds | `[FILL]` | | | |
| **SkyFinder** (test) | lds_fds | `[FILL]` | | | |

> (prelim, SkyFinder val) lds_fds: overall **−21.4%**, many −22.6%, medium +24.7%, few +25.8%.
> **The signature:** tail (medium/few) transfers — even larger than the paper — but **overall/many is
> sign-flipped** (paper helps, SkyFinder regresses). FDS ≈ 0 on every bin. `[FILL: confirm on aligned test]`

## T3. Significance — paired bootstrap CIs (method − baseline, common folds)   `[FILL boot_*.txt]`
| model | bin | LDS Δ [CI] | FDS Δ [CI] | LDS+FDS Δ [CI] |
|---|---|---|---|---|
| resnet50 | overall | `[FILL]` | | |
| resnet50 | few | `[FILL]` | | |

> (prelim, val) LDS few Δ=−3.73 [−5.31, −2.15] *; **FDS few Δ=−0.18 [−0.91, +0.44] (CI ∋ 0 → no effect)**.

## Interpretation (Part I)
`[FILL: Does DIR transfer? Lead with: tail-direction yes (larger % than paper), overall-direction no`
`(sign-flipped, because SkyFinder is ~99% body so tail reweight costs the body); FDS contributes nothing;`
`LDS+FDS ≈ LDS. State the answer plainly, with the T3 CIs as the evidence.]`

---

# PART II — distribution shift (the deeper finding)

## T4. Validation vs LOCO test gap   `[FILL agg_val vs agg_test]`
| method | val overall | test overall | gap |
|---|---|---|---|
| baseline resnet50 | `[FILL]` | `[FILL]` | |

> (prelim) val ≈ 2.8 °C, LOCO test ≈ 7.3 °C — they measure different things (memorisation vs generalisation).

## T5. Metadata C2 vs CNN on LOCO test   `[FILL agg_test + run_baselines]`
| | All | Many | Medium |
|---|---|---|---|
| best CNN | `[FILL]` | | |
| **C2 metadata GBM** | `[FILL]` | | |

> (prelim) C2 **6.36** beats best CNN **7.31** overall; medium 17.42 vs 21.96. **Headline of Part II.**

## T6. Per-camera failure — climate vs geography   `[FILL percam_*.txt]`
- climate-distance r = `[FILL]` (per-cam test MAE vs |cam_mean_T − train_mean_T|)
- geo-distance r = `[FILL]`
- worst 5 cameras: `[FILL]`

> (prelim, report) climate r ≈ **+0.56**, geo r ≈ **−0.17** → failure is "no per-camera prior", not geography.

## T7. Random-split control   `[FILL rand_val.txt / rand_test.txt]`
| split | val overall | test overall |
|---|---|---|
| LOCO | `[FILL T4]` | `[FILL T4]` |
| random | `[FILL]` | `[FILL]` |

> **Decision rule:** if random val ≈ test but LOCO val ≪ test → the gap is **camera distribution shift**,
> not sample variance. `[FILL the verdict]`

## Interpretation (Part II)
`[FILL: image-only regression fails under LOCO; metadata wins; per-camera climate distance predicts`
`failure; random-split control confirms camera-shift. Conclude: a distribution-shift problem.]`

---

# PART III — can anything close the LOCO gap? (ablations)

## T8. αC2 + CNN ensemble   `[FILL ensemble.txt]`
best α = `[FILL]` (α=0 pure CNN, α=1 pure C2). CNN-only `[FILL]` / C2-only `[FILL]` / best blend `[FILL]`.
> Reads: α≈1 → image adds nothing; α≈0.5 → complementary (fusion is the project).

## T9. Camera-conditioned head   `[FILL camcond_test.txt]`
| | val overall | test overall |
|---|---|---|
| baseline | `[FILL T1]` | `[FILL T1]` |
| cam-conditioned | `[FILL]` | `[FILL]` |
> Reads: test → C2 (~6.4) means the per-camera prior is the missing signal; test stays ~7.3 means image content is the bottleneck.

## T10. Image+metadata fusion (GBM on CNN-feats ++ metadata)   `[FILL fusion job log]`
val `[FILL]` / test `[FILL]` vs CNN-alone `[FILL]` vs C2-alone `[FILL]`.

## T11. DINOv2 frozen probe + Ridge   `[FILL dino job log]`
val `[FILL]` / test `[FILL]` vs trained ResNet `[FILL T1]`.
> Reads: frozen DINOv2 ≥ trained ResNet on test → fine-tuning is harmful (major reframe).

---

## 4. Answer to the PI question — "does DIR work on SkyFinder?"
`[FILL: layered. (a) In-camera val: LDS trades body for tail (helps medium/few, regresses overall/many),`
`FDS nil. (b) LOCO test: image-only DIR does not close the gap; metadata beats every CNN. (c) Transfer`
`verdict: partial, sign-flipped on overall; the few-bin is too sparse for a precise rate. (d) The real`
`story is distribution shift across cameras, not label imbalance.]`

## 5. Limitations
- **Few-bin underpowered** (test n≈220; some folds 0) → tail % are directional, not precise; report n.
- **One seed per fold.** The 5 LOCO folds (different cameras) are the uncertainty source — the *right* one
  for generalisation — but optimisation-noise CIs would need ≥3 seeds on the headline configs.
- **FDS / ViT** historically on fewer folds; the clean re-run fixes that — confirm all 5 folds present.
- bin width 1.0 °C chosen on evidence (U3); the few-region is genuinely sparse on SkyFinder.

## 6. Provenance
- Code: `skyfinder-dir` @ `[FILL git rev]` (training package + analysis + ablations).
- Data: 47 cams / ~81k images; splits `loco_5fold.json` (+ `random_5fold.json` for T7).
- Reference repo (history + Q&A + design): `DIR_Code` @ `refactor-2026-05`, `experiments/restart-2026-05-24/`.
- Paper reference (T2): Yang et al. 2021, *Delving into Deep Imbalanced Regression*, ICML (arXiv:2102.09554).
- Reproduce: `AGENT_PLAN.md`.
