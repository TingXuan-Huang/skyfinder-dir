# AGENT_PLAN.md — run the SkyFinder DIR experiments on Hyak (Klone)

Self-contained runbook for an automated agent. You start in a clone of `skyfinder-dir` on a Klone
login node. **`git pull` first.** Follow the dependency order; do not improvise on the science.
GPU steps are SLURM jobs — **SUBMIT → WAIT for completion → VERIFY outputs** before any dependent step.

## Goal (context)
Evaluate DIR (LDS/FDS) on SkyFinder temperature regression, two layers:
(I) **transfer eval** — do LDS/FDS help, and how do the per-bin %-improvements compare to the paper
(IMDB-WIKI/AgeDB)? (II) **distribution-shift** — under LOCO test, does image-only fail and metadata win?
Deliverable = the 10-condition val+test table with CIs, the transfer table, per-camera diagnosis,
and the four extra ablations.

## Dependency graph (run order)
```
0 preflight ─► A main sweep (40) ─► B infer_test ─┬─► E analysis (val+test)
                                    C baselines ──┘
              D1 random-split ─► (its own infer_test) ─► E (results_random)
              D2 cam-conditioned (writes val+test itself) ─► E
              D3 fusion  (needs A checkpoints) ─► prints MAE
              D4 dino    (independent)          ─► prints MAE
```
Cross-method comparison is INVALID until every config has the SAME folds — finish A fully first.

## 0. Conventions
- `cd $REPO` (your skyfinder-dir checkout) for everything.
- Env prefix: `./.conda/skyfinder` (built once, step 1).
- Klone flags: `ACCOUNT=stf`, `PARTITION=gpu-rtx6k`. If that partition is full/unavailable, export
  another (`gpu-a40`, `gpu-l40s`, `gpu-2080ti`, or `ckpt-all`): `export PARTITION=...`.
- **WAIT for job J** = poll `squeue -u $USER` until J and its array tasks are gone, then
  `sacct -j J --format=JobID,State,Elapsed` → expect `COMPLETED` (investigate any `FAILED`/`TIMEOUT`).
- **GPU one-off template** (used for infer_test / fusion / dino — they aren't array jobs):
  ```bash
  gpu_run () {  # gpu_run "<python command>"
    sbatch -A "${ACCOUNT:-stf}" -p "${PARTITION:-gpu-rtx6k}" --gpus=1 -c8 --mem=120G -t8:00:00 \
      -o slurm/logs/%x-%j.out --job-name=skyaux \
      --wrap="set -e; cd $PWD; module load conda; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate ./.conda/skyfinder; $1"
  }
  ```

## 1. Preflight (login node)
```bash
git pull
ls data/labels_with_images.csv data/splits/loco_5fold.json data/images >/dev/null   # data present?
```
- If data missing, build it (login node has internet):
  `python data/prep_labels.py && python data/download_images.py && python data/filter_to_images.py && python data/splits.py`
- Build the env once: `sbatch slurm/setup_env.slurm`; WAIT; then verify:
  `conda run -p ./.conda/skyfinder python -c "import torch, sklearn, matplotlib; print('env ok')"`
- `python run_sweep.py --config configs/main.yaml --list`  → **must print 40 cells**.
- Confirm a GPU partition is available: `sinfo -p ${PARTITION:-gpu-rtx6k} -h` non-empty.

## 2. Phase A — main sweep (the headline 40 cells)
```bash
bash submit_sweep.sh 1 configs/smoke.yaml      # GPU smoke; WAIT; verify a results/smoke* JSON appears
bash submit_sweep.sh 2 configs/main.yaml       # the 40-cell array; WAIT for the whole array
python recover_results_from_last.py --task-id 0-39 --dry-run   # any train-but-unsaved? recover them:
# python recover_results_from_last.py --task-id <ids printed above>
bash submit_sweep.sh 2 configs/main.yaml       # resubmit; --skip-existing reruns ONLY missing cells
```
Repeat the last line until complete.
**ACCEPTANCE:** `ls results/*/*.json | wc -l` ≥ 40; every run JSON has `final_val` + `val_preds`;
**all 8 configs have all 5 folds** (fold alignment — required before any cross-method analysis).

## 3. Phase B — LOCO test inference
```bash
gpu_run "python infer_test.py --config configs/main.yaml"      # WAIT
```
**ACCEPTANCE:** each `results/*/*.json` now has `test_preds` + `test_final`.

## 4. Phase C — CPU metadata baselines (independent; login or a CPU node)
```bash
python run_baselines.py --config configs/main.yaml
```
**ACCEPTANCE:** `results/_analysis/{c1_constants,c2_metadata_only}.json` exist.

## 5. Phase D — the four extra ablations (mutually independent)
```bash
# D1 random-split control
python data/splits_random.py                                   # login node -> data/splits/random_5fold.json
bash submit_sweep.sh 2 configs/main_random.yaml ; # WAIT; resubmit missing as in Phase A
gpu_run "python infer_test.py --config configs/main_random.yaml"

# D2 camera-conditioned head (writes val+test itself; one job per fold)
for k in 0 1 2 3 4; do gpu_run "python cam_cond_train.py --config configs/cam_cond.yaml --task-id $k"; done

# D3 image+metadata fusion (needs Phase A baseline checkpoints)
gpu_run "python -m skyfinder.analysis.fusion --cnn baseline_resnet50 --img-dir data/images"

# D4 DINOv2 frozen probe  (torch.hub needs internet — see gotcha #4; pre-cache on login node first)
python -c "import torch; torch.hub.load('facebookresearch/dinov2','dinov2_vits14')"   # login node, caches weights
gpu_run "python -m skyfinder.analysis.dino_probe --img-dir data/images --variant dinov2_vits14"
```
**ACCEPTANCE:** D1 → `results_random/*/*.json` (val+test); D2 → `results_cam_cond/*/*.json` (val+test);
D3/D4 print per-fold + summary MAE to their job logs.

## 6. Phase E — analysis (after A+B; C needed for ensemble). Capture all stdout.
```bash
mkdir -p analysis_outputs figures
python -m skyfinder.analysis.aggregate --split val   | tee analysis_outputs/agg_val.txt
python -m skyfinder.analysis.aggregate --split test  | tee analysis_outputs/agg_test.txt
python -m skyfinder.analysis.bootstrap --model resnet50 --bin overall --split test | tee analysis_outputs/boot_rn_overall.txt
python -m skyfinder.analysis.bootstrap --model resnet50 --bin few     --split test | tee analysis_outputs/boot_rn_few.txt
python -m skyfinder.analysis.transfer_table --split test --model resnet50 | tee analysis_outputs/transfer_rn.txt
python -m skyfinder.analysis.transfer_table --split test --model vit_b_16 | tee analysis_outputs/transfer_vit.txt
python -m skyfinder.analysis.per_camera --cnn baseline_resnet50  | tee analysis_outputs/percam_base.txt
python -m skyfinder.analysis.per_camera --cnn lds_fds_resnet50   | tee analysis_outputs/percam_ldsfds.txt
python -m skyfinder.analysis.plots --model resnet50 --split test --out figures
python -m skyfinder.analysis.ensemble --cnn lds_fds_resnet50     | tee analysis_outputs/ensemble.txt
# random-split: is val ~= test? (camera-shift confirmation)
python -m skyfinder.analysis.aggregate --results results_random --splits data/splits/random_5fold.json --split val  | tee analysis_outputs/rand_val.txt
python -m skyfinder.analysis.aggregate --results results_random --splits data/splits/random_5fold.json --split test | tee analysis_outputs/rand_test.txt
# cam-conditioned: aggregate its results_cam_cond/ for val+test
python -m skyfinder.analysis.aggregate --results results_cam_cond --split val  | tee analysis_outputs/camcond_val.txt
python -m skyfinder.analysis.aggregate --results results_cam_cond --split test | tee analysis_outputs/camcond_test.txt
```

## 7. Collect + report back
- `results/` is gitignored. To return data: `tar czf results_$(date +%Y%m%d).tgz results results_random results_cam_cond results/_analysis analysis_outputs figures`
  and copy it off-cluster, OR push `analysis_outputs/` + `figures/` to a `results` branch
  (`git switch -c results-<date>; git add -f analysis_outputs figures; git commit; git push`).
- Write `SUMMARY.md` containing: the 10-condition val+test per-bin table (agg_*), the transfer table,
  the bootstrap CIs, per-camera climate r, random-split val-vs-test, and ablation MAEs
  (cam-cond / fusion / dino vs baseline and vs C2).

## 8. Gotchas / failure handling
1. **Fold alignment** — never compare methods across different fold sets. `bootstrap.compare_to_baseline`
   auto-aligns to common folds, but the headline table needs all 5 folds per config. Finish Phase A fully.
2. **JSON-save** — already fixed (`_json_default`). If an old run shows only `<run>_last.pt` and no JSON,
   recover with `recover_results_from_last.py` (do not retrain).
3. **Partition unavailable** — `export PARTITION=<available>`; `GPUS` / `GPU_FLAG` are also overridable
   (e.g. `export GPU_FLAG=--gpus-per-node=2080ti:1`).
4. **No internet on compute nodes** — DINOv2 `torch.hub` and any data download must happen on a LOGIN node.
   Pre-cache DINOv2 weights on login (`TORCH_HOME` defaults to scratch via run_sweep.slurm; export it the
   same way for `gpu_run` if needed), then the compute job loads from cache.
5. **Time/cost** — ResNet cell ~3–4 GPU-h, ViT more; 12h/task limit is ample per cell; the array uses %2
   concurrency. Whole main sweep ≈ 1–2 days wall.
6. **Do NOT interpret partial/misaligned results as final.** Report them as in-progress.

## 9. Acceptance (whole run complete)
- 40 main + 10 random + 5 cam-cond run JSONs, each with `val_preds`+`test_preds`.
- `results/_analysis/{c1,c2}` present; fusion + dino summaries captured.
- All Phase-E tables/figures in `analysis_outputs/` + `figures/`.
- `SUMMARY.md` written; artifacts returned per §7.
