# Running on UW Hyak Klone

This checkout is designed to run from group scratch, for example:

```bash
cd /gscratch/stf/$USER/skyfinder-dir
```

Hyak Klone exposes `/gscratch` and `/mmfs1/gscratch` as the same storage. Keep the
repo, Conda env, data, caches, and results under `/gscratch` rather than `$HOME`.

## 1. Link Existing SkyFinder Data

If the old checkout already has prepared data, link it into this repo instead of
downloading again:

```bash
ln -s /gscratch/stf/$USER/DIR-SkyFinder/data/images data/images
ln -s /gscratch/stf/$USER/DIR-SkyFinder/data/splits data/splits
ln -s /gscratch/stf/$USER/DIR-SkyFinder/data/labels_with_images.csv data/labels_with_images.csv
```

The expected files are:

```text
data/labels_with_images.csv
data/splits/loco_5fold.json
data/images/<CamId>/<Filename>
```

## 2. Build the Conda Env

On Klone, module commands are available on compute nodes, so create the env with
Slurm rather than from the login node:

```bash
sbatch --account=stf --partition=compute slurm/setup_env.slurm
```

To verify CUDA during setup, submit the same setup job on a GPU partition:

```bash
sbatch --account=stf --partition=gpu-rtx6k --gpus=1 slurm/setup_env.slurm
```

This creates `.conda/skyfinder` inside the new checkout. By default it clones
the existing package environment at
`/gscratch/stf/$USER/DIR-SkyFinder/.conda/skyfinder`, then installs this new
checkout with `pip install --no-deps -e .`. That avoids relying on package-index
downloads from compute nodes. It also prefetches the torchvision ResNet-50 and
ViT-B/16 weights into `/gscratch/stf/$USER/.cache/torch` by default, which keeps
array tasks from trying to download model weights mid-run. Set
`PREFETCH_WEIGHTS=0` only if the weights are already cached and package
installation is the only setup step you want.

## 3. Run a Smoke Job

Submit a one-epoch, small-subset GPU smoke job before the full sweep:

```bash
bash submit_sweep.sh 1 configs/smoke.yaml
```

Watch logs in `slurm/logs/`. A successful smoke job writes under:

```text
results/smoke_resnet50/smoke_resnet50_fold0.json
```

## 4. Submit the Full Sweep

Default submission targets account `stf`, partition `gpu-rtx6k`, and one GPU:

```bash
bash submit_sweep.sh 2 configs/main.yaml
```

Override Hyak resources without editing scripts:

```bash
ACCOUNT=stf PARTITION=gpu-l40s GPUS=1 bash submit_sweep.sh 2 configs/main.yaml
GPU_FLAG=--gpus-per-node=2080ti:1 PARTITION=ckpt-all bash submit_sweep.sh 2 configs/main.yaml
```

Use `hyakalloc` and `sinfo -s` on Hyak to confirm the account and GPU partitions
available to you before submitting the full run.

## 5. Recover Results JSON from Completed Checkpoints

If a run finished training and wrote `<run_name>.pt` plus `<run_name>_last.pt`,
but failed while saving `<run_name>.json`, recover the JSON without retraining:

```bash
python recover_results_from_last.py --config configs/main.yaml --task-id 20
```

For many tasks, use the CPU-only Slurm wrapper:

```bash
sbatch --array=0-1,20-39%6 slurm/recover_json_cpu.slurm
```

The recovery reads the full `_last.pt` training state, reconstructs the metrics
and validation predictions, writes the nested results JSON, and keeps `_last.pt`
by default. Add `--remove-last` only after confirming the recovered JSON files
are valid.
