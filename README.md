# NAS-ADE for Unsupervised Time-Series Anomaly Detection

This repository implements a UAD-oriented neural architecture search pipeline for multivariate time series. It combines:

- TS-TCC self-supervised pretraining
- DeepSVDD-based one-class scoring
- AdaptNAS-style bilevel optimization for target-aware architecture selection

The current codebase supports two modes:

- `uad_source`: source-only UAD using `train_normal` and DeepSVDD scoring
- `adaptnas_combined`: source-normal plus unlabeled target-pool adaptation with reliability-aware weighting and unlabeled bi-level search

It also now supports multiple model families:

- `default_nasade`: the current TS-TCC + CNN/Transformer-GRU-TCN + DeepSVDD / AdaptNAS pipeline
- `omni_anomaly`: paper-faithful OmniAnomaly family for raw SMD machine-by-machine source-only runs
- `usad`: paper-style USAD family for raw SWaT source-only runs with fixed-baseline vs partial-NAS comparison

## What Is Implemented

The current implementation now aligns with the intended project design in three important ways:

- TS-TCC pretraining is used to initialize candidate encoders before search/final training instead of being trained and then ignored.
- `arch_params` now affect the actual forward pass by weighting encoder-depth features.
- Combined mode now requires a separate `target_pool_unlabeled.npz`, so architecture search no longer reuses labeled validation data as target adaptation input.

## Project Layout

```text
src/
  pipeline.py                # main end-to-end entrypoint
  data/
    omni_smd.py              # raw SMD loader + sliding-window helpers for Omni family
    swat.py                  # raw SWaT loader + flatten-window helpers for USAD family
  adaptnas/
    search_space.py          # discrete search space sampling
    trainer.py               # bilevel training loop
    optimizer.py             # lower/upper-level optimizers
  models/
    tscnn.py                 # CNN encoder blocks
    transformer.py           # Transformer sequence encoder
    classifier.py            # MLP classifier head
    discriminator.py         # domain discriminator
    deepsvdd.py              # DeepSVDD soft-boundary objective
  ts_tcc/                    # TS-TCC components
  utils/
    metrics.py               # AUROC, AP, Best-F1, POT-like F1, event-F1, delay

scripts/
  preprocess_smd.py          # raw SMD -> source.npz / target.npz
  make_uad_smd.py            # build a real-data UAD experiment split (temporal or cross-machine)
  build_domain_shift_smd.py  # build a temporal/cross-machine domain-shift suite + manifest
  run_all_smd.py             # batch runner for all SMD machines
  run_pipeline.sh            # single-run helper
```

## Environment Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

If you already created `venv/`, update it after pulling new changes:

```bash
./venv/Scripts/python.exe -m pip install -r requirements.txt
```

### Optional: enable NVIDIA GPU on Windows

If your machine has an NVIDIA GPU, reinstall PyTorch from the official CUDA wheel index inside the project `venv`:

```bash
./venv/Scripts/python.exe -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify that CUDA is available:

```bash
./venv/Scripts/python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

`run.ps1` defaults to `-Device auto`, so once CUDA is available it will automatically run on GPU.

## Family Selection

The pipeline now separates:

- `mode`: training/search protocol
- `family`: model/loss/scoring family

Current families:

- `default_nasade`
- `omni_anomaly`
- `usad`

Current Omni family scope:

- implemented only for `uad_source`
- skips TS-TCC pretraining
- skips DeepSVDD
- runs on raw SMD `train/test/test_label` machine-by-machine
- uses sliding-window last-point reconstruction scoring
- includes planar normalizing flows in the posterior path
- supports Monte Carlo test-time scoring via `--omni_test_n_z`
- uses Omni-style adjusted best-F1 / POT-like evaluation on normal-score thresholds

Current USAD family scope:

- implemented only for `uad_source`
- runs on raw SWaT normal/attack CSV files
- follows the original encoder + two-decoder adversarial training recipe
- freezes the core USAD macro-architecture and searches only a small width/bottleneck subspace
- reports both the fixed paper-style baseline and a searched partial-NAS variant
- uses window-level flattened inputs with last-point label alignment for evaluation

For paper-faithful Omni runs, prefer raw SMD over the `.npz` UAD protocol.

## Dataset Protocols

### 1. `uad_source`

Use:

```text
train_normal.npz,val_mixed.npz[,test_mixed.npz]
```

- `train_normal.npz`: normal-only source windows
- `val_mixed.npz`: mixed normal/anomaly windows with labels
- `test_mixed.npz`: optional mixed test split with labels

### 2. `adaptnas_combined`

Required protocol:

```text
train_normal.npz,target_pool_unlabeled.npz,val_mixed.npz[,test_mixed.npz]
```

- `train_normal.npz`: normal-only source windows
- `target_pool_unlabeled.npz`: unlabeled target windows used for adaptation, weighting, and unlabeled upper-level architecture search
- `val_mixed.npz`: labeled mixed split used only for evaluation/model selection after training, not for architecture-gradient updates
- `test_mixed.npz`: optional labeled evaluation split

## Data Preparation

### Raw SMD for `family=omni_anomaly`

The paper-faithful Omni family reads the official raw SMD layout directly:

```text
data/ServerMachineDataset/
  train/
  test/
  test_label/
```

You do not need to build `train_normal.npz` / `val_mixed.npz` / `test_mixed.npz` for this family.

### Raw SWaT for `family=usad`

The USAD family reads the original SWaT CSV protocol directly:

```text
data/SWaT/
  SWaT_Dataset_Normal_v1.csv
  SWaT_Dataset_Attack_v0.csv
```

It follows the public USAD notebook preprocessing:

- read the normal CSV and the attack CSV
- drop `Timestamp` and `Normal/Attack`
- convert locale-formatted decimal strings to floats
- fit `MinMaxScaler` on the normal train split and transform the attack split
- optionally downsample the series before building windows

You do not need to build `train_normal.npz` / `val_mixed.npz` / `test_mixed.npz` for this family.

### Raw SMD to windowed `source.npz` / `target.npz`

```bash
python scripts/preprocess_smd.py --raw_root data/ServerMachineDataset --out_root data/smd
```

This creates, per machine:

- `source.npz`
- `target.npz`

### Important constraint

The new experiment builder does not generate synthetic data. It only reorganizes real windows that already exist in SMD:

- `train_normal` always comes from the source machine `source.npz`
- `target_pool_unlabeled`, `val_mixed`, and `test_mixed` always come from a real target machine `target.npz`
- domain shift is created either by a later target timeline segment or by using another real machine as the target domain

### Build a same-machine temporal-shift experiment

```bash
python scripts/make_uad_smd.py \
  --machine_dir data/smd/machine-1-1 \
  --split_mode search \
  --shift_level medium \
  --target_pool_frac 0.2 \
  --val_frac 0.3 \
  --guard 4
```

This creates a new folder like:

```text
data/smd_experiments/temporal_medium/machine-1-1/
```

with:

- `train_normal.npz`
- `target_pool_unlabeled.npz`
- `val_mixed.npz`
- `test_mixed.npz`
- `split_metadata.json`

`split_metadata.json` includes:

- source machine / target machine
- hidden anomaly ratio inside `target_pool_unlabeled`
- anomaly counts in `val_mixed` and `test_mixed`
- domain-shift proxy scores such as source-vs-target-pool `domain_auc`

### Build a cross-machine domain-shift experiment

This is the strongest form of deployment shift in this repository:

- train on machine A normal data
- adapt/evaluate on machine B target data

Example:

```bash
python scripts/make_uad_smd.py \
  --machine_dir data/smd/machine-1-1 \
  --target_machine_dir data/smd/machine-1-2 \
  --split_mode search \
  --shift_level medium \
  --guard 4
```

This creates:

```text
data/smd_experiments/cross_machine_medium/machine-1-1__to__machine-1-2/
```

### Build a benchmark suite automatically

Generate a small suite:

```bash
python scripts/build_domain_shift_smd.py \
  --machines machine-1-1,machine-1-2 \
  --shift_levels medium \
  --build_temporal \
  --build_cross_machine \
  --same_family_only \
  --topk_cross 1
```

Generate a larger suite across all prepared SMD machines:

```bash
python scripts/build_domain_shift_smd.py \
  --shift_levels medium,hard \
  --build_temporal \
  --build_cross_machine \
  --same_family_only \
  --topk_cross 1
```

This writes experiment folders under `data/smd_experiments/` and a global manifest at:

- `data/smd_experiments/manifest.json`

## Running the Pipeline

If you use `run.ps1`, the script now defaults to `-Device auto` and falls back to `cpu` automatically when the local PyTorch build does not support CUDA.

`run.ps1` also accepts `-DataDir`, so you can point it directly to any generated experiment folder.

### UAD source mode

```bash
python -m src.pipeline \
  --dataset_or_paths data/smd/machine-1-1/train_normal.npz,data/smd/machine-1-1/val_mixed.npz,data/smd/machine-1-1/test_mixed.npz \
  --mode uad_source \
  --family default_nasade \
  --epochs_pretrain 50 \
  --search_candidates 20 \
  --batch_size 128 \
  --device cuda
```

### Combined mode with separate target pool

```bash
python -m src.pipeline \
  --dataset_or_paths data/smd/machine-1-1/train_normal.npz,data/smd/machine-1-1/target_pool_unlabeled.npz,data/smd/machine-1-1/val_mixed.npz,data/smd/machine-1-1/test_mixed.npz \
  --mode adaptnas_combined \
  --family default_nasade \
  --epochs_pretrain 50 \
  --search_candidates 20 \
  --batch_size 128 \
  --device cuda
```

### OmniAnomaly family in source mode

```bash
python -m src.pipeline \
  --mode uad_source \
  --family omni_anomaly \
  --raw_smd_root data/ServerMachineDataset \
  --machine machine-1-1 \
  --search_candidates 20 \
  --omni_window_length 100 \
  --omni_batch_size 50 \
  --omni_test_n_z 1 \
  --device cuda
```

### USAD family on raw SWaT

```bash
python -m src.pipeline \
  --mode uad_source \
  --family usad \
  --swat_train_csv data/SWaT/SWaT_Dataset_Normal_v1.csv \
  --swat_test_csv data/SWaT/SWaT_Dataset_Attack_v0.csv \
  --search_candidates 10 \
  --usad_window_length 12 \
  --usad_downsample 5 \
  --usad_latent_size 40 \
  --usad_batch_size 128 \
  --device cuda
```

### Run a generated experiment folder with `run.ps1`

Temporal-shift example:

```powershell
.\run.ps1 -Mode uad_source -Family default_nasade -DataDir data\smd_experiments\temporal_medium\machine-1-1
```

Cross-machine example:

```powershell
.\run.ps1 -Mode adaptnas_combined -Family default_nasade -DataDir data\smd_experiments\cross_machine_medium\machine-1-1__to__machine-1-2
```

OmniAnomaly family example:

```powershell
.\run.ps1 -Mode uad_source -Family omni_anomaly -RawSmdRoot data\ServerMachineDataset -Machine machine-1-1
```

USAD family example:

```powershell
.\run.ps1 -Mode uad_source -Family usad -SwatTrainCsv data\SWaT\SWaT_Dataset_Normal_v1.csv -SwatTestCsv data\SWaT\SWaT_Dataset_Attack_v0.csv
```

Batch-run a few raw SMD machines with the Omni family:

```bash
python scripts/run_all_smd.py \
  --mode uad_source \
  --family omni_anomaly \
  --raw_smd_root data/ServerMachineDataset \
  --machines machine-1-1,machine-1-2
```

Save a USAD-on-SWaT run into `outputs/benchmarks/...`:

```bash
python scripts/run_usad_swat.py \
  --train_csv data/SWaT/SWaT_Dataset_Normal_v1.csv \
  --test_csv data/SWaT/SWaT_Dataset_Attack_v0.csv \
  --search_candidates 10 \
  --tag swat_partial
```

## End-to-End Flow

### TS-TCC pretraining

- `uad_source`: pretrains on `train_normal`
- `adaptnas_combined`: pretrains on available unlabeled windows, preferring multi-machine SMD when available

The pretrained TS-TCC encoder is then used to initialize candidate encoder weights before search and final training. It is not kept as a parallel branch during candidate inference.

### Candidate model

Each sampled candidate contains:

- 1 to 3 temporal CNN blocks
- a projection to `d_model`
- one sequence family: `Transformer`, `GRU`, or `TCN`
- classifier head
- domain discriminator

`arch_params` now control:

- the mixture over encoder depths

### `uad_source`

For each sampled architecture:

1. Build candidate features on source-normal windows
2. Fit DeepSVDD on source-normal features
3. Score held-out source-normal windows by mean SVDD distance
4. Select the architecture with the smallest validation compactness objective

Final anomaly scores are SVDD distances on the selected architecture.

### `uad_source` with `family=omni_anomaly`

For each raw SMD machine:

1. Load raw `train`, `test`, and `test_label`
2. Normalize using train statistics only
3. Split raw train contiguously into inner-train / inner-validation
4. Train the fixed paper-faithful Omni baseline
5. Sample partial-NAS Omni architectures around the fixed baseline
6. Select by smallest validation anomaly score on the inner-validation series
7. Refit/evaluate on the full raw train / raw test machine split

Final anomaly scores are the negative last-point reconstruction log-probabilities on sliding windows with last-point alignment.

### `uad_source` with `family=usad`

For raw SWaT:

1. Load the raw normal and attack CSV files
2. Drop `Timestamp` / `Normal/Attack`, convert numeric strings, and fit train-only normalization
3. Downsample the series if requested, then split raw train contiguously into inner-train / inner-validation
4. Train the fixed paper-style USAD baseline
5. Sample partial-NAS USAD architectures around the fixed encoder/decoder family
   Only `latent_size` and an overall hidden-width scale are searched; the encoder/decoder topology, adversarial losses, and anomaly score remain fixed to the USAD design.
6. Select by the smallest validation anomaly score on the inner-validation normal series
7. Refit/evaluate on the full normalized SWaT train / attack split

Final anomaly scores are `alpha * MSE(x, w1) + beta * MSE(x, w3)` on flattened sliding windows with last-point label alignment.

### `adaptnas_combined`

For each sampled architecture:

1. Warm up on source-normal labels (`0`) to stabilize features
2. Fit DeepSVDD on source-normal features
3. Score `target_pool_unlabeled` and convert distances into reliability weights
4. Run lower-level training with:
   - source CE
   - weighted target entropy minimization
   - GRL-based domain loss with weighted target contribution
5. Run upper-level updates on:
   - source holdout normal windows
   - unlabeled weighted target-pool windows
6. Select the architecture by the unlabeled upper objective, not by target labels

Final anomaly scores are SVDD distances on the adapted selected architecture.

## Metrics

Implemented metrics:

- AUROC
- Average Precision
- Best-F1
- POT-like F1 using thresholds estimated from source-normal scores
- Event-F1
- Detection delay (mean and median)

## Outputs

Main artifacts are written to `outputs/`:

- `outputs/results.json`
- `outputs/baselines/*.json`
- `outputs/baselines_summary.json`
- `outputs/checkpoints/*.pt`
- `outputs/figures/*.png`

## Batch Running on SMD

Run all prepared machines in source mode:

```bash
python scripts/run_all_smd.py --mode uad_source
```

Run all prepared machines in combined mode:

```bash
python scripts/run_all_smd.py --mode adaptnas_combined
```

Run all generated experiment folders under a custom root:

```bash
python scripts/run_all_smd.py --mode adaptnas_combined --data_root data/smd_experiments/cross_machine_medium
```

For `adaptnas_combined`, the batch runner now expects the full 4-file protocol and skips folders that do not contain `target_pool_unlabeled.npz`.

To run only a few generated `default_nasade` experiment folders, use `--cases`:

```bash
python scripts/run_all_smd.py \
  --mode uad_source \
  --family default_nasade \
  --data_root data/smd_experiments/cross_machine_medium \
  --cases machine-1-1__to__machine-1-2,machine-1-1__to__machine-1-6
```

### Compare `uad_source` vs `adaptnas_combined`

Run both modes on the same benchmark folders, then compare the saved benchmark outputs:

```bash
python scripts/run_all_smd.py --mode uad_source --family default_nasade --data_root data/smd_experiments/cross_machine_medium --tag cross_medium_source
python scripts/run_all_smd.py --mode adaptnas_combined --family default_nasade --data_root data/smd_experiments/cross_machine_medium --tag cross_medium_combined
```

```bash
python scripts/compare_mode_benchmarks.py \
  --source_dirs outputs/benchmarks/default_nasade-uad_source-cross_medium_source \
  --combined_dirs outputs/benchmarks/default_nasade-adaptnas_combined-cross_medium_combined \
  --out outputs/benchmarks/compare_cross_medium.json
```

You can compare multiple shift roots together by passing comma-separated benchmark directories to `--source_dirs` and `--combined_dirs`, for example to summarize `cross_machine_medium` and `cross_machine_hard` together.

## Notes

- A working Python environment is required; this repository does not bundle one.
- `scripts/run_pipeline.sh` and `scripts/run_all_smd.py` now use the current CLI based on `--mode`.
- `omni_anomaly` currently supports paper-faithful `uad_source` on raw SMD machine-by-machine runs.
- `omni_anomaly` does not yet support `adaptnas_combined`.
- `usad` currently supports paper-style `uad_source` on raw SWaT runs.
- `usad` does not yet support `adaptnas_combined`.
- Combined mode now keeps `target_pool_unlabeled` strictly separate from `val_mixed` and `test_mixed` to avoid target-label leakage during architecture search.
- For domain-shift experiments, prefer the new `data/smd_experiments/...` folders instead of overwriting the original `data/smd/machine-*` directories.
- Cross-machine experiments are a valid domain-shift setting in SMD because the feature schema is aligned, while the machine operating distributions can differ substantially.
