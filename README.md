# NAS--for--UAD

This repository contains experiments for unsupervised anomaly detection (UAD) on multivariate time series, centered on:

- a NAS-based UAD pipeline for processed window datasets
- paper-faithful raw-data baselines for SMD and SWaT
- domain-shift benchmarking utilities for source/target transfer experiments

The codebase currently supports four model families:

- `default_nasade`: TS-TCC + CNN/GRU/TCN/Transformer candidate search with pluggable one-class scoring
- `omni_anomaly`: paper-style OmniAnomaly on raw SMD
- `tranad`: upstream-faithful TranAD on raw SMD
- `usad`: paper-style USAD on raw SWaT

## Main Workflows

### 1. `default_nasade`

This is the main NAS-oriented workflow in the repo. It supports two modes:

- `uad_source`
  Uses `train_normal.npz` and selects architectures from source-normal data only.
- `adaptnas_combined`
  Uses `train_normal.npz` plus `target_pool_unlabeled.npz` for target-aware unlabeled adaptation/search.

Expected input protocols:

```text
uad_source:
  train_normal.npz,val_mixed.npz[,test_mixed.npz]

adaptnas_combined:
  train_normal.npz,target_pool_unlabeled.npz,val_mixed.npz[,test_mixed.npz]
```

`default_nasade` now supports pluggable one-class backends through `--oneclass_method`:

- `deepsvdd` (default)
- `autoencoder`
- `knn_distance`
- `oneclass_svm`
- `svdd`
- `prototype_oneclass`
- `mahalanobis_head`
- `gmm_head`

Framework diagram for the current `adaptnas_combined` implementation:

- [docs/docs/adaptnas_combined_framework_diagram.md](docs/docs/adaptnas_combined_framework_diagram.md)

### 2. Paper-faithful source-only baselines

- `omni_anomaly`
  Raw SMD, machine-by-machine, source-only.
- `tranad`
  Raw SMD, machine-by-machine, source-only.
- `usad`
  Raw SWaT CSVs, source-only.

These families are implemented only for `--mode uad_source`.

### 3. Domain-shift experiment tooling

The repo also includes utilities to:

- preprocess raw SMD into per-machine `.npz` caches
- preprocess raw SMAP/MSL into per-channel `.npz` caches
- build temporal-shift and cross-machine source/target splits
- build temporal-shift and cross-entity source/target splits
- rank source-target pairs using JEPA-style latent features + PAD
- run case matrices for high-shift vs low-shift comparisons

## Repository Layout

```text
README.md
requirements.txt
setup.py
run.ps1

src/
  pipeline.py
  adaptnas/
  data/
    datasets.py
    omni_smd.py
    tranad_smd.py
    swat.py
  families/
    omni_anomaly.py
    tranad.py
    usad.py
  models/
  shift/
    pad.py
    ts_jepa.py
  ts_tcc/
  utils/
    data_paths.py
    metrics.py

scripts/
  preprocess_smd.py
  preprocess_smap.py
  preprocess_msl.py
  make_uad_smd.py
  make_uad_smap.py
  make_uad_msl.py
  build_domain_shift_smd.py
  build_domain_shift_smap.py
  build_domain_shift_msl.py
  pilot_rank_jepa_pad_smd.py
  run_domain_shift_case_matrix.py
  run_all_smd.py
  run_tranad_smd.py
  run_tranad_upstream_smd.py
  run_usad_swat.py
  run_usad_upstream_swat.py
  compare_mode_benchmarks.py
  export_figures.py

docs/
  docs/
    *.md
    *.tex
```

## Environment Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

If you are using the local Windows `venv/` in this repo:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Optional CUDA reinstall on Windows:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA:

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Data Expectations

### Raw SMD

Raw SMD is expected in a layout like:

```text
data/ServerMachineDataset/
  train/
  test/
  test_label/
```

or:

```text
.../
  train/
  test/
  labels/
```

The repo now resolves raw SMD roots automatically from common locations such as:

- `data/ServerMachineDataset`
- `external/OmniAnomaly/ServerMachineDataset`
- `external/tranad_upstream/data/SMD`

### Raw SWaT

USAD expects:

```text
data/SWaT/
  SWaT_Dataset_Normal_v1.csv
  SWaT_Dataset_Attack_v0.csv
```

### Raw SMAP

Raw SMAP is expected in a Telemanom-style layout like:

```text
data/SMAP_MSL/
  train/
  test/
  labeled_anomalies.csv
```

The repo resolves raw SMAP roots automatically from common locations such as:

- `data/SMAP_MSL`
- `external/tranad_upstream/data/SMAP_MSL`
- `external/OmniAnomaly/data`

### Raw MSL

Raw MSL uses the same Telemanom-style layout:

```text
data/SMAP_MSL/
  train/
  test/
  labeled_anomalies.csv
```

The repo resolves raw MSL roots from the same common locations as SMAP.

### Processed SMD cache

For `default_nasade` experiments, raw SMD is typically converted into per-machine caches:

```text
data/smd/machine-*/source.npz
data/smd/machine-*/target.npz
```

### Processed SMAP cache

For `default_nasade` experiments, raw SMAP is typically converted into per-channel caches:

```text
data/smap/<channel-id>/source.npz
data/smap/<channel-id>/target.npz
```

### Processed MSL cache

For `default_nasade` experiments, raw MSL is typically converted into per-channel caches:

```text
data/msl/<channel-id>/source.npz
data/msl/<channel-id>/target.npz
```

## Quick Start

### 1. Preprocess SMD into per-machine caches

```powershell
.\venv\Scripts\python.exe scripts\preprocess_smd.py --raw_root data/ServerMachineDataset --out_root data/smd --machine machine-1-1 --window 128 --stride 64
```

### 2. Build a cross-machine SMD UAD experiment split

```powershell
.\venv\Scripts\python.exe scripts\make_uad_smd.py `
  --machine_dir data/smd/machine-1-1 `
  --target_machine_dir data/smd/machine-1-3 `
  --split_mode search `
  --shift_level hard `
  --target_pool_frac 0.2 `
  --val_frac 0.3 `
  --guard 4 `
  --out_dir data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3
```

### 3. Preprocess SMAP into per-channel caches

```powershell
.\venv\Scripts\python.exe scripts\preprocess_smap.py --raw_root data/SMAP_MSL --out_root data/smap --channel A-1 --window 128 --stride 64
```

### 4. Build a cross-entity SMAP covariate-shift split

```powershell
.\venv\Scripts\python.exe scripts\make_uad_smap.py `
  --channel_dir data/smap/A-1 `
  --target_channel_dir data/smap/A-3 `
  --split_mode search `
  --shift_level auto `
  --target_pool_frac 0.1 `
  --val_frac 0.2 `
  --out_dir data/smap_experiments/cross_entity_auto/A-1__to__A-3
```

### 5. Preprocess MSL into per-channel caches

```powershell
.\venv\Scripts\python.exe scripts\preprocess_msl.py --raw_root data/SMAP_MSL --out_root data/msl --channel M-1 --window 128 --stride 64
```

### 6. Build a cross-entity MSL covariate-shift split

```powershell
.\venv\Scripts\python.exe scripts\make_uad_msl.py `
  --channel_dir data/msl/M-1 `
  --target_channel_dir data/msl/M-7 `
  --split_mode search `
  --shift_level auto `
  --target_pool_frac 0.1 `
  --val_frac 0.2 `
  --out_dir data/msl_experiments/cross_entity_auto/M-1__to__M-7
```

### 7. Run `default_nasade` in source-only mode

```powershell
.\venv\Scripts\python.exe -m src.pipeline `
  --dataset_or_paths data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/train_normal.npz,data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/val_mixed.npz,data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/test_mixed.npz `
  --mode uad_source `
  --family default_nasade `
  --oneclass_method deepsvdd `
  --epochs_pretrain 10 `
  --search_candidates 5 `
  --batch_size 64 `
  --device cuda
```

### 8. Run `default_nasade` in combined mode

```powershell
.\venv\Scripts\python.exe -m src.pipeline `
  --dataset_or_paths data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/train_normal.npz,data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/target_pool_unlabeled.npz,data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/val_mixed.npz,data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-3/test_mixed.npz `
  --mode adaptnas_combined `
  --family default_nasade `
  --oneclass_method autoencoder `
  --epochs_pretrain 10 `
  --search_candidates 5 `
  --batch_size 64 `
  --device cuda
```

The same `src.pipeline` commands work for SMAP/MSL once `dataset_or_paths` points to a generated folder under `data/smap_experiments/` or `data/msl_experiments/`.

### 9. Run OmniAnomaly on raw SMD

```powershell
.\venv\Scripts\python.exe -m src.pipeline `
  --mode uad_source `
  --family omni_anomaly `
  --raw_smd_root data/ServerMachineDataset `
  --machine machine-1-1 `
  --search_candidates 20 `
  --device cuda
```

### 10. Run TranAD on raw SMD

```powershell
.\venv\Scripts\python.exe -m src.pipeline `
  --mode uad_source `
  --family tranad `
  --raw_smd_root data/ServerMachineDataset `
  --machine machine-1-1 `
  --search_candidates 4 `
  --tranad_epochs 5 `
  --tranad_final_epochs 5 `
  --device cuda
```

Or use the dedicated helper:

```powershell
.\venv\Scripts\python.exe scripts\run_tranad_smd.py --machine machine-1-1 --device cuda
```

### 11. Run USAD on raw SWaT

```powershell
.\venv\Scripts\python.exe -m src.pipeline `
  --mode uad_source `
  --family usad `
  --swat_train_csv data/SWaT/SWaT_Dataset_Normal_v1.csv `
  --swat_test_csv data/SWaT/SWaT_Dataset_Attack_v0.csv `
  --search_candidates 5 `
  --device cuda
```

Or use the helper:

```powershell
.\venv\Scripts\python.exe scripts\run_usad_swat.py --device cuda
```

## `run.ps1` Wrapper

`run.ps1` is the main Windows convenience wrapper for:

- `default_nasade`
- `omni_anomaly`
- `usad`

Examples:

```powershell
.\run.ps1 -Mode uad_source -Family default_nasade -DataDir data\smd_experiments\cross_machine_hard\machine-1-1__to__machine-1-3
```

```powershell
.\run.ps1 -Mode adaptnas_combined -Family default_nasade -DataDir data\smd_experiments\cross_machine_hard\machine-1-1__to__machine-1-3
```

```powershell
.\run.ps1 -Mode uad_source -Family omni_anomaly -RawSmdRoot data\ServerMachineDataset -Machine machine-1-1
```

```powershell
.\run.ps1 -Mode uad_source -Family usad -SwatTrainCsv data\SWaT\SWaT_Dataset_Normal_v1.csv -SwatTestCsv data\SWaT\SWaT_Dataset_Attack_v0.csv
```

Note:

- `run.ps1` currently does not expose `family=tranad`
- it auto-resolves device from `-Device auto`
- it auto-resolves raw SMD roots from common locations

## Domain-Shift Tooling

### Build a temporal/cross-machine suite

```powershell
.\venv\Scripts\python.exe scripts\build_domain_shift_smd.py `
  --shift_levels medium,hard `
  --build_temporal `
  --build_cross_machine `
  --same_family_only `
  --topk_cross 1
```

This writes experiment folders under `data/smd_experiments/` and a manifest file.

### Rank source-target pairs with JEPA-style latent PAD

```powershell
.\venv\Scripts\python.exe scripts\pilot_rank_jepa_pad_smd.py `
  --raw_smd_root data/ServerMachineDataset `
  --device cuda
```

Output is written under:

- `outputs/benchmarks/jepa_pad_pilot/`

### Run a domain-shift case matrix

```powershell
.\venv\Scripts\python.exe scripts\run_domain_shift_case_matrix.py `
  --cases L1,H2,H3 `
  --include_anchor `
  --seeds 42 `
  --device cuda
```

Output is written under:

- `outputs/benchmarks/domain_shift_matrix/`

### Compare `uad_source` vs `adaptnas_combined`

```powershell
.\venv\Scripts\python.exe scripts\compare_mode_benchmarks.py `
  --source_dirs outputs/benchmarks/default_nasade-uad_source-cross_medium_source `
  --combined_dirs outputs/benchmarks/default_nasade-adaptnas_combined-cross_medium_combined `
  --out outputs/benchmarks/compare_cross_medium.json
```

## Batch Runners

### Batch-run generated `default_nasade` experiment folders

```powershell
.\venv\Scripts\python.exe scripts\run_all_smd.py --mode uad_source --family default_nasade --data_root data/smd_experiments/cross_machine_hard
```

```powershell
.\venv\Scripts\python.exe scripts\run_all_smd.py --mode adaptnas_combined --family default_nasade --data_root data/smd_experiments/cross_machine_hard
```

### Batch-run OmniAnomaly on raw SMD

```powershell
.\venv\Scripts\python.exe scripts\run_all_smd.py --mode uad_source --family omni_anomaly --raw_smd_root data/ServerMachineDataset --machines machine-1-1,machine-1-2
```

## Outputs

Generated outputs are typically written to:

- `outputs/results.json`
- `outputs/baselines_summary.json`
- `outputs/baselines/`
- `outputs/checkpoints/`
- `outputs/figures/`
- `outputs/logs/`
- `outputs/benchmarks/`

These generated artifacts are now ignored by Git in this repo, so they can be used locally without polluting commits.

## Documentation

Experiment notes and report drafts live under [docs/docs](docs/docs). This includes:

- method notes for Ben-David + JEPA + PAD ranking
- domain-shift setup reports
- detailed Vietnamese writeups and advisor-facing report drafts

## Notes and Limitations

- `default_nasade` is the only family that supports `adaptnas_combined`.
- `omni_anomaly`, `tranad`, and `usad` are source-only families.
- `run.ps1` currently supports `default_nasade`, `omni_anomaly`, and `usad`, but not `tranad`.
- `run_all_smd.py` currently supports `default_nasade` and `omni_anomaly`.
- Raw datasets are not bundled by Git.
- For cross-machine experiments, prefer using generated folders under `data/smd_experiments/` rather than editing the original per-machine caches.
- For SMAP covariate-shift experiments, the default builder keeps transfer within the same channel-prefix family unless you explicitly allow cross-prefix pairs.
- For MSL covariate-shift experiments, the default builder also keeps transfer within the same channel-prefix family unless you explicitly allow cross-prefix pairs.
