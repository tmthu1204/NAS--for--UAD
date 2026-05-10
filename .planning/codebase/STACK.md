# Technology Stack

**Analysis Date:** 2026-05-10

## Languages

**Primary:**
- Python 3.12.13 - active interpreter in `venv\Scripts\python.exe`; all core training, data preparation, benchmarking, and evaluation code lives under `src/` and `scripts/`

**Secondary:**
- PowerShell - Windows execution wrapper and CUDA/device selection in `run.ps1`
- Bash - POSIX execution wrapper in `scripts/run_pipeline.sh`
- Jupyter Notebook - exploratory and upstream reference artifacts in `flowchart.ipynb`, `external/usad_upstream/USAD.ipynb`, and `src/ts_tcc/data_preprocessing/fault_diagnosis/Data_preprocessing.ipynb`
- MATLAB - upstream TS-TCC preprocessing artifact in `src/ts_tcc/data_preprocessing/fault_diagnosis/Data_loading_segmentation.m`

## Runtime

**Environment:**
- CPython 3.12.13 detected in `venv\Scripts\python.exe`
- `setup.py` declares `python_requires=">=3.8"` for the package `adapt_ts_project`
- Runtime device selection is local-process based: `run.ps1` probes `torch.cuda.is_available()` and `src/pipeline.py` defaults to `cuda` when available, otherwise `cpu`

**Package Manager:**
- `pip` 25.0.1 inside `venv`
- `setuptools` packaging via `setup.py`
- Lockfile: missing

## Frameworks

**Core:**
- PyTorch 2.11.0 - neural network models, training loops, checkpointing, and dataloaders in `src/pipeline.py`, `src/models/*.py`, `src/families/*.py`, `src/adaptnas/*.py`, and `src/ts_tcc/`
- NumPy 2.4.3 - array transforms, `.npz` protocol handling, and window construction in `src/data/*.py` and `scripts/*.py`
- scikit-learn 1.8.0 - metrics, normalization, and domain-shift scoring in `src/utils/metrics.py`, `src/data/swat.py`, and `scripts/make_uad_smd.py`
- Vendored TS-TCC implementation - self-supervised pretraining stack in `src/ts_tcc/`, integrated through `src.ts_tcc.trainer.trainer.TSTrainer`

**Testing:**
- Not detected

**Build/Dev:**
- `setuptools` - package metadata and install surface in `setup.py`
- PowerShell and Bash wrappers - reproducible local runs via `run.ps1` and `scripts/run_pipeline.sh`
- Python orchestration scripts - benchmark runners in `scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_smd.py`, and `scripts/run_usad_upstream_swat.py`
- Matplotlib 3.10.8 - offline plots and saved figures in `src/utils/visualization.py` and `scripts/export_figures.py`

## Key Dependencies

**Critical:**
- `torch==2.11.0` - every model family and training path depends on it; see `src/pipeline.py`, `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`, and `src/models/deepsvdd.py`
- `numpy==2.4.3` - primary numeric container for raw series, windows, labels, and saved `.npz` artifacts in `src/data/omni_smd.py`, `src/data/swat.py`, `src/data/tranad_smd.py`, and `scripts/preprocess_smd.py`
- `scikit-learn==1.8.0` - anomaly metrics and preprocessing helpers in `src/utils/metrics.py`, `src/data/swat.py`, and `scripts/make_uad_smd.py`
- `pandas==3.0.1` - CSV and TXT ingestion for SWaT and SMD in `src/data/swat.py`, `src/data/omni_smd.py`, and `scripts/preprocess_smd.py`
- `scipy==1.17.1` - POT/SPOT threshold modeling through `scipy.stats.genpareto` in `src/families/omni_spot.py`
- `einops==0.8.1` - tensor reshaping for TS-TCC attention blocks in `src/ts_tcc/models/attention.py`

**Infrastructure:**
- `matplotlib==3.10.8` - plot generation in `src/utils/visualization.py` and `scripts/export_figures.py`
- `torchvision==0.26.0` - installed by `requirements.txt`; not directly imported by the current pipeline, but provisioned alongside PyTorch in the repo environment
- Optional `mne` - used only for Sleep-EDF preprocessing in `scripts/preprocess_sleepedf.py` and `src/ts_tcc/data_preprocessing/sleep-edf/preprocess_sleep_edf.py`; not pinned in `requirements.txt`
- Bundled upstream USAD code - local benchmark-only import surface in `external/usad_upstream/` and `scripts/run_usad_upstream_swat.py`

## Configuration

**Environment:**
- Main experiment configuration is CLI-driven through `src/pipeline.py`; wrappers `run.ps1` and `scripts/run_pipeline.sh` assemble mode, family, data paths, and hyperparameters rather than loading YAML or TOML config files
- Dataset defaults are hard-coded in code paths: raw SMD under `data/ServerMachineDataset` in `src/pipeline.py`, raw SWaT under `data/SWaT` in `src/pipeline.py`, and generated `.npz` windows under `data/smd` and `data/smd_experiments` in `scripts/preprocess_smd.py` and `scripts/build_domain_shift_smd.py`
- Upstream TS-TCC keeps Python config modules in `src/ts_tcc/config_files/*.py`
- No repo-local skills were detected under `.codex/skills/` or `.agents/skills/`
- No `.env` files were detected at repo root during this scan

**Build:**
- Dependency manifest: `requirements.txt`
- Package metadata: `setup.py`
- Windows runner: `run.ps1`
- POSIX runner: `scripts/run_pipeline.sh`
- No `pyproject.toml`, no lockfile, and no container build files were detected

## Platform Requirements

**Development:**
- Windows-first local workflow with `run.ps1` and the checked-in `venv\Scripts\python.exe`
- POSIX shell support is also present via `scripts/run_pipeline.sh`
- Optional NVIDIA GPU acceleration is supported through PyTorch CUDA detection in `run.ps1` and `src/pipeline.py`
- Local dataset directories must exist on disk for the chosen workflow: `data/ServerMachineDataset`, `data/SWaT`, `data/smd`, and `data/smd_experiments`
- Checked-in environment artifacts under `external\miniconda3`, `external\conda-envs\omni36`, and `external\Miniconda3-latest-Windows-x86_64.exe` exist in the repo, but the main execution path uses the project `venv`

**Production:**
- No deployed service target was detected
- The repo operates as a local research and benchmarking pipeline that writes artifacts to `outputs/` and generated dataset folders under `data/`

---

*Stack analysis: 2026-05-10*
