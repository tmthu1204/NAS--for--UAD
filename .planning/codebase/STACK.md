# Technology Stack

**Analysis Date:** 2026-04-27

## Languages

**Primary:**
- Python 3.12.13 is the active runtime in the checked-in virtual environment at `venv/`, and all first-party implementation code lives under `src/` and `scripts/`.

**Secondary:**
- PowerShell drives the main Windows launcher in `run.ps1`.
- Bash is supported by the lightweight Unix-style wrapper in `scripts/run_pipeline.sh`.
- Markdown is used for operational and research docs in `README.md` and `src/ts_tcc/README.md`.
- Jupyter notebook and MATLAB artifacts exist for research support in `flowchart.ipynb`, `src/ts_tcc/data_preprocessing/fault_diagnosis/Data_preprocessing.ipynb`, and `src/ts_tcc/data_preprocessing/fault_diagnosis/Data_loading_segmentation.m`.

## Runtime

**Environment:**
- Local `venv/` is the primary runtime. `run.ps1` invokes `venv/Scripts/python.exe` directly and fails fast if that interpreter is missing.
- `setup.py` declares `python_requires=">=3.8"`, but the checked-in environment is already on Python 3.12.
- Optional CUDA acceleration is documented in `README.md` and is selected dynamically by `run.ps1` after checking `torch.cuda.is_available()`.
- A legacy baseline environment exists at `external/conda-envs/omni36`; `external/conda-envs/omni36/conda-meta/history` shows it was created from `external/miniconda3` with Python 3.6 for older Omni-related tooling.

**Package Manager:**
- `pip` is the active installer path in `README.md`, `requirements.txt`, and the environment snapshot `pip_list.txt`.
- `setuptools` packages the project through `setup.py` with `find_packages(where="src")`.
- Lockfile: missing. The repo has `requirements.txt`, but no `pyproject.toml`, `poetry.lock`, `Pipfile.lock`, or `uv.lock`.

## Frameworks

**Core:**
- PyTorch 2.11.0 and `torchvision` 0.26.0 are the core ML runtime for `src/pipeline.py`, `src/models/`, `src/adaptnas/`, and `src/families/`.
- NumPy 2.4.3, pandas 3.0.1, SciPy 1.17.1, and scikit-learn 1.8.0 support array processing, preprocessing, statistics, and evaluation in `src/data/`, `src/utils/metrics.py`, and `scripts/make_uad_smd.py`.
- TS-TCC is vendored under `src/ts_tcc/` and reused from `src/pipeline.py` through `src.ts_tcc.trainer.trainer.TSTrainer`.
- The main pipeline supports four families in `src/pipeline.py`: `default_nasade`, `omni_anomaly`, `usad`, and `tranad`.

**Testing:**
- Not detected. No `pytest`, `unittest` suite, or dedicated test-runner config is present at the repo root or under `src/`.

**Build/Dev:**
- `argparse` is the common CLI layer for `src/pipeline.py`, `src/ts_tcc/main.py`, and the scripts in `scripts/`.
- `matplotlib` 3.10.8 drives figure generation in `src/utils/visualization.py` and `scripts/export_figures.py`.
- `setuptools` is the only first-party packaging layer in `setup.py`.

## Key Dependencies

**Critical:**
- `torch==2.11.0` in `requirements.txt` powers training, scoring, checkpointing, and device selection across `src/pipeline.py` and `src/families/`.
- `numpy==2.4.3` in `requirements.txt` is the dominant data interchange type for `.npz` datasets, raw-window transforms, and metric inputs across `src/` and `scripts/`.
- `scikit-learn==1.8.0` in `requirements.txt` is required by `src/utils/metrics.py`, `src/data/swat.py`, and `scripts/make_uad_smd.py`.
- `pandas==3.0.1` in `requirements.txt` is required for raw SMD and SWaT ingestion in `src/data/omni_smd.py` and `src/data/swat.py`.
- `einops==0.8.1` in `requirements.txt` is required by TS-TCC attention modules in `src/ts_tcc/models/attention.py`.

**Infrastructure:**
- `matplotlib==3.10.8` in `requirements.txt` backs local plotting and saved figures in `outputs/figures/`.
- `joblib==1.5.3`, `pillow==12.1.1`, and `python-dateutil==2.9.0.post0` are pinned in `requirements.txt` and appear in `pip_list.txt` as environment support packages.
- `scripts/preprocess_sleepedf.py` requires `mne`, which is not declared in `requirements.txt`.
- `src/ts_tcc/README.md` documents additional TS-TCC-side preprocessing dependencies such as `openpyxl`, `mne==0.20.7`, and `mat4py`.
- Vendored upstream baselines bring their own dependency stacks through `external/OmniAnomaly/requirements.txt` and `external/tranad_upstream/requirements.txt`.

## Configuration

**Environment:**
- Configuration is CLI-first. The canonical argument surface is in `src/pipeline.py`, and `run.ps1` mirrors the most common options for Windows usage.
- No root `.env*` file is present. Runtime configuration is path-driven and flag-driven rather than secret-driven.
- The only environment variable written by first-party Python code is `PYTHONHASHSEED` in `src/pipeline.py` for determinism.
- `scripts/run_pipeline.sh` exports `CUDA_VISIBLE_DEVICES` for local GPU selection.
- `run.ps1` exposes `default_nasade`, `omni_anomaly`, and `usad`, while `src/pipeline.py` additionally supports `tranad`.

**Build:**
- `requirements.txt` and `setup.py` are the only first-party dependency manifests.
- `pip_list.txt` is an environment snapshot, not an installable lockfile.
- No `pyproject.toml`, `tox.ini`, `noxfile.py`, Dockerfile, or CI workflow configuration exists at the repo root.

## Platform Requirements

**Development:**
- Windows is the primary platform because `run.ps1` assumes `venv/Scripts/python.exe`.
- Unix-like shells are partially supported through `scripts/run_pipeline.sh`.
- NVIDIA GPU support is optional and depends on a CUDA-capable PyTorch install as described in `README.md`.
- Meaningful runs require large local datasets under `data/`, especially `data/ServerMachineDataset`, `data/smd`, `data/smd_experiments`, and `data/SWaT`.
- Local benchmark comparisons also depend on vendored upstream code in `external/OmniAnomaly`, `external/tranad_upstream`, and `external/usad_upstream`.

**Production:**
- Not a deployed service. The repo is structured for offline research runs, preprocessing jobs, and benchmark generation that write artifacts to `outputs/`, `results/`, and `data/`.

---

*Stack analysis: 2026-04-27*
