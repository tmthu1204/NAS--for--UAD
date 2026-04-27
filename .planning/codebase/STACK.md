# Technology Stack

**Analysis Date:** 2026-04-27

## Languages

**Primary:**
- Python 3.12.13 in the checked-in virtual environment (`venv/pyvenv.cfg`, `./venv/Scripts/python.exe --version`); `setup.py` declares `python_requires=">=3.8"` for the packaged project in `src/`.

**Secondary:**
- PowerShell for the main Windows launcher in `run.ps1`.
- Bash for the lightweight Unix-style launcher in `scripts/run_pipeline.sh`.
- Markdown for operational and research documentation in `README.md` and `src/ts_tcc/README.md`.
- Notebook and MATLAB artifacts for research preprocessing in `flowchart.ipynb`, `src/ts_tcc/data_preprocessing/fault_diagnosis/Data_preprocessing.ipynb`, and `src/ts_tcc/data_preprocessing/fault_diagnosis/Data_loading_segmentation.m`.

## Runtime

**Environment:**
- Local `venv/` is the primary runtime expected by `run.ps1`, with `venv/Scripts/python.exe` invoked directly.
- Optional CUDA acceleration is documented in `README.md` via a manual PyTorch wheel reinstall against the official `cu118` index.
- A second legacy runtime exists under `external/conda-envs/omni36`; `external/conda-envs/omni36/conda-meta/history` shows it was created as a Python 3.6.13 Conda environment for older Omni-related tooling.

**Package Manager:**
- `pip` is the active installer path in `README.md`, `requirements.txt`, and `pip_list.txt`.
- `setuptools` packaging is defined in `setup.py` with `find_packages(where="src")`.
- Lockfile: missing. `requirements.txt` is present, but no `pyproject.toml`, `Pipfile.lock`, `poetry.lock`, or `uv.lock` is present at the repo root.

## Frameworks

**Core:**
- PyTorch 2.11.0 and `torchvision` 0.26.0 are the core ML stack in `requirements.txt`, `src/pipeline.py`, `src/models/`, and `src/families/`.
- NumPy 2.4.3, pandas 3.0.1, SciPy 1.17.1, and scikit-learn 1.8.0 support preprocessing, normalization, metrics, and shift scoring in `requirements.txt`, `src/data/`, `src/utils/metrics.py`, and `scripts/make_uad_smd.py`.
- TS-TCC is vendored into `src/ts_tcc/` and integrated into the main pipeline through `src.ts_tcc.trainer.trainer.TSTrainer` in `src/pipeline.py`.

**Testing:**
- Not detected. No `pytest`, `unittest`, or dedicated test runner config was found in the repo root or `src/`.

**Build/Dev:**
- `argparse` drives all first-party CLIs in `src/pipeline.py`, `src/ts_tcc/main.py`, and `scripts/*.py`.
- `matplotlib` 3.10.8 is used for local plot generation in `src/utils/visualization.py` and `scripts/export_figures.py`.
- `setuptools` is the only first-party packaging/build layer in `setup.py`.

## Dependency Groups

**Main runtime:**
- `requirements.txt` pins `torch`, `torchvision`, `numpy`, `scikit-learn`, `matplotlib`, `pandas`, `scipy`, `joblib`, `pillow`, `python-dateutil`, and `einops` for the primary `src/` pipeline.

**Optional preprocessing:**
- `scripts/preprocess_sleepedf.py` imports `mne`, which is not pinned in `requirements.txt`.
- `src/ts_tcc/README.md` documents extra TS-TCC-side requirements such as `openpyxl`, `mne==0.20.7`, and `mat4py` for auxiliary dataset preparation.

**Vendored baseline runtimes:**
- `external/OmniAnomaly/requirements.txt` carries a TensorFlow-era stack including `tensorflow-gpu==1.12.0`, `tensorflow_probability==0.5.0`, and Git installs for `zhusuan` and `tfsnippet`.
- `external/tranad_upstream/requirements.txt` adds `dgl`, `SciencePlots`, and `xlrd==1.2.0`.
- `external/usad_upstream` ships code and weights directly in the repo and is imported by `scripts/run_usad_upstream_swat.py` through `sys.path`, not via `pip`.

## Key Dependencies

**Critical:**
- `torch==2.11.0` in `requirements.txt` powers the end-to-end pipeline in `src/pipeline.py` and the model families in `src/families/`.
- `numpy==2.4.3` in `requirements.txt` is the common array format for raw loaders, `.npz` datasets, and metrics across `src/` and `scripts/`.
- `scikit-learn==1.8.0` in `requirements.txt` is required by `src/utils/metrics.py`, `src/data/swat.py`, and `scripts/make_uad_smd.py`.
- `pandas==3.0.1` in `requirements.txt` is required for SMD and SWaT ingestion in `src/data/omni_smd.py`, `src/data/swat.py`, and `scripts/preprocess_smd.py`.
- `einops==0.8.1` in `requirements.txt` is used by TS-TCC attention code in `src/ts_tcc/models/attention.py`.

**Infrastructure:**
- `matplotlib==3.10.8` in `requirements.txt` backs plotting in `src/utils/visualization.py` and `scripts/export_figures.py`.
- `joblib==1.5.3`, `pillow==12.1.1`, and `python-dateutil==2.9.0.post0` are pinned in `requirements.txt` and visible in `pip_list.txt` as environment support packages.
- `subprocess` orchestration is used in `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, and `scripts/run_usad_swat.py` to batch local experiment runs.

## Execution Entry Points

**Primary:**
- `src/pipeline.py`: canonical CLI entrypoint for `python -m src.pipeline` across `default_nasade`, `omni_anomaly`, `usad`, and `tranad`.
- `run.ps1`: Windows-first launcher that validates datasets, resolves CPU vs CUDA, and invokes `venv/Scripts/python.exe -m src.pipeline`.
- `scripts/run_pipeline.sh`: minimal Bash wrapper for source-mode and combined-mode runs.

**Data preparation:**
- `scripts/preprocess_smd.py`: converts raw SMD or legacy CSV layouts into `data/smd/machine-*/source.npz` and `target.npz`.
- `scripts/make_uad_smd.py`: builds `train_normal.npz`, `target_pool_unlabeled.npz`, `val_mixed.npz`, `test_mixed.npz`, and `split_metadata.json`.
- `scripts/build_domain_shift_smd.py`: batch-builds temporal and cross-machine experiment suites plus `data/smd_experiments/manifest.json`.
- `scripts/preprocess.py` and `scripts/preprocess_sleepedf.py`: auxiliary preprocessors for generic `.npz`/`.npy` inputs and Sleep-EDF data.

**Benchmarking and reporting:**
- `scripts/run_all_smd.py`: batch runner for prepared SMD experiments and raw Omni runs.
- `scripts/run_tranad_smd.py` and `scripts/run_usad_swat.py`: family-specific wrappers around `src/pipeline.py`.
- `scripts/run_tranad_upstream_smd.py` and `scripts/run_usad_upstream_swat.py`: upstream-faithful baseline runners using local vendored code.
- `scripts/compare_mode_benchmarks.py` and `scripts/export_figures.py`: result summarization and figure export.
- `src/ts_tcc/main.py`: standalone TS-TCC training entrypoint separate from the main NAS-ADE pipeline.

## Configuration

**Environment:**
- Configuration is CLI-first. `src/pipeline.py` exposes the main argument surface, while `run.ps1` mirrors the same knobs for Windows usage.
- No root `.env*` files are present. Runtime configuration is path-based and argument-based rather than secret-based.
- The only environment variable mutated by first-party Python code is `PYTHONHASHSEED` in `src/pipeline.py` for determinism.

**Build:**
- `requirements.txt` and `setup.py` are the only first-party dependency manifests.
- No `pyproject.toml`, `tox.ini`, `noxfile.py`, `.github/workflows/`, or Dockerfile was detected at the repo root.

## Platform Requirements

**Development:**
- Windows is a first-class platform because `run.ps1` assumes `venv/Scripts/python.exe`.
- Unix-like shells are partially supported through `scripts/run_pipeline.sh`.
- NVIDIA GPU support is optional and documented in `README.md`; CPU fallback is built into `run.ps1`.
- Meaningful runs require large local datasets under `data/`, especially `data/ServerMachineDataset`, `data/smd`, `data/smd_experiments`, and `data/SWaT`.

**Production:**
- Not a deployed service. The repo is structured for offline research runs, local preprocessing, and batch experiments that write to `outputs/`, `results/`, and `data/`.

---

*Stack analysis: 2026-04-27*
