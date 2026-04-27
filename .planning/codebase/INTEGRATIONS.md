# External Integrations

**Analysis Date:** 2026-04-27

## APIs & External Services

**Runtime APIs:**
- None detected in first-party code under `src/`, `scripts/`, `run.ps1`, or `README.md`.
  - SDK/Client: Not applicable
  - Auth: Not applicable

**Datasets on disk:**
- Raw SMD dataset under `data/ServerMachineDataset/` is the source for paper-style Omni and TranAD runs.
  - SDK/Client: `src/data/omni_smd.py`, `src/data/tranad_smd.py`, `scripts/preprocess_smd.py`
  - Auth: None
- Prepared SMD `.npz` splits under `data/smd/` and experiment suites under `data/smd_experiments/` are the canonical inputs for `default_nasade` and `adaptnas_combined`.
  - SDK/Client: `src/pipeline.py`, `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`
  - Auth: None
- Raw SWaT CSV files under `data/SWaT/` are the input for `family=usad`.
  - SDK/Client: `src/data/swat.py`, `src/pipeline.py`, `run.ps1`
  - Auth: None
- Auxiliary TS-TCC datasets are kept locally under `data/UCI HAR Dataset/`, `data/uci_har/`, and the preprocessing paths referenced by `src/ts_tcc/README.md`.
  - SDK/Client: `src/ts_tcc/data_preprocessing/uci_har/preprocess_har.py`, `scripts/preprocess_sleepedf.py`
  - Auth: None

**Vendored upstream baselines:**
- `external/OmniAnomaly` is a local copy of the OmniAnomaly reference implementation used for comparison and environment reference.
  - SDK/Client: local Python package tree in `external/OmniAnomaly/` and `external/OmniAnomaly/requirements.txt`
  - Auth: None
- `external/tranad_upstream` is a local copy of the TranAD reference implementation used for comparison and dependency reference.
  - SDK/Client: local Python package tree in `external/tranad_upstream/` and `external/tranad_upstream/requirements.txt`
  - Auth: None
- `external/usad_upstream` is imported directly by `scripts/run_usad_upstream_swat.py` through `sys.path` to run an upstream-faithful USAD benchmark.
  - SDK/Client: `scripts/run_usad_upstream_swat.py`, `external/usad_upstream/usad.py`
  - Auth: None
- `external/miniconda3` and `external/conda-envs/omni36` provide a local legacy Conda toolchain for older baseline workflows.
  - SDK/Client: `external/miniconda3/`, `external/conda-envs/omni36/conda-meta/history`
  - Auth: None

## Data Storage

**Databases:**
- None. No SQL, NoSQL, vector store, or ORM integration was detected in `src/` or `scripts/`.
  - Connection: Not applicable
  - Client: Not applicable

**File Storage:**
- Local filesystem only.
- Raw inputs live under `data/ServerMachineDataset/`, `data/SWaT/`, and TS-TCC-related dataset folders in `data/`.
- Prepared experiment inputs live under `data/smd/` and `data/smd_experiments/`.
- Run artifacts are written to `outputs/results.json`, `outputs/baselines/*.json`, `outputs/baselines_summary.json`, `outputs/checkpoints/*.pt`, `outputs/figures/*.png`, `outputs/logs/`, and benchmark folders in `outputs/benchmarks/`.
- Standalone TS-TCC runs write experiment artifacts under `results/` and the log directories managed by `src/ts_tcc/main.py`.

**Caching:**
- None. Reusable state is persisted as files on disk: checkpoints, `.npz` datasets, `.npy` diagnostics, JSON summaries, and figures.

## Authentication & Identity

**Auth Provider:**
- None.
  - Implementation: No API keys, OAuth flows, tokens, or user/session code were detected in `src/`, `scripts/`, `run.ps1`, or `README.md`.

## Monitoring & Observability

**Error Tracking:**
- None.

**Logs:**
- Batch wrappers in `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, `scripts/run_tranad_upstream_smd.py`, `scripts/run_usad_swat.py`, and `scripts/run_usad_upstream_swat.py` write local log files under `outputs/logs/`.
- `src/pipeline.py` writes structured result payloads to `outputs/results.json` and related JSON artifacts, and `scripts/export_figures.py` turns them into plots.

## CI/CD & Deployment

**Hosting:**
- Not applicable. No web host, daemon, scheduler, or service deployment target is configured.

**CI Pipeline:**
- None detected. No `.github/workflows/`, `.gitlab-ci.yml`, `azure-pipelines.yml`, `tox.ini`, or `noxfile.py` is present at the repo root.

## Environment Configuration

**Required env vars:**
- None required by first-party application code.
- `src/pipeline.py` sets `PYTHONHASHSEED` internally for deterministic seeding.
- `scripts/run_pipeline.sh` exports `CUDA_VISIBLE_DEVICES` for local GPU selection.
- `run.ps1` resolves CPU vs CUDA from CLI flags and local PyTorch availability instead of secret-backed configuration.
- Practical path note: `src/pipeline.py` and `run.ps1` default SWaT paths to `data/SWaT/SWaT_Dataset_Normal_v1.csv` and `data/SWaT/SWaT_Dataset_Attack_v0.csv`, while this checkout currently contains `data/SWaT/normal.csv` and `data/SWaT/attack.csv`.

**Secrets location:**
- Not detected for first-party code.
- No root `.env*` files are present, and no secret-management system is wired into `src/` or `scripts/`.

## Webhooks & Callbacks

**Incoming:**
- None.

**Outgoing:**
- None.

---

*Integration audit: 2026-04-27*
