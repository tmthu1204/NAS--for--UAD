# External Integrations

**Analysis Date:** 2026-05-10

## APIs & External Services

**Datasets:**
- Server Machine Dataset (SMD) local copy - raw time-series source for `family=omni_anomaly`, `family=tranad`, and SMD preprocessing utilities
  - SDK/Client: local filesystem readers in `src/data/omni_smd.py`, `src/data/tranad_smd.py`, and `scripts/preprocess_smd.py`
  - Auth: None
- SWaT local CSV copy - raw train/test source for `family=usad`
  - SDK/Client: local filesystem readers in `src/data/swat.py`
  - Auth: None

**Bundled Research Code:**
- TS-TCC upstream implementation - vendored self-supervised encoder pretraining stack reused by the main pipeline
  - SDK/Client: local package in `src/ts_tcc/`
  - Auth: None
- USAD upstream reference implementation - benchmark-only import used for side-by-side comparison runs
  - SDK/Client: local vendored module in `external/usad_upstream/`, loaded by `scripts/run_usad_upstream_swat.py`
  - Auth: None

**Network Services:**
- Not detected in `src/` or `scripts/` main execution paths
  - SDK/Client: Not applicable
  - Auth: Not applicable
- `external/usad_upstream/gdrivedl.py` contains URL-fetching code for upstream data download, but no repo runner imports or calls it
  - SDK/Client: `urllib`
  - Auth: None

## Data Storage

**Databases:**
- None
  - Connection: Not applicable
  - Client: Not applicable

**File Storage:**
- Local filesystem only
- Raw SMD input layout:
  - `data/ServerMachineDataset/train/*.txt`
  - `data/ServerMachineDataset/test/*.txt`
  - `data/ServerMachineDataset/test_label/*.txt`
- Raw SWaT input layout:
  - `data/SWaT/SWaT_Dataset_Normal_v1.csv`
  - `data/SWaT/SWaT_Dataset_Attack_v0.csv`
- Generated windowed datasets:
  - `data/smd/machine-*/source.npz`
  - `data/smd/machine-*/target.npz`
  - `data/smd_experiments/**/train_normal.npz`
  - `data/smd_experiments/**/target_pool_unlabeled.npz`
  - `data/smd_experiments/**/val_mixed.npz`
  - `data/smd_experiments/**/test_mixed.npz`
  - `data/smd_experiments/**/split_metadata.json`
  - `data/smd_experiments/manifest.json`
- Runtime and benchmark outputs:
  - `outputs/results.json`
  - `outputs/baselines/*.json`
  - `outputs/baselines_summary.json`
  - `outputs/benchmarks/**/*.json`
  - `outputs/logs/*.txt`
  - `outputs/checkpoints/*.pt`
  - `outputs/figures/*.png`

**Caching:**
- None
- The nearest equivalent is reusable model state saved under `outputs/checkpoints/`

## Authentication & Identity

**Auth Provider:**
- None
  - Implementation: the repo has no user accounts, tokens, OAuth flows, API keys, or secret-backed identity layer in `src/` or `scripts/`

## Monitoring & Observability

**Error Tracking:**
- None

**Logs:**
- Plain stdout/stderr redirected into local files by `scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, and `scripts/run_tranad_smd.py`
- Structured experiment summaries are written as JSON by `src/pipeline.py`, `scripts/run_usad_upstream_swat.py`, `scripts/run_tranad_upstream_smd.py`, and `scripts/compare_mode_benchmarks.py`

## CI/CD & Deployment

**Hosting:**
- None

**CI Pipeline:**
- None detected

## Environment Configuration

**Required env vars:**
- None required by the main pipeline
- Optional `CUDA_VISIBLE_DEVICES` is set by `scripts/run_pipeline.sh`
- `PYTHONHASHSEED` is set programmatically by `src/pipeline.py`

**Secrets location:**
- No secret-management files or secret-backed integration points were detected in the scanned repo paths
- No `.env` files were detected at repo root during this scan

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

---

*Integration audit: 2026-05-10*
