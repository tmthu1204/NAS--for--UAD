# Codebase Structure

**Analysis Date:** 2026-04-27

## Directory Layout

```text
[project-root]/
├── `src/`                  # Main Python source, orchestrator, families, data adapters, TS-TCC
├── `scripts/`              # Dataset builders, benchmark runners, family wrappers
├── `data/`                 # Raw datasets and generated `.npz` experiment bundles
├── `outputs/`              # Generated logs, checkpoints, figures, benchmark JSON, summaries
├── `results/`              # TS-TCC trainer artifacts (`saved_models/`)
├── `external/`             # Vendored upstream repos and environment payloads
├── `docs/`                 # Research notes, papers, and writeups
├── `.planning/codebase/`   # Generated codebase maps
├── `venv/`                 # Local Python environment
├── `run.ps1`               # Windows launcher for common runs
├── `README.md`             # Usage, protocols, and project layout notes
└── `setup.py`              # Package metadata for the `src/` tree
```

Source-of-truth project code lives in `src/` and `scripts/`. Treat `data/`, `outputs/`, `results/`, `venv/`, and most of `external/` as inputs, generated artifacts, or vendored references rather than places to add new project logic.

## Directory Purposes

**`src/`:**
- Purpose: Holds the active project implementation.
- Contains: `src/pipeline.py`, plus subpackages `src/adaptnas/`, `src/data/`, `src/families/`, `src/models/`, `src/ts_tcc/`, and `src/utils/`.
- Key files: `src/pipeline.py`, `src/adaptnas/search_space.py`, `src/data/omni_smd.py`, `src/families/usad.py`.

**`src/adaptnas/`:**
- Purpose: Default-family architecture-search helpers.
- Contains: search-space sampling, bi-level trainer, optimizer, and an older standalone `AdaptNASModel`.
- Key files: `src/adaptnas/search_space.py`, `src/adaptnas/trainer.py`, `src/adaptnas/optimizer.py`, `src/adaptnas/model_adaptnas.py`.

**`src/data/`:**
- Purpose: Data ingestion and tensor-shaping boundary.
- Contains: raw SMD/SWaT readers, contiguous splits, sliding-window helpers, and generic `ArrayDataset`.
- Key files: `src/data/omni_smd.py`, `src/data/swat.py`, `src/data/tranad_smd.py`, `src/data/datasets.py`.

**`src/families/`:**
- Purpose: One-file-per-backbone family implementations.
- Contains: `omni_anomaly`, `usad`, `tranad`, plus Omni evaluation utilities.
- Key files: `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`, `src/families/omni_eval.py`.

**`src/models/`:**
- Purpose: Reusable blocks used by the default pipeline.
- Contains: CNN encoder stages, transformer encoder, classifier head, domain discriminator, and DeepSVDD network.
- Key files: `src/models/tscnn.py`, `src/models/transformer.py`, `src/models/classifier.py`, `src/models/discriminator.py`, `src/models/deepsvdd.py`.

**`src/ts_tcc/`:**
- Purpose: Embedded TS-TCC subsystem reused for self-supervised pretraining.
- Contains: upstream-style configs, dataloaders, trainer, preprocessing utilities, models, and a standalone legacy CLI.
- Key files: `src/ts_tcc/trainer/trainer.py`, `src/ts_tcc/dataloader/dataloader.py`, `src/ts_tcc/models/model.py`, `src/ts_tcc/main.py`.

**`scripts/`:**
- Purpose: Operational CLIs outside the main package.
- Contains: raw-data preprocessing, `.npz` experiment builders, family-specific wrappers, upstream comparison runners, and batch benchmark execution.
- Key files: `scripts/preprocess_smd.py`, `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`, `scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_smd.py`.

**`data/`:**
- Purpose: Store both raw datasets and generated protocol-ready bundles.
- Contains: raw `data/ServerMachineDataset/`, raw `data/SWaT/`, normalized `data/smd/`, and generated `data/smd_experiments/`.
- Key files: `data/smd/machine-1-1/source.npz`, `data/smd/machine-1-1/target.npz`, `data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-7/split_metadata.json`.

**`outputs/`:**
- Purpose: Central sink for run artifacts.
- Contains: `outputs/results.json`, baseline JSON, benchmark folders, figures, logs, checkpoints, and summaries.
- Key files: `outputs/results.json`, `outputs/baselines_summary.json`, `outputs/benchmarks/`, `outputs/checkpoints/`, `outputs/logs/`.

**`external/`:**
- Purpose: Keep vendored upstream repos and environment payloads out of `src/`.
- Contains: `external/usad_upstream/`, `external/tranad_upstream/`, `external/OmniAnomaly/`, `external/miniconda3/`, `external/conda-envs/`.
- Key files: `external/usad_upstream/`, `external/tranad_upstream/`, `external/OmniAnomaly/`.

## Key File Locations

**Entry Points:**
- `src/pipeline.py`: Main end-to-end entrypoint for all project-native experiment modes and families.
- `run.ps1`: Windows convenience launcher that resolves the Python interpreter and device before calling `src.pipeline`.
- `scripts/run_pipeline.sh`: Minimal shell wrapper for direct `src.pipeline` runs.
- `scripts/run_all_smd.py`: Batch runner that loops over experiment folders or raw SMD machines.
- `src/ts_tcc/main.py`: Standalone TS-TCC upstream-style CLI, separate from the active project orchestrator.

**Configuration:**
- `setup.py`: Package metadata for the `src/` package layout.
- `README.md`: High-level protocol documentation, directory overview, and usage examples.
- `src/ts_tcc/config_files/HAR_Configs.py`: The config object that `src/pipeline.py` currently reuses for TS-TCC pretraining.
- `run.ps1`: Operational defaults for common modes, families, datasets, and CUDA auto-detection.

**Core Logic:**
- `src/pipeline.py`: Active orchestration logic, default-family `CandidateModel`, family dispatch, search loops, and output serialization.
- `src/adaptnas/trainer.py`: Default-family bi-level optimization loop.
- `src/families/usad.py`, `src/families/tranad.py`, `src/families/omni_anomaly.py`: Family-native training, validation, and scoring.
- `src/adaptnas/model_adaptnas.py`: Older standalone search model definition that is present in the tree but not the active model used by `src/pipeline.py`.

**Testing:**
- Not detected in a dedicated `tests/` package.
- No `*.test.py` or `*.spec.py` modules were found under `src/` or `scripts/` during this scan.

## Naming Conventions

**Files:**
- Use lower snake case for Python modules and scripts, such as `src/pipeline.py`, `src/data/omni_smd.py`, and `scripts/build_domain_shift_smd.py`.
- Family files are named after the model or paper family they implement, such as `src/families/usad.py`, `src/families/tranad.py`, and `src/families/omni_anomaly.py`.
- Wrapper and operational scripts follow verb-led names like `scripts/run_all_smd.py`, `scripts/preprocess_smd.py`, and `scripts/make_uad_smd.py`.

**Directories:**
- Python packages are lower case or snake case, such as `src/adaptnas/`, `src/families/`, and `src/ts_tcc/`.
- Raw and normalized machine directories use `machine-<group>-<id>` naming, such as `data/smd/machine-1-1`.
- Cross-machine experiment folders use `<source>__to__<target>` naming, such as `data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-7`.

## Where to Add New Code

**New Feature:**
- Primary code: Put end-to-end protocol changes in `src/pipeline.py` only when they affect orchestration; put reusable logic in `src/data/`, `src/models/`, `src/adaptnas/`, or `src/families/` instead.
- Tests: Not detected; if tests are introduced later, create a dedicated `tests/` tree rather than adding ad hoc validation into `outputs/` or `data/`.

**New Component/Module:**
- Implementation: Add a new backbone family to `src/families/<family_name>.py`, a new raw-data adapter to `src/data/<dataset_name>.py`, and shared neural blocks to `src/models/<module_name>.py`.
- Implementation: If the new family needs a user-facing benchmark CLI, add a thin wrapper under `scripts/run_<family>_<dataset>.py` that either calls `python -m src.pipeline` or clearly labels itself as an upstream reproduction script.

**Utilities:**
- Shared helpers: `src/utils/` for generic metrics, plotting, and schedules.
- Shared helpers: `src/ts_tcc/` only for TS-TCC-specific augmentations, config objects, and trainer pieces.

## Special Directories

**`data/smd/`:**
- Purpose: Machine-by-machine normalized/windowed SMD bundles created from raw data.
- Generated: Yes
- Committed: Yes

**`data/smd_experiments/`:**
- Purpose: Search/evaluation protocol folders with `train_normal.npz`, `target_pool_unlabeled.npz`, `val_mixed.npz`, `test_mixed.npz`, and `split_metadata.json`.
- Generated: Yes
- Committed: Yes

**`outputs/`:**
- Purpose: Run-time artifacts including `outputs/results.json`, figures, logs, checkpoint files, and benchmark summaries.
- Generated: Yes
- Committed: Yes

**`external/`:**
- Purpose: Vendored third-party baselines and large environment payloads that should stay separate from project source code.
- Generated: No
- Committed: Yes

**`src/ts_tcc/`:**
- Purpose: Embedded upstream TS-TCC source tree reused as an internal subsystem.
- Generated: No
- Committed: Yes

---

*Structure analysis: 2026-04-27*
