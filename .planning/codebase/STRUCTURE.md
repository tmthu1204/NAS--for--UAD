# Codebase Structure

**Analysis Date:** 2026-04-27

## Directory Layout

```text
[project-root]/
|-- src/                  # Active Python implementation
|   |-- adaptnas/         # Default-family NAS and bilevel optimization
|   |-- data/             # Raw readers and dataset adapters
|   |-- families/         # Omni/USAD/TranAD implementations
|   |-- models/           # Default-family shared neural blocks
|   |-- ts_tcc/           # Embedded TS-TCC subsystem
|   `-- utils/            # Metrics, schedulers, plotting
|-- scripts/              # Data builders and benchmark wrappers
|-- data/                 # Raw datasets and generated `.npz` bundles
|-- outputs/              # Logs, figures, checkpoints, benchmark JSON
|-- results/              # TS-TCC checkpoint sink
|-- external/             # Vendored upstream baselines and env payloads
|-- docs/                 # Research notes and papers
|-- .planning/codebase/   # Generated codebase maps
|-- run.ps1               # Windows launcher
|-- README.md             # Usage and protocol documentation
|-- requirements.txt      # Python dependency list
`-- setup.py              # Package metadata
```

Treat `src/` and `scripts/` as the source-of-truth code. Treat `data/`, `outputs/`, `results/`, `external/`, and `venv/` as inputs, generated artifacts, or vendored dependencies unless a phase explicitly targets them.

## Directory Purposes

**`src/`:**
- Purpose: Hold the active implementation used by project-native runs.
- Contains: `src/pipeline.py` plus `src/adaptnas/`, `src/data/`, `src/families/`, `src/models/`, `src/ts_tcc/`, and `src/utils/`.
- Key files: `src/pipeline.py`, `src/adaptnas/search_space.py`, `src/families/usad.py`.

**`src/adaptnas/`:**
- Purpose: Own the default-family search space and bilevel optimization mechanics.
- Contains: `ArchConfig`, the active optimizer and trainer, and an older standalone model prototype.
- Key files: `src/adaptnas/search_space.py`, `src/adaptnas/trainer.py`, `src/adaptnas/optimizer.py`, `src/adaptnas/model_adaptnas.py`.

**`src/data/`:**
- Purpose: Form the data-ingestion boundary between raw files or `.npz` bundles and PyTorch tensors/windows.
- Contains: raw SMD/SWaT readers, split helpers, window builders, and `ArrayDataset`.
- Key files: `src/data/datasets.py`, `src/data/omni_smd.py`, `src/data/swat.py`, `src/data/tranad_smd.py`.

**`src/families/`:**
- Purpose: Keep family-native implementations separate from the default stack.
- Contains: one file per family plus Omni-specific evaluation utilities.
- Key files: `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`, `src/families/omni_eval.py`.

**`src/models/`:**
- Purpose: Provide reusable building blocks for `default_nasade`.
- Contains: CNN stages, transformer encoder, classifier head, domain discriminator, and DeepSVDD network.
- Key files: `src/models/tscnn.py`, `src/models/transformer.py`, `src/models/classifier.py`, `src/models/discriminator.py`, `src/models/deepsvdd.py`.

**`src/ts_tcc/`:**
- Purpose: Embed the TS-TCC subsystem reused for self-supervised pretraining.
- Contains: configs, augmentations, dataloaders, models, trainer code, and legacy upstream preprocessing helpers.
- Key files: `src/ts_tcc/dataloader/dataloader.py`, `src/ts_tcc/models/model.py`, `src/ts_tcc/models/TC.py`, `src/ts_tcc/trainer/trainer.py`, `src/ts_tcc/main.py`.

**`scripts/`:**
- Purpose: Hold operational CLIs outside the main package.
- Contains: raw-data preprocessors, experiment-split builders, batch wrappers, and upstream comparison runners.
- Key files: `scripts/preprocess_smd.py`, `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`, `scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_smd.py`.

**`data/`:**
- Purpose: Store both raw datasets and generated protocol-ready bundles.
- Contains: `data/ServerMachineDataset/`, `data/SWaT/`, `data/smd/`, `data/smd_experiments/`, and legacy TS-TCC data folders such as `data/UCI HAR Dataset/`.
- Key files: `data/smd/machine-1-1/source.npz`, `data/smd/machine-1-1/target.npz`, `data/smd_experiments/manifest.json`.

**`outputs/`:**
- Purpose: Act as the main sink for run artifacts.
- Contains: `outputs/results.json`, `outputs/baselines/`, `outputs/benchmarks/`, `outputs/checkpoints/`, `outputs/figures/`, and `outputs/logs/`.
- Key files: `outputs/results.json`, `outputs/baselines_summary.json`.

**`results/`:**
- Purpose: Preserve TS-TCC trainer checkpoints outside the main `outputs/` tree.
- Contains: `results/saved_models/`.
- Key files: `results/saved_models/ckp_last.pt`.

**`external/`:**
- Purpose: Keep vendored baselines and environment payloads out of the active source tree.
- Contains: `external/OmniAnomaly/`, `external/usad_upstream/`, `external/tranad_upstream/`, `external/miniconda3/`, `external/conda-envs/`.
- Key files: `external/usad_upstream/`, `external/tranad_upstream/`, `external/OmniAnomaly/`.

**`.planning/codebase/`:**
- Purpose: Store generated mapping documents consumed by later planning and execution phases.
- Contains: `ARCHITECTURE.md`, `STRUCTURE.md`, and the other codebase-map documents.
- Key files: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`.

## Key File Locations

**Entry Points:**
- `src/pipeline.py`: Main project-native CLI and orchestration hub.
- `run.ps1`: Windows launcher for common runs; currently exposes `default_nasade`, `omni_anomaly`, and `usad`.
- `scripts/run_pipeline.sh`: Minimal shell wrapper around `python -m src.pipeline`.
- `scripts/run_all_smd.py`: Batch runner for `default_nasade` and `omni_anomaly`.
- `scripts/run_usad_swat.py`: Project-native USAD wrapper on raw SWaT.
- `scripts/run_tranad_smd.py`: Project-native TranAD wrapper on raw SMD.
- `scripts/preprocess_smd.py`: Raw SMD to `source.npz`/`target.npz` converter.
- `scripts/make_uad_smd.py`: Search/eval experiment-split builder.

**Configuration:**
- `README.md`: Usage, family selection, data protocols, and repo layout.
- `requirements.txt`: Python dependency list for local environments.
- `setup.py`: Package metadata for the `src/` package tree.
- `src/ts_tcc/config_files/HAR_Configs.py`: TS-TCC config object reused by the default-family pretraining stage.
- `run.ps1`: Operational defaults for device selection, dataset path resolution, and wrapper-exposed family options.

**Core Logic:**
- `src/pipeline.py`: Default-family orchestration, family dispatch, and final metric serialization.
- `src/adaptnas/search_space.py`: Searchable config definition for `default_nasade`.
- `src/adaptnas/trainer.py`: Bilevel lower/upper training loop.
- `src/adaptnas/optimizer.py`: Loss computation and upper-objective optimization helpers.
- `src/data/omni_smd.py`, `src/data/swat.py`, `src/data/tranad_smd.py`: Raw dataset adapters.
- `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`: Family-native models and training loops.
- `src/ts_tcc/trainer/trainer.py`: TS-TCC self-supervised training implementation.

**Testing:**
- Not detected as a dedicated tree.
- No `tests/`, `*.test.py`, or `*.spec.py` modules were found in the active source tree during this scan.

## Naming Conventions

**Files:**
- Use lower snake case for Python modules and scripts, such as `src/data/omni_smd.py` and `scripts/build_domain_shift_smd.py`.
- Name family modules after the implemented backbone or paper family, such as `src/families/usad.py` and `src/families/tranad.py`.
- Use verb-led names for operational scripts, such as `scripts/preprocess_smd.py`, `scripts/run_all_smd.py`, and `scripts/export_figures.py`.
- Keep protocol bundle filenames stable: `source.npz`, `target.npz`, `train_normal.npz`, `target_pool_unlabeled.npz`, `val_mixed.npz`, `test_mixed.npz`, and `split_metadata.json`.

**Directories:**
- Use lower case package names under `src/`, such as `src/adaptnas/`, `src/families/`, and `src/utils/`.
- Use `machine-<group>-<id>` for per-machine dataset folders, such as `data/smd/machine-1-1`.
- Use `temporal_<shift>` and `cross_machine_<shift>` for experiment groups under `data/smd_experiments/`.
- Use `<source>__to__<target>` for cross-machine case folders, such as `data/smd_experiments/cross_machine_hard/machine-1-1__to__machine-1-7`.

## Where to Add New Code

**New Feature:**
- Primary code: Add new orchestration only to `src/pipeline.py` when the change is truly launch-flow logic; put reusable behavior in `src/data/`, `src/models/`, `src/adaptnas/`, or `src/families/`.
- Primary code: For a new model family, add `src/families/<family_name>.py`, then wire that family into `src/pipeline.py` and the relevant wrapper script in `scripts/`.
- Tests: No formal test tree exists; if automated tests are introduced, create a dedicated `tests/` package instead of writing assertions into `scripts/` or artifact folders.

**New Component/Module:**
- Implementation: Put default-family model blocks in `src/models/` and default-family search logic in `src/adaptnas/`.
- Implementation: Put raw-reader or window-protocol changes in `src/data/`.
- Implementation: Put TS-TCC-specific changes in `src/ts_tcc/`, not in generic helpers.
- Implementation: Avoid extending `src/adaptnas/model_adaptnas.py` unless a phase explicitly revives the older standalone mixed-op model; the active default-family model is `CandidateModel` inside `src/pipeline.py`.

**Utilities:**
- Shared helpers: `src/utils/` for generic metrics, plotting, and schedules.
- Operational wrappers: `scripts/` for one-off CLIs, sweep drivers, and artifact post-processing.
- Do not add new project logic to `data/`, `outputs/`, `results/`, `external/`, or `venv/`.

## Special Directories

**`data/smd/`:**
- Purpose: Store machine-by-machine normalized/windowed SMD bundles.
- Generated: Yes
- Committed: Yes

**`data/smd_experiments/`:**
- Purpose: Store protocol-ready experiment folders with `train_normal.npz`, `target_pool_unlabeled.npz`, `val_mixed.npz`, `test_mixed.npz`, and `split_metadata.json`.
- Generated: Yes
- Committed: Yes

**`outputs/`:**
- Purpose: Store runtime metrics, logs, figures, checkpoints, and benchmark summaries.
- Generated: Yes
- Committed: Yes

**`results/saved_models/`:**
- Purpose: Store TS-TCC checkpoint artifacts written by `src/ts_tcc/trainer/trainer.py`.
- Generated: Yes
- Committed: Yes

**`external/`:**
- Purpose: Store vendored upstream baselines and environment payloads that should remain separate from project-native source code.
- Generated: No
- Committed: Yes

---

*Structure analysis: 2026-04-27*
