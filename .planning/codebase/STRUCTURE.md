# Codebase Structure

**Analysis Date:** 2026-05-10

## Directory Layout

```text
[project-root]/
|-- .codex/               # Local Codex/GSD metadata, not project source
|-- .planning/codebase/   # Generated codebase maps
|-- external/             # Untracked upstream repos and conda payloads
|-- outputs/              # Tracked run artifacts and benchmark JSON
|-- results/              # Tracked TS-TCC checkpoints
|-- scripts/              # Operational CLIs and post-processing helpers
|-- src/                  # Active Python package
|-- venv/                 # Local virtual environment
|-- README.md             # Usage and protocol documentation
|-- requirements.txt      # Python dependency list
|-- run.ps1               # Windows launcher
`-- setup.py              # Package metadata
```

Top-level `data/` is an expected runtime root referenced by `run.ps1`, `src/pipeline.py`, and `scripts/*.py`, but it is gitignored by `.gitignore` and is not present in the tracked repo snapshot.

## Directory Purposes

**`.planning/codebase/`:**
- Purpose: Store generated architecture, structure, stack, convention, testing, and concern maps for later planning work.
- Contains: `ARCHITECTURE.md`, `STRUCTURE.md`, `STACK.md`, `INTEGRATIONS.md`, `CONVENTIONS.md`, `TESTING.md`, `CONCERNS.md`.
- Key files: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`

**`.codex/`:**
- Purpose: Hold local Codex/GSD runtime metadata for this workspace.
- Contains: `.codex/agents/`, `.codex/get-shit-done/`, `.codex/hooks/`, and `.codex/skills.local-backup/`.
- Key files: `.codex/config.toml`, `.codex/gsd-file-manifest.json`

**`src/`:**
- Purpose: Hold the active implementation used by project-native runs.
- Contains: `src/pipeline.py` plus `src/adaptnas/`, `src/data/`, `src/families/`, `src/models/`, `src/ts_tcc/`, and `src/utils/`.
- Key files: `src/pipeline.py`, `src/__init__.py`

**`src/adaptnas/`:**
- Purpose: Own the `default_nasade` search space and bilevel optimization mechanics.
- Contains: `ArchConfig`, the active optimizer and trainer, and an older standalone model prototype.
- Key files: `src/adaptnas/search_space.py`, `src/adaptnas/trainer.py`, `src/adaptnas/optimizer.py`, `src/adaptnas/model_adaptnas.py`

**`src/data/`:**
- Purpose: Form the data-ingestion boundary between raw files or `.npz` bundles and PyTorch tensors or windows.
- Contains: raw SMD/SWaT readers, sliding-window helpers, normalization logic, and `ArrayDataset`.
- Key files: `src/data/datasets.py`, `src/data/omni_smd.py`, `src/data/swat.py`, `src/data/tranad_smd.py`

**`src/families/`:**
- Purpose: Keep family-native implementations separate from the default stack.
- Contains: one file per family plus Omni-specific evaluation helpers and package exports.
- Key files: `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`, `src/families/omni_eval.py`, `src/families/omni_spot.py`

**`src/models/`:**
- Purpose: Provide reusable building blocks for `default_nasade`.
- Contains: CNN stages, transformer encoder, classifier head, domain discriminator, and DeepSVDD network.
- Key files: `src/models/tscnn.py`, `src/models/transformer.py`, `src/models/classifier.py`, `src/models/discriminator.py`, `src/models/deepsvdd.py`

**`src/ts_tcc/`:**
- Purpose: Embed the TS-TCC subsystem reused for self-supervised pretraining.
- Contains: configs, augmentations, dataloaders, models, trainer code, and upstream preprocessing examples.
- Key files: `src/ts_tcc/dataloader/dataloader.py`, `src/ts_tcc/models/model.py`, `src/ts_tcc/models/TC.py`, `src/ts_tcc/trainer/trainer.py`, `src/ts_tcc/main.py`

**`src/utils/`:**
- Purpose: Hold small shared helpers used across training and evaluation.
- Contains: metric functions, GRL scheduler curves, and simple plotting.
- Key files: `src/utils/metrics.py`, `src/utils/schedulers.py`, `src/utils/visualization.py`

**`scripts/`:**
- Purpose: Hold operational CLIs outside the main package.
- Contains: raw-data preprocessors, experiment-split builders, batch runners, and artifact post-processing scripts.
- Key files: `scripts/preprocess_smd.py`, `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`, `scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_smd.py`

**`outputs/`:**
- Purpose: Act as the main sink for run artifacts.
- Contains: `outputs/results.json`, `outputs/baselines/`, `outputs/benchmarks/`, `outputs/checkpoints/`, `outputs/figures/`, and `outputs/logs/`.
- Key files: `outputs/results.json`, `outputs/baselines_summary.json`

**`results/`:**
- Purpose: Preserve TS-TCC checkpoints outside the main `outputs/` tree.
- Contains: `results/saved_models/`.
- Key files: `results/saved_models/ckp_last.pt`

**`external/`:**
- Purpose: Keep upstream repos and local conda payloads out of the active source tree.
- Contains: `external/OmniAnomaly/`, `external/tranad_upstream/`, `external/usad_upstream/`, `external/miniconda3/`, and `external/conda-envs/`.
- Key files: `external/OmniAnomaly/`, `external/tranad_upstream/`, `external/usad_upstream/`

## Key File Locations

**Entry Points:**
- `run.ps1`: Windows wrapper that resolves device selection, dataset-path contracts, and calls `python -m src.pipeline`.
- `src/pipeline.py`: Main project-native CLI and orchestration hub.
- `scripts/preprocess_smd.py`: Raw SMD to `source.npz` / `target.npz` converter.
- `scripts/make_uad_smd.py`: Protocol split builder for `train_normal.npz`, `target_pool_unlabeled.npz`, `val_mixed.npz`, and `test_mixed.npz`.
- `scripts/build_domain_shift_smd.py`: Temporal and cross-machine experiment-suite builder.
- `scripts/run_all_smd.py`: Batch runner for `default_nasade` and `omni_anomaly`.
- `scripts/run_usad_swat.py`: Project-native USAD wrapper on raw SWaT.
- `scripts/run_tranad_smd.py`: Project-native TranAD wrapper on raw SMD.
- `src/ts_tcc/main.py`: Legacy TS-TCC CLI retained inside the embedded subsystem.

**Configuration:**
- `README.md`: Usage, family selection, data protocols, outputs, and end-to-end flow.
- `requirements.txt`: Python dependency list for local environments.
- `setup.py`: Package metadata for the `src/` tree.
- `.gitignore`: Runtime directories excluded from the tracked tree, especially `data/`, `external/`, `.codex/`, and `venv/`.
- `src/ts_tcc/config_files/HAR_Configs.py`: TS-TCC config object reused by the default-family pretraining stage.

**Core Logic:**
- `src/pipeline.py`: Default-family orchestration, family dispatch, and final metric serialization.
- `src/adaptnas/search_space.py`: Searchable config definition for `default_nasade`.
- `src/adaptnas/trainer.py`: Bilevel lower/upper training loop.
- `src/adaptnas/optimizer.py`: Loss computation and unlabeled upper-objective helpers.
- `src/data/omni_smd.py`, `src/data/swat.py`, `src/data/tranad_smd.py`: Raw dataset adapters.
- `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`: Family-native models and training loops.
- `src/models/tscnn.py`, `src/models/transformer.py`, `src/models/classifier.py`, `src/models/discriminator.py`, `src/models/deepsvdd.py`: Default-family reusable blocks.
- `src/ts_tcc/trainer/trainer.py`: TS-TCC self-supervised training implementation.

**Testing:**
- Not detected as a dedicated tree.
- No `tests/`, `*.test.py`, or `*.spec.py` modules were found in the active source tree during this scan.

## Naming Conventions

**Files:**
- Use lower snake case for Python modules and scripts, such as `src/data/omni_smd.py` and `scripts/build_domain_shift_smd.py`.
- Name family modules after the implemented backbone or paper family, such as `src/families/usad.py`, `src/families/tranad.py`, and `src/families/omni_anomaly.py`.
- Use verb-led names for operational scripts, such as `scripts/preprocess_smd.py`, `scripts/run_all_smd.py`, `scripts/export_figures.py`, and `scripts/compare_mode_benchmarks.py`.
- Keep protocol bundle filenames stable: `source.npz`, `target.npz`, `train_normal.npz`, `target_pool_unlabeled.npz`, `val_mixed.npz`, `test_mixed.npz`, and `split_metadata.json`.
- Benchmark folders follow `<family>-<mode>[-tag]` under `outputs/benchmarks/`, while cross-machine case folders use `<source>__to__<target>`.

**Directories:**
- Use lower case package names under `src/`, such as `src/adaptnas/`, `src/families/`, and `src/utils/`.
- Use `machine-<group>-<id>` for per-machine dataset folders and case IDs referenced by scripts.
- Use `temporal_<shift>` and `cross_machine_<shift>` for generated experiment suites under the expected `data/smd_experiments/` root.

## Where to Add New Code

**New Feature:**
- Primary code: Put new default-family runtime wiring in `src/pipeline.py` only when the change is true dispatch or orchestration logic. Put reusable behavior in `src/adaptnas/`, `src/models/`, or `src/data/`.
- Primary code: For a new model family, add `src/families/<family_name>.py`, then wire that family into `src/pipeline.py`, `run.ps1`, and the relevant wrapper in `scripts/`.
- Primary code: For dataset-protocol changes, update `scripts/make_uad_smd.py` or `scripts/build_domain_shift_smd.py` and the matching runtime loader in `src/data/`.
- Tests: No formal test tree exists. If automated tests are introduced, create a dedicated root `tests/` package instead of writing assertions into `scripts/` or artifact folders.

**New Component/Module:**
- Implementation: Put default-family model blocks in `src/models/`.
- Implementation: Put default-family search or training logic in `src/adaptnas/`.
- Implementation: Put raw-reader, normalization, or windowing changes in `src/data/`.
- Implementation: Put TS-TCC-only changes in `src/ts_tcc/`, not in generic helpers.
- Implementation: Avoid adding new active default-family model logic to `src/adaptnas/model_adaptnas.py` unless a phase explicitly revives that standalone prototype. The active runtime model is `CandidateModel` inside `src/pipeline.py`.

**Utilities:**
- Shared helpers: `src/utils/` for metrics, plotting, and scheduler helpers.
- Operational wrappers and reports: `scripts/` for CLIs, sweep drivers, and artifact post-processing.
- Do not add source code to `outputs/`, `results/`, `external/`, `.codex/`, or `venv/`.

## Special Directories

**`.planning/codebase/`:**
- Purpose: Store generated codebase map documents consumed by future planning and execution phases.
- Generated: Yes
- Committed: Yes

**`.codex/`:**
- Purpose: Store local Codex/GSD runtime metadata for this workspace.
- Generated: Yes
- Committed: No

**`external/`:**
- Purpose: Store upstream repos and local conda payloads that sit outside the active source tree.
- Generated: No
- Committed: No

**`outputs/`:**
- Purpose: Store runtime metrics, logs, figures, checkpoints, and benchmark summaries.
- Generated: Yes
- Committed: Yes

**`results/saved_models/`:**
- Purpose: Store TS-TCC checkpoint artifacts written by `src/ts_tcc/trainer/trainer.py`.
- Generated: Yes
- Committed: Yes

**`data/`:**
- Purpose: Expected runtime root for raw datasets and generated protocol bundles such as `data/smd/` and `data/smd_experiments/`.
- Generated: No
- Committed: No

**`venv/`:**
- Purpose: Local Python environment used by `run.ps1`.
- Generated: Yes
- Committed: No

---

*Structure analysis: 2026-05-10*
