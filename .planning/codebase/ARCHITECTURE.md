<!-- refreshed: 2026-05-10 -->
# Architecture

**Analysis Date:** 2026-05-10

## System Overview

```text
+-------------------------------------------------------------------+
| Launch and data-prep layer                                        |
+-------------------+---------------------+-------------------------+
| `run.ps1`         | `scripts/run_*.py`  | `scripts/preprocess_*` |
| wrapper CLI       | benchmark runners   | and split builders     |
+---------+---------+----------+----------+------------+------------+
          |                    |                       |
          v                    v                       v
+-------------------------------------------------------------------+
| Main orchestrator                                                 |
| `src/pipeline.py`                                                 |
| - parses `--mode` and `--family`                                  |
| - loads protocol `.npz` bundles or raw dataset paths              |
| - dispatches to default or family-native execution paths          |
+----------------------+-----------------------------+--------------+
                       |                             |
                       v                             v
+----------------------+-----------+   +----------------------------+
| Default NAS-ADE stack            |   | Raw family stacks          |
| `src/adaptnas/`                  |   | `src/families/*.py`       |
| `src/models/`                    |   | `src/data/*.py`           |
| `src/ts_tcc/`                    |   +----------------------------+
+----------------------+-----------+
                       |
                       v
+-------------------------------------------------------------------+
| Artifacts and runtime data                                        |
| `outputs/`, `results/saved_models/`, gitignored `data/`           |
+-------------------------------------------------------------------+
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Main orchestrator | Own CLI parsing, mode/family dispatch, default-family search loops, metric packaging, and final JSON writes. | `src/pipeline.py` |
| Default-family search layer | Define the searchable `ArchConfig` surface and run lower/upper bilevel optimization for `default_nasade`. | `src/adaptnas/search_space.py`, `src/adaptnas/trainer.py`, `src/adaptnas/optimizer.py` |
| Default-family model blocks | Provide the CNN encoder, sequence encoder, classifier, discriminator, and DeepSVDD blocks used by the active `CandidateModel`. | `src/models/tscnn.py`, `src/models/transformer.py`, `src/models/classifier.py`, `src/models/discriminator.py`, `src/models/deepsvdd.py` |
| Family-native implementations | Keep paper-specific configs, models, training loops, validation logic, and scoring rules outside the default stack. | `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`, `src/families/omni_eval.py`, `src/families/omni_spot.py` |
| Data adapters | Read raw SMD or SWaT data, normalize it, and build the tensor/window protocol each family expects. | `src/data/datasets.py`, `src/data/omni_smd.py`, `src/data/swat.py`, `src/data/tranad_smd.py` |
| TS-TCC subsystem | Provide self-supervised pretraining used only by the `default_nasade` branch before search and final scoring. | `src/ts_tcc/models/model.py`, `src/ts_tcc/models/TC.py`, `src/ts_tcc/dataloader/dataloader.py`, `src/ts_tcc/trainer/trainer.py` |
| Dataset-build scripts | Convert raw datasets into `source.npz`/`target.npz`, then slice them into protocol-ready experiment folders with metadata. | `scripts/preprocess_smd.py`, `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py` |
| Batch wrappers | Fan out repeated runs, capture logs, and copy `outputs/results.json` into case-scoped benchmark files. | `run.ps1`, `scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_smd.py` |
| Upstream reference payloads | Preserve external repos and local conda payloads used for comparison or manual reference, not for the active import graph. | `external/OmniAnomaly/`, `external/tranad_upstream/`, `external/usad_upstream/`, `external/miniconda3/`, `external/conda-envs/` |

## Pattern Overview

**Overall:** Monolithic experiment orchestrator with pluggable family modules and file-based artifacts

**Key Characteristics:**
- `src/pipeline.py` is the runtime hub for all project-native runs.
- The `default_nasade` path mixes orchestration and implementation details in the same module.
- `omni_anomaly`, `usad`, and `tranad` bypass TS-TCC and DeepSVDD and execute through dedicated family modules.
- Data preparation and benchmark sweeping live in `scripts/` and communicate with the runtime only through the filesystem.
- Persistent state is local to files under `outputs/`, `results/`, and the expected but untracked `data/` root.

## Layers

**Launch layer:**
- Purpose: Turn user input into repeatable `python -m src.pipeline` invocations.
- Location: `run.ps1`, `scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_smd.py`, `scripts/run_pipeline.sh`
- Contains: wrapper CLIs, device resolution, subprocess execution, and log redirection.
- Depends on: `src/pipeline.py`, the project `venv`, and repo-relative dataset paths.
- Used by: local runs, benchmark sweeps, and case-by-case family experiments.

**Orchestration layer:**
- Purpose: Choose the execution path for a run and glue together data loading, model construction, search, training, evaluation, and artifact writes.
- Location: `src/pipeline.py`
- Contains: `main()`, family dispatch, the active `CandidateModel`, TS-TCC hookup, default-family search loops, and final result serialization.
- Depends on: `src/data/`, `src/adaptnas/`, `src/models/`, `src/families/`, `src/ts_tcc/`, and `src/utils/metrics.py`.
- Used by: every project-native launcher and wrapper.

**Default NAS-ADE layer:**
- Purpose: Implement the `default_nasade` research path: TS-TCC initialization, architecture search, target-aware adaptation, and DeepSVDD scoring.
- Location: `src/pipeline.py`, `src/adaptnas/`, `src/models/`, `src/utils/metrics.py`
- Contains: `CandidateModel`, `ArchConfig`, bilevel lower/upper updates, discriminator-aware objectives, and final SVDD metrics.
- Depends on: PyTorch, numpy, `src/data/datasets.py`, and pretrained TS-TCC weights from `src/ts_tcc/`.
- Used by: the `family=default_nasade` branch in `src/pipeline.py`.

**Family-native execution layer:**
- Purpose: Preserve paper-specific training loops and anomaly-score definitions for raw-data families.
- Location: `src/families/`, supported by `src/data/omni_smd.py`, `src/data/swat.py`, and `src/data/tranad_smd.py`
- Contains: family dataclasses, model classes, `train_*`, `validate_*`, and `score_*` functions.
- Depends on: raw-data adapters in `src/data/` and shared metrics in `src/utils/metrics.py`.
- Used by: the `omni_anomaly`, `usad`, and `tranad` branches in `src/pipeline.py`.

**Dataset preparation layer:**
- Purpose: Build the protocol bundle layout required by `default_nasade` and generate temporal or cross-machine experiment suites.
- Location: `scripts/preprocess_smd.py`, `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`
- Contains: raw readers, split search, domain-shift ranking, metadata generation, and manifest generation.
- Depends on: the gitignored `data/` tree, numpy, pandas, and scikit-learn.
- Used by: `run.ps1`, `scripts/run_all_smd.py`, and direct `python -m src.pipeline` runs that consume `.npz` bundles.

**Artifact and reference layer:**
- Purpose: Persist checkpoints, metrics, logs, figures, benchmark summaries, and upstream comparison payloads.
- Location: `outputs/`, `results/saved_models/`, `external/`
- Contains: `outputs/results.json`, `outputs/baselines/`, `outputs/benchmarks/`, `outputs/logs/`, `outputs/checkpoints/`, and vendored upstream trees.
- Depends on: every launcher and training path.
- Used by: downstream comparison scripts, plotting scripts, and manual reference work.

## Data Flow

### Primary Request Path

1. A launcher resolves paths and shells into `python -m src.pipeline` (`run.ps1:135`, `run.ps1:272`, `scripts/run_all_smd.py:160`, `scripts/run_all_smd.py:297`).
2. `main()` parses `--mode` and `--family`, validates the input contract, dispatches raw-family branches, or loads `.npz` arrays for `default_nasade` and normalizes every window to length 128 (`src/pipeline.py:1879`, `src/pipeline.py:1997`, `src/pipeline.py:2052`, `src/pipeline.py:2061`, `src/pipeline.py:2110`).
3. For `default_nasade`, TS-TCC pretraining builds a self-supervised dataset and trains `base_Model` plus `TC` so default-family candidates can inherit convolutional weights (`src/pipeline.py:2132`, `src/pipeline.py:2198`, `src/pipeline.py:2204`, `src/ts_tcc/dataloader/dataloader.py:9`, `src/ts_tcc/trainer/trainer.py:169`).
4. The search stage samples `ArchConfig`, instantiates `CandidateModel`, and optimizes either source-only SVDD compactness or the combined unlabeled upper objective (`src/pipeline.py:2214`, `src/pipeline.py:2235`, `src/pipeline.py:2274`, `src/pipeline.py:2292`, `src/pipeline.py:2343`, `src/adaptnas/trainer.py:18`, `src/adaptnas/optimizer.py:285`).
5. The final stage fits DeepSVDD or replays final-only baselines, computes UAD metrics, and overwrites shared JSON artifacts under `outputs/` (`src/pipeline.py:2413`, `src/pipeline.py:2430`, `src/pipeline.py:2472`, `src/pipeline.py:2528`, `src/utils/metrics.py:4`).

### Raw Family Request Path

1. `src/pipeline.py` branches on `--family` and calls `run_omni_uad_source_family_raw`, `run_usad_uad_source_family_raw`, or `run_tranad_uad_source_family_raw` (`src/pipeline.py:1997`, `src/pipeline.py:2016`, `src/pipeline.py:2033`).
2. The branch-specific data adapter loads raw SMD or SWaT data, applies family-specific normalization or windowing, and creates inner-train/validation splits (`src/pipeline.py:1256`, `src/pipeline.py:1467`, `src/pipeline.py:1691`, `src/data/tranad_smd.py:31`, `src/data/swat.py:124`, `src/data/omni_smd.py:144`).
3. Each family module runs a fixed baseline plus partial NAS search inside its own train/validate/score loop and returns metrics to the orchestrator (`src/pipeline.py:1305`, `src/pipeline.py:1530`, `src/pipeline.py:1736`, `src/families/tranad.py:246`, `src/families/usad.py:207`, `src/families/omni_anomaly.py:327`).
4. The orchestrator serializes the selected family result into `outputs/results.json`, and wrapper scripts may copy that shared file into case-scoped benchmark JSONs (`src/pipeline.py:2010`, `src/pipeline.py:2027`, `src/pipeline.py:2046`, `scripts/run_all_smd.py:31`, `scripts/run_all_smd.py:309`).

**State Management:**
- Runtime state is mostly local numpy arrays, PyTorch tensors, and dataclass configs passed directly between functions inside one process.
- Persistent shared state is filesystem-based: `outputs/results.json`, `outputs/baselines/`, `outputs/checkpoints/`, `outputs/logs/`, and `results/saved_models/ckp_last.pt`.
- No database, queue, RPC service, or long-lived worker process is part of the active architecture.

## Key Abstractions

**Search config dataclasses:**
- Purpose: Represent the searchable architecture knobs for the default path and each paper family.
- Examples: `src/adaptnas/search_space.py`, `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`
- Pattern: Small dataclasses plus `get_fixed_*` and `sample_*` helper constructors.

**`CandidateModel`:**
- Purpose: Active `default_nasade` model that combines CNN stages, a sequence block, a classifier, a domain discriminator, and learnable depth weights.
- Examples: `src/pipeline.py`
- Pattern: Orchestrator-owned composite model assembled from `src/models/` primitives plus `arch_params`.

**Dataset wrappers and raw-dataset records:**
- Purpose: Normalize labeled, unlabeled, weighted, or sliding-window access for training loops.
- Examples: `src/data/datasets.py`, `src/data/omni_smd.py`, `src/data/swat.py`
- Pattern: Thin `torch.utils.data.Dataset` adapters and dataclasses over numpy arrays or raw time series.

**Family modules:**
- Purpose: Keep family-specific model definitions, training, validation, and scoring together.
- Examples: `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`
- Pattern: One file per family with a config dataclass, model class, `train_*`, `validate_*`, and `score_*` functions.

**TS-TCC pretraining stack:**
- Purpose: Provide self-supervised initialization for the default family before search and final scoring.
- Examples: `src/ts_tcc/models/model.py`, `src/ts_tcc/models/TC.py`, `src/ts_tcc/trainer/trainer.py`
- Pattern: Embedded upstream subsystem reused as an internal backbone rather than as the main experiment CLI.

## Entry Points

**Main project CLI:**
- Location: `src/pipeline.py`
- Triggers: `python -m src.pipeline ...`
- Responsibilities: All project-native training, search, family dispatch, evaluation, and final JSON serialization.

**Windows launcher:**
- Location: `run.ps1`
- Triggers: `powershell -File run.ps1 ...`
- Responsibilities: Validate wrapper-supported arguments, resolve the device, build dataset-path contracts, and call the project `venv`.

**Dataset build CLIs:**
- Location: `scripts/preprocess_smd.py`, `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`
- Triggers: Manual dataset preparation before `default_nasade` runs.
- Responsibilities: Create normalized machine bundles and protocol-ready experiment folders.

**Batch benchmark runners:**
- Location: `scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_smd.py`
- Triggers: Repeated machine-by-machine or case-by-case sweeps.
- Responsibilities: Spawn runs, store logs, and copy `outputs/results.json` into benchmark folders.

**Standalone comparison path:**
- Location: `src/ts_tcc/main.py`, `scripts/run_usad_upstream_swat.py`, `scripts/run_tranad_upstream_smd.py`
- Triggers: Legacy TS-TCC runs or upstream-style comparison experiments.
- Responsibilities: Execute embedded or comparison-specific flows outside the default project-native wrapper surface.

## Architectural Constraints

- **Threading:** Active training code is single-process Python with PyTorch execution on CPU or CUDA. Parallelism comes from separate CLI invocations, not internal worker orchestration.
- **Global state:** `outputs/results.json`, `outputs/baselines_summary.json`, `outputs/checkpoints/`, `outputs/logs/`, and `results/saved_models/ckp_last.pt` are shared write targets across runs.
- **Protocol contract:** `default_nasade` expects an exact file-order contract for `--dataset_or_paths`, and `adaptnas_combined` always requires a separate `target_pool_unlabeled.npz` input (`src/pipeline.py:2068`, `src/pipeline.py:2077`).
- **Runtime data root:** Top-level `data/` is required by defaults in `run.ps1`, `src/pipeline.py`, and `scripts/*.py`, but `data/` is gitignored in `.gitignore` and is not part of the tracked repo snapshot.
- **Wrapper parity:** `src/pipeline.py` accepts `default_nasade`, `omni_anomaly`, `usad`, and `tranad`, but `run.ps1` omits `tranad`, and `scripts/run_all_smd.py` omits `usad` and `tranad`.
- **TS-TCC config reuse:** The default-family path hardcodes `src.ts_tcc.config_files.HAR_Configs.Config` for pretraining setup, so TS-TCC hyperparameter changes route through that config object even for SMD-based runs (`src/pipeline.py:2135`).
- **Circular imports:** No active circular import chain is detected. `src/pipeline.py` imports leaf modules, but those modules do not import `src/pipeline.py` back.
- **Project-local skills:** No repo-local `.codex/skills/` or `.agents/skills/` directory is present. Current `.codex/` content is local tooling metadata, not runtime architecture.

## Anti-Patterns

### Orchestrator-Owned Default Stack

**What happens:** `src/pipeline.py` contains the CLI, family dispatch, the active `CandidateModel`, TS-TCC initialization, default-family search loops, and final result serialization.
**Why it's wrong:** Default-family runtime behavior is tightly coupled to launch orchestration, so changes to search, scoring, or model assembly all land in the same large module.
**Do this instead:** Keep `src/pipeline.py` focused on dispatch and move reusable default-family model/search code into `src/models/` or `src/adaptnas/`.

### Duplicate Default-Model Surfaces

**What happens:** The active runtime model is `CandidateModel` inside `src/pipeline.py`, while `src/adaptnas/model_adaptnas.py` defines a different `AdaptNASModel` that is not used by the project-native execution path.
**Why it's wrong:** The module boundary for the default family is unclear, so future edits can easily land in the inactive model instead of the runtime path.
**Do this instead:** Keep one authoritative default-family model implementation in `src/models/` or `src/adaptnas/`, and import it from `src/pipeline.py`.

### Shared Artifact Sink

**What happens:** Every successful run overwrites the same `outputs/results.json` and may also reuse the same checkpoint and log locations.
**Why it's wrong:** Repeated or parallel runs can clobber each other, and wrappers must copy artifacts immediately to avoid losing the previous result.
**Do this instead:** Write case-scoped artifacts directly under a per-run directory such as `outputs/benchmarks/<run>/` and keep `outputs/results.json` as an optional summary mirror only.

## Error Handling

**Strategy:** Fail fast on missing files, invalid shapes, unsupported family/mode combinations, or broken protocol inputs, and let wrappers surface failures through exit codes plus log files.

**Patterns:**
- Data-boundary helpers raise `FileNotFoundError` or `ValueError` close to the offending input (`src/data/omni_smd.py`, `src/data/swat.py`, `src/data/tranad_smd.py`).
- `src/pipeline.py` raises early on unsupported family/mode combinations or malformed dataset contracts before training starts (`src/pipeline.py:1998`, `src/pipeline.py:2017`, `src/pipeline.py:2034`, `src/pipeline.py:2069`).
- Batch wrappers redirect stdout and stderr to `outputs/logs/` and gate benchmark persistence on subprocess return codes (`scripts/run_all_smd.py:23`, `scripts/run_all_smd.py:307`).
- Dataset builders warn and continue when sweeping many cases, but they raise on an individual case when a required split cannot be produced (`scripts/make_uad_smd.py:426`, `scripts/build_domain_shift_smd.py:190`).

## Cross-Cutting Concerns

**Logging:** Runtime logging is `print()`-driven in `src/pipeline.py` and the family scripts, while sweep wrappers redirect full command output into `outputs/logs/`.
**Validation:** Input validation is split between `argparse` choices, explicit file checks, protocol-size checks, class-balance checks, and split-search constraints.
**Authentication:** Not applicable. The active architecture is offline, filesystem-driven research code.

---

*Architecture analysis: 2026-05-10*
