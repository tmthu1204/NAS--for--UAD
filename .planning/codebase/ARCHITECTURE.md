<!-- refreshed: 2026-04-27 -->
# Architecture

**Analysis Date:** 2026-04-27

## System Overview

```text
+-----------------------------+      +----------------------------------+
| Data preparation CLIs       |----->| Protocol-ready datasets          |
| `scripts/preprocess_smd.py` |      | `data/smd/`                      |
| `scripts/make_uad_smd.py`   |      | `data/smd_experiments/`          |
| `scripts/build_domain_...`  |      +----------------------------------+
+--------------+--------------+
               |
               v
+--------------+--------------+
| Launchers and wrappers       |
| `run.ps1`                    |
| `scripts/run_all_smd.py`     |
| `scripts/run_usad_swat.py`   |
| `scripts/run_tranad_smd.py`  |
+--------------+--------------+
               |
               v
+--------------+-----------------------------------------------+
| Main orchestrator                                              |
| `src/pipeline.py`                                              |
| - parses `--mode` and `--family`                               |
| - loads either `.npz` protocol bundles or raw dataset paths    |
| - dispatches into default or family-native training paths      |
+--------------+-----------------------------+------------------+
               |                             |
               v                             v
+--------------+--------------+   +----------+-------------------+
| `default_nasade` stack      |   | Raw family stacks            |
| `src/ts_tcc/`               |   | `src/families/omni_...py`   |
| `src/models/`               |   | `src/families/usad.py`      |
| `src/adaptnas/`             |   | `src/families/tranad.py`    |
| `src/utils/metrics.py`      |   | `src/data/*.py`             |
+--------------+--------------+   +----------+-------------------+
               \                             /
                \                           /
                 v                         v
          +----------------------------------------+
          | Shared artifact sinks                  |
          | `outputs/` and `results/saved_models/` |
          +----------------------------------------+
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Main orchestrator | Own CLI parsing, mode/family dispatch, default-family search loops, metric packaging, and final JSON writes. | `src/pipeline.py` |
| Default-family NAS layer | Sample `ArchConfig` values and optimize lower/upper bilevel objectives for `default_nasade`. | `src/adaptnas/search_space.py`, `src/adaptnas/trainer.py`, `src/adaptnas/optimizer.py` |
| Default-family model primitives | Provide the CNN encoder, sequence encoder, classifier, discriminator, and DeepSVDD blocks used by `CandidateModel`. | `src/models/tscnn.py`, `src/models/transformer.py`, `src/models/classifier.py`, `src/models/discriminator.py`, `src/models/deepsvdd.py` |
| Family-native implementations | Keep paper-specific forward passes, training loops, validation objectives, and scoring rules separate from the default family. | `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`, `src/families/omni_eval.py` |
| Data adapters | Read raw SMD or SWaT files, normalize them, and build the tensor/window protocol each family expects. | `src/data/datasets.py`, `src/data/omni_smd.py`, `src/data/swat.py`, `src/data/tranad_smd.py` |
| TS-TCC subsystem | Provide self-supervised pretraining used only by the default family before search/final training. | `src/ts_tcc/models/model.py`, `src/ts_tcc/models/TC.py`, `src/ts_tcc/dataloader/dataloader.py`, `src/ts_tcc/trainer/trainer.py` |
| Dataset-build scripts | Convert raw datasets into `source.npz`/`target.npz`, then into search/eval experiment bundles with metadata. | `scripts/preprocess_smd.py`, `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py` |
| Batch wrappers | Fan out repeated runs, capture logs, and copy `outputs/results.json` into case-scoped benchmark files. | `run.ps1`, `scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_smd.py` |

## Pattern Overview

**Overall:** Single-orchestrator research pipeline with family-specific execution modules and file-based artifacts

**Key Characteristics:**
- `src/pipeline.py` is the only active orchestrator for project-native runs; almost every runtime branch starts there.
- `default_nasade` composes TS-TCC pretraining, inline `CandidateModel` construction, DeepSVDD scoring, and AdaptNAS bilevel search in one flow.
- `omni_anomaly`, `usad`, and `tranad` bypass the default stack and keep their own model/training/scoring rules in `src/families/`.
- Data preparation is a separate pre-run layer in `scripts/`, not part of `src/pipeline.py`.
- Persistent state is filesystem-based rather than service-based: datasets, logs, checkpoints, and metrics are all written to repo directories.

## Layers

**Data preparation layer:**
- Purpose: Convert raw datasets into the repo's `.npz` protocol and search/eval experiment folders.
- Location: `scripts/preprocess_smd.py`, `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`
- Contains: raw readers, split search, domain-shift scoring, manifest generation.
- Depends on: `data/ServerMachineDataset/`, `data/SWaT/`, `data/smd/`, numpy, scikit-learn.
- Used by: `run.ps1`, `scripts/run_all_smd.py`, and direct `python -m src.pipeline` runs that consume `.npz` bundles.

**Launch layer:**
- Purpose: Turn user arguments into repeatable `python -m src.pipeline` invocations.
- Location: `run.ps1`, `scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_smd.py`, `scripts/run_pipeline.sh`
- Contains: wrapper CLIs, device detection, subprocess execution, log redirection.
- Depends on: `src/pipeline.py`, `venv/Scripts/python.exe`, and repo-relative dataset paths.
- Used by: local interactive runs and benchmark sweeps.

**Orchestration layer:**
- Purpose: Choose the execution path for a run and glue together data loading, model construction, search, training, evaluation, and artifact writes.
- Location: `src/pipeline.py`
- Contains: `main()`, family dispatch, the inline `CandidateModel`, TS-TCC hookup, default-family search loops, and final result serialization.
- Depends on: `src/data/`, `src/adaptnas/`, `src/models/`, `src/families/`, `src/ts_tcc/`, `src/utils/metrics.py`.
- Used by: every project-native launcher and wrapper.

**Default-family search layer:**
- Purpose: Implement `default_nasade` architecture search and adaptation behavior.
- Location: `src/adaptnas/`, `src/models/`, `src/utils/metrics.py`
- Contains: `ArchConfig`, bilevel lower/upper updates, discriminator-aware objectives, DeepSVDD fitting/scoring, and reusable encoder/sequence blocks.
- Depends on: PyTorch, numpy, `src/data/datasets.py`, and TS-TCC-pretrained weights from `src/pipeline.py`.
- Used by: the `family=default_nasade` branch in `src/pipeline.py`.

**Family-native execution layer:**
- Purpose: Preserve paper-specific training loops and anomaly-score definitions for non-default families.
- Location: `src/families/`, supported by `src/data/omni_smd.py`, `src/data/swat.py`, and `src/data/tranad_smd.py`
- Contains: family dataclasses, model classes, `train_*`, `validate_*`, and `score_*` functions.
- Depends on: raw-data adapters in `src/data/` and shared metrics in `src/utils/metrics.py`.
- Used by: the `omni_anomaly`, `usad`, and `tranad` branches in `src/pipeline.py`.

**Artifact layer:**
- Purpose: Persist datasets, manifests, checkpoints, figures, logs, and benchmark summaries.
- Location: `data/smd/`, `data/smd_experiments/`, `outputs/`, `results/saved_models/`
- Contains: `.npz` bundles, `split_metadata.json`, `outputs/results.json`, `outputs/benchmarks/*.json`, plots, and checkpoints.
- Depends on: every launcher and training path.
- Used by: downstream comparison scripts, plotting scripts, and future benchmark analysis.

## Data Flow

### Primary Request Path

1. A launcher resolves paths and shells into `python -m src.pipeline` (`run.ps1:131`, `scripts/run_all_smd.py:297`).
2. `main()` parses `--mode` and `--family`, validates the input contract, loads `.npz` arrays for `default_nasade`, and normalizes every window to length 128 (`src/pipeline.py:1879`, `src/pipeline.py:2061`, `src/pipeline.py:2110`).
3. TS-TCC pretraining builds a self-supervised dataset and trains `base_Model` plus `TC` so the default-family candidates can inherit convolutional weights (`src/pipeline.py:2132`, `src/ts_tcc/dataloader/dataloader.py:8`, `src/ts_tcc/trainer/trainer.py:144`).
4. The search stage samples `ArchConfig`, instantiates `CandidateModel`, and optimizes either source-only SVDD compactness or the combined unlabeled upper objective (`src/pipeline.py:2214`, `src/pipeline.py:2292`, `src/adaptnas/trainer.py:17`).
5. The final stage fits DeepSVDD or replays final-only baselines, computes UAD metrics, and overwrites shared JSON artifacts under `outputs/` (`src/pipeline.py:2413`, `src/pipeline.py:2491`, `src/utils/metrics.py:4`).

### Dataset Preparation Flow

1. `scripts/preprocess_smd.py` converts raw machine files into normalized per-machine `source.npz` and `target.npz` bundles (`scripts/preprocess_smd.py:55`, `scripts/preprocess_smd.py:118`).
2. `scripts/make_uad_smd.py` slices those bundles into `train_normal.npz`, `target_pool_unlabeled.npz`, `val_mixed.npz`, `test_mixed.npz`, and `split_metadata.json` (`scripts/make_uad_smd.py:426`).
3. `scripts/build_domain_shift_smd.py` fans that split logic out across temporal and cross-machine cases and writes `data/smd_experiments/manifest.json` (`scripts/build_domain_shift_smd.py:123`, `scripts/build_domain_shift_smd.py:168`).

**State Management:**
- Runtime state is mostly local numpy arrays, PyTorch tensors, and dataclass configs passed directly between functions.
- Mutable global state is filesystem-based: `outputs/results.json`, `outputs/baselines/`, `outputs/checkpoints/`, `outputs/logs/`, and `results/saved_models/ckp_last.pt` are shared across runs.
- No database, queue, RPC service, or long-lived process state is present in the active architecture.

## Key Abstractions

**Search config dataclasses:**
- Purpose: Represent the searchable architecture knobs for the default path and each paper family.
- Examples: `src/adaptnas/search_space.py`, `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`
- Pattern: Small dataclasses with `get_fixed_*` and `sample_*` helper constructors.

**`CandidateModel`:**
- Purpose: Active `default_nasade` model that combines CNN stages, a sequence block, a classifier, a domain discriminator, and learnable depth weights.
- Examples: `src/pipeline.py`
- Pattern: Inline orchestrator-owned model assembled from `src/models/` primitives plus `arch_params`.

**Dataset wrappers:**
- Purpose: Normalize labeled, unlabeled, and weighted data access for training loops.
- Examples: `src/data/datasets.py`, `src/data/omni_smd.py`, `src/data/swat.py`
- Pattern: Thin `torch.utils.data.Dataset` adapters around numpy arrays or raw time series.

**Family modules:**
- Purpose: Keep family-specific model definitions, training, validation, and scoring together.
- Examples: `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`
- Pattern: One file per family with a dataclass, model class, `train_*`, `validate_*`, and `score_*` functions.

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
- Responsibilities: Validate common arguments, resolve the device, build the command line, and call the project `venv`.

**Dataset build CLIs:**
- Location: `scripts/preprocess_smd.py`, `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`
- Triggers: Manual dataset preparation before `default_nasade` runs.
- Responsibilities: Create normalized machine bundles and protocol-ready search/eval experiment folders.

**Batch benchmark runners:**
- Location: `scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_smd.py`
- Triggers: Repeated machine-by-machine or case-by-case sweeps.
- Responsibilities: Spawn runs, store logs, and copy `outputs/results.json` into benchmark folders.

**Standalone comparison paths:**
- Location: `scripts/run_usad_upstream_swat.py`, `scripts/run_tranad_upstream_smd.py`, `src/ts_tcc/main.py`
- Triggers: Upstream-faithful comparison runs outside the project-native orchestration path.
- Responsibilities: Reproduce family-specific baselines or legacy TS-TCC flows without going through the default launcher surface.

## Architectural Constraints

- **Threading:** Active training code is single-process Python with PyTorch execution on CPU or CUDA; parallelism comes from separate CLI invocations, not internal worker orchestration.
- **Global state:** `outputs/results.json` is a shared sink overwritten by every successful `src/pipeline.py` run, and `outputs/checkpoints/` plus `results/saved_models/` are also shared write targets.
- **Family dispatch:** `default_nasade` is the only branch that uses TS-TCC pretraining, `CandidateModel`, and DeepSVDD; `omni_anomaly`, `usad`, and `tranad` bypass that stack entirely.
- **Wrapper surface area:** `src/pipeline.py` accepts `default_nasade`, `omni_anomaly`, `usad`, and `tranad`, but `run.ps1` validates only `default_nasade`, `omni_anomaly`, and `usad`, and `scripts/run_all_smd.py` exposes only `default_nasade` and `omni_anomaly`.
- **TS-TCC config reuse:** The default-family path hardcodes `src.ts_tcc.config_files.HAR_Configs.Config` for pretraining setup, so TS-TCC hyperparameter changes currently route through that config object even for SMD-based runs.
- **Inactive alternates:** `src/adaptnas/model_adaptnas.py` and `src/ts_tcc/main.py` are present in the repo but are not part of the active project-native execution path driven by `src/pipeline.py`.

## Anti-Patterns

### Orchestrator-Owned Model Logic

**What happens:** `src/pipeline.py` contains the CLI, family dispatch, the active `CandidateModel`, TS-TCC initialization, search loops, and final serialization.
**Why it's wrong:** Any change to the default-family model, search behavior, or result packaging forces edits in the same large module, which couples research logic to launch orchestration.
**Do this instead:** Keep `src/pipeline.py` focused on dispatch and move reusable model/search pieces into `src/models/`, `src/adaptnas/`, or `src/families/`.

### Shared Artifact Sink

**What happens:** Every project-native run writes to the same `outputs/results.json`, `outputs/checkpoints/`, and `results/saved_models/` locations.
**Why it's wrong:** Repeated or parallel runs can clobber each other, and wrappers must copy artifacts immediately to avoid losing the previous result.
**Do this instead:** Write case-scoped outputs under `outputs/benchmarks/<run>/` or another per-run directory and only mirror a summary file when needed.

### Wrapper/Core Drift

**What happens:** Wrapper CLIs do not expose the same family surface as `src/pipeline.py`; for example, `run.ps1` omits `tranad`, and `scripts/run_all_smd.py` omits both `usad` and `tranad`.
**Why it's wrong:** The public entry points lag the actual architecture, so users can add a core capability without making it reachable through the expected operational scripts.
**Do this instead:** When adding or changing a family or mode, update `src/pipeline.py`, `run.ps1`, and the relevant `scripts/run_*.py` wrappers in the same phase.

## Error Handling

**Strategy:** Fail fast on missing files, invalid shapes, unsupported family/mode combinations, or broken protocol inputs, and let wrappers surface failures through exit codes plus log files.

**Patterns:**
- Data-boundary helpers raise `FileNotFoundError` or `ValueError` close to the offending input (`src/data/omni_smd.py`, `src/data/swat.py`, `src/data/tranad_smd.py`, `src/pipeline.py`).
- Batch wrappers redirect stdout and stderr to `outputs/logs/` and gate benchmark persistence on subprocess return codes (`scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_smd.py`).
- Dataset builders warn and continue when sweeping many cases, but they raise on an individual case when a required split cannot be produced (`scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`).

## Cross-Cutting Concerns

**Logging:** Runtime logging is `print()`-driven in `src/pipeline.py` and family scripts, while sweep wrappers redirect full command output into `outputs/logs/`.
**Validation:** Input validation is split between `argparse` choices, explicit file checks, protocol-size checks, class-balance checks, and split-search constraints.
**Authentication:** Not applicable; the active architecture is offline, filesystem-driven research code.

---

*Architecture analysis: 2026-04-27*
