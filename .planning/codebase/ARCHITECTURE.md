<!-- refreshed: 2026-04-27 -->
# Architecture

**Analysis Date:** 2026-04-27

## System Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Entry Points / Experiment Wrappers                      │
├──────────────────┬──────────────────┬──────────────────────────────────────┤
│ `run.ps1`        │ `scripts/*.py`   │ `src/pipeline.py`                    │
│ Windows launcher │ Batch/wrapper CLIs│ Main project orchestrator CLI       │
└────────┬─────────┴────────┬─────────┴──────────────┬───────────────────────┘
         │                  │                        │
         ▼                  ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              Mode / Family Orchestration Layer                              │
│ `src/pipeline.py` branches by `--mode` and `--family`                      │
│ - `default_nasade` -> TS-TCC + CandidateModel + DeepSVDD + AdaptNAS        │
│ - `omni_anomaly` -> raw SMD + Omni family                                  │
│ - `usad` -> raw SWaT + USAD family                                         │
│ - `tranad` -> raw SMD + TranAD family                                      │
└────────┬──────────────────────────┬──────────────────────────┬──────────────┘
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                Data / Training / Backbone Subsystems                        │
│ `src/data/` `src/families/` `src/adaptnas/` `src/models/` `src/ts_tcc/`   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Raw Data, Experiment Bundles, and Outputs                   │
│ `data/ServerMachineDataset/` `data/SWaT/` `data/smd*/` `outputs/`          │
│ `results/`                                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Pipeline orchestrator | Parses CLI flags, chooses protocol/family, runs search/training/evaluation, writes `outputs/results.json` | `src/pipeline.py` |
| AdaptNAS search layer | Samples searchable default architectures and applies lower/upper bi-level updates | `src/adaptnas/search_space.py`, `src/adaptnas/trainer.py`, `src/adaptnas/optimizer.py` |
| Family modules | Encapsulate paper-faithful or partial-NAS model families for `omni_anomaly`, `usad`, and `tranad` | `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py` |
| Data adapters | Load raw SMD/SWaT inputs, normalize them, and build family-specific windows | `src/data/omni_smd.py`, `src/data/swat.py`, `src/data/tranad_smd.py`, `src/data/datasets.py` |
| Model primitives | Provide reusable encoder, sequence, classifier, discriminator, and DeepSVDD blocks for `default_nasade` | `src/models/tscnn.py`, `src/models/transformer.py`, `src/models/classifier.py`, `src/models/discriminator.py`, `src/models/deepsvdd.py` |
| TS-TCC subsystem | Supplies self-supervised pretraining backbone, augmentations, configs, and trainer reused by the default family | `src/ts_tcc/models/`, `src/ts_tcc/dataloader/`, `src/ts_tcc/trainer/trainer.py` |
| Experiment runners | Convert datasets, build benchmark splits, and run many cases while capturing logs | `scripts/preprocess_smd.py`, `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`, `scripts/run_all_smd.py` |

## Pattern Overview

**Overall:** Monolithic research orchestrator with pluggable family modules

**Key Characteristics:**
- `src/pipeline.py` is the single active orchestration hub for project-native runs and contains both CLI parsing and default-family model wiring.
- Family branching is explicit: `default_nasade` uses `src/adaptnas/`, `src/models/`, and `src/ts_tcc/`, while `omni_anomaly`, `usad`, and `tranad` each route into dedicated `src/families/*.py` implementations.
- `scripts/` is split between thin wrappers that shell into `python -m src.pipeline` and standalone reproduction scripts that call family modules directly.

## Layers

**Entry-point layer:**
- Purpose: Start runs from Windows, shell, or batch automation.
- Location: `run.ps1`, `scripts/run_pipeline.sh`, `scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_smd.py`
- Contains: CLI argument parsing, device resolution, subprocess spawning, log-file routing.
- Depends on: `src.pipeline`, filesystem layout under `data/` and `outputs/`.
- Used by: Interactive local runs and benchmark sweeps.

**Orchestration layer:**
- Purpose: Coordinate end-to-end experiment flow for all supported modes and families.
- Location: `src/pipeline.py`
- Contains: `main()`, family-specific runner functions, default-family `CandidateModel`, TS-TCC pretraining hookup, search loops, final metric serialization.
- Depends on: `src/data/`, `src/families/`, `src/adaptnas/`, `src/models/`, `src/ts_tcc/`, `src/utils/metrics.py`.
- Used by: `run.ps1`, `scripts/run_pipeline.sh`, `scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_smd.py`.

**Family and training layer:**
- Purpose: Implement paper-specific training loops and scoring rules.
- Location: `src/families/`, `src/adaptnas/`, `src/ts_tcc/trainer/trainer.py`
- Contains: Omni ELBO training, USAD dual-loss training, TranAD two-phase reconstruction training, AdaptNAS lower/upper updates, TS-TCC contrastive pretraining.
- Depends on: `src/data/`, `src/models/`, PyTorch, numpy.
- Used by: `src/pipeline.py` and the upstream reproduction scripts in `scripts/`.

**Data adapter layer:**
- Purpose: Turn raw files or `.npz` bundles into the tensor shapes each family expects.
- Location: `src/data/`
- Contains: raw SMD/SWaT readers, normalization helpers, contiguous train/validation splitting, sliding-window datasets, weighted/unlabeled array datasets.
- Depends on: numpy, pandas, scikit-learn, torch.
- Used by: `src/pipeline.py`, `src/families/*.py`, `scripts/preprocess_smd.py`, `scripts/run_*_upstream_*.py`.

**Artifact layer:**
- Purpose: Persist generated datasets, checkpoints, plots, benchmark JSON, and logs.
- Location: `data/smd/`, `data/smd_experiments/`, `outputs/`, `results/`
- Contains: protocol-ready `.npz` bundles, `split_metadata.json`, `outputs/results.json`, baseline summaries, training curves, saved models.
- Depends on: every runner that writes files.
- Used by: `src/pipeline.py`, `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`, `scripts/run_all_smd.py`.

## Data Flow

### Primary Request Path

1. A launcher builds a `python -m src.pipeline` command and resolves paths/device selection (`run.ps1:131-272`, `scripts/run_pipeline.sh:19-24`).
2. `main()` parses `--mode` and `--family`, validates inputs, and branches into either raw-family or default-family execution (`src/pipeline.py:1879-2053`).
3. For `family=default_nasade`, `.npz` inputs are loaded, binarized, normalized to fixed length 128, and mapped into source/target/eval arrays (`src/pipeline.py:2055-2129`).
4. TS-TCC pretraining builds a self-supervised dataset with augmentations, trains `base_Model` plus `TC`, and keeps the pretrained backbone for later initialization (`src/pipeline.py:2132-2208`, `src/ts_tcc/dataloader/dataloader.py:9-42`, `src/ts_tcc/trainer/trainer.py:144-254`).
5. Candidate architectures are sampled from `ArchConfig`, instantiated as `CandidateModel`, and searched either by source-only SVDD compactness or by unlabeled AdaptNAS upper objective (`src/adaptnas/search_space.py:6-80`, `src/pipeline.py:246-352`, `src/pipeline.py:2214-2411`, `src/adaptnas/trainer.py:18-249`, `src/adaptnas/optimizer.py:285-365`).
6. Final scoring fits or reuses DeepSVDD, computes project metrics, writes per-baseline JSON when needed, and always overwrites `outputs/results.json` (`src/pipeline.py:2413-2530`, `src/models/deepsvdd.py:5-41`, `src/utils/metrics.py:4-108`).

### Raw Family Path

1. `main()` short-circuits early when `--family` is `omni_anomaly`, `usad`, or `tranad` and dispatches to a dedicated runner (`src/pipeline.py:1997-2050`).
2. Each runner loads raw dataset files through its adapter: `RawSMDMachine.from_root()` for Omni, `RawSWaTDataset.from_csvs()` for USAD, and `load_raw_tranad_smd_machine()` for TranAD (`src/pipeline.py:1256-1875`, `src/data/omni_smd.py:137-157`, `src/data/swat.py:247-276`, `src/data/tranad_smd.py:31-56`).
3. The runner builds the correct window protocol and arch dataclass, evaluates a fixed baseline, then optionally runs a small partial-NAS search inside the family module (`src/pipeline.py:1272-1464`, `src/pipeline.py:1480-1688`, `src/pipeline.py:1704-1875`, `src/families/omni_anomaly.py:17-405`, `src/families/usad.py:13-306`, `src/families/tranad.py:14-350`).

**State Management:**
- Runtime state is mostly local numpy arrays, PyTorch modules, and dataclass configs passed down the call stack rather than long-lived services.
- Mutable global outputs live on disk: `outputs/results.json`, `outputs/baselines/`, `outputs/checkpoints/`, `outputs/figures/`, and `results/saved_models/`.
- Search history and metrics are accumulated in Python dict/list structures and serialized to JSON at the end of each run.

## Key Abstractions

**Search arch dataclasses:**
- Purpose: Describe the searchable architecture knobs for both the default pipeline and family-specific branches.
- Examples: `src/adaptnas/search_space.py`, `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`
- Pattern: Small `@dataclass` config objects with `get_fixed_*` and `sample_*` constructors.

**`CandidateModel`:**
- Purpose: Active default-family model that combines CNN depth search, a sequence block, a classifier, and a domain discriminator.
- Examples: `src/pipeline.py:246-352`
- Pattern: Inline research model defined inside the orchestrator and parameterized by `ArchConfig`.

**Dataset wrappers:**
- Purpose: Normalize how labeled, unlabeled, and weighted batches are presented to training code.
- Examples: `src/data/datasets.py:7-24`, `src/data/omni_smd.py:105-157`, `src/data/swat.py:215-276`
- Pattern: Thin `torch.utils.data.Dataset` adapters around numpy arrays or raw series.

**Family modules:**
- Purpose: Package paper-faithful forward pass, training loop, validation objective, and anomaly score for a specific backbone family.
- Examples: `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`
- Pattern: One file per family with config dataclass, model class, `train_*`, `validate_*`, and `score_*` functions.

**TS-TCC pretraining stack:**
- Purpose: Provide reusable self-supervised initialization for the default pipeline.
- Examples: `src/ts_tcc/models/model.py`, `src/ts_tcc/models/TC.py`, `src/ts_tcc/trainer/trainer.py`
- Pattern: Embedded upstream subsystem reused as a backbone rather than as the primary experiment entrypoint.

## Entry Points

**Main project CLI:**
- Location: `src/pipeline.py`
- Triggers: `python -m src.pipeline ...`
- Responsibilities: End-to-end run orchestration across all modes and families.

**Windows launcher:**
- Location: `run.ps1`
- Triggers: `powershell -File run.ps1 ...`
- Responsibilities: Validate common arguments, detect CUDA, construct the `src.pipeline` command, and launch it from the project `venv`.

**Benchmark sweep runner:**
- Location: `scripts/run_all_smd.py`
- Triggers: Manual batch benchmark execution.
- Responsibilities: Iterate machine folders or raw machines, spawn repeated pipeline runs, and save benchmark JSON into `outputs/benchmarks/`.

**Dataset build CLIs:**
- Location: `scripts/preprocess_smd.py`, `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`
- Triggers: Manual data preparation before experiments.
- Responsibilities: Convert raw SMD into `source.npz`/`target.npz`, then build protocol-specific experiment folders under `data/smd_experiments/`.

**Standalone upstream-style CLIs:**
- Location: `scripts/run_tranad_upstream_smd.py`, `scripts/run_usad_upstream_swat.py`, `src/ts_tcc/main.py`
- Triggers: Family reproduction runs that do not use the full project orchestrator.
- Responsibilities: Reproduce or compare against upstream family behavior with their own training/evaluation loops.

## Architectural Constraints

- **Threading:** Active code is single-process Python with PyTorch tensor execution on CPU or CUDA; parallel sweeps are done by sequential subprocess spawning in `scripts/run_all_smd.py`.
- **Global state:** `outputs/results.json` is a shared single-run sink overwritten by every successful invocation of `src/pipeline.py`; `outputs/checkpoints/`, `outputs/figures/`, and `results/saved_models/` are also shared write targets.
- **Circular imports:** No active circular dependency chain was detected in `src/`; imports flow mainly from `src/pipeline.py` downward into `src/data/`, `src/families/`, `src/models/`, `src/adaptnas/`, and `src/ts_tcc/`.
- **Family branching:** `default_nasade` is the only family that uses TS-TCC pretraining and DeepSVDD; `omni_anomaly`, `usad`, and `tranad` bypass that stack and rely on family-native objectives.
- **Path assumptions:** Several scripts mutate `sys.path` or assume fixed repo-relative locations like `external/usad_upstream`, `data/ServerMachineDataset`, and `./venv/Scripts/python.exe`.

## Anti-Patterns

### Orchestrator-Embedded Model Logic

**What happens:** `src/pipeline.py` defines `TCNBlock`, `CandidateModel`, three family runners, the full CLI, search loops, and final serialization in one file.
**Why it's wrong:** Adding a new family or changing the default model forces edits in the same large module, which couples model internals to CLI flow and makes reuse harder.
**Do this instead:** Put reusable model code under `src/models/` or `src/families/`, and keep `src/pipeline.py` limited to dispatch and high-level wiring.

### Duplicate Execution Paths for Similar Work

**What happens:** `scripts/run_usad_swat.py` and `scripts/run_tranad_smd.py` shell into `src.pipeline`, while `scripts/run_usad_upstream_swat.py` and `scripts/run_tranad_upstream_smd.py` reimplement training/evaluation directly.
**Why it's wrong:** Metrics, preprocessing, and output formats can drift between wrappers and reproduction scripts because there is no single shared orchestration contract.
**Do this instead:** Use `src/pipeline.py` plus `src/data/` and `src/utils/metrics.py` for project-native runs, and keep direct-upstream reproductions isolated and clearly labeled as comparisons.

## Error Handling

**Strategy:** Fail fast on missing files, invalid shapes, invalid ratios, or unsupported family/mode combinations, and let wrappers surface failures through exit codes and log files.

**Patterns:**
- Loaders raise `FileNotFoundError` or `ValueError` close to the data boundary (`src/data/omni_smd.py`, `src/data/swat.py`, `src/data/tranad_smd.py`, `src/pipeline.py`).
- Batch wrappers capture stdout/stderr into `outputs/logs/` and check subprocess return codes before saving benchmark JSON (`scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_smd.py`).

## Cross-Cutting Concerns

**Logging:** Console `print()` calls dominate the main pipeline, while batch runners redirect full runs to `outputs/logs/`; TS-TCC also writes checkpoints into `results/saved_models/`.
**Validation:** Argument validation is split between `argparse` choices and explicit runtime checks on file existence, class balance, window lengths, and dataset protocol completeness.
**Authentication:** Not applicable; the codebase is local/offline research code with filesystem-based inputs.

---

*Architecture analysis: 2026-04-27*
