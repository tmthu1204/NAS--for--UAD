# Coding Conventions

**Analysis Date:** 2026-04-27

## Naming Patterns

**Files:**
- Use `snake_case.py` for runtime modules and experiment scripts: `src/pipeline.py`, `src/data/omni_smd.py`, `src/families/omni_anomaly.py`, `scripts/build_domain_shift_smd.py`.
- Use family-specific suffixes to signal protocol scope: `src/data/tranad_smd.py`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`.
- Use `_upstream_` in filenames when a script wraps an external reference implementation instead of the local family code: `scripts/run_tranad_upstream_smd.py`, `scripts/run_usad_upstream_swat.py`.

**Functions:**
- Use `snake_case` for helpers and orchestration functions: `set_global_seed`, `load_npz_if_exists`, `compute_domain_shift_metrics`, `score_tranad_windows`.
- Use verb-first names for operational helpers: `build_dataset_arg`, `save_json`, `load_raw_swat`, `create_dataset`.
- Use `get_fixed_paper_*` and `sample_*_arch` as the standard factory pair for family configs in `src/families/omni_anomaly.py`, `src/families/usad.py`, and `src/families/tranad.py`.

**Variables:**
- Use short research-oriented tensor names inside loops and scoring code: `xb`, `yb`, `xt`, `yt`, `Fs`, `Xs_norm`, `X_target_pool`.
- Use uppercase module constants for directories and ranges: `SHIFT_RANGES` in `scripts/make_uad_smd.py`, `LOGS` and `BENCHMARKS` in runner scripts.

**Types:**
- Use `PascalCase` for models, datasets, and dataclasses: `CandidateModel`, `RawSMDMachine`, `RawSWaTDataset`, `OmniArchConfig`, `UsadArchConfig`, `TranADArchConfig`.
- Use dataclasses for architecture and dataset configuration objects instead of YAML or TOML configs: `src/adaptnas/search_space.py`, `src/families/*.py`, `src/data/*.py`.

## Code Style

**Formatting:**
- No repo-level formatter configuration is detected. `pyproject.toml`, `setup.cfg`, `.flake8`, `.pylintrc`, `pytest.ini`, and `tox.ini` are not present at the repository root.
- Source uses 4-space indentation and mostly PEP 8 spacing, but blank-line density varies noticeably between files such as `src/data/datasets.py`, `src/models/transformer.py`, and `src/ts_tcc/utils.py`.
- JSON artifacts are written with `indent=2` and usually `ensure_ascii=False`: `src/pipeline.py`, `scripts/run_all_smd.py`, `scripts/compare_mode_benchmarks.py`, `scripts/run_tranad_upstream_smd.py`.

**Linting:**
- No enforced lint tool is detected.
- Code relies on local discipline plus runtime validation rather than automated style checks.

## Import Organization

**Order:**
1. Standard library imports first: `argparse`, `json`, `os`, `random`, `sys`, `pathlib`.
2. Third-party imports second: `numpy`, `torch`, `pandas`, `sklearn`.
3. Local imports last, usually rooted at `src.`: `from src.data.swat import RawSWaTDataset`.

**Path Aliases:**
- No import alias system is configured.
- Mainline code uses explicit absolute package imports from `src`: `src/pipeline.py`, `scripts/run_tranad_upstream_smd.py`.
- The `src/ts_tcc` subtree keeps standalone script-style relative imports such as `from utils import _logger` in `src/ts_tcc/main.py`.
- Several scripts mutate `sys.path` instead of relying on packaging: `scripts/build_domain_shift_smd.py`, `scripts/run_tranad_upstream_smd.py`, `scripts/run_usad_upstream_swat.py`.

## CLI and Operational Patterns

- Use `argparse` for every Python entrypoint, with long `--snake_case` flags and a `main()` guard: `src/pipeline.py`, `scripts/run_all_smd.py`, `scripts/compare_mode_benchmarks.py`, `scripts/make_uad_smd.py`.
- Keep protocol selection on the CLI instead of config files. The central switches are `--mode`, `--family`, and dataset-path arguments in `src/pipeline.py`.
- Mirror the Python CLI in PowerShell for Windows operation. `run.ps1` exposes `PascalCase` parameters like `$Mode`, `$Family`, and `$Device`, then translates them into the Python flag set.
- Use benchmark tags in output folder names rather than a registry. Existing artifact names such as `outputs/benchmarks/default_nasade-uad_source-cross_medium_source_quick` and `outputs/benchmarks/tranad-upstream-smoke` show the naming pattern.

## Tensor and Device Handling

- Pass device as a plain string through function boundaries and move models/tensors at call sites: `model.to(device)`, `xb.to(device)`, `torch.tensor(..., device=device)` in `src/pipeline.py`, `src/adaptnas/optimizer.py`, `src/families/*.py`.
- Default device selection is opportunistic CUDA, falling back to CPU: `src/pipeline.py`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`.
- `run.ps1` adds an operational `auto` mode and explicitly downgrades `cuda` to `cpu` when CUDA is unavailable.
- Default tensor dtype is `float32` across the main pipeline and data modules, but the TranAD path intentionally switches to `double`: `src/families/tranad.py`, `src/data/tranad_smd.py`, `src/pipeline.py`.
- Dataset boundaries are where NumPy becomes Torch. `src/data/datasets.py` and family-local `_window_loader` helpers are the common conversion points.

## Config Conventions

- Treat architecture choices as Python dataclasses or dataclass-like objects, not serialized config files: `ArchConfig` in `src/adaptnas/search_space.py`, plus `OmniArchConfig`, `UsadArchConfig`, and `TranADArchConfig`.
- Treat experiment configuration as CLI flags plus derived metadata written beside artifacts. `scripts/make_uad_smd.py` emits `split_metadata.json`; `scripts/build_domain_shift_smd.py` emits `data/smd_experiments/manifest.json`.
- Reuse per-family helper pairs for fixed baselines and partial NAS search space construction: `get_fixed_paper_*` and `sample_*_arch` in `src/families/*.py`.
- Keep run summaries as JSON payloads with primitive metrics, search history, and family notes: `outputs/results.json`, `outputs/benchmarks/tranad-upstream-smoke/machine-1-1.json`.

## Error Handling

**Patterns:**
- Raise explicit `FileNotFoundError`, `ValueError`, and `RuntimeError` for bad inputs, unsupported modes, missing labels, or broken assumptions: `src/pipeline.py`, `src/data/swat.py`, `src/data/omni_smd.py`, `scripts/preprocess_smd.py`.
- Use warning-style continuation for batch dataset creation and benchmark sweeps, especially when some cases are expected to fail: `scripts/build_domain_shift_smd.py`, `scripts/run_all_smd.py`, `scripts/make_uad_smd.py`.
- Sanitize non-finite floats before JSON export instead of letting dumps fail: `_sanitize_json` in `scripts/run_all_smd.py`, `scripts/run_tranad_upstream_smd.py`, `scripts/run_usad_upstream_swat.py`.

## Logging

**Framework:** Mixed `print` and `logging`

**Patterns:**
- The main pipeline and most scripts use `print` with status prefixes like `[INFO]`, `[WARN]`, `[OK]`, and `[FAIL]`: `src/pipeline.py`, `scripts/preprocess_smd.py`, `scripts/run_all_smd.py`.
- The TS-TCC subsystem uses a file-plus-console logger via `_logger` in `src/ts_tcc/utils.py`, and writes timestamped log files under its experiment directory.
- Batch runners redirect subprocess stdout/stderr to `outputs/logs/*.txt` and treat the log file as the debugging source of truth: `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`.

## Comments

**When to Comment:**
- Use module docstrings or block comments to explain protocol semantics, family fidelity, and research assumptions: `src/pipeline.py`, `src/families/omni_anomaly.py`, `src/families/tranad.py`.
- Use inline comments to pin behavior to paper or upstream semantics: `src/families/usad.py`, `src/data/swat.py`, `scripts/make_uad_smd.py`.

**JSDoc/TSDoc:**
- Not applicable.
- Python docstrings are present on many public helpers, but consistency drops in older or imported upstream-style code such as `src/ts_tcc/main.py`.

## Function Design

**Size:** Large orchestration functions are accepted. `src/pipeline.py` is the canonical example and centralizes CLI parsing, family dispatch, pretraining, search, scoring, and output writing.

**Parameters:**
- Public helpers prefer explicit keyword-friendly parameters instead of opaque config blobs: `compute_domain_shift_metrics`, `train_usad_source`, `validate_tranad_on_windows`.
- Scripts commonly pass many CLI values straight through to helper functions without an intermediate config layer.

**Return Values:**
- Favor plain dictionaries of scalar metrics and training curves over custom result classes: `src/pipeline.py`, `scripts/run_tranad_upstream_smd.py`, `scripts/compare_mode_benchmarks.py`.

## Module Design

**Exports:** Runtime modules typically expose direct functions and classes. `src/families/*.py` and `src/data/*.py` are imported explicitly rather than through a rich package API.

**Barrel Files:** Limited and inconsistent.
- `src/families/__init__.py` exists as a barrel, but it exports names such as `score_usad_series` and `validate_tranad_on_series` that do not match the current function names in `src/families/usad.py` and `src/families/tranad.py`.
- Other packages such as `src/utils` and `src/models` rely mostly on direct module imports.

## Operational Conventions

- The local skill pack under `.codex/skills/` encodes a phase-based workflow rather than a Python runtime convention. `gsd-add-tests` and `gsd-verify-work` describe post-implementation test generation and conversational verification, which matches the repository’s artifact-first validation style.
- Experiment outputs are part of the working convention. `outputs/results.json`, `outputs/benchmarks/.../*.json`, `outputs/logs/*.txt`, and `outputs/checkpoints/*.pt` are treated as first-class evidence of a run.

## Notable Inconsistencies

- `src/pipeline.py` and most scripts use `src.` absolute imports, while `src/ts_tcc/main.py` uses script-local imports and its own logging/runtime structure.
- Determinism conventions diverge. `src/pipeline.py` and `scripts/run_tranad_upstream_smd.py` set `torch.backends.cudnn.deterministic = True`, but `src/ts_tcc/main.py` sets it to `False`.
- Text encoding is inconsistent. Several comments contain mojibake or mixed Vietnamese/English text in `src/utils/metrics.py`, `src/adaptnas/search_space.py`, and `src/adaptnas/optimizer.py`.
- Batch-family support is uneven. `src/pipeline.py` supports `default_nasade`, `omni_anomaly`, `usad`, and `tranad`, but `scripts/run_all_smd.py` only exposes `default_nasade` and `omni_anomaly`.

---

*Convention analysis: 2026-04-27*
