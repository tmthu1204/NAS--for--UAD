# Coding Conventions

**Analysis Date:** 2026-04-27

## Naming Patterns

**Files:**
- Use `snake_case.py` for runtime modules and scripts: `src/pipeline.py`, `src/data/omni_smd.py`, `src/families/omni_anomaly.py`, `scripts/build_domain_shift_smd.py`.
- Encode dataset, family, or protocol directly in the filename when the behavior is specialized: `src/data/tranad_smd.py`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`.
- Use `_upstream_` in script names only when the script wraps an external reference implementation instead of the repo-native family code: `scripts/run_tranad_upstream_smd.py`, `scripts/run_usad_upstream_swat.py`.

**Functions:**
- Use `snake_case` for helpers, loaders, scorers, and training routines: `set_global_seed`, `compute_domain_shift_metrics`, `validate_omni_on_series`, `train_usad_source`, `build_dataset_arg`.
- Use verb-first names for operational helpers in `scripts/`: `load_results_json`, `save_json`, `run_cmd`, `split_dirs`, `create_dataset`.
- Keep the `get_fixed_paper_*` and `sample_*_arch` pair for family-specific fixed baselines and NAS sampling in `src/families/omni_anomaly.py`, `src/families/usad.py`, and `src/families/tranad.py`.

**Variables:**
- Use short tensor and batch names inside training code: `xb`, `yb`, `xt`, `yt`, `d_s`, `d_t` in `src/adaptnas/trainer.py` and `src/adaptnas/optimizer.py`.
- Use descriptive NumPy and dataset names at pipeline boundaries: `X_train_norm`, `X_target_pool`, `y_test_aligned`, `train_windows_full` in `src/pipeline.py` and `scripts/run_tranad_upstream_smd.py`.
- Use uppercase module constants for directories and range tables: `ROOT`, `PROJ`, `LOGS`, `BENCHMARKS` in runner scripts and `SHIFT_RANGES` in `scripts/make_uad_smd.py`.

**Types:**
- Use `PascalCase` for models, datasets, and dataclasses: `CandidateModel`, `ArrayDataset`, `RawSWaTDataset`, `RawSMDMachine`, `UsadArchConfig`, `TranADArchConfig`, `OmniArchConfig`.
- Newer code in `src/data/*.py` and `src/families/*.py` uses explicit type hints such as `str | Path`, `Tuple[np.ndarray, np.ndarray]`, and `Dict[str, float]`; match that style when editing those areas.
- Older model code in `src/models/*.py` and parts of `src/adaptnas/*.py` is lightly typed or untyped. Preserve local style when making targeted fixes, but prefer the newer typed style for net-new code.

## Code Style

**Formatting:**
- No repo-level formatter configuration is detected. `pyproject.toml`, `setup.cfg`, `.flake8`, `.pylintrc`, `pytest.ini`, `tox.ini`, and `noxfile.py` are not present at the repository root.
- Use 4-space indentation and mostly PEP 8 spacing. That pattern is consistent in `src/data/swat.py`, `src/data/omni_smd.py`, `src/families/usad.py`, and `scripts/build_domain_shift_smd.py`.
- Formatting consistency drops in older files such as `src/data/datasets.py`, `src/models/transformer.py`, and `src/ts_tcc/utils.py`, where blank-line spacing and import grouping are looser.
- JSON artifacts are written with `indent=2` and usually `ensure_ascii=False` in `src/pipeline.py`, `scripts/run_all_smd.py`, `scripts/compare_mode_benchmarks.py`, `scripts/run_tranad_upstream_smd.py`, and `scripts/run_usad_upstream_swat.py`.

**Linting:**
- No enforced lint tool is detected.
- Runtime validation and manual review act as the effective quality gate.
- When adding new code, prefer the cleaner typed style already present in `src/data/*.py`, `src/families/*.py`, and the newer runner scripts, because no formatter will normalize it afterward.

## Import Organization

**Order:**
1. Use `from __future__ import annotations` first when the module relies on modern Python type syntax, as in `src/data/swat.py`, `src/data/omni_smd.py`, and `src/families/*.py`.
2. Import standard-library modules next: `argparse`, `json`, `math`, `os`, `random`, `sys`, `pathlib`.
3. Import third-party modules after that: `numpy`, `torch`, `pandas`, `sklearn`.
4. Import local modules last, typically from the `src.` package root.

**Path Aliases:**
- No import alias system is configured.
- Mainline code prefers explicit absolute imports rooted at `src`, for example in `src/pipeline.py`, `scripts/run_tranad_upstream_smd.py`, and `scripts/run_usad_upstream_swat.py`.
- The `src/ts_tcc/*` subtree uses local script-style imports such as `from utils import _logger` in `src/ts_tcc/main.py`. Keep that style isolated to the TS-TCC subtree.
- Standalone runner scripts sometimes mutate `sys.path` before importing local packages. That pattern appears in `scripts/build_domain_shift_smd.py`, `scripts/run_tranad_upstream_smd.py`, and `scripts/run_usad_upstream_swat.py`.

## Error Handling

**Patterns:**
- Validate inputs early and raise built-in exceptions with actionable messages. Common exceptions are `FileNotFoundError`, `ValueError`, `RuntimeError`, and `NotImplementedError` in `src/pipeline.py`, `src/data/swat.py`, `src/data/omni_smd.py`, `src/data/tranad_smd.py`, `scripts/preprocess_smd.py`, and `scripts/make_uad_smd.py`.
- Prefer explicit shape and protocol checks over silent coercion. Examples include:
  - `src/data/swat.py` rejects non-2-D series and invalid `window_length` or `stride`.
  - `src/families/usad.py` rejects non-flattened window arrays.
  - `src/pipeline.py` rejects missing labeled validation data for evaluation paths.
- Use warning-and-continue behavior only for batch build or sweep scripts where one failed case should not abort the full run, as in `scripts/build_domain_shift_smd.py`, `scripts/run_all_smd.py`, and `scripts/make_uad_smd.py`.
- Use `assert` only for low-level invariants inside model code or imported upstream utilities, such as `src/models/tscnn.py`, `src/ts_tcc/models/loss.py`, and `src/ts_tcc/models/attention.py`. For repo-owned public APIs, prefer explicit exceptions.

## Logging

**Framework:** Mostly `print`; legacy TS-TCC code uses `logging`

**Patterns:**
- Use prefix-tagged console output for operational progress, for example `[INFO]`, `[WARN]`, `[OK]`, and `[FAIL]` in `src/pipeline.py`, `scripts/preprocess_smd.py`, `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, and `scripts/run_usad_swat.py`.
- Batch runners redirect subprocess stdout and stderr to `outputs/logs/*.txt`. Those log files are the primary debugging artifacts for sweep failures in `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, and `scripts/run_usad_swat.py`.
- `src/ts_tcc/utils.py` defines `_logger` for file-plus-console logging. That pattern is local to `src/ts_tcc/*` and is not reused by the main UAD pipeline.

## Comments

**When to Comment:**
- Use module docstrings or block comments to explain protocol semantics, research assumptions, and tensor layout expectations. Good examples are `src/pipeline.py`, `src/families/usad.py`, `src/families/tranad.py`, and `src/families/omni_anomaly.py`.
- Use short inline comments only for non-obvious behavior, such as layout conversions in `src/models/tscnn.py`, paper-fidelity notes in `src/families/usad.py`, and dataset-splitting rules in `scripts/make_uad_smd.py`.
- Avoid comments that restate obvious tensor moves or assignments.

**JSDoc/TSDoc:**
- Not applicable.
- Prefer Python docstrings for public helpers and family modules, following the style already used in `src/data/swat.py`, `src/data/omni_smd.py`, and `src/families/*.py`.

## Function Design

**Size:**
- Large orchestration functions are accepted for CLI entrypoints. `src/pipeline.py` and `scripts/run_all_smd.py` centralize parsing, dispatch, training, evaluation, and artifact writing.
- Reusable logic is factored into smaller helpers around those entrypoints, especially in `src/data/*.py`, `src/families/*.py`, and `scripts/make_uad_smd.py`.

**Parameters:**
- Use explicit parameters rather than opaque config dictionaries for reusable helpers: `compute_domain_shift_metrics` in `scripts/make_uad_smd.py`, `train_usad_source` in `src/families/usad.py`, and `validate_tranad_on_windows` in `src/families/tranad.py`.
- Use dataclasses for stable architecture/config bundles: `ArchConfig` in `src/adaptnas/search_space.py` and the `*ArchConfig` classes in `src/families/*.py`.
- CLI-facing functions and scripts use long `--snake_case` options via `argparse` or `ValidateSet` in `run.ps1`.

**Return Values:**
- Return plain dictionaries of scalar metrics, curves, or metadata from training and validation helpers, for example `train_bilevel` in `src/adaptnas/trainer.py`, `validate_usad_on_windows` in `src/families/usad.py`, and `compute_metrics` helpers in runner scripts.
- Return `np.ndarray` or tuple outputs from loaders and preprocessing helpers unless a module already uses a dataclass wrapper, such as `RawSWaTDataset` in `src/data/swat.py` or `RawSMDMachine` in `src/data/omni_smd.py`.

## Module Design

**Exports:**
- Import concrete functions and classes from their owning module instead of relying on wildcard exports. `src/pipeline.py` imports directly from `src.data.*`, `src.families.*`, `src.models.*`, and `src.utils.metrics`.
- Keep family modules self-contained. Each of `src/families/usad.py`, `src/families/tranad.py`, and `src/families/omni_anomaly.py` owns its architecture dataclass, model class, scorer, validator, and training loop.

**Barrel Files:**
- Barrel files exist but are lightweight: `src/families/__init__.py`, `src/models/__init__.py`, `src/data/__init__.py`, and `src/utils/__init__.py`.
- Mainline code still prefers direct submodule imports, so add new exports to the owning module first and update the barrel only when the package already exposes that surface.

## Validation Conventions

- Data loaders validate file existence, shapes, and protocol assumptions before training starts. Follow the patterns in `src/data/swat.py`, `src/data/omni_smd.py`, `src/data/tranad_smd.py`, and `scripts/preprocess.py`.
- Metric helpers handle degenerate evaluation cases instead of crashing. `src/utils/metrics.py` falls back to `0.5` AUROC when only one class is present or metric computation fails.
- Benchmark writers sanitize non-finite floats before serializing JSON. Reuse the `_sanitize_json` pattern from `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`, and `scripts/run_usad_upstream_swat.py`.
- Windows PowerShell automation uses `ValidateSet`, explicit file existence checks, and `throw` for invalid state in `run.ps1`.

## Subtree Differences

- Prefer the newer typed style used in `src/data/*.py`, `src/families/*.py`, and most `scripts/*.py` for new work.
- Treat `src/models/*.py`, `src/adaptnas/*.py`, and especially `src/ts_tcc/*` as legacy research code. Those areas use looser typing, denser scripts, more inline comments, and a different logging/import style.
- Keep edits aligned with the local subtree instead of restyling the whole repo. For example, do not convert `src/ts_tcc/main.py` to `src.` absolute imports unless the task explicitly targets that subsystem.

## Notable Inconsistencies

- Determinism policy differs by subtree. `src/pipeline.py` and `scripts/run_tranad_upstream_smd.py` enable deterministic CuDNN settings, while `src/ts_tcc/main.py` sets `torch.backends.cudnn.deterministic = False`.
- Text comments contain mixed English/Vietnamese and some mojibake in `src/utils/metrics.py`, `src/adaptnas/search_space.py`, and `src/adaptnas/optimizer.py`.
- Runner coverage is uneven. `src/pipeline.py` supports `default_nasade`, `omni_anomaly`, `usad`, and `tranad`, but `scripts/run_all_smd.py` exposes only `default_nasade` and `omni_anomaly`.

## Not Detected

- No repo-local project skill directory under `.codex/skills/` or `.agents/skills/`.
- No repo-level formatter, linter, type checker, or test runner configuration.

---

*Convention analysis: 2026-04-27*
