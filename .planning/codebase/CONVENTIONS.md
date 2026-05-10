# Coding Conventions

**Analysis Date:** 2026-05-10

## Naming Patterns

**Files:**
- Use `snake_case.py` for first-party Python modules in `src/` and `scripts/`, for example `src/data/omni_smd.py`, `src/families/omni_anomaly.py`, and `scripts/make_uad_smd.py`.
- Keep vendored `src/ts_tcc/` names in their upstream style, including mixed-case filenames such as `src/ts_tcc/config_files/HAR_Configs.py` and `src/ts_tcc/models/TC.py`.
- Use `__init__.py` only to mark packages or re-export a small public surface, as in `src/families/__init__.py`.

**Functions:**
- Use `snake_case` for helpers, loaders, and training routines, such as `set_global_seed` in `src/pipeline.py`, `load_raw_swat` in `src/data/swat.py`, and `train_bilevel` in `src/adaptnas/trainer.py`.
- Prefix internal helpers with `_` when they are file-local, such as `_read_txt_matrix` in `src/data/omni_smd.py` and `_sanitize_json` in `scripts/run_usad_upstream_swat.py`.
- Use a top-level `main()` for CLI scripts, then guard it with `if __name__ == "__main__":`, as in `src/pipeline.py`, `scripts/run_all_smd.py`, and `scripts/build_domain_shift_smd.py`.

**Variables:**
- Use descriptive dataset and artifact names for pipeline state, such as `target_pool_unlabeled`, `valid_ratio`, `search_candidates`, and `metrics_uad` in `src/pipeline.py` and `scripts/run_all_smd.py`.
- Use short tensor names only inside tight training loops, such as `xb`, `yb`, `xt`, and `yt_w` in `src/adaptnas/trainer.py`.
- Public PowerShell parameters use `PascalCase` names such as `-Mode`, `-Family`, and `-DataDir` in `run.ps1`, even though Python flags remain `--snake_case`.

**Types:**
- Use `PascalCase` for classes and dataclasses, such as `RawSMDMachine` in `src/data/omni_smd.py`, `RawSWaTDataset` in `src/data/swat.py`, and `UsadArchConfig` in `src/families/usad.py`.
- When a module is in the newer first-party style, annotate return types and container types explicitly, as in `src/data/swat.py`, `src/families/tranad.py`, and `src/families/omni_spot.py`.

## Code Style

**Formatting:**
- No repo-level formatter config is detected. There is no `pyproject.toml`, `ruff.toml`, `setup.cfg`, `.flake8`, `.prettierrc`, or `eslint` config at the repo root.
- Use 4-space indentation and standard Python spacing. This is consistent in `src/pipeline.py`, `src/data/omni_smd.py`, and `scripts/preprocess.py`.
- Prefer triple-double-quoted module docstrings when a file needs context, as in `src/pipeline.py`, `scripts/preprocess.py`, and `src/families/usad.py`.
- Preserve the file-local style instead of forcing a global cleanup. `src/data/` and `src/families/` use a newer, typed style with grouped imports and restrained blank lines, while `src/models/` and `src/ts_tcc/` are looser and more compact.

**Linting:**
- No enforced lint runner is detected in the repo. `requirements.txt` contains runtime libraries only.
- Inline suppression appears only where compatibility workarounds need it, such as `# type: ignore` and `# noqa` in `scripts/run_usad_upstream_swat.py`.
- Keep new code warning-free without adding broad suppressions, because there is no project linter to police them later.

## Import Organization

**Order:**
1. Standard library imports first, as in `src/pipeline.py` and `scripts/run_usad_upstream_swat.py`.
2. Third-party libraries second, such as `numpy`, `pandas`, `torch`, and `sklearn` in `src/data/swat.py` and `src/families/usad.py`.
3. Local package imports last, using either absolute `src.*` imports in top-level orchestrators or relative imports inside a package.

**Path Aliases:**
- No custom import aliasing is configured.
- Top-level entrypoints import via the package root, for example `from src.data.swat import RawSWaTDataset` in `src/pipeline.py`.
- Package-internal modules prefer relative imports, for example `from .optimizer import AdaptNASOptimizer` in `src/adaptnas/trainer.py` and `from .omni_spot import SPOT` in `src/families/omni_eval.py`.

## Error Handling

**Patterns:**
- Validate file existence and shape assumptions up front, then raise specific exceptions. See `src/data/omni_smd.py`, `src/data/swat.py`, `scripts/preprocess_smd.py`, `scripts/preprocess.py`, and `src/families/usad.py`.
- Use `ValueError` for bad arguments or invalid shapes, `FileNotFoundError` for missing inputs, and `RuntimeError` for impossible runtime states. This is the dominant pattern in `src/pipeline.py`, `src/families/omni_spot.py`, and `src/adaptnas/optimizer.py`.
- Treat `assert` as a legacy or low-level invariant check only. It appears in `src/models/tscnn.py` and vendored `src/ts_tcc/` files, but newer first-party data and family modules prefer explicit exceptions.
- Reserve broad `except Exception` blocks for optional environment compatibility or safe metric fallbacks, such as UTF-8 console reconfiguration in `src/pipeline.py`, AUROC fallback in `src/utils/metrics.py`, the optional `seaborn` stub in `scripts/run_usad_upstream_swat.py`, and GPD fitting fallback in `src/families/omni_spot.py`.

## Logging

**Framework:** Mixed `print` and `logging`

**Patterns:**
- Use plain `print` for first-party CLI progress, search summaries, and artifact locations. This is the dominant pattern in `src/pipeline.py`, `scripts/run_all_smd.py`, `scripts/preprocess_smd.py`, and `scripts/make_uad_smd.py`.
- Use bracketed status prefixes when printing operational messages, such as `[INFO]`, `[WARN]`, `[FAIL]`, and `[OK]` in `src/pipeline.py`, `scripts/run_usad_swat.py`, and `scripts/build_domain_shift_smd.py`.
- Keep `logging` scoped to the vendored TS-TCC subsystem. `src/ts_tcc/utils.py` builds a file+console logger, and `src/ts_tcc/main.py` passes it into `src/ts_tcc/trainer/trainer.py`.
- Do not introduce a second logging abstraction for new first-party modules unless the entire caller chain already uses it. Match the surrounding file.

## Comments

**When to Comment:**
- Comment paper alignment, data protocol, tensor shapes, and phase boundaries. Examples include the top-of-file overview in `src/pipeline.py`, architecture docstrings in `src/families/usad.py` and `src/families/tranad.py`, and shape notes in `src/models/tscnn.py`.
- Keep comments short and local. First-party modules mostly avoid line-by-line commentary.
- Keep new comments ASCII. A few copied comments in `src/utils/metrics.py`, `src/data/datasets.py`, and `src/adaptnas/search_space.py` show encoding artifacts and are not a style to repeat.

**JSDoc/TSDoc:**
- Not applicable.
- Python docstrings are selective rather than comprehensive. Add them to entrypoints, dataclasses, and non-obvious helpers, not every small tensor operation.

## Validation and Reproducibility

- Validate CLI inputs at the boundary. Python entrypoints rely on `argparse` defaults and `choices` in `src/pipeline.py`, `scripts/build_domain_shift_smd.py`, and `scripts/run_all_smd.py`, while `run.ps1` uses `ValidateSet` and explicit `throw`.
- Validate data shapes, window lengths, and label alignment before model code runs. See `src/data/omni_smd.py`, `src/data/swat.py`, `scripts/preprocess.py`, and `scripts/preprocess_smd.py`.
- Seed randomness explicitly for reproducibility. Use `set_global_seed` in `src/pipeline.py` and the matching helper in `scripts/run_tranad_upstream_smd.py`.
- Preserve best model state with in-memory clones before reloading it. This pattern appears in `src/adaptnas/trainer.py`, `src/families/usad.py`, `src/families/tranad.py`, and `src/pipeline.py`.
- Persist validation artifacts as JSON, NPZ, PT, and PNG outputs instead of transient console-only reporting. See `src/pipeline.py`, `scripts/run_all_smd.py`, `scripts/make_uad_smd.py`, and `src/ts_tcc/trainer/trainer.py`.

## Function Design

**Size:**
- Keep low-level helpers small and single-purpose, as in `src/utils/schedulers.py`, `src/data/omni_smd.py`, and `src/data/swat.py`.
- Accept that orchestration files are large. `src/pipeline.py` centralizes end-to-end family flows instead of delegating each branch into a separate service layer.
- When adding new logic to a large file, prefer a new helper function near the call site instead of expanding an already large branch inline.

**Parameters:**
- Prefer explicit parameters or dataclass-based config objects over hidden globals. See `ArchConfig` in `src/adaptnas/search_space.py`, `UsadArchConfig` in `src/families/usad.py`, and `RawSWaTDataset.from_csvs` in `src/data/swat.py`.
- Expose model-family knobs directly in the CLI layer, then pass them down unchanged. `src/pipeline.py` is the authoritative example.
- Use keyword-only parameters when a function accepts several tuning knobs, as in `_window_loader` helpers in `src/families/usad.py` and `src/families/tranad.py`.

**Return Values:**
- Return NumPy arrays from loaders and scoring helpers, such as `load_raw_swat`, `build_upstream_usad_flat_windows`, and `score_tranad_windows`.
- Return plain dictionaries for metrics, histories, and serialized outputs, such as `validate_usad_on_windows`, `train_bilevel`, and the JSON payload builders in `scripts/run_usad_upstream_swat.py`.
- Dataset objects should return tuples that match `DataLoader` expectations, as in `ArrayDataset` in `src/data/datasets.py`.

## Module Design

**Exports:**
- Import most modules directly by file path. `src/pipeline.py` imports concrete symbols from `src.data.*`, `src.models.*`, and `src.families.*` rather than relying on package-wide re-export layers.
- Use a barrel only when a package truly exposes a public family surface. `src/families/__init__.py` is the only deliberate example and defines `__all__`.

**Barrel Files:**
- `src/families/__init__.py` is the only package that actively re-exports symbols.
- `src/__init__.py`, `src/models/__init__.py`, `src/utils/__init__.py`, and `src/adaptnas/__init__.py` are minimal or empty. Do not assume package-level imports exist unless the file already exports them.

---
*Convention analysis: 2026-05-10*
