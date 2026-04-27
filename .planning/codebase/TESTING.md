# Testing Patterns

**Analysis Date:** 2026-04-27

## Test Framework

**Runner:**
- Not detected for repo-native automated tests.
- No `pytest`, `unittest`, `nose`, `tox`, or `nox` configuration is present at the repository root.

**Assertion Library:**
- Not applicable for a dedicated test suite.
- Runtime validation mostly happens through metric thresholds, shape checks, and exception guards in `src/pipeline.py`, `src/utils/metrics.py`, and the `src/data/*.py` loaders.

**Run Commands:**
```bash
python -m src.pipeline --mode uad_source --family default_nasade --dataset_or_paths ...
python scripts/run_all_smd.py --mode uad_source --family default_nasade --data_root data/smd_experiments/cross_machine_medium
python scripts/run_tranad_upstream_smd.py --machine machine-1-1 --tag smoke
python scripts/run_usad_upstream_swat.py --train_csv data/SWaT/... --test_csv data/SWaT/... --tag smoke
python scripts/compare_mode_benchmarks.py --source_dirs outputs/benchmarks/... --combined_dirs outputs/benchmarks/... --out outputs/benchmarks/compare.json
```

## Test File Organization

**Location:**
- No first-party `tests/` directory is detected under the repo-owned source tree.
- Validation entrypoints live in `scripts/` and in the `main()` block of `src/pipeline.py`.
- Benchmark evidence is stored under `outputs/benchmarks/` and `outputs/logs/`.

**Naming:**
- Script names describe dataset and fidelity level rather than test granularity: `scripts/run_tranad_smd.py`, `scripts/run_tranad_upstream_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_usad_upstream_swat.py`.
- Benchmark directories follow `<family>-<mode>-<tag>` or `<family>-upstream-<tag>` naming, for example `outputs/benchmarks/default_nasade-uad_source-cross_medium_source_quick` and `outputs/benchmarks/tranad-upstream-smoke`.

**Structure:**
```text
scripts/                     # dataset builders, benchmark runners, comparison tools
outputs/logs/*.txt           # full subprocess logs from sweeps
outputs/results.json         # latest single-run payload
outputs/benchmarks/*/*.json  # per-case benchmark artifacts
data/smd_experiments/*/      # generated evaluation datasets and metadata
```

## Test Structure

**Suite Organization:**
```python
def main():
    ap = argparse.ArgumentParser()
    ...
    args = ap.parse_args()
    ...
    result = {..., "metrics_uad": metrics_uad}
    save_json(...)

if __name__ == "__main__":
    main()
```

**Patterns:**
- Validation is benchmark-first and script-driven rather than assertion-first.
- Each runner builds inputs, trains or evaluates a model, computes anomaly metrics, and writes a JSON artifact.
- Comparison is done by aggregating saved benchmark directories instead of re-running inside a test runner: `scripts/compare_mode_benchmarks.py`.

## Mocking

**Framework:** Not used

**Patterns:**
```python
try:
    import seaborn  # type: ignore
except Exception:
    sys.modules["seaborn"] = types.ModuleType("seaborn")
```

**What to Mock:**
- Only lightweight import shims appear, such as the `seaborn` fallback in `scripts/run_usad_upstream_swat.py` to keep the upstream script importable.

**What NOT to Mock:**
- Core evaluation paths use real datasets, real metrics, and real output files.
- `scripts/run_all_smd.py` runs `python -m src.pipeline` as a subprocess and captures full logs instead of faking pipeline behavior.

## Fixtures and Factories

**Test Data:**
```python
metadata = build_metadata(...)
save_npz(str(out_train), X_train, y_train)
save_npz(str(out_target_pool), split["X_pool"], None)
save_npz(str(out_val), split["X_val"], split["y_val"])
save_npz(str(out_test), split["X_test"], split["y_test"])
```

**Location:**
- Experiment fixtures are generated, not hand-written.
- `scripts/make_uad_smd.py` creates `train_normal.npz`, `target_pool_unlabeled.npz`, `val_mixed.npz`, `test_mixed.npz`, and `split_metadata.json`.
- `scripts/build_domain_shift_smd.py` builds whole suites and writes `data/smd_experiments/manifest.json`.

## Coverage

**Requirements:** None enforced

**View Coverage:**
```bash
Not applicable
```

## Test Types

**Unit Tests:**
- Not detected in repo-owned code.
- Searches of repo-owned files did not find `pytest` or `unittest` usage, and no `tests/` package exists outside vendored environments.

**Integration Tests:**
- The dominant validation mode is full-pipeline or per-family benchmark execution.
- `src/pipeline.py` validates `default_nasade`, `omni_anomaly`, `usad`, and `tranad` by producing `outputs/results.json`.
- `scripts/run_all_smd.py` sweeps many cases, stores logs in `outputs/logs/`, and archives successful JSON results under `outputs/benchmarks/`.

**E2E Tests:**
- Browser-style E2E tests are not used.
- Operational end-to-end checks are shell-driven via `run.ps1` and `scripts/run_pipeline.sh`.

## Validation Approach

- `default_nasade` in `uad_source` validates architectures on held-out normal windows using an SVDD objective before reporting anomaly metrics on `val_mixed` or `test_mixed`: `src/pipeline.py`.
- `default_nasade` in `adaptnas_combined` chooses architectures using an unlabeled upper objective and only uses labeled splits for post-training reporting: `src/pipeline.py`.
- `omni_anomaly`, `usad`, and `tranad` each implement family-local validation helpers and compare a fixed baseline against searched variants: `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`.
- Metric calculation is centralized in `src/utils/metrics.py` for AP, AUROC, POT-like F1, best F1, event F1, and delay. Omni keeps a separate evaluator in `src/families/omni_eval.py`.

## Benchmark Artifacts

- Single-run payloads are written to `outputs/results.json`.
- Batch and comparison artifacts are written under `outputs/benchmarks/`, with observed examples such as:
  - `outputs/benchmarks/default_nasade-uad_source-cross_medium_source_quick/machine-1-1__to__machine-1-2.json`
  - `outputs/benchmarks/omni_anomaly-uad_source/summary.json`
  - `outputs/benchmarks/tranad-upstream-smoke/machine-1-1.json`
- Log capture is explicit. `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, and `scripts/run_usad_swat.py` write per-case logs to `outputs/logs/*.txt`.
- Existing root outputs such as `outputs/baselines_summary.json`, `outputs/checkpoints/*.pt`, and `outputs/figures/*.png` are part of the validation trail.

## Reproducibility Signals

- Most runtime paths expose a seed or hard-code `42`: `src/pipeline.py`, `scripts/build_domain_shift_smd.py`, `scripts/make_uad_smd.py`, `scripts/run_tranad_upstream_smd.py`.
- Dataset-generation scripts persist split metadata and domain-shift diagnostics, which makes evaluation inputs auditable after the fact: `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`.
- Benchmark payloads preserve the chosen architecture, search curves, final curves, and computed metrics: `outputs/results.json`, `outputs/benchmarks/tranad-upstream-smoke/machine-1-1.json`.
- Reproducibility is partial rather than uniform because `src/ts_tcc/main.py` uses a different determinism policy from the rest of the codebase.

## Common Patterns

**Async Testing:**
```python
Not applicable
```

**Error Testing:**
```python
if not os.path.exists(path):
    raise FileNotFoundError(path)
if "X" not in data:
    raise ValueError(f"{path} missing key 'X'")
```

- Failure handling in benchmark sweeps is operational rather than assert-based. Scripts print `[FAIL]` or `[WARN]` and continue where that makes sense: `scripts/run_all_smd.py`, `scripts/build_domain_shift_smd.py`, `scripts/make_uad_smd.py`.

## Current Gaps

- There is no automated regression suite for loaders, metrics, architecture samplers, or JSON output schemas.
- There is no CI workflow or coverage gate detected in the repository root.
- Benchmark runners do not cover every declared family uniformly. `scripts/run_all_smd.py` omits `usad` and `tranad`, so those families depend on separate manual runner scripts.
- The local GSD workflow under `.codex/skills/gsd-add-tests/SKILL.md` and `.codex/skills/gsd-verify-work/SKILL.md` assumes tests and UAT can be layered on after implementation, which matches the current repository practice but also explains why first-party Python tests are missing.

---

*Testing analysis: 2026-04-27*
