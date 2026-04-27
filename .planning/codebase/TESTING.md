# Testing Patterns

**Analysis Date:** 2026-04-27

## Test Framework

**Runner:**
- No repo-native automated test runner is detected. The repository root does not contain `pytest`, `unittest`, `tox`, `nox`, or CI workflow configuration.
- Validation is script-driven through `src/pipeline.py`, `run.ps1`, `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_upstream_smd.py`, and `scripts/run_usad_upstream_swat.py`.

**Assertion Library:**
- Not applicable for a dedicated test suite.
- The effective quality gate is successful execution plus metric output, exception checks, and artifact generation in `outputs/results.json`, `outputs/benchmarks/*`, and `outputs/logs/*`.

**Run Commands:**
```bash
python -m src.pipeline --mode uad_source --family default_nasade --dataset_or_paths ...
python scripts/run_all_smd.py --mode uad_source --family default_nasade --data_root data/smd_experiments/cross_machine_medium
python scripts/run_tranad_smd.py --machine machine-1-1 --tag smoke
python scripts/run_usad_swat.py --tag smoke
python scripts/run_tranad_upstream_smd.py --machine machine-1-1 --tag smoke
python scripts/run_usad_upstream_swat.py --tag smoke
python scripts/compare_mode_benchmarks.py --source_dirs outputs/benchmarks/... --combined_dirs outputs/benchmarks/... --out outputs/benchmarks/compare.json
.\run.ps1 -Mode uad_source -Family default_nasade -DataDir data\smd_experiments\temporal_medium\machine-1-1
```

## Test File Organization

**Location:**
- No top-level `tests/` package is present.
- Verification entrypoints live in `scripts/`, `src/pipeline.py`, and `run.ps1`.
- Generated verification artifacts live under `outputs/results.json`, `outputs/logs/*.txt`, `outputs/benchmarks/*/*.json`, and `data/smd_experiments/*`.

**Naming:**
- Runner names encode family, dataset, and fidelity level: `scripts/run_tranad_smd.py`, `scripts/run_tranad_upstream_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_usad_upstream_swat.py`.
- Benchmark directories follow `<family>-<mode>-<tag>` or `<family>-upstream-<tag>`, for example `outputs/benchmarks/default_nasade-uad_source-...` and `outputs/benchmarks/tranad-upstream-smoke`.
- `src/ts_tcc/data_preprocessing/sleep-edf/generate_train_val_test.py` is a dataset-preparation utility despite its name; it is not part of an automated test harness.

**Structure:**
```text
scripts/                     # verification runners, comparison tools, dataset builders
run.ps1                      # Windows wrapper for end-to-end runs
outputs/results.json         # latest single-run payload
outputs/logs/*.txt           # captured logs from scripted runs
outputs/benchmarks/*/*.json  # archived per-case benchmark artifacts
data/smd_experiments/*/      # generated evaluation fixtures and metadata
```

## Test Structure

**Suite Organization:**
```python
def main():
    ap = argparse.ArgumentParser()
    ...
    rc = run_cmd(cmd, log_file)
    if rc != 0:
        sys.exit(rc)

    result = load_results_json()
    if result is None:
        sys.exit(1)

    save_json(os.path.join(bench_dir, "...json"), result)

if __name__ == "__main__":
    main()
```

**Patterns:**
- Verification is benchmark-first and script-first rather than assertion-first.
- A typical run does four things:
  1. Build or select a dataset protocol from `data/` or `data/smd_experiments/`.
  2. Execute `src.pipeline` directly or through a family-specific runner.
  3. Compute anomaly metrics from the produced scores.
  4. Save JSON artifacts for later comparison.
- Comparison happens by reading saved benchmark directories in `scripts/compare_mode_benchmarks.py`, not by rerunning models inside a unit-test framework.
- Optional result visualization is handled after the run by `scripts/export_figures.py`.

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
- Only optional import shims appear, such as the `seaborn` fallback in `scripts/run_usad_upstream_swat.py` so the upstream USAD code can import cleanly.

**What NOT to Mock:**
- Core validation paths use real datasets, real subprocesses, real metric functions, and real output files.
- `scripts/run_all_smd.py` executes `python -m src.pipeline` and captures full logs instead of simulating the pipeline.

## Fixtures and Factories

**Test Data:**
```python
save_npz(str(out_train), X_train, y_train)
save_npz(str(out_target_pool), split["X_pool"], None)
save_npz(str(out_val), split["X_val"], split["y_val"])
save_npz(str(out_test), split["X_test"], split["y_test"])
```

**Location:**
- `scripts/preprocess_smd.py` builds per-machine `source.npz` and `target.npz` fixtures under `data/smd/*`.
- `scripts/make_uad_smd.py` builds experiment fixtures under `data/smd_experiments/*` and writes `split_metadata.json`.
- `scripts/build_domain_shift_smd.py` expands those fixtures into benchmark suites and writes `data/smd_experiments/manifest.json`.
- Raw family-specific fixtures live under `data/ServerMachineDataset` for `omni_anomaly` and `tranad`, and under `data/SWaT` for `usad`.

## Coverage

**Requirements:** None enforced

**View Coverage:**
```bash
Not applicable
```

- No coverage report generator, threshold, or HTML/XML artifact is detected in the repository root.

## Test Types

**Unit Tests:**
- Not detected in repo-owned code.
- The closest unit-like checks are inline validation guards in `src/data/swat.py`, `src/data/omni_smd.py`, `src/data/tranad_smd.py`, `src/utils/metrics.py`, and `scripts/preprocess.py`.

**Integration Tests:**
- Integration-style benchmarking is the dominant validation mode.
- `src/pipeline.py` validates `default_nasade`, `omni_anomaly`, `usad`, and `tranad` by producing `outputs/results.json`.
- `scripts/run_all_smd.py` sweeps many cases, writes `outputs/logs/*.txt`, and archives successful case JSONs under `outputs/benchmarks/`.
- `scripts/run_tranad_smd.py` and `scripts/run_usad_swat.py` run targeted family checks when a full sweep is unnecessary.

**E2E Tests:**
- Browser-style E2E tests are not used.
- Operational end-to-end checks are CLI-driven through `run.ps1`, `scripts/run_pipeline.sh`, and the benchmark runners in `scripts/`.

## Validation and Verification Workflow

- For `default_nasade`, the current workflow is:
  1. Build or select an experiment folder with `scripts/preprocess_smd.py`, `scripts/make_uad_smd.py`, or `scripts/build_domain_shift_smd.py`.
  2. Run `src.pipeline` directly, `run.ps1`, or `scripts/run_all_smd.py`.
  3. Inspect `outputs/results.json` and any `outputs/logs/*.txt` files.
  4. Optionally archive comparison-ready results under `outputs/benchmarks/`.
- For `usad` and `tranad`, use the dedicated family runners `scripts/run_usad_swat.py` and `scripts/run_tranad_smd.py`. `scripts/run_all_smd.py` does not cover those families.
- For upstream verification, run `scripts/run_usad_upstream_swat.py` or `scripts/run_tranad_upstream_smd.py` and compare the saved JSON payloads against repo-native runs.
- For mode comparisons, use `scripts/compare_mode_benchmarks.py` on saved source-only and combined benchmark directories instead of relying on a single in-process assertion.
- For visual inspection, use `scripts/export_figures.py` to render search curves, ROC plots, and saved training curves from `outputs/results.json`.

## Benchmark Artifacts

- Single-run payloads are written to `outputs/results.json`.
- Archived benchmark evidence is written under `outputs/benchmarks/*/*.json`.
- Sweep and subprocess diagnostics are written to `outputs/logs/*.txt`.
- Dataset metadata used for reproducibility is stored in `data/smd_experiments/*/split_metadata.json` and `data/smd_experiments/manifest.json`.
- Auxiliary verification outputs such as figures and checkpoints are stored under `outputs/figures/`, `outputs/checkpoints/`, and `outputs/baselines*.json`.

## Reproducibility Signals

- Most benchmark-facing paths expose a seed or hard-code `42`, including `src/pipeline.py`, `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`, and `scripts/run_tranad_upstream_smd.py`.
- Dataset builders persist split metadata and domain-shift diagnostics so evaluation inputs remain auditable after the run.
- Benchmark payloads preserve architecture choices, curves, and metric summaries in JSON rather than only printing them to stdout.
- Reproducibility is partial rather than uniform because `src/ts_tcc/main.py` uses a different CuDNN determinism policy than the main UAD pipeline.

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
if rc != 0:
    sys.exit(rc)
```

- Loader and preprocessing validation rejects bad files and incompatible shapes early in `src/data/*.py` and `scripts/preprocess*.py`.
- Metric code handles degenerate labels and threshold edge cases in `src/utils/metrics.py` and `src/families/omni_eval.py`.
- Batch scripts continue through `[WARN]` or `[FAIL]` cases where partial success is useful, especially in `scripts/build_domain_shift_smd.py` and `scripts/run_all_smd.py`.

## Current Gaps

- No automated regression suite exists for loaders, metric helpers, architecture samplers, JSON schemas, or benchmark comparisons.
- No CI workflow, coverage gate, or single `test` command is defined at the repository root.
- Verification coverage is uneven across families because `scripts/run_all_smd.py` omits `usad` and `tranad`.
- Legacy `src/ts_tcc/*` utilities and preprocessing scripts do not have repo-native regression checks around them.

---

*Testing analysis: 2026-04-27*
