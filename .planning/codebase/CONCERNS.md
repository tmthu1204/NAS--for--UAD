# Codebase Concerns

**Analysis Date:** 2026-04-27

Confirmed concerns come from direct inspection of `src/`, `scripts/`, root entrypoints, packaging metadata, and current repository state. Inferred concerns are marked explicitly where the failure mode is strong but not reproduced end-to-end in this pass.

## Tech Debt

**Repository hygiene and artifact sprawl [Confirmed]:**
- Issue: generated artifacts and benchmark data live in the main repo surface, and several of them are tracked. `git status --short` currently shows modified generated files such as `outputs/results.json`, `outputs/figures/train_curve_loss.png`, `outputs/checkpoints/Base_CNN_GRU_final_best.pt`, `results/saved_models/ckp_last.pt`, and `data/smd_experiments/manifest.json`. `git ls-files` also shows tracked datasets and outputs under `data/ServerMachineDataset/`, `data/smd/`, `data/smd_experiments/`, `outputs/benchmarks/`, and `outputs/checkpoints/`.
- Files: `.gitignore`, `data/`, `outputs/`, `results/`, `docs/`, `docs.rar`
- Impact: review noise, frequent dirty worktrees after experiments, accidental publication of datasets/results, and weak provenance because reruns overwrite tracked artifacts.
- Fix approach: move datasets and run artifacts outside the repo or ignore them aggressively; keep only manifests, lightweight fixtures, and curated reference outputs under version control.

**Monolithic orchestration and wrapper drift [Confirmed]:**
- Issue: `src/pipeline.py` is 2219 lines and mixes CLI parsing, TS-TCC pretraining, default NAS-ADE search, raw-family adapters, metric calculation, and file output. Behavior is then partially reimplemented in `run.ps1`, `scripts/run_all_smd.py`, `scripts/run_pipeline.sh`, `scripts/run_tranad_smd.py`, and `scripts/run_usad_swat.py`. Documentation has also drifted: `README.md` documents `default_nasade`, `omni_anomaly`, and `usad`, but not `tranad`, even though `src/pipeline.py` supports it.
- Files: `src/pipeline.py`, `run.ps1`, `scripts/run_all_smd.py`, `scripts/run_pipeline.sh`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`, `README.md`
- Impact: feature additions require touching multiple shells and runners, and supported families/options can silently differ by entrypoint.
- Fix approach: split orchestration into reusable runner modules, keep one canonical CLI schema, and generate shell wrappers from that schema instead of hand-maintaining them.

**Broken packaging/install contract [Confirmed]:**
- Issue: runtime code imports `src.*` and `src.ts_tcc.*`, but `setup.py` declares `packages=find_packages(where="src")`. Running `find_packages(where='src')` in the project venv returns `['adaptnas', 'data', 'families', 'models', 'utils']`, which omits both the `src` namespace and the `ts_tcc` subtree.
- Files: `setup.py`, `src/__init__.py`, `src/pipeline.py`, `src/ts_tcc/`
- Impact: `pip install .` does not reproduce the import layout that `python -m src.pipeline` expects, so packaged use outside the repo root is brittle or broken.
- Fix approach: adopt a consistent package layout with a real `src` package or a standard `src/` packaging pattern, then add an install smoke test that imports the supported entrypoints.

## Known Bugs

**PowerShell wrapper rejects a supported family [Confirmed]:**
- Symptoms: `run.ps1` rejects `tranad`, even though `src/pipeline.py` accepts `--family tranad` and `scripts/run_tranad_smd.py` depends on it.
- Files: `run.ps1`, `src/pipeline.py`, `scripts/run_tranad_smd.py`
- Trigger: running `run.ps1 -Family tranad ...`.
- Workaround: call `python -m src.pipeline --family tranad ...` or `scripts/run_tranad_smd.py` directly.

**Sleep-EDF preprocessing writes fake labels [Confirmed]:**
- Symptoms: the preprocessing script writes all-zero `y` arrays for both source and target outputs.
- Files: `scripts/preprocess_sleepedf.py`
- Trigger: running `python scripts/preprocess_sleepedf.py`.
- Workaround: avoid using the generated `data/sleepedf/source.npz` and `data/sleepedf/target.npz` until real labels are implemented.

**Shared `outputs/results.json` creates cross-run race risk [Confirmed]:**
- Symptoms: all families write the same `outputs/results.json`, and batch wrappers immediately read that file as an implicit handoff.
- Files: `src/pipeline.py`, `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`
- Trigger: running experiments in parallel, rerunning after failure, or inspecting results after another job completed.
- Workaround: run jobs strictly serially and copy outputs to per-run paths immediately after each invocation.

## Security Considerations

**Unsafe local deserialization surfaces [Confirmed]:**
- Risk: `.npz` loading uses `allow_pickle=True`, and the TS-TCC path loads `.pt` files with `torch.load()`. Both trust workspace contents as executable data boundaries.
- Files: `src/pipeline.py`, `scripts/make_uad_smd.py`, `src/ts_tcc/main.py`, `src/ts_tcc/dataloader/dataloader.py`
- Current mitigation: none detected beyond local-path assumptions.
- Recommendations: remove `allow_pickle=True` where object arrays are unnecessary, document checkpoint trust boundaries, and treat `.pt` and pickled `.npz` files as privileged inputs.

**Dynamic import via `exec` [Confirmed]:**
- Risk: TS-TCC config selection imports modules with `exec(...)` based on runtime strings.
- Files: `src/ts_tcc/main.py`
- Current mitigation: dataset names are expected to be local and known.
- Recommendations: replace `exec` with an explicit module map and normal imports.

**Tracked datasets and benchmark artifacts raise redistribution/privacy risk [Inferred]:**
- Risk: raw datasets and generated benchmark outputs are stored directly under the repo tree, including `data/ServerMachineDataset/`, `data/SWaT/`, `data/smd_experiments/`, `outputs/logs/`, and `outputs/benchmarks/`.
- Files: `data/`, `outputs/`, `.gitignore`, `README.md`
- Current mitigation: no licensing/privacy boundary is enforced in repository automation.
- Recommendations: move raw datasets to external storage, keep only manifests/checksums in git, and document which artifacts are safe to share.

## Performance Bottlenecks

**Candidate evaluation repeats expensive full-fit loops [Confirmed]:**
- Problem: each default NAS-ADE candidate redoes warmup, feature extraction, SVDD fitting, target weighting, bilevel training, and final evaluation. The raw-family adapters also retrain full Omni/USAD/TranAD models per candidate.
- Files: `src/pipeline.py`, `src/adaptnas/trainer.py`, `src/adaptnas/optimizer.py`, `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`
- Cause: search is implemented as repeated end-to-end training with little checkpoint reuse or intermediate caching.
- Improvement path: cache reusable feature banks/checkpoints, split cheap ranking from full retraining, and add resumable search state.

**Memory amplification in dataset and window construction [Confirmed]:**
- Problem: `ArrayDataset` eagerly copies full arrays into tensors, `build_validation()` restacks subsets into new arrays, and `build_tranad_windows()` allocates one padded history window per timestep. The same repo also stores large raw files under `data/`, which increases local I/O and memory pressure during experiments.
- Files: `src/data/datasets.py`, `src/pipeline.py`, `src/data/tranad_smd.py`, `src/data/omni_smd.py`, `src/data/swat.py`, `data/`
- Cause: preprocessing and loaders favor full materialization over streaming, memmap, or on-the-fly slicing.
- Improvement path: move to lazy datasets, memory-mapped arrays, and batched sliding-window generation.

## Fragile Areas

**Benchmark generation is label-informed and warning-driven [Confirmed]:**
- Files: `scripts/build_domain_shift_smd.py`, `scripts/make_uad_smd.py`, `data/smd_experiments/manifest.json`
- Why fragile: cross-machine target ranking uses anomaly counts and `domain_auc` before the benchmark is frozen, `search_best_split()` enforces anomaly-count constraints using target labels, `build_domain_shift_smd.py` hardcodes `strict=False`, and batch generation catches exceptions with warning-only logging. Manifest creation can therefore bias toward easier targets and still succeed with partial outputs.
- Safe modification: separate benchmark curation from evaluation labels, default batch builds to fail-fast mode, and emit a structured failure report when any case is skipped.
- Test coverage: no regression tests detected for split search, manifest stability, or batch-builder failure handling.

**TS-TCC snapshot carries independent execution semantics [Confirmed]:**
- Files: `src/ts_tcc/main.py`, `src/ts_tcc/dataloader/dataloader.py`, `src/ts_tcc/trainer/trainer.py`, `src/pipeline.py`
- Why fragile: this subtree has its own CLI, dynamic config import, checkpoint loading path, expected `train.pt`/`val.pt`/`test.pt` layout, and its own reproducibility settings. It also sets `torch.backends.cudnn.deterministic = False`, which conflicts with the top-level pipeline's deterministic setup in `src/pipeline.py`.
- Safe modification: isolate TS-TCC behind a narrow adapter layer with explicit input/output contracts, then pin and test that boundary separately.
- Test coverage: no project-local tests detected for the adapter boundary.

**Legacy preprocessing utilities are unfinished and warning-prone [Confirmed]:**
- Files: `scripts/preprocess_sleepedf.py`, `src/ts_tcc/data_preprocessing/sleep-edf/dhedfreader.py`, `src/ts_tcc/data_preprocessing/sleep-edf/generate_train_val_test.py`
- Why fragile: `scripts/preprocess_sleepedf.py` still contains placeholder labels, and `python -m compileall src scripts` emits `SyntaxWarning` messages from `dhedfreader.py` about invalid escape sequences. These paths also depend on dataset-specific assumptions that are not covered by the main pipeline.
- Safe modification: quarantine legacy preprocessing behind explicit "experimental" boundaries, fix parser warnings, and add smoke tests for generated file formats before reusing them.
- Test coverage: none detected.

## Scaling Limits

**Workspace-local execution model [Confirmed]:**
- Current capacity: one local machine, one local Python environment, shared `outputs/` handoff files, and sequential subprocess loops in the batch runners.
- Limit: as the number of cases grows under `data/smd_experiments/` and `outputs/benchmarks/`, shared filenames such as `outputs/results.json` and global figure/checkpoint names become collision-prone, and batch throughput is capped by serial execution.
- Scaling path: run each case in an isolated run directory with immutable artifact paths, then aggregate results from manifests instead of shared globals.

**Repo-embedded datasets and vendored runtimes [Confirmed]:**
- Current capacity: the repo already includes local datasets/results plus vendored runtime trees such as `external/conda-envs/`, `external/miniconda3/`, `external/OmniAnomaly/`, `external/tranad_upstream/`, and `external/usad_upstream/`.
- Limit: checkout size, storage cost, onboarding friction, and CI/container reproducibility all degrade as the repo accumulates binary state.
- Scaling path: replace vendored runtimes with reproducible environment specs, fetch upstream repos explicitly, and keep datasets outside the source repo.

## Dependencies at Risk

**Locally vendored upstream snapshots [Confirmed]:**
- Risk: family behavior depends on copied upstream code and local runtime trees rather than a small reproducible dependency graph.
- Impact: upstream fixes are hard to absorb, local results can depend on undeclared environment drift, and security updates are easy to miss.
- Files: `external/OmniAnomaly/`, `external/tranad_upstream/`, `external/usad_upstream/`, `external/miniconda3/`, `external/conda-envs/`
- Migration plan: replace copied repos with pinned submodules or extracted adapters, and replace vendored runtimes with environment lockfiles plus documented install steps.

## Missing Critical Features

**No isolated run manifest / run ID contract [Confirmed]:**
- Problem: results, plots, checkpoints, and summaries are keyed by shared filenames instead of immutable run IDs.
- Blocks: safe parallel benchmarking, exact reruns, and post-hoc provenance audits.
- Files: `src/pipeline.py`, `src/adaptnas/trainer.py`, `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`

**No CI or project-local automated test suite for core flows [Confirmed]:**
- Problem: no dedicated project tests were detected under `src/`, `scripts/`, or `tests/`, and no `.github/workflows/` directory is present. The only project file that matches a test-like naming pattern is `src/ts_tcc/data_preprocessing/sleep-edf/generate_train_val_test.py`, which is a preprocessing script rather than a test.
- Blocks: safe refactoring of search logic, wrapper defaults, preprocessing, and security-sensitive loaders.
- Files: `src/`, `scripts/`, `tests/`, `.github/`

**No packaged-install smoke path [Confirmed]:**
- Problem: there is no verification that the install metadata in `setup.py` can produce a working import surface for the documented entrypoints.
- Blocks: reliable use outside the repo root, reproducible packaging, and any future CLI distribution.
- Files: `setup.py`, `README.md`, `src/pipeline.py`

## Test Coverage Gaps

**Pipeline orchestration, wrapper parity, and installability [Confirmed]:**
- What's not tested: `--family` parity across `src/pipeline.py`, `run.ps1`, `scripts/run_all_smd.py`, and `scripts/run_pipeline.sh`; artifact naming collisions; and package-install smoke imports.
- Files: `src/pipeline.py`, `run.ps1`, `scripts/run_all_smd.py`, `scripts/run_pipeline.sh`, `setup.py`
- Risk: supported families can diverge by entrypoint, and packaged use can break without any automated signal.
- Priority: High

**Dataset split generation and benchmark ranking [Confirmed]:**
- What's not tested: `create_dataset()`, `search_best_split()`, cross-machine target ranking, and manifest completeness when some cases fail.
- Files: `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`
- Risk: label-informed selection bias, partial manifests, or invalid split metadata can slip into published experiments unnoticed.
- Priority: High

**Family adapters, reproducibility settings, and unsafe loaders [Confirmed]:**
- What's not tested: Omni/USAD/TranAD entrypoint parity, TS-TCC determinism, safe `.pt`/`.npz` loading boundaries, and legacy preprocessing scripts.
- Files: `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`, `src/ts_tcc/main.py`, `src/ts_tcc/dataloader/dataloader.py`, `scripts/preprocess_sleepedf.py`
- Risk: the same benchmark launched through different entrypoints can produce materially different behavior, and unsafe input assumptions remain unguarded.
- Priority: Medium

---

*Concerns audit: 2026-04-27*
