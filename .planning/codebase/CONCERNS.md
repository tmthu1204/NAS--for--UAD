# Codebase Concerns

**Analysis Date:** 2026-05-10

## Tech Debt

**Monolithic orchestration and runner drift:**
- Issue: `src/pipeline.py` owns CLI parsing, dataset loading, TS-TCC pretraining, NAS search, family dispatch, metric calculation, and artifact writes. `run.ps1`, `scripts/run_all_smd.py`, `scripts/run_pipeline.sh`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`, and `scripts/run_tranad_upstream_smd.py` each mirror part of that contract with separate defaults.
- Files: `src/pipeline.py`, `run.ps1`, `scripts/run_all_smd.py`, `scripts/run_pipeline.sh`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_upstream_smd.py`
- Impact: feature support, defaults, and output handling drift across entrypoints, and even small behavioral changes require multi-file edits.
- Fix approach: split family-specific run logic into importable modules and keep shell or helper scripts as thin pass-through wrappers over one canonical CLI.

**Packaging and dependency contract drift:**
- Issue: `setup.py` advertises `find_packages(where="src")`, `python_requires=">=3.8"`, a placeholder author string, and a looser dependency set than `requirements.txt`, while runtime commands import `src.*` and execute `python -m src.pipeline`.
- Files: `setup.py`, `requirements.txt`, `README.md`, `src/__init__.py`, `src/pipeline.py`
- Impact: packaged installs, editable installs, and repo-root execution do not share one reliable environment contract.
- Fix approach: choose one supported packaging model, publish one authoritative dependency lock, and add an install smoke test for `python -m src.pipeline`.

**Tracked generated artifacts in the source repo:**
- Issue: `.gitignore` ignores `external/` and `data/`, but tracked files still include `outputs/results.json`, `outputs/baselines/*.json`, `outputs/checkpoints/*.pt`, `outputs/logs/*.txt`, `outputs/benchmarks/**/*.json`, `results/saved_models/ckp_last.pt`, and `src/ts_tcc/data_preprocessing/epilepsy/data_files/data.csv`.
- Files: `.gitignore`, `outputs/`, `results/saved_models/ckp_last.pt`, `src/ts_tcc/data_preprocessing/epilepsy/data_files/data.csv`
- Impact: the repo accumulates noisy diffs, heavyweight artifacts, and ambiguous experiment provenance because generated state sits beside source code.
- Fix approach: keep only curated fixtures and manifests in git; move checkpoints, logs, and benchmark outputs to ignored run directories.

## Known Bugs

**PowerShell entrypoint blocks a supported family:**
- Symptoms: `run.ps1` rejects `tranad` because its `ValidateSet` allows only `default_nasade`, `omni_anomaly`, and `usad`, while `src/pipeline.py` accepts `--family tranad`.
- Files: `run.ps1`, `src/pipeline.py`
- Trigger: running `.\run.ps1 -Mode uad_source -Family tranad ...`.
- Workaround: call `python -m src.pipeline --family tranad ...` or `scripts/run_tranad_smd.py` directly.

**Sleep-EDF preprocessing writes fake labels:**
- Symptoms: the script saves all-zero `y` arrays for both source and target outputs.
- Files: `scripts/preprocess_sleepedf.py`
- Trigger: running `python scripts/preprocess_sleepedf.py`.
- Workaround: do not use the generated `.npz` outputs for supervised or evaluated flows.

**USAD upstream helper points at a different default CSV contract:**
- Symptoms: `scripts/run_usad_upstream_swat.py` defaults to `data/SWaT/normal.csv` and `data/SWaT/attack.csv`, while `run.ps1` and `README.md` use `data/SWaT/SWaT_Dataset_Normal_v1.csv` and `data/SWaT/SWaT_Dataset_Attack_v0.csv`.
- Files: `scripts/run_usad_upstream_swat.py`, `run.ps1`, `README.md`
- Trigger: running `python scripts/run_usad_upstream_swat.py` without explicit `--train_csv` and `--test_csv`.
- Workaround: pass the intended SWaT CSV paths explicitly.

## Security Considerations

**Unsafe local deserialization surfaces:**
- Risk: `np.load(..., allow_pickle=True)` accepts pickled objects from workspace files, and TS-TCC loaders use `torch.load` on `.pt` datasets and checkpoints.
- Files: `src/pipeline.py`, `scripts/make_uad_smd.py`, `src/ts_tcc/main.py`, `src/ts_tcc/dataloader/dataloader.py`
- Current mitigation: inputs are treated as trusted local files; no validation boundary is enforced in code.
- Recommendations: remove `allow_pickle=True` where object arrays are unnecessary, prefer safer formats, and treat `.pt` and pickled `.npz` files as privileged inputs.

**Dynamic code execution for config selection:**
- Risk: `src/ts_tcc/main.py` imports configs through `exec(...)` using the CLI-selected dataset name.
- Files: `src/ts_tcc/main.py`
- Current mitigation: none beyond the expectation that callers pass known dataset names.
- Recommendations: replace `exec` with an explicit mapping of dataset names to config modules.

## Performance Bottlenecks

**Search paths retrain many models per run:**
- Problem: search loops run `N_ITERS` times and retrain `args.search_candidates` models per iteration, then combined mode reruns final-only baselines for `Base_CNN_GRU`, `Base_CNN_TCN`, `Base_CNN_TRF`, and `NAS_BestArch`.
- Files: `src/pipeline.py`, `src/adaptnas/trainer.py`, `src/adaptnas/optimizer.py`, `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`
- Cause: candidate ranking relies on repeated warmup, feature extraction, SVDD fitting, bilevel training, and final rescoring with little reuse of intermediate state.
- Improvement path: cache reusable feature banks and checkpoints, prune candidates early, and persist resumable search state.

**Memory amplification from eager array and tensor materialization:**
- Problem: arrays are repeatedly copied through `fix_length()`, `load_all_smd_for_pretrain()`, `build_validation()`, `ArrayDataset`, and `build_tranad_windows()`.
- Files: `src/pipeline.py`, `src/data/datasets.py`, `src/data/tranad_smd.py`, `src/data/omni_smd.py`, `src/data/swat.py`
- Cause: loaders and preprocessing favor full in-memory `np.concatenate`, `np.stack`, and `torch.tensor(...)` copies instead of lazy slicing or memory mapping.
- Improvement path: use lazy datasets, generator-based windowing, and shared-memory views where possible.

**TranAD runs double-precision end to end:**
- Problem: TranAD windows are built as `np.float64`, converted to `torch.double`, and the model is forced to `.double()`.
- Files: `src/data/tranad_smd.py`, `src/families/tranad.py`, `scripts/run_tranad_upstream_smd.py`
- Cause: the implementation mirrors upstream numeric types instead of using a project-wide `float32` path.
- Improvement path: validate a `float32` path and keep double precision only if a benchmark requires it.

## Fragile Areas

**Label-informed benchmark construction and partial manifests:**
- Files: `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`
- Why fragile: `search_best_split()` ranks target splits using hidden target labels, anomaly-count thresholds, and `domain_auc`, then writes hidden anomaly counts into metadata. `scripts/build_domain_shift_smd.py` builds with `strict=False`, catches exceptions, prints warnings, and still writes a manifest.
- Safe modification: separate dataset curation from hidden labels, fail the batch when any split is invalid, or mark partial manifests as incomplete in machine-readable metadata.
- Test coverage: no project-local regression tests cover split selection, manifest completeness, or target-ranking behavior.

**TS-TCC subtree has independent execution semantics:**
- Files: `src/ts_tcc/main.py`, `src/ts_tcc/dataloader/dataloader.py`, `src/ts_tcc/trainer/trainer.py`, `src/pipeline.py`
- Why fragile: the vendored subtree has its own CLI, its own checkpoint and dataset layout, dynamic imports, `torch.load`ed datasets, and a different reproducibility setting (`torch.backends.cudnn.deterministic = False`) than the top-level pipeline.
- Safe modification: isolate TS-TCC behind a narrow adapter layer and test that boundary separately instead of mixing its conventions into `src/pipeline.py`.
- Test coverage: no project-local adapter or parity tests are present.

**Shared artifact handoff contract:**
- Files: `src/pipeline.py`, `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`
- Why fragile: all major entrypoints write or read the same `outputs/results.json`, and training also writes shared `outputs/figures` and `outputs/checkpoints` paths.
- Safe modification: require a run ID or output directory argument for every invocation and aggregate results from immutable per-run files.
- Test coverage: no tests enforce artifact isolation or parallel-safe runs.

**Repo-local environment assumptions:**
- Files: `run.ps1`, `README.md`, `scripts/run_all_smd.py`, `scripts/run_usad_swat.py`, `scripts/run_tranad_smd.py`
- Why fragile: the primary Windows wrapper hardcodes `.\venv\Scripts\python.exe`, default dataset roots under `data/`, and relative output folders, while helper scripts assume they are launched from the repo checkout.
- Safe modification: derive the interpreter from `sys.executable` or an environment variable, and require explicit data and output roots for automation.
- Test coverage: no smoke tests validate entrypoints outside one local workstation layout.

## Scaling Limits

**Single-workspace artifact namespace:**
- Current capacity: one local workspace writes into one shared `outputs/` tree and uses serial subprocess loops to sweep cases.
- Limit: concurrent runs clobber shared JSON, checkpoints, and figures, while long sweeps serialize behind one results namespace.
- Scaling path: isolate each run under a unique output root and add an aggregation pass that merges finished run manifests.

**RAM-bound preprocessing and search:**
- Current capacity: datasets fit only while the workspace can afford multiple in-memory copies across `numpy` and `torch`.
- Limit: larger SMD-style collections or longer windows hit memory ceilings before compute is saturated.
- Scaling path: introduce streaming or memory-mapped datasets, chunked feature extraction, and on-disk caches for reusable intermediate features.

## Dependencies at Risk

**Divergent Python dependency declarations:**
- Risk: `requirements.txt` pins one environment while `setup.py` declares a looser, shorter dependency set and an older Python floor.
- Impact: editable installs, packaged installs, and README-based environment setup can resolve different libraries.
- Files: `requirements.txt`, `setup.py`, `README.md`
- Migration plan: keep one locked environment spec and generate secondary install metadata from it.

**Vendored upstream TS-TCC snapshot:**
- Risk: `src/ts_tcc/` is a copied upstream subtree with its own docs, loaders, preprocessing scripts, and runtime assumptions, but no explicit sync mechanism.
- Impact: upstream fixes, security patches, and behavior changes are easy to miss or merge incorrectly.
- Files: `src/ts_tcc/README.md`, `src/ts_tcc/main.py`, `src/ts_tcc/dataloader/dataloader.py`, `src/ts_tcc/trainer/trainer.py`
- Migration plan: track the upstream source explicitly via subtree or submodule, or extract a stable local adapter with a documented upgrade path.

## Missing Critical Features

**No run-scoped artifact contract:**
- Problem: the codebase has no required `--out_dir` or run-id contract for the main pipeline, so experiments default to shared filenames and directories.
- Blocks: safe parallel benchmarking, reliable provenance, and restartable long-running sweeps.
- Files: `src/pipeline.py`, `run.ps1`, `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`

**No project-local automated test suite or CI pipeline:**
- Problem: no project-local test files or workflow configuration are present under tracked source paths, and no `.github/` workflow directory exists in the tracked tree.
- Blocks: safe refactoring of search logic, packaging, wrappers, and dataset builders.
- Files: `src/`, `scripts/`, `setup.py`, `README.md`

**No portable bootstrap or install path:**
- Problem: the documented and scripted flows depend on a repo-local `venv/` plus repo-relative data directories instead of a portable installer or environment bootstrap command.
- Blocks: reproducible setup in CI, containers, and clean workstations.
- Files: `run.ps1`, `README.md`, `requirements.txt`, `setup.py`

## Test Coverage Gaps

**Pipeline orchestration and wrapper parity:**
- What's not tested: argument parity across `src/pipeline.py`, `run.ps1`, `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, and `scripts/run_usad_swat.py`, plus artifact-path collisions.
- Files: `src/pipeline.py`, `run.ps1`, `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`
- Risk: supported families and defaults drift silently, and concurrent runs overwrite each other.
- Priority: High

**Dataset builders and benchmark curation:**
- What's not tested: `search_best_split()`, `create_dataset()`, cross-machine target ranking, and manifest completeness when some cases fail.
- Files: `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`
- Risk: label-informed selection bias and partial datasets can slip into experiments without an automated signal.
- Priority: High

**TS-TCC adapter boundary and unsafe loader assumptions:**
- What's not tested: TS-TCC import and adapter compatibility, deterministic behavior, `.pt` and pickled `.npz` trust boundaries, and legacy preprocessing outputs such as `scripts/preprocess_sleepedf.py`.
- Files: `src/ts_tcc/main.py`, `src/ts_tcc/dataloader/dataloader.py`, `src/pipeline.py`, `scripts/preprocess_sleepedf.py`, `scripts/make_uad_smd.py`
- Risk: data-loading security assumptions, reproducibility, and integration correctness regress without notice.
- Priority: Medium

---

*Concerns audit: 2026-05-10*
