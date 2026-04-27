# Codebase Concerns

**Analysis Date:** 2026-04-27

Confirmed concerns come from directly inspected code, repository state, and file layout. Inferred concerns are marked explicitly when the risk is strong but the failure was not reproduced in this pass.

## Tech Debt

**Repository hygiene and artifact sprawl [Confirmed]:**
- Issue: `git status --short` currently reports 173 entries (20 modified, 153 untracked), and generated experiment data/results live beside source. `git ls-files data` reports 306 tracked paths and `git ls-files outputs` reports 109 tracked paths.
- Files: `.gitignore`, `data/`, `outputs/`, `results/`, `docs.rar`
- Impact: review noise, merge conflicts, accidental publication of datasets/results, and weak experiment provenance because working tree state becomes part of the run.
- Fix approach: move raw data and run artifacts outside the repo or ignore them aggressively; commit only curated manifests and small reference outputs.

**Monolithic orchestration and wrapper drift [Confirmed]:**
- Issue: `src/pipeline.py` is 2219 lines and mixes CLI parsing, dataset loading, TS-TCC pretraining, default NAS-ADE search, Omni/USAD/TranAD family logic, metric reporting, and artifact writing. Behavior also diverges across `run.ps1`, `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, and `scripts/run_usad_swat.py`.
- Files: `src/pipeline.py`, `run.ps1`, `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`
- Impact: any feature change has a wide blast radius and entrypoints can silently stop matching one another.
- Fix approach: split orchestration into shared config/schema, family runners, dataset protocol helpers, and result writers; keep one source of truth for defaults.

**Search budget and artifact naming are partially hard-coded [Confirmed]:**
- Issue: the default NAS-ADE path fixes `N_ITERS = 3` in `src/pipeline.py`, and bilevel training writes global figure names such as `outputs/figures/train_curve_loss.png` and `outputs/figures/train_curve_upper_obj.png` for every candidate/final run.
- Files: `src/pipeline.py`, `src/adaptnas/trainer.py`
- Impact: search cost is not fully controlled from CLI, and later runs overwrite earlier diagnostics.
- Fix approach: expose the default-family search iteration count and namespace artifacts by family, mode, case, and architecture.

## Known Bugs

**PowerShell wrapper excludes a supported family [Confirmed]:**
- Symptoms: `run.ps1` rejects `tranad`, even though `src/pipeline.py` accepts `--family tranad`.
- Files: `run.ps1`, `src/pipeline.py`
- Trigger: running `run.ps1 -Family tranad ...`.
- Workaround: call `python -m src.pipeline` or `scripts/run_tranad_smd.py` directly.

**USAD defaults differ by entrypoint [Confirmed]:**
- Symptoms: `run.ps1` injects `--usad_latent_size 40`, while `src/pipeline.py` defaults `--usad_latent_size` to `0`, which resolves to the paper-style latent size in `src/families/usad.py` (`window_length * 100`, 1200 at the default window length of 12).
- Files: `run.ps1`, `src/pipeline.py`, `src/families/usad.py`
- Trigger: running `run.ps1` with `-Family usad` and relying on defaults.
- Workaround: pass `-UsadLatentSize 0` explicitly or avoid the wrapper until defaults are unified.

**Shared `outputs/results.json` creates cross-run race risk [Confirmed]:**
- Symptoms: every family runner writes `outputs/results.json`, and batch wrappers immediately read the same file as an implicit handoff.
- Files: `src/pipeline.py`, `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`
- Trigger: running multiple experiments in parallel, resuming after interruption, or inspecting stale outputs after a failed run.
- Workaround: run jobs strictly serially and copy results out immediately after each invocation.

## Security Considerations

**Unsafe local deserialization surfaces [Confirmed]:**
- Risk: `.npz` loading uses `allow_pickle=True`, and TS-TCC paths use `torch.load()` on `.pt` files. Both assume the workspace contents are trusted.
- Files: `src/pipeline.py`, `scripts/make_uad_smd.py`, `src/ts_tcc/main.py`, `src/ts_tcc/dataloader/dataloader.py`
- Current mitigation: none detected in code beyond local-path assumptions.
- Recommendations: remove `allow_pickle=True` where object arrays are unnecessary, restrict checkpoint loading to trusted artifacts, and document that `.pt`/`.npz` inputs are code-execution boundaries.

**Dynamic import via `exec` [Confirmed]:**
- Risk: TS-TCC config selection uses `exec(...)` to import config modules based on runtime strings.
- Files: `src/ts_tcc/main.py`
- Current mitigation: dataset choices appear to be local and expected.
- Recommendations: replace `exec` with an explicit module map and normal imports.

**Tracked raw datasets may create redistribution/privacy risk [Inferred]:**
- Risk: raw operational datasets and generated experiment splits are kept under `data/` inside the repo tree, including `data/SWaT/` and `data/ServerMachineDataset/`.
- Files: `data/`, `.gitignore`, `README.md`
- Current mitigation: not detected in repo metadata or automation.
- Recommendations: move raw datasets to external storage, track only manifests/checksums, and document the legal/privacy boundary for dataset sharing.

## Performance Bottlenecks

**Candidate evaluation repeats heavy fit loops [Confirmed]:**
- Problem: each default NAS-ADE candidate is warmed up, feature-extracted, SVDD-fitted, target-scored, bilevel-trained, and sometimes SVDD-fitted again; Omni/USAD/TranAD search loops also retrain family models per candidate.
- Files: `src/pipeline.py`, `src/adaptnas/trainer.py`, `src/adaptnas/optimizer.py`, `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`
- Cause: search is implemented as repeated end-to-end training with little caching or reuse.
- Improvement path: cache feature banks/checkpoints per source split, separate cheap ranking from full retraining, and add resumable search state.

**Memory amplification in data handling [Confirmed]:**
- Problem: `ArrayDataset` eagerly copies whole arrays into tensors, `build_validation()` re-materializes subsets with `np.stack`, and `build_tranad_windows()` allocates one padded history window per timestep. Large raw files are already in-tree, including `data/SWaT/normal.csv` at roughly 383 MB.
- Files: `src/data/datasets.py`, `src/pipeline.py`, `src/data/tranad_smd.py`, `data/SWaT/`
- Cause: preprocessing favors in-memory copies over streaming or memory-mapped access.
- Improvement path: switch to lazy datasets, memory-mapped arrays, and batched window generation.

## Fragile Areas

**Benchmark generation is label-informed [Confirmed]:**
- Files: `scripts/build_domain_shift_smd.py`, `scripts/make_uad_smd.py`
- Why fragile: cross-machine target ranking uses full-target anomaly counts and domain AUC before the split is frozen, and split search enforces anomaly-count constraints using labels. That is workable for curation, but it biases suite construction toward label-rich/easier targets.
- Safe modification: separate benchmark discovery from evaluation labels and freeze manifests once created.
- Test coverage: no regression tests detected for suite generation or manifest stability.

**TS-TCC snapshot carries independent execution semantics [Confirmed]:**
- Files: `src/ts_tcc/main.py`, `src/ts_tcc/dataloader/dataloader.py`, `src/ts_tcc/trainer/trainer.py`
- Why fragile: this subtree has its own CLI, checkpoint loading, dataset assumptions (`train.pt`/`val.pt`/`test.pt`), and seeding behavior. It also sets `torch.backends.cudnn.deterministic = False`, which conflicts with the top-level pipeline's deterministic setup.
- Safe modification: isolate it behind a narrow adapter boundary or vendor it cleanly with pinned interfaces and separate tests.
- Test coverage: no local tests detected for the adapter boundary.

**Batch builders prefer partial success over fail-fast behavior [Confirmed]:**
- Files: `scripts/build_domain_shift_smd.py`, `scripts/make_uad_smd.py`
- Why fragile: suite generation hardcodes `strict=False` when building per-dataset args and catches exceptions with warning-only logging, so manifest creation can silently produce partial corpora.
- Safe modification: default to strict mode in batch builders and emit a structured failure report.
- Test coverage: not detected.

## Scaling Limits

**Workspace-local execution model [Confirmed]:**
- Current capacity: one machine, one local Python environment, local filesystem datasets, and shared `outputs/` handoff files.
- Limit: as the number of experiment folders grows (`data/smd_experiments/` already contains hundreds of generated case folders), sequential subprocess loops and shared output paths become slow and collision-prone.
- Scaling path: run each case in an isolated run directory with immutable result paths, then aggregate via manifest-driven collectors.

**Vendored environments and upstream repos inside `external/` [Confirmed]:**
- Current capacity: the repo already contains local vendor trees such as `external/conda-envs/`, `external/miniconda3/`, `external/OmniAnomaly/`, `external/tranad_upstream/`, and `external/usad_upstream/`.
- Limit: storage bloat, inconsistent runtime state across developers, and harder onboarding/debugging.
- Scaling path: replace checked-in environments/installers with reproducible environment specs and documented fetch steps.

## Dependencies at Risk

**Locally vendored upstream snapshots [Inferred]:**
- Risk: family behavior depends on copied upstream code and ad hoc local environments rather than a small pinned dependency graph.
- Impact: upstream bug fixes or security fixes are hard to absorb, and local benchmark results can depend on undeclared environment drift.
- Migration plan: treat upstream repos as explicit submodules or extracted adapters with version tags and reproducible environment files.

## Missing Critical Features

**No isolated run manifest / run ID contract [Confirmed]:**
- Problem: results, plots, checkpoints, and summaries are mostly keyed by shared filenames rather than immutable run IDs.
- Blocks: safe parallel benchmarking, exact reruns, and post-hoc provenance audits.
- Files: `src/pipeline.py`, `src/adaptnas/trainer.py`, `scripts/run_all_smd.py`, `scripts/run_tranad_smd.py`, `scripts/run_usad_swat.py`

**No CI or project-local test suite for core flows [Confirmed]:**
- Problem: no project tests were detected under `src/` or `scripts/`, and no `.github/workflows/` directory is present.
- Blocks: safe refactoring of search logic, data split generation, wrapper defaults, and security-sensitive loaders.
- Files: `src/`, `scripts/`, `.github/`

## Test Coverage Gaps

**Pipeline orchestration and file protocol validation [Confirmed]:**
- What's not tested: parsing of `--dataset_or_paths`, family/mode gating, per-family default parity, and artifact naming collisions.
- Files: `src/pipeline.py`, `run.ps1`, `scripts/run_all_smd.py`
- Risk: wrapper drift or silent protocol mismatches will only appear after long experiments.
- Priority: High

**Dataset split generation and benchmark ranking [Confirmed]:**
- What's not tested: `create_dataset()`, `search_best_split()`, and cross-machine ranking behavior.
- Files: `scripts/make_uad_smd.py`, `scripts/build_domain_shift_smd.py`
- Risk: label-informed selection bias or broken manifests can slip into published results unnoticed.
- Priority: High

**Family adapters and reproducibility settings [Confirmed]:**
- What's not tested: Omni/USAD/TranAD entrypoint parity, TS-TCC determinism, and safe loading of `.pt`/`.npz` assets.
- Files: `src/families/omni_anomaly.py`, `src/families/usad.py`, `src/families/tranad.py`, `src/ts_tcc/main.py`, `src/ts_tcc/dataloader/dataloader.py`
- Risk: the same benchmark launched through different entrypoints can produce materially different behavior.
- Priority: Medium

---

*Concerns audit: 2026-04-27*
