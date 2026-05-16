# Ben-David + JEPA Latent PAD Implementation Spec

## 1. Goal

Implement a pilot pipeline that:

1. trains a global JEPA-style time-series encoder on SMD normal windows,
2. extracts latent features for each machine,
3. computes `PAD_latent` for same-family directed source-target pairs,
4. produces a ranking of machine pairs with the largest domain shift.

The pilot should be lightweight, reproducible, and kept separate from the existing `summary_auc` path until the metric is validated.

## 2. New Files

### 2.1. `src/shift/ts_jepa.py`

Responsibility:

- defines a lightweight JEPA-style time-series encoder,
- trains a global latent predictor with EMA target encoder,
- exposes feature extraction for downstream PAD scoring.

Main components:

- `TSJepaConfig`
- `TSJepaEncoder`
- `TSJepaPredictor`
- `TSJepaModel`
- `train_ts_jepa(...)`
- `extract_jepa_features(...)`

### 2.2. `src/shift/pad.py`

Responsibility:

- converts latent source-target separability into `PAD_latent`.

Main functions:

- `proxy_a_distance_from_accuracy(...)`
- `compute_pad_from_latents(...)`

### 2.3. `scripts/pilot_rank_jepa_pad_smd.py`

Responsibility:

- resolves raw SMD path,
- builds source-normal and target-normal windows for all machines,
- pretrains the global JEPA encoder,
- caches latents,
- ranks same-family directed pairs by `PAD_latent`,
- saves checkpoint and JSON outputs.

## 3. Reused Existing Files

### 3.1. `src/data/omni_smd.py`

Used for:

- loading raw SMD machine data after path resolution,
- applying the existing per-machine normalization path.

### 3.2. `scripts/make_uad_smd.py`

Used for:

- optional baseline comparison through the existing handcrafted summary proxy.

### 3.3. `src/utils/data_paths.py`

Used for:

- automatic fallback to `external/OmniAnomaly/ServerMachineDataset`.

## 4. Data Flow

### Step 1. Raw machine loading

For each machine:

- load normalized train and test series,
- convert train series into source-normal windows,
- convert test series into target windows,
- keep only target windows whose window label is normal.

### Step 2. Global JEPA pretraining set

Build one global pretraining array by concatenating sampled source-normal windows from all 28 machines.

Pilot default:

- window length: `128`
- stride: `64`
- max train windows per machine: `1000`

In practice this gives the full available source windows for each machine under the pilot stride.

### Step 3. JEPA pretraining

The pilot JEPA uses:

- a 1D convolutional encoder,
- a bidirectional GRU predictor,
- an EMA target encoder,
- masked latent prediction loss implemented by masking time spans in the input and predicting target encoder latents at masked positions.

Pilot default:

- `epochs = 10`
- `batch_size = 128`
- `lr = 1e-3`
- `mask_ratio = 0.4`
- `mask_span = 8`
- `ema_momentum = 0.99`

### Step 4. Latent caching

For each machine:

- cache source latent vectors from source-normal windows,
- cache target latent vectors from target-normal windows.

The pilot uses mean pooling over token latents to produce one latent vector per window.

### Step 5. PAD scoring

For each directed pair `(source_machine -> target_machine)` inside the same family:

- train a regularized logistic domain classifier on source vs target latents,
- compute domain accuracy on a held-out split,
- convert accuracy into `PAD_latent`.

Pilot default:

- `PAD_latent = clip(4 * acc_d - 2, 0, 2)`
- `logreg C = 0.01`

The regularization is intentionally stronger than a default unconstrained linear separator because many same-family SMD pairs are nearly perfectly separable even with simple features. The stronger regularization reduces PAD saturation and makes the pilot ranking more informative.

## 5. Outputs

### 5.1. Checkpoint

Saved to:

- `outputs/checkpoints/ts_jepa_pilot.pt`

Contents:

- JEPA config
- model weights
- training history

### 5.2. Ranking JSON

Saved to:

- `outputs/benchmarks/jepa_pad_pilot/pilot_rankings.json`

Main fields:

- `method`
- `notes`
- `config`
- `machine_stats`
- `train_history`
- `pair_rankings`
- `top_pair_overall`
- `top_pairs_by_source`

Each pair row contains:

- `source_machine`
- `target_machine`
- `family`
- `n_source_windows`
- `n_target_normal_windows`
- `pad_latent`
- optional `summary_baseline`

## 6. Pilot Command

```powershell
.\venv\Scripts\python.exe scripts\pilot_rank_jepa_pad_smd.py `
  --epochs 10 `
  --batch_size 128 `
  --max_train_windows_per_machine 1000 `
  --max_eval_windows_per_machine 1000 `
  --max_pad_samples 1000 `
  --pad_logreg_c 0.01 `
  --include_summary_baseline
```

## 7. Preliminary Pilot Status

The pilot is considered successful if:

1. the JEPA model trains end-to-end,
2. machine latents are extracted for all 28 machines,
3. same-family pair ranking is produced for all directed pairs,
4. training loss decreases across epochs,
5. the resulting PAD distribution is not completely degenerate.

## 8. Known Limitations of the Pilot

### 8.1. Simple JEPA backbone

This is a lightweight time-series JEPA implementation inspired by JEPA/V-JEPA principles, not a literal port of the video architecture.

### 8.2. One-seed ranking

The pilot uses a single seed. Before final paper selection, at least one additional seed should be checked.

### 8.3. Remaining PAD saturation

Some SMD same-family pairs still saturate near `PAD = 2`. This does not break the pilot, but it means the final version may benefit from:

- multi-seed averaging,
- an optional tie-breaker,
- or a slightly different regularization setting.

## 9. Next Integration Step

After validating the pilot ranking:

1. add a reusable `shift_metric=pad_latent_jepa` path to the SMD experiment builders,
2. generate top-ranked experiment folders automatically,
3. run `uad_source` and `adaptnas_combined` on the selected pairs,
4. report whether adaptation gains increase with `PAD_latent`.
