# Ben-David + JEPA Latent PAD Method Spec

## 1. Problem

We want to compare `adaptnas_combined` and `uad_source` in a realistic same-family deployment setting on SMD.

The practical scenario is:

- a model is trained on one source machine,
- deployed to a different unlabeled target machine,
- both machines belong to the same system family,
- but they may differ because of operating condition, environment, calibration, measurement time, maintenance, or aging.

The goal is not only to ask whether transfer works, but to identify **which cross-machine pairs exhibit the strongest source-target mismatch**, so that the comparison between `adaptnas_combined` and `uad_source` is performed in a domain-shift setting that is both meaningful and defensible.

## 2. Papers Used and Their Roles

### 2.1. Ben-David et al.

Role in this method:

- provides the theoretical reason to focus on source-target divergence,
- motivates using a proxy for the `H`-divergence / `A-distance`,
- justifies a ranking score based on how separable source and target are.

What we take from it:

- the main paper-facing metric should target the divergence term,
- not only a heuristic reconstruction or prediction loss.

### 2.2. DANN

Role in this method:

- gives the standard operational view of proxy `A-distance`,
- shows how a domain classifier can be used to estimate source-target separability in feature space.

What we take from it:

- train a domain classifier on latent features,
- convert domain classification accuracy into a proxy `A-distance`.

### 2.3. AdaptNAS

Role in this method:

- aligns the method with the project goal,
- supports the use of latent-space source-target discrepancy to reason about when adaptation-aware NAS is beneficial.

What we take from it:

- latent discrepancy is relevant to NAS under transfer,
- using `A-distance` style ranking is consistent with the adaptation claim.

### 2.4. I-JEPA and V-JEPA

Role in this method:

- provide the representation-learning idea,
- motivate learning a structured latent space by predicting masked content in latent space rather than reconstructing raw input.

What we take from them:

- use a JEPA-style encoder as the representation backbone,
- keep JEPA as the feature-learning component,
- do **not** use JEPA loss itself as the final domain-shift score.

## 3. Final Method Choice

The selected method is:

1. Train a **global JEPA-style time-series encoder** on normal source windows from SMD.
2. Extract latent features from each source machine and each target machine.
3. Compute **PAD_latent** on those latent features.
4. Rank same-family directed cross-machine pairs by `PAD_latent` in descending order.

This is stronger than using JEPA loss alone because:

- JEPA learns the representation,
- Ben-David style PAD measures the actual source-target separability in that representation.

## 4. Target Protocol

The chosen target protocol is:

- `source normal` vs `target normal`

Why:

- the benchmark is curated offline,
- the purpose is to measure machine/domain mismatch cleanly,
- using target normal windows avoids mixing domain shift with anomaly contamination.

For SMD, the pilot implementation uses:

- source normal windows from the machine train split,
- target normal windows from the target test split filtered by window label `0`.

## 5. Pair Scope

The selected pair scope for the main experiment is:

- **same-family cross-machine only**

Why:

- this matches the real deployment story better than fully heterogeneous transfer,
- it avoids ranking pairs mainly by family identity,
- it keeps the claim focused on machine-level and condition-level mismatch.

For SMD this means:

- `machine-1-*` pairs only inside family 1,
- `machine-2-*` pairs only inside family 2,
- `machine-3-*` pairs only inside family 3.

## 6. What the Metric Measures

The primary metric is:

- `PAD_latent`

This measures domain shift from the **representation / separability view**.

Interpretation:

- if source and target latent features are easy to separate,
- the source-target divergence is large in latent space,
- and the pair is a stronger domain-shift candidate.

This does **not** directly measure:

- label-function mismatch,
- anomaly semantics,
- full Ben-David target-risk bound.

So the correct claim is:

- this method measures **latent distribution shift**,
- not the full target risk decomposition.

## 7. Mathematical Form

For each machine pair `(s, t)`:

1. Learn latent features with a JEPA-style encoder:

   - `z_s = E(x_s)`
   - `z_t = E(x_t)`

2. Train a domain classifier to separate `z_s` from `z_t`.

3. Let domain accuracy be `acc_d`.

4. Define proxy `A-distance`:

   - `PAD_latent = clip(4 * acc_d - 2, 0, 2)`

Large `PAD_latent` means larger source-target mismatch.

## 8. Why We Do Not Use JEPA Loss Alone

If we use JEPA loss only, then the score mainly measures:

- predictive mismatch,
- temporal dependency mismatch,
- dynamics mismatch.

That is useful, but weaker as a paper-facing domain-shift metric because:

- high JEPA loss can come from model weakness,
- masking choices,
- optimization noise,
- or target complexity unrelated to domain divergence.

Therefore:

- JEPA loss is better treated as a supplementary signal,
- while `PAD_latent` remains the primary ranking metric.

## 9. Pilot Scope

The pilot implementation keeps the method minimal:

- a lightweight 1D JEPA-style encoder,
- global pretraining on source-normal windows,
- same-family pair ranking,
- latent PAD ranking with a regularized logistic domain classifier.

The pilot is meant to answer:

- does the end-to-end pipeline run,
- does the metric produce a preliminary ranking,
- and can we already identify high-shift cross-machine candidates for the main experiment.

## 10. Expected Next Step

After the pilot:

1. inspect whether top-ranked pairs are stable,
2. optionally rerun with more than one seed,
3. choose top high-shift pairs and one lower-shift control pair,
4. run `uad_source` and `adaptnas_combined` on those exact pairs.
