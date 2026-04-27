# Practical Domain Shift for NAS-based UAD: Reasoning, Recommended Measurement Protocol, and A/A* Reading Guide

## 1. Purpose of This Document

This document is written for the practical transfer setting behind this project:

- a detector or NAS-selected detector is trained on source data,
- then transferred to an unlabeled target domain,
- source and target usually belong to the same system family rather than unrelated domains,
- but the target may differ because of machine identity, operating regime, time, environment, calibration, maintenance, or aging,
- and the target pool is unlabeled and may contain a small amount of anomaly contamination.

The goal of this document is to provide a paper-safe and implementation-oriented answer to five questions:

1. How should this problem be characterized?
2. What should be measured when we say "domain shift"?
3. Which views are primary, optional, or only auxiliary?
4. Which A/A* conference papers best support each view?
5. How should the resulting shift scores be used to support the claim that `adaptnas_combined` can outperform `uad_source` under source-target mismatch?

This document is intentionally broader than SMD. The same reasoning is meant to remain valid for other industrial or multivariate time-series datasets, with only dataset-specific details changing.

## 2. Recommended Characterization of the Problem

### 2.1. Core Claim

The practical source-to-target setting should be described as:

**same-family deployment shift under temporal nonstationarity, where the dominant observable effect is covariate shift, while concept drift is possible but not assumed**

Short form:

**a practical deployment shift in time series, primarily expressed as covariate shift under nonstationarity**

### 2.2. What Should Be Claimed and What Should Not Be Claimed

The following claim is appropriate:

$$
P_s(X) \neq P_t(X)
$$

This is the main formulation because the source and target mainly differ in feature distribution, temporal dynamics, and inter-variable dependency structure.

The following claim may be true, but should not be assumed without evidence:

$$
P_s(Y \mid X) \neq P_t(Y \mid X)
$$

That is, concept drift is possible, but it should only be claimed if there is direct evidence that the semantics of normal vs. anomalous behavior changed across domains.

The following claim is not the main focus here:

$$
P_s(Y) \neq P_t(Y)
$$

Label shift is not the primary formulation for this project because the main concern is mismatch between source-normal behavior and target behavior, not a change in anomaly prevalence itself.

### 2.3. Severity Should Not Be Hard-Coded as "Mild"

The previous version of this note leaned too strongly toward "mild covariate shift." That is too narrow for a project that may include different datasets and a range of source-target gaps.

The safer framing is:

- the setting is **same-family** rather than fully heterogeneous cross-domain transfer,
- the dominant effect is still **covariate shift under nonstationarity**,
- but the shift severity can range from **mild to hard** depending on the machine pair, time gap, or operating regime mismatch.

This wording is general enough for SMD and other datasets, while staying faithful to the actual experimental protocol.

### 2.4. Important Practical Caveat: The Target Pool Is Unlabeled

In this project, the target pool is not a clean normal-only reference set. It is unlabeled and can contain hidden anomalies.

Therefore:

- domain-shift measurement must not silently assume that every target window is normal,
- otherwise the shift score may mix together true domain gap and anomaly contamination,
- and any paper claim based on the score becomes much harder to defend.

The correct statement is:

**we measure source-target mismatch using source-normal data and a reliable target subset, or equivalently a reliability-weighted target pool**

This is a crucial design choice.

## 3. Recommended Measurement Protocol

### 3.1. Default Protocol

The recommended default protocol keeps two primary views:

1. **Representation shift**
2. **Dependency shift**

Two additional views are not part of the default protocol:

1. **Spectral shift**: optional, activated only when periodic or frequency-related regime changes are likely.
2. **Domain separability**: auxiliary only, used as a sanity check rather than a main ranking metric.

### 3.2. Why These Two Views Should Be Primary

This choice is the most defensible across datasets because:

- representation discrepancy captures temporal semantics beyond raw summary statistics,
- dependency discrepancy captures changes in inter-variable structure that are especially important in multivariate industrial systems,
- the pair remains broad enough for datasets beyond SMD,
- and it avoids over-engineering the protocol before there is evidence that frequency or adversarial separability adds decisive value.

### 3.3. Naming Recommendation

For generality across datasets, the structural second view should be named:

**dependency shift**

not simply "correlation shift" as a conceptual label.

However, in the default protocol this view is operationalized by:

**CORAL-style correlation mismatch**

This distinction matters:

- **dependency shift** is the broader concept,
- **CORAL on shared features** is the default measurement choice,
- so the wording remains general without overclaiming that the chosen metric captures every kind of relation.

## 4. View 1: Representation Shift

### 4.1. Role of This View

Representation shift is the primary view.

The reason is that in practical time-series transfer, source-target mismatch often does not appear clearly in raw mean or variance alone. It appears more clearly in a feature space that encodes local temporal semantics, subsequence context, and multivariate dynamics.

### 4.2. A/A* Papers to Read First

1. **TS2Vec: Towards Universal Representation of Time Series**, AAAI 2022  
   Why it matters: a strong default for extracting temporally meaningful representations.

2. **Contrastive Learning for Unsupervised Domain Adaptation of Time Series (CLUDA)**, ICLR 2023  
   Why it matters: supports the idea that time-series transfer should be studied in contextual representation space rather than only in raw summary space.

3. **Drift Doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection (D3R)**, NeurIPS 2023  
   Why it matters: directly motivates the importance of nonstationarity and distribution drift in multivariate time-series anomaly detection.

### 4.3. Recommended Computation

Let:

- $X_s^N$ be source-normal windows,
- $\widehat{X}_t^N$ be a reliable target-normal subset or a reliability-weighted target pool,
- $f(\cdot)$ be a fixed time-series encoder,
- $Z_s = f(X_s^N)$ and $Z_t = f(\widehat{X}_t^N)$ be the corresponding embeddings.

Then define:

$$
D_{\mathrm{rep}}(s,t) = \mathrm{MMD}(Z_s, Z_t)
$$

Practical notes:

- The encoder should be fixed across all machine pairs if the goal is pair ranking.
- TS2Vec is a strong default, but another encoder can be used if it is fixed and justified.
- The target side should be filtered or weighted; using the entire unlabeled pool without caution is not recommended.

### 4.4. What This View Really Measures

This view measures mismatch in latent temporal representation space.

It is best interpreted as:

- a proxy for how different the target windows look from the source in temporal feature space,
- not a direct proof of concept drift,
- and not by itself a guarantee that adaptation will help.

It is therefore a strong component score, but it should still be validated against downstream behavior.

## 5. View 2: Dependency Shift

### 5.1. Role of This View

Dependency shift is the second primary view.

In multivariate time series, shift is not only about each variable individually. It also appears in how variables co-vary, coordinate, or decouple under new regimes. This is especially important for sensor systems, KPI graphs, and machine states.

### 5.2. A/A* Papers to Read First

1. **Return of Frustratingly Easy Domain Adaptation**, AAAI 2016  
   Why it matters: provides the classical second-order alignment perspective behind CORAL.

2. **CauDiTS: Causal Disentangled Domain Adaptation of Multivariate Time Series**, ICML 2024  
   Why it matters: argues that cross-domain transfer in multivariate time series involves domain-common causal rationales and domain-specific correlations among variables.

3. **SARAD: Spatial Association-Aware Anomaly Detection and Diagnosis for Multivariate Time Series**, NeurIPS 2024  
   Why it matters: strongly supports the practical importance of inter-feature associations in multivariate anomaly detection.

### 5.3. Recommended Computation

Using the same source and target embeddings as above, compute covariance or correlation statistics in the shared feature space:

$$
C_s = \mathrm{Cov}(Z_s), \qquad C_t = \mathrm{Cov}(Z_t)
$$

Then define the default dependency score by CORAL-style mismatch:

$$
D_{\mathrm{dep}}(s,t) = \frac{1}{4d^2}\lVert C_s - C_t \rVert_F^2
$$

Interpretation:

- conceptually, this is the default measurement of **dependency shift**,
- operationally, it is a **CORAL-style second-order discrepancy**,
- and for multivariate datasets this is more general than calling the view only "correlation shift."

### 5.4. When This View Is Not Applicable

If the dataset is effectively univariate or has negligible cross-channel structure, dependency shift should be treated as:

- not applicable,
- or a secondary optional score rather than a default requirement.

This makes the protocol portable across datasets instead of SMD-specific.

## 6. View 3: Spectral Shift (Optional)

### 6.1. Role of This View

Spectral shift should not be part of the default protocol.

It should only be added when there is concrete reason to believe that the shift is expressed through:

- periodicity changes,
- vibration regimes,
- workload cycles,
- rotating machinery signatures,
- or other frequency-sensitive patterns.

### 6.2. A/A* Papers to Read First

1. **Domain Adaptation for Time Series Under Feature and Label Shifts (RAINCOAT)**, ICML 2023  
   Why it matters: explicitly argues that time and frequency views can shift differently across domains.

2. **Boosting Transferability and Discriminability for Time Series Domain Adaptation (ACON)**, NeurIPS 2024  
   Why it matters: emphasizes that temporal features are often more transferable, while frequency features can be more discriminative within a domain.

### 6.3. Recommended Computation

If this view is activated:

1. compute PSD or another stable spectral representation on source-normal and reliable target-normal windows,
2. compare the spectral distributions channel-wise or in shared spectral features,
3. summarize the discrepancy into a single optional score $D_{\mathrm{spec}}(s,t)$.

Important note:

- this document does not recommend a single mandatory spectral metric across all datasets,
- because the correct spectral representation depends strongly on the application,
- therefore spectral shift is an optional extension, not a default pillar.

## 7. Auxiliary View: Domain Separability

### 7.1. Role of This View

Domain separability has value, but it should not be a primary ranking metric in this project.

Its best use is:

- sanity check,
- experiment construction support,
- and rough confirmation that source and target are distinguishable.

### 7.2. A/A* Paper to Read

1. **Unsupervised Domain Adaptation by Backpropagation (DANN)**, ICML 2015  
   Why it matters: practical entry point for domain discrimination and proxy A-distance reasoning.

### 7.3. Why It Should Not Be the Main Score

Under same-family shift, shallow domain classification can easily become:

- too coarse,
- too sensitive to trivial summary statistics,
- or saturated at very high discrimination accuracy.

When this happens, it may still separate "easy" from "hard" in a broad sense, but it becomes poor at fine-grained pair ranking.

Therefore the safest stance is:

**keep separability as an auxiliary sanity check, not as the main evidence of transfer difficulty**

## 8. Reliable Target Subset and Reliability Weighting

### 8.1. Why This Matters

Because the target pool is unlabeled, a robust domain-shift protocol should explicitly control target reliability.

Otherwise:

- anomalous target windows may inflate the measured domain gap,
- the score may no longer reflect deployment shift alone,
- and the result can become unstable across machine pairs.

### 8.2. A/A* Paper to Read

1. **Confidence Score for Source-Free Unsupervised Domain Adaptation (CoWA-JMDS)**, ICML 2022  
   Why it matters: although not a time-series anomaly-detection paper, it strongly supports the general principle that target samples should not be treated as equally reliable under unlabeled adaptation.

### 8.3. Recommended Practical Rule

Use one of the following:

1. **Reliable target-normal subset**  
   Keep only windows with high pseudo-normal confidence or low anomaly score.

2. **Reliability weighting**  
   Keep all target windows, but weight them according to a reliability score when computing shift.

The exact selection rule can vary by dataset, but the document should always state that target reliability is controlled explicitly.

## 9. Should There Be One Combined Score?

### 9.1. Recommended Default Answer

For the main protocol, the safest default is:

**report component scores separately**

That means reporting:

- $D_{\mathrm{rep}}(s,t)$
- $D_{\mathrm{dep}}(s,t)$

This is more defensible than hard-coding a single weighted score too early.

### 9.2. When a Combined Score Is Acceptable

A combined score is acceptable if the goal is:

- ranking candidate machine pairs,
- selecting a benchmark subset,
- or providing an exploratory one-number summary.

In that case:

1. normalize each component score across all candidate pairs, preferably by percentile rank,
2. combine them only after normalization,
3. and clearly state that the combined score is used for ranking rather than as a fundamental law of transfer.

### 9.3. Recommended Formula If a Single Score Is Needed

Let $\widetilde{D}_{\mathrm{rep}}$ and $\widetilde{D}_{\mathrm{dep}}$ be normalized scores. Then:

$$
D_{\mathrm{pair}}(s,t) = \alpha \widetilde{D}_{\mathrm{rep}}(s,t) + (1-\alpha)\widetilde{D}_{\mathrm{dep}}(s,t)
$$

However, unlike the previous version of this note, this document does **not** recommend fixing $\alpha = 0.7$ as a paper default.

### 9.4. How $\alpha$ Should Be Chosen

The recommended choice is empirical:

1. choose a target evaluation metric $M$, preferably **AP** or **AUROC**,
2. define downstream gain for each pair:

$$
G(i,j) = M_{\mathrm{adaptnas\_combined}}(i,j) - M_{\mathrm{uad\_source}}(i,j)
$$

3. choose $\alpha$ to maximize the rank correlation between $D_{\mathrm{pair}}^{(\alpha)}(i,j)$ and $G(i,j)$ on held-out or pilot pairs.

Recommended correlation:

- Spearman
- or Kendall

If no such validation is available, then:

- use the component scores directly in the main analysis,
- and treat any combined score as exploratory only.

## 10. How the Shift Scores Should Support the Paper Claim

### 10.1. What Should Be Verified

If the goal is to support the claim that `adaptnas_combined` can outperform `uad_source` under domain shift, the shift score should not only be "large." It should be useful in explaining downstream behavior.

Two hypotheses are worth testing:

1. **Higher shift tends to hurt source-only transfer**

$$
D(i,j) \uparrow \quad \Rightarrow \quad M_{\mathrm{uad\_source}}(i,j) \downarrow
$$

2. **Higher shift increases the potential value of adaptation**

$$
D(i,j) \uparrow \quad \Rightarrow \quad G(i,j) \uparrow
$$

where $G(i,j)$ is the performance gain of `adaptnas_combined` over `uad_source`.

### 10.2. Which Metric Should Be Chosen for $M$

The main metric should be fixed in advance.

For this project, the safest choices are usually:

1. **AP**
2. **AUROC**

If a threshold-dependent F1 metric is used, the document should explicitly justify why that thresholding protocol is the right one for the paper.

### 10.3. Recommended Reporting Style

For each machine pair, report:

- the source-only performance,
- the combined-model performance,
- the performance gain,
- the representation shift score,
- the dependency shift score,
- and optionally the combined ranking score.

Then report:

- rank correlation between shift and source-only performance,
- rank correlation between shift and adaptation gain,
- and a small qualitative discussion for the strongest success and strongest failure cases.

## 11. A/A* Reading Priority

If you want a short reading list that best supports this document, use this order:

1. **TS2Vec**, AAAI 2022
2. **CLUDA**, ICLR 2023
3. **D3R**, NeurIPS 2023
4. **CORAL**, AAAI 2016
5. **RAINCOAT**, ICML 2023
6. **CauDiTS**, ICML 2024
7. **SARAD**, NeurIPS 2024
8. **ACON**, NeurIPS 2024
9. **CoWA-JMDS**, ICML 2022
10. **DEV**, ICML 2019

If you only have time for the minimum set, read:

- TS2Vec
- CLUDA
- CORAL
- CauDiTS
- SARAD
- CoWA-JMDS

## 12. Final Conclusion

### 12.1. Final Position

The problem should be framed as:

**same-family deployment shift under temporal nonstationarity, primarily expressed as covariate shift**

not as a generic large cross-domain problem, and not as a setting that is always mild.

### 12.2. Final Protocol Recommendation

The default protocol should be:

1. **Representation shift** as the primary score
2. **Dependency shift** as the second primary score
3. **Spectral shift** only when there is clear frequency-related evidence
4. **Domain separability** only as an auxiliary sanity check

### 12.3. Final Guidance on Combined Scores

The default paper-safe choice is to report the component scores separately.

If a single ranking score is needed, combine normalized scores only after selecting the weight empirically against downstream gain. Do not present a fixed manual weight as if it were supported directly by the literature.

### 12.4. A Statement That Can Be Reused in a Paper

We characterize the practical source-to-target transfer setting as same-family deployment shift under temporal nonstationarity, where the dominant observable effect is covariate shift. Accordingly, we measure source-target mismatch primarily through representation discrepancy and dependency discrepancy in a shared time-series feature space, while spectral shift is treated as optional and domain separability is used only as an auxiliary sanity check. Because the target pool is unlabeled and may contain hidden anomalies, target reliability is explicitly controlled when computing shift scores.

Shorter version:

For our practical deployment setting, domain shift is measured primarily through representation and dependency mismatch, with explicit control of target reliability and without assuming that the unlabeled target pool is clean.
