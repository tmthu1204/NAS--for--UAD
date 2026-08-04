# SMD Explainable Paper-Safe Rule

This report freezes an explainable SMD pair-selection rule and audits it against the full all-pair benchmark.
The audit label `good` is not used by the rule itself; it is defined as `combined.auroc > uad_source.auroc and best_NAS.auroc > best_fixed.auroc`.

## Main Rule

`buildable cross-entity SMD pair; 38 <= global_pad_rank <= 80; precheck_feature_mean_l2 <= 5.0832; precheck_domain_auc <= 0.999897`

Interpretation: keep the adaptation-window region rather than the most extreme PAD pairs. The pair must be buildable, sit in the middle-high PAD-rank band, avoid raw feature-scale outliers, and avoid a saturated TS-JEPA domain classifier.

## Threshold Sources

| Profile | Threshold | Quantile definition | Value |
| --- | --- | --- | ---: |
| main | global_pad_rank_min | ceil(Q30(global_pad_rank | audited buildable SMD pairs)) | 38 |
| main | global_pad_rank_max | floor(Q70(global_pad_rank | audited buildable SMD pairs)) | 80 |
| main | precheck_feature_mean_l2_max | Q40(precheck_feature_mean_l2 | audited buildable SMD pairs) | 5.083209 |
| main | precheck_domain_auc_max | Q60(precheck_domain_auc | audited buildable SMD pairs) | 0.999897 |
| strict | global_pad_rank_min | ceil(Q30(global_pad_rank | audited buildable SMD pairs)) | 38 |
| strict | global_pad_rank_max | floor(Q70(global_pad_rank | audited buildable SMD pairs)) | 80 |
| strict | precheck_feature_mean_l2_max | Q30(precheck_feature_mean_l2 | audited buildable SMD pairs) | 4.933366 |
| strict | pool_feature_mean_l2_max | Q75(pool_feature_mean_l2 | audited buildable SMD pairs) | 14.035165 |
| strict | val_count_max | floor(Q75(val_count | audited buildable SMD pairs)) | 50 |
| coverage | global_pad_rank_min | ceil(Q20(global_pad_rank | audited buildable SMD pairs)) | 27 |
| coverage | precheck_feature_mean_l2_max | Q50(precheck_feature_mean_l2 | audited buildable SMD pairs) | 5.327197 |
| coverage | pool_feature_mean_l2_max | Q40(pool_feature_mean_l2 | audited buildable SMD pairs) | 10.486414 |

## Audit Result

| Profile | Selected | TP | FP | Precision | Recall over good | F1 | mean d(combined-source) | mean d(NAS-fixed) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 12 | 9 | 3 | 0.7500 | 0.2812 | 0.4091 | 0.0767 | 0.0776 |
| strict | 7 | 7 | 0 | 1.0000 | 0.2188 | 0.3590 | 0.1438 | 0.0900 |
| coverage | 25 | 15 | 10 | 0.6000 | 0.4688 | 0.5263 | 0.1198 | 0.1151 |

- Audited SMD buildable pairs: 100
- Good pairs in audited universe: 32
- Main selected pairs: 12

## Main Selected Pairs

| PAD rank | Pair | precheck L2 | precheck AUC | pool L2 | val count | d combined-source | d NAS-fixed | good |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 46 | machine-1-4__to__machine-2-4 | 4.5505 | 0.999881 | 9.7183 | 34 | 0.2046 | 0.0076 | yes |
| 47 | machine-3-10__to__machine-1-7 | 4.2260 | 0.999881 | 7.9338 | 47 | 0.0601 | 0.0913 | yes |
| 55 | machine-1-3__to__machine-2-1 | 4.4887 | 0.999790 | 14.4407 | 36 | -0.4750 | -0.5137 | no |
| 56 | machine-2-1__to__machine-2-7 | 5.0410 | 0.999895 | 13.0869 | 50 | 0.1448 | 0.0466 | yes |
| 61 | machine-1-3__to__machine-2-4 | 4.7930 | 0.999881 | 9.8390 | 34 | 0.2708 | 0.1146 | yes |
| 66 | machine-1-4__to__machine-2-1 | 4.1973 | 0.996634 | 13.6698 | 36 | 0.1892 | 0.1916 | yes |
| 67 | machine-2-7__to__machine-2-1 | 4.8948 | 0.999264 | 12.3743 | 36 | 0.0692 | 0.0072 | yes |
| 68 | machine-3-10__to__machine-2-1 | 4.9481 | 0.998948 | 14.4661 | 36 | 0.1948 | 0.0225 | yes |
| 69 | machine-3-10__to__machine-2-7 | 5.0796 | 0.999579 | 6.5158 | 50 | 0.1745 | -0.0079 | no |
| 70 | machine-2-5__to__machine-3-10 | 4.9836 | 0.999787 | 8.1414 | 67 | 0.1721 | 0.5577 | yes |
| 71 | machine-2-7__to__machine-3-10 | 4.6928 | 0.999787 | 9.1499 | 67 | -0.2342 | 0.2971 | no |
| 76 | machine-1-3__to__machine-1-7 | 3.8597 | 0.999761 | 5.9718 | 47 | 0.1490 | 0.1166 | yes |

## Paper Explanation

The rule operationalizes a moderate-shift adaptation window. Very top PAD-ranked SMD pairs are often trivially separable and can induce negative transfer, while very low-ranked pairs may not provide enough cross-domain adaptation signal. The selected band therefore uses PAD rank quantiles and filters out raw-scale outliers with `precheck_feature_mean_l2`. The `precheck_domain_auc` upper bound avoids pairs where the TS-JEPA domain classifier is already saturated. These quantities are available before running AdaptNAS and before observing test outcomes.

The `strict` profile is a conservative audit variant with zero false positives in this run. The `coverage` profile is a broader variant that recovers more good pairs at lower precision.
