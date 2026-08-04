# MSL Moderate-Shift Proxy-Only Main Rule

This report defines the current main MSL pair-selection rule.
A pair is labeled `good` only for audit/reporting, using `combined.auroc > uad_source.auroc and best_NAS.auroc > best_fixed.auroc`.

## Rule

`same-prefix buildable; PAD >= Q25(PAD | domain_auc < 1) = 0.8000; TS-JEPA domain_auc <= Q60(domain_auc | domain_auc < 1) = 0.9286; 7.0e+06 <= precheck_feature_mean_l2 <= 2.5e+07`

## Threshold Sources

| Threshold | Definition | Value |
| --- | --- | ---: |
| PAD min | Q25(PAD | domain_auc < 1) | 0.8000 |
| domain_auc max | Q60(domain_auc | domain_auc < 1) | 0.9286 |
| precheck L2 min | Q25(precheck_feature_mean_l2 | all buildable pairs) | 7.0e+06 |
| precheck L2 max | Q75(precheck_feature_mean_l2 | domain_auc < 1) | 2.5e+07 |

## Audit Result

- Audited MSL pairs considered: 51
- Good pairs in audited universe: 14
- Selected pairs: 6
- True positives: 5
- False positives: 1
- Precision: 0.8333
- Recall over good pairs: 0.3571
- F1 over good pairs: 0.5000
- Mean combined-source AUROC delta: 0.0657
- Mean NAS-fixed AUROC delta: 0.1094

## Selected Pairs

| PAD rank | Prefix | Pair | PAD | Domain AUC | precheck L2 | val anom | d combined-source | d NAS-fixed | good |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 23 | M | M-2__to__M-5 | 1.7647 | 0.9028 | 1.2e+07 | 4 | 0.1111 | 0.1111 | yes |
| 36 | M | M-1__to__M-5 | 1.2941 | 0.9028 | 1.2e+07 | 4 | 0.1667 | 0.0741 | yes |
| 41 | T | T-12__to__T-4 | 1.2000 | 0.9200 | 2.5e+07 | 1 | 0.0385 | 0.1923 | yes |
| 42 | F | F-5__to__F-7 | 1.0000 | 0.8472 | 7.0e+06 | 5 | 0.2011 | 0.1250 | yes |
| 43 | T | T-12__to__T-13 | 0.8000 | 0.7200 | 2.0e+07 | 3 | -0.2000 | 0.0000 | no |
| 44 | T | T-13__to__T-4 | 0.8000 | 0.7600 | 2.5e+07 | 1 | 0.0769 | 0.1538 | yes |

## Paper Explanation

The rule keeps source-target pairs in a moderate domain-shift band: the proxy shift is not too weak by PAD, not trivially separable by TS-JEPA domain AUC, and not an extreme raw-scale outlier by `precheck_feature_mean_l2`. All thresholds are computed from pre-benchmark candidate statistics.
