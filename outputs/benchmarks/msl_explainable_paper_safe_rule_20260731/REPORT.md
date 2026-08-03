# MSL Explainable Paper-Safe Rule Search

This report searches simple, explainable rule families over the audited MSL same-prefix buildable universe.
A pair is labeled `good` iff `combined.auroc > uad_source.auroc and best_NAS.auroc > best_fixed.auroc`.

- Audited MSL pairs considered: 51
- Good pairs in the audited universe: 14

## MSL Feature Note

- On MSL, `target_pool_hidden_anomaly_ratio` is nearly degenerate and `val_count` is almost constant across pairs.
- The useful discriminators are therefore dominated by `PAD`, TS-JEPA latent `domain_auc`, `precheck_feature_mean_l2`, and `val_anomaly_count`.

## Recommended Paper-Safe Profile

`Bucket A: TS-JEPA domain_auc < 1.0000, precheck_feature_mean_l2 <= 1.2e+07; within Bucket A, keep at most 1 source per target, ordered by buildable PAD rank. Bucket B override: keep any pair with val_anomaly_count >= 5`

- Selected pairs: 7
- True positives: 6
- False positives: 1
- Precision: 0.8571
- Recall over good pairs: 0.4286
- F1 over good pairs: 0.5714

## Recommended Balanced Profile

`Bucket A: PAD >= 1.5000, TS-JEPA domain_auc < 1.0000, val_anomaly_count >= 2. Bucket B override: keep any pair with val_anomaly_count >= 5`

- Selected pairs: 10
- True positives: 7
- False positives: 3
- Precision: 0.7000
- Recall over good pairs: 0.5000
- F1 over good pairs: 0.5833

## Strict Zero-FP Profile

`Bucket A: PAD >= 1.5000, TS-JEPA domain_auc < 1.0000, val_anomaly_count >= 4. Bucket B override: keep any pair with val_anomaly_count >= 5`

- Selected pairs: 5
- True positives: 5
- False positives: 0
- Precision: 1.0000
- Recall over good pairs: 0.3571
- F1 over good pairs: 0.5263

## Pareto Frontier

| Profile | TP | FP | Precision | Recall | Complexity | Rule |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
|  | 14 | 36 | 0.2800 | 1.0000 | 2 | Bucket A: PAD >= 0.5000. Bucket B override: keep any pair with val_anomaly_count >= 5 |
|  | 13 | 34 | 0.2766 | 0.9286 | 3 | Bucket A: PAD >= 0.5000, precheck_feature_mean_l2 <= 3.0e+07. Bucket B override: keep any pair with val_anomaly_count >= 5 |
|  | 12 | 31 | 0.2791 | 0.8571 | 2 | Bucket A: PAD >= 1.0000. Bucket B override: keep any pair with val_anomaly_count >= 5 |
|  | 11 | 15 | 0.4231 | 0.7857 | 3 | Bucket A: PAD >= 0.5000, TS-JEPA domain_auc < 1.0000. Bucket B override: keep any pair with val_anomaly_count >= 5 |
|  | 9 | 11 | 0.4500 | 0.6429 | 3 | Bucket A: PAD >= 1.0000, TS-JEPA domain_auc < 1.0000. Bucket B override: keep any pair with val_anomaly_count >= 5 |
|  | 8 | 7 | 0.5333 | 0.5714 | 3 | Bucket A: TS-JEPA domain_auc < 1.0000, precheck_feature_mean_l2 <= 1.2e+07. Bucket B override: keep any pair with val_anomaly_count >= 5 |
| balanced | 7 | 3 | 0.7000 | 0.5000 | 4 | Bucket A: PAD >= 1.5000, TS-JEPA domain_auc < 1.0000, val_anomaly_count >= 2. Bucket B override: keep any pair with val_anomaly_count >= 5 |
| paper_safe | 6 | 1 | 0.8571 | 0.4286 | 4 | Bucket A: TS-JEPA domain_auc < 1.0000, precheck_feature_mean_l2 <= 1.2e+07; within Bucket A, keep at most 1 source per target, ordered by buildable PAD rank. Bucket B override: keep any pair with val_anomaly_count >= 5 |
| strict | 5 | 0 | 1.0000 | 0.3571 | 4 | Bucket A: PAD >= 1.5000, TS-JEPA domain_auc < 1.0000, val_anomaly_count >= 4. Bucket B override: keep any pair with val_anomaly_count >= 5 |

## Paper-Safe Selected Pairs

| PAD rank | Prefix | Pair | PAD | Domain AUC | precheck L2 | val anom | d combined-source | d NAS-fixed | good |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | F | F-4__to__F-7 | 1.8095 | 1.0000 | 7.0e+06 | 5 | 0.0707 | 0.1440 | yes |
| 23 | M | M-2__to__M-5 | 1.7647 | 0.9028 | 1.2e+07 | 4 | 0.1111 | 0.1111 | yes |
| 25 | M | M-2__to__M-3 | 1.7500 | 0.9688 | 6.7084 | 2 | 0.1250 | -0.0312 | no |
| 28 | P | P-11__to__P-15 | 1.6667 | 0.9861 | 4.7e+06 | 2 | 0.5500 | 0.8500 | yes |
| 31 | P | P-10__to__P-11 | 1.5714 | 0.9745 | 5.3e+06 | 4 | 0.3750 | 0.2232 | yes |
| 42 | F | F-5__to__F-7 | 1.0000 | 0.8472 | 7.0e+06 | 5 | 0.2011 | 0.1250 | yes |
| 50 | F | F-8__to__F-7 | 0.1935 | 0.6750 | 7.0e+06 | 5 | 0.2255 | 0.0190 | yes |

## Balanced Selected Pairs

| PAD rank | Prefix | Pair | PAD | Domain AUC | precheck L2 | val anom | d combined-source | d NAS-fixed | good |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | F | F-4__to__F-7 | 1.8095 | 1.0000 | 7.0e+06 | 5 | 0.0707 | 0.1440 | yes |
| 23 | M | M-2__to__M-5 | 1.7647 | 0.9028 | 1.2e+07 | 4 | 0.1111 | 0.1111 | yes |
| 25 | M | M-2__to__M-3 | 1.7500 | 0.9688 | 6.7084 | 2 | 0.1250 | -0.0312 | no |
| 27 | M | M-6__to__M-7 | 1.7143 | 0.9184 | 2.7e+07 | 3 | -0.3667 | 0.3667 | no |
| 28 | P | P-11__to__P-15 | 1.6667 | 0.9861 | 4.7e+06 | 2 | 0.5500 | 0.8500 | yes |
| 31 | P | P-10__to__P-11 | 1.5714 | 0.9745 | 5.3e+06 | 4 | 0.3750 | 0.2232 | yes |
| 33 | M | M-2__to__M-7 | 1.5294 | 0.9306 | 2.7e+07 | 3 | 0.0333 | 0.3000 | yes |
| 34 | M | M-5__to__M-3 | 1.5000 | 0.8594 | 9.3959 | 2 | -0.3125 | 0.0312 | no |
| 42 | F | F-5__to__F-7 | 1.0000 | 0.8472 | 7.0e+06 | 5 | 0.2011 | 0.1250 | yes |
| 50 | F | F-8__to__F-7 | 0.1935 | 0.6750 | 7.0e+06 | 5 | 0.2255 | 0.0190 | yes |

## Strict Selected Pairs

| PAD rank | Prefix | Pair | PAD | Domain AUC | precheck L2 | val anom | d combined-source | d NAS-fixed | good |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | F | F-4__to__F-7 | 1.8095 | 1.0000 | 7.0e+06 | 5 | 0.0707 | 0.1440 | yes |
| 23 | M | M-2__to__M-5 | 1.7647 | 0.9028 | 1.2e+07 | 4 | 0.1111 | 0.1111 | yes |
| 31 | P | P-10__to__P-11 | 1.5714 | 0.9745 | 5.3e+06 | 4 | 0.3750 | 0.2232 | yes |
| 42 | F | F-5__to__F-7 | 1.0000 | 0.8472 | 7.0e+06 | 5 | 0.2011 | 0.1250 | yes |
| 50 | F | F-8__to__F-7 | 0.1935 | 0.6750 | 7.0e+06 | 5 | 0.2255 | 0.0190 | yes |
