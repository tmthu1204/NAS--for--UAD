# SMAP Explainable Paper-Safe Rule Search

This report searches simple rule families over the audited SMAP same-prefix buildable universe.
A pair is labeled `good` iff `combined.auroc > uad_source.auroc and best_NAS.auroc > best_fixed.auroc`.

- Audited SMAP pairs considered: 299
- Good pairs in the audited universe: 68
- Candidate rules searched: 26357

## Recommended Paper-Safe Profile

`buildable PAD rank <= 250, val_anomaly_count <= 4, val_anomaly_ratio >= 0.4000`

- Selected pairs: 10
- True positives: 8
- False positives: 2
- Precision: 0.8000
- Recall over good pairs: 0.1176
- F1 over good pairs: 0.2051
- Mean delta combined-source AUROC: 0.1550
- Mean delta NAS-fixed AUROC: 0.2529

## Balanced Profile

`source-vs-pool feature-mean L2 >= 12, val_count >= 12, val_anomaly_ratio <= 0.2500; keep at most 3 pair(s) per target, ordered by buildable PAD rank`

- Selected pairs: 32
- True positives: 17
- False positives: 15
- Precision: 0.5312
- Recall over good pairs: 0.2500
- F1 over good pairs: 0.3400
- Mean delta combined-source AUROC: 0.2418
- Mean delta NAS-fixed AUROC: 0.1288

## Strict Zero-FP Profile

`summary feature-mean L2 <= 1.5e+06, source-vs-pool feature-mean L2 >= 15, val_anomaly_ratio >= 0.2000; keep at most 2 pair(s) per target, ordered by buildable PAD rank`

- Selected pairs: 5
- True positives: 5
- False positives: 0
- Precision: 1.0000
- Recall over good pairs: 0.0735
- F1 over good pairs: 0.1370
- Mean delta combined-source AUROC: 0.4657
- Mean delta NAS-fixed AUROC: 0.3388

## Broad High-Recall Profile

`buildable PAD rank <= 275, source-vs-pool feature-mean L2 >= 7, val_count >= 12`

- Selected pairs: 85
- True positives: 34
- False positives: 51
- Precision: 0.4000
- Recall over good pairs: 0.5000
- F1 over good pairs: 0.4444
- Mean delta combined-source AUROC: 0.1154
- Mean delta NAS-fixed AUROC: 0.0512

## Why This Is Paper-Safe

- The rule uses only pre-benchmark pair descriptors: TS-JEPA/PAD rank, source-vs-pool feature shift, target-pool size, and validation-label density.
- It does not use test labels, test metrics, model winner identity, or entity-name exceptions as selection inputs.
- The final benchmark outcomes are used only to audit the selected profile and report TP/FP trade-offs.

## Pareto Frontier

| Profile | TP | FP | Precision | Recall | Complexity | Rule |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
|  | 68 | 204 | 0.2500 | 1.0000 | 2 | summary feature-mean L2 <= 2.0e+07, target_pool_count >= 11 |
|  | 67 | 200 | 0.2509 | 0.9853 | 3 | TS-JEPA domain_auc <= 1.0000, summary feature-mean L2 <= 2.0e+07, target_pool_count >= 11 |
|  | 66 | 183 | 0.2651 | 0.9706 | 3 | buildable PAD rank <= 275, summary feature-mean L2 <= 2.0e+07, target_pool_count >= 11 |
|  | 64 | 174 | 0.2689 | 0.9412 | 3 | buildable PAD rank <= 275, source-vs-pool feature-mean L2 >= 7, target_pool_count >= 11 |
|  | 63 | 161 | 0.2812 | 0.9265 | 3 | buildable PAD rank <= 250, summary feature-mean L2 <= 2.0e+07, target_pool_count >= 11 |
|  | 61 | 152 | 0.2864 | 0.8971 | 3 | buildable PAD rank <= 250, source-vs-pool feature-mean L2 >= 7, target_pool_count >= 11 |
|  | 58 | 148 | 0.2816 | 0.8529 | 3 | buildable PAD rank <= 275, source-vs-pool feature-mean L2 >= 7, target_pool_count >= 12 |
|  | 57 | 142 | 0.2864 | 0.8382 | 2 | buildable PAD rank <= 250, target_pool_count >= 12 |
|  | 55 | 126 | 0.3039 | 0.8088 | 3 | buildable PAD rank <= 250, source-vs-pool feature-mean L2 >= 7, target_pool_count >= 12 |
|  | 51 | 119 | 0.3000 | 0.7500 | 3 | TS-JEPA domain_auc >= 0.9000, source-vs-pool feature-mean L2 >= 7, target_pool_count >= 12 |
|  | 50 | 117 | 0.2994 | 0.7353 | 3 | buildable PAD rank <= 250, target_pool_count >= 12, val_anomaly_count <= 6 |
|  | 46 | 108 | 0.2987 | 0.6765 | 3 | buildable PAD rank <= 250, target_pool_count >= 12, val_anomaly_ratio <= 0.5000 |
|  | 43 | 102 | 0.2966 | 0.6324 | 3 | TS-JEPA domain_auc >= 0.9500, source-vs-pool feature-mean L2 >= 7, target_pool_count >= 12 |
|  | 42 | 95 | 0.3066 | 0.6176 | 3 | buildable PAD rank <= 275, source-vs-pool feature-mean L2 >= 8, target_pool_count >= 12 |
|  | 40 | 87 | 0.3150 | 0.5882 | 2 | buildable PAD rank <= 250, val_anomaly_count >= 3 |

## Paper-Safe Selected Pairs

| Rank | Pair | Good | PAD | Pool L2 | Val anomalies | d_combined-source AUROC | d_NAS-fixed AUROC |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 134 | E-5__to__E-9 | N | 1.8519 | 8.0820 | 4 | -0.0286 | -0.0357 |
| 173 | E-1__to__E-9 | Y | 1.7037 | 7.7534 | 4 | 0.1786 | 0.4714 |
| 175 | E-10__to__E-9 | Y | 1.7037 | 7.7178 | 4 | 0.0929 | 0.4071 |
| 177 | E-11__to__E-9 | Y | 1.7037 | 7.7166 | 4 | 0.0143 | 0.4714 |
| 192 | E-7__to__E-9 | Y | 1.6923 | 7.5895 | 4 | 0.0286 | 0.0357 |
| 204 | E-12__to__E-9 | Y | 1.5556 | 7.7160 | 4 | 0.0071 | 0.4429 |
| 206 | E-13__to__E-9 | Y | 1.5556 | 8.2817 | 4 | 0.6714 | 0.2000 |
| 222 | E-2__to__E-9 | Y | 1.4074 | 8.1060 | 4 | 0.4286 | 0.0071 |
| 223 | E-3__to__E-9 | N | 1.4074 | 7.9414 | 4 | -0.0357 | 0.0643 |
| 226 | E-6__to__E-9 | Y | 1.4074 | 7.9255 | 4 | 0.1929 | 0.4643 |

## Balanced Selected Pairs

| Rank | Pair | Good | PAD | Pool L2 | Val anomalies | d_combined-source AUROC | d_NAS-fixed AUROC |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 5 | A-2__to__A-1 | Y | 2.0000 | 93.1496 | 2 | 0.0339 | 0.0339 |
| 8 | A-3__to__A-1 | Y | 2.0000 | 93.2270 | 2 | 0.0339 | 0.0339 |
| 11 | A-4__to__A-1 | Y | 2.0000 | 93.3848 | 2 | 0.0339 | 1.0000 |
| 21 | A-7__to__A-4 | N | 2.0000 | 15.1283 | 1 | -0.0288 | -0.2115 |
| 25 | A-8__to__A-4 | Y | 2.0000 | 18.9828 | 1 | 0.5769 | 0.6731 |
| 28 | A-9__to__A-2 | N | 2.0000 | 18.0480 | 2 | 0.1900 | -0.0500 |
| 29 | A-9__to__A-4 | Y | 2.0000 | 18.9810 | 1 | 0.0673 | 0.0385 |
| 32 | D-1__to__D-11 | Y | 2.0000 | 18.3925 | 2 | 1.0000 | 0.0851 |
| 46 | D-2__to__D-8 | N | 2.0000 | 19.2271 | 2 | -0.0673 | 0.0673 |
| 47 | D-3__to__D-11 | N | 2.0000 | 19.4220 | 2 | 1.0000 | 0.0000 |
| 53 | D-5__to__D-8 | N | 2.0000 | 18.9322 | 2 | -0.0481 | 0.0481 |
| 54 | D-6__to__D-11 | Y | 2.0000 | 18.4589 | 2 | 1.0000 | 1.0000 |
| 57 | D-6__to__D-8 | N | 2.0000 | 18.4849 | 2 | 0.0000 | 0.0000 |
| 104 | G-4__to__G-3 | Y | 2.0000 | 1.7e+07 | 2 | 0.1000 | 0.6000 |
| 107 | G-6__to__G-3 | Y | 2.0000 | 1.7e+07 | 2 | 0.2545 | 0.6182 |
| 108 | G-7__to__G-1 | N | 2.0000 | 19.9499 | 2 | 0.0185 | -0.7963 |
| 109 | G-7__to__G-2 | N | 2.0000 | 18.3659 | 1 | 0.5700 | 0.0000 |
| 110 | G-7__to__G-3 | N | 2.0000 | 1.7e+07 | 2 | -0.2364 | 0.0727 |
| 120 | P-7__to__P-4 | Y | 2.0000 | 15.0546 | 5 | 0.1250 | 0.0016 |
| 121 | T-1__to__T-3 | N | 2.0000 | 24.3591 | 4 | -0.0323 | -0.0215 |
| 123 | T-2__to__T-3 | N | 2.0000 | 18.8925 | 4 | -0.0932 | -0.0251 |
| 139 | F-3__to__F-1 | N | 1.8519 | 19.5072 | 2 | -0.7660 | -0.0638 |
| 140 | G-6__to__G-1 | N | 1.8519 | 12.6909 | 2 | 0.2407 | -0.1852 |
| 142 | A-7__to__A-2 | Y | 1.8462 | 13.5650 | 2 | 0.2400 | 0.0200 |
| 143 | A-7__to__A-3 | N | 1.8462 | 13.4620 | 3 | 0.2788 | -0.1827 |
| 154 | G-1__to__G-7 | N | 1.8462 | 18.6191 | 3 | 0.1983 | -0.8764 |
| 167 | G-3__to__G-7 | Y | 1.8333 | 19.1325 | 3 | 0.7385 | 0.0057 |
| 170 | G-4__to__G-1 | Y | 1.8261 | 13.7634 | 2 | 0.4259 | 0.4722 |
| 199 | G-2__to__G-7 | Y | 1.6522 | 18.1098 | 3 | 0.2902 | 0.9511 |
| 230 | A-8__to__A-2 | Y | 1.3333 | 18.0488 | 2 | 0.2100 | 0.1000 |
| 231 | A-8__to__A-3 | Y | 1.3333 | 17.7913 | 3 | 0.6731 | 0.0288 |
| 233 | A-9__to__A-3 | Y | 1.3333 | 17.7885 | 3 | 0.7115 | 0.6827 |

## Strict Selected Pairs

| Rank | Pair | Good | PAD | Pool L2 | Val anomalies | d_combined-source AUROC | d_NAS-fixed AUROC |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 113 | P-1__to__P-7 | Y | 2.0000 | 18.5385 | 8 | 0.5833 | 0.5667 |
| 115 | P-3__to__P-7 | Y | 2.0000 | 18.5154 | 8 | 0.2357 | 0.4143 |
| 120 | P-7__to__P-4 | Y | 2.0000 | 15.0546 | 5 | 0.1250 | 0.0016 |
| 231 | A-8__to__A-3 | Y | 1.3333 | 17.7913 | 3 | 0.6731 | 0.0288 |
| 233 | A-9__to__A-3 | Y | 1.3333 | 17.7885 | 3 | 0.7115 | 0.6827 |
