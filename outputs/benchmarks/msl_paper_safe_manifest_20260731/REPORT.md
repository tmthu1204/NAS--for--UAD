# msl_paper_safe_manifest_20260731 manifest benchmark

| Pair | PAD rank | PAD | Winner | Best fixed | Best NAS | uad_source (AP/AUROC/F1) | combined (AP/AUROC/F1) | best_NAS (AP/AUROC/F1) | best_fixed (AP/AUROC/F1) | d_combined-source AUROC | d_bestNAS-fixed AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F-4__to__F-7 | None |  | NAS_TopK_02 | Base_CNN_TCN | NAS_TopK_02 | 0.2719 / 0.4810 / 0.2727 | 0.1995 / 0.5516 / 0.3529 | 0.1995 / 0.5516 / 0.3529 | 0.2564 / 0.4076 / 0.2623 | 0.0707 | 0.1440 |
| M-2__to__M-5 | None |  | NAS_TopK_03 | Base_CNN_GRU | NAS_TopK_03 | 0.2241 / 0.2593 / 0.4000 | 0.2576 / 0.3704 / 0.4286 | 0.2576 / 0.3704 / 0.4286 | 0.2205 / 0.2593 / 0.4286 | 0.1111 | 0.1111 |
| M-2__to__M-3 | None |  | NAS_TopK_04 | Base_CNN_TCN | NAS_TopK_04 | 0.6875 / 0.6875 / 0.7500 | 0.8500 / 0.8125 / 0.8571 | 0.8500 / 0.8125 / 0.8571 | 0.8125 / 0.8438 / 0.7500 | 0.1250 | -0.0312 |
| P-11__to__P-15 | None |  | NAS_TopK_05 | Base_CNN_GRU | NAS_TopK_05 | 0.0714 / 0.3500 / 0.1333 | 0.3333 / 0.9000 / 0.5000 | 0.3333 / 0.9000 / 0.5000 | 0.0500 / 0.0500 / 0.0952 | 0.5500 | 0.8500 |
| P-10__to__P-11 | None |  | NAS_TopK_03 | Base_CNN_TCN | NAS_TopK_03 | 0.1009 / 0.2411 / 0.2222 | 0.3958 / 0.6161 / 0.4000 | 0.3958 / 0.6161 / 0.4000 | 0.1393 / 0.3929 / 0.2667 | 0.3750 | 0.2232 |
| F-5__to__F-7 | None |  | NAS_TopK_05 | Base_CNN_TCN | NAS_TopK_05 | 0.2593 / 0.4022 / 0.2727 | 0.3144 / 0.6033 / 0.4000 | 0.3144 / 0.6033 / 0.4000 | 0.2796 / 0.4783 / 0.3333 | 0.2011 | 0.1250 |
| F-8__to__F-7 | None |  | NAS_TopK_04 | Base_CNN_GRU | NAS_TopK_04 | 0.2631 / 0.4484 / 0.2857 | 0.5283 / 0.6739 / 0.5455 | 0.5283 / 0.6739 / 0.5455 | 0.5647 / 0.6549 / 0.6154 | 0.2255 | 0.0190 |

## Means

| Metric | uad_source | combined | best_NAS | best_fixed |
| --- | --- | --- | --- | --- |
| ap | 0.2683 | 0.4113 | 0.4113 | 0.3319 |
| auroc | 0.4099 | 0.6468 | 0.6468 | 0.4409 |
| f1_best | 0.3338 | 0.4977 | 0.4977 | 0.3931 |
| f1_pot | 0.1760 | 0.2586 | 0.2586 | 0.3127 |
| event_f1 | 0.4698 | 0.4626 | 0.4626 | 0.8154 |
| delay_mean | 1.9000 | 0.9167 | 0.9167 | 0.9286 |

## NAS Surrogate Alignment

| Pair | Best NAS | NAS eval count | corr(-upper_obj, val AUROC) | corr(-proxy_obj, val AUROC) | upper top1 hit | proxy top1 hit |
| --- | --- | --- | --- | --- | --- | --- |
| F-4__to__F-7 | NAS_TopK_02 | 5 | 0.2000 | 0.2000 | N | N |
| M-2__to__M-5 | NAS_TopK_03 | 5 | -0.5000 | 0.6000 | N | N |
| M-2__to__M-3 | NAS_TopK_04 | 5 | -0.1026 | -0.5643 | N | N |
| P-11__to__P-15 | NAS_TopK_05 | 5 | -0.3000 | -0.4000 | N | N |
| P-10__to__P-11 | NAS_TopK_03 | 5 | -0.5000 | 0.1000 | N | Y |
| F-5__to__F-7 | NAS_TopK_05 | 5 | -0.4104 | 0.2052 | N | N |
| F-8__to__F-7 | NAS_TopK_04 | 5 | -0.5000 | -0.3000 | N | N |
