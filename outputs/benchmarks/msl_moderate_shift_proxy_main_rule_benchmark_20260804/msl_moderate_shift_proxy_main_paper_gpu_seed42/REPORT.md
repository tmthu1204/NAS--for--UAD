# msl_moderate_shift_proxy_main_paper_gpu_seed42 manifest benchmark

| Pair | PAD rank | PAD | Winner | Best fixed | Best NAS | uad_source (AP/AUROC/F1) | combined (AP/AUROC/F1) | best_NAS (AP/AUROC/F1) | best_fixed (AP/AUROC/F1) | d_combined-source AUROC | d_bestNAS-fixed AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M-2__to__M-5 | None |  | NAS_TopK_03 | Base_CNN_GRU | NAS_TopK_03 | 0.2241 / 0.2593 / 0.4000 | 0.2576 / 0.3704 / 0.4286 | 0.2576 / 0.3704 / 0.4286 | 0.2205 / 0.2593 / 0.4286 | 0.1111 | 0.1111 |
| M-1__to__M-5 | None |  | NAS_TopK_02 | Base_CNN_GRU | NAS_TopK_02 | 0.2219 / 0.2593 / 0.4286 | 0.2905 / 0.4259 / 0.4615 | 0.2905 / 0.4259 / 0.4615 | 0.2407 / 0.3519 / 0.4615 | 0.1667 | 0.0741 |
| T-12__to__T-4 | None |  | NAS_TopK_02 | Base_CNN_TCN | NAS_TopK_02 | 0.1224 / 0.1923 / 0.2667 | 0.1270 / 0.2308 / 0.2500 | 0.1270 / 0.2308 / 0.2500 | 0.1051 / 0.0385 / 0.2353 | 0.0385 | 0.1923 |
| F-5__to__F-7 | None |  | NAS_TopK_05 | Base_CNN_TCN | NAS_TopK_05 | 0.2593 / 0.4022 / 0.2727 | 0.3144 / 0.6033 / 0.4000 | 0.3144 / 0.6033 / 0.4000 | 0.2796 / 0.4783 / 0.3333 | 0.2011 | 0.1250 |
| T-12__to__T-13 | None |  | Base_CNN_TCN | Base_CNN_TCN | NAS_TopK_03 | 0.2784 / 0.4462 / 0.5000 | 0.2217 / 0.2462 / 0.4348 | 0.2205 / 0.2462 / 0.4545 | 0.2217 / 0.2462 / 0.4348 | -0.2000 | -0.0000 |
| T-13__to__T-4 | None |  | NAS_TopK_03 | Base_CNN_TRF | NAS_TopK_03 | 0.1333 / 0.2692 / 0.2857 | 0.1465 / 0.3462 / 0.3077 | 0.1465 / 0.3462 / 0.3077 | 0.1224 / 0.1923 / 0.2667 | 0.0769 | 0.1538 |

## Means

| Metric | uad_source | combined | best_NAS | best_fixed |
| --- | --- | --- | --- | --- |
| ap | 0.2065 | 0.2263 | 0.2261 | 0.1983 |
| auroc | 0.3047 | 0.3704 | 0.3704 | 0.2610 |
| f1_best | 0.3589 | 0.3804 | 0.3837 | 0.3600 |
| f1_pot | 0.1184 | 0.1172 | 0.1172 | 0.1656 |
| event_f1 | 0.1852 | 0.2143 | 0.2143 | 0.3846 |
| delay_mean | 1.7500 | 0.5000 | 0.5000 | 0.8333 |

## NAS Surrogate Alignment

| Pair | Best NAS | NAS eval count | corr(-upper_obj, val AUROC) | corr(-proxy_obj, val AUROC) | upper top1 hit | proxy top1 hit |
| --- | --- | --- | --- | --- | --- | --- |
| M-2__to__M-5 | NAS_TopK_03 | 5 | -0.5000 | 0.6000 | N | N |
| M-1__to__M-5 | NAS_TopK_02 | 5 | -0.1000 | 0.5000 | N | N |
| T-12__to__T-4 | NAS_TopK_02 | 5 | -0.1539 | 0.8208 | N | N |
| F-5__to__F-7 | NAS_TopK_05 | 5 | -0.4104 | 0.2052 | N | N |
| T-12__to__T-13 | NAS_TopK_03 | 5 | -0.3591 | 0.8208 | N | Y |
| T-13__to__T-4 | NAS_TopK_03 | 5 | -0.6669 | 0.9747 | N | N |
