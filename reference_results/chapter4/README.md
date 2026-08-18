# Chapter 4 reference results

`aggregate_metrics.csv` and `pair_auroc.csv` contain the values reported in
Chapter 4. Full runs write fresh reports to `outputs/chapter4/`.

The experiment uses seed 42. Small numerical differences can still occur across
GPU models, CUDA/cuDNN versions, and operating systems. `--quick` output must not
be compared with these values because quick mode intentionally uses one pair and
one-epoch settings.

Pair-level values are rounded to four decimal places as in the thesis tables.
Aggregate values were computed from the original unrounded metrics, so averaging
the displayed pair values can differ by 0.0001.
