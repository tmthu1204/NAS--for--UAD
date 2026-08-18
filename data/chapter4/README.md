# Chapter 4 benchmark data

This directory contains only the 28 preprocessed source-target pairs used in
Chapter 4 of the thesis:

- SMD: 12 pairs, 38 channels
- MSL: 6 pairs, 55 channels
- SMAP: 10 pairs, 25 channels

Each pair contains `train_normal.npz`, `target_pool_unlabeled.npz`,
`val_mixed.npz`, `test_mixed.npz`, and `split_metadata.json`. The windows have
length 128. Raw datasets, unused entities, ranking caches, and model outputs are
not included.

SMD is derived from the public Server Machine Dataset. MSL and SMAP are derived
from the public NASA anomaly-detection dataset. These files are included solely
to make the thesis benchmark reproducible; retain the original dataset
attribution when redistributing the repository.
