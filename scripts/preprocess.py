"""Preprocess helpers: normalization and windowing for dataset creation.

This script expects an input npz or npy with key/array 'X' and optional 'y'.
It performs global z-score normalization and non-overlapping windowing.

Usage:
python scripts/preprocess.py --in_npz raw/input.npz --out_npz data/uci_har/source.npz --window 128
"""
import numpy as np
import argparse
import os

def zscore(x: np.ndarray, eps=1e-8):
    # normalize per-channel across all samples and time-steps
    mean = x.mean(axis=(0,1), keepdims=True)
    std = x.std(axis=(0,1), keepdims=True)
    return (x - mean) / (std + eps)

def windowing(x: np.ndarray, w: int):
    # x: (N, T, C)
    N, T, C = x.shape
    if T < w:
        raise ValueError(f"Series length T={T} < window {w}.")
    n_win = T // w
    x = x[:, : n_win * w, :]
    x = x.reshape(N * n_win, w, C)
    return x

def load_input(path: str):
    # support .npz (with arrays) or .npy raw array
    if path.endswith('.npz'):
        data = np.load(path)
        if 'X' in data:
            X = data['X']
            y = data['y'] if 'y' in data else None
            return X, y
        else:
            # try to load first array
            keys = list(data.keys())
            if len(keys) == 0:
                raise ValueError("Empty npz")
            X = data[keys[0]]
            y = data[keys[1]] if len(keys) > 1 else None
            return X, y
    elif path.endswith('.npy'):
        X = np.load(path)
        return X, None
    else:
        raise ValueError("Unsupported input format. Use .npz or .npy")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_npz', required=True, help='Input .npz or .npy containing X (and optional y)')
    parser.add_argument('--out_npz', required=True, help='Output .npz path')
    parser.add_argument('--window', type=int, default=128)
    args = parser.parse_args()

    X, y = load_input(args.in_npz)
    if X.ndim != 3:
        raise ValueError(f"Input X must be shape (N,T,C); got {X.shape}")

    Xn = zscore(X.astype(np.float32))
    Xw = windowing(Xn, args.window)

    out = {'X': Xw}
    if y is not None:
        # expand/collapse labels that align with windows
        # if original y shape is (N,) (per series), repeat per window
        if y.ndim == 1 and y.shape[0] == X.shape[0]:
            n_win = Xw.shape[0] // X.shape[0]
            yw = np.repeat(y, n_win)
            out['y'] = yw.astype(int)
        else:
            # try to window labels same as X
            y = np.array(y)
            if y.shape[1] == X.shape[1]:
                yw = (y[:, : (y.shape[1] // args.window) * args.window]
                      .reshape(-1, args.window).mean(axis=1) > 0.5)
                out['y'] = yw.astype(int)
            else:
                # fallback: drop labels
                print("Warning: cannot align labels to windows, omitting y in output.")
    os.makedirs(os.path.dirname(args.out_npz) or '.', exist_ok=True)
    np.savez(args.out_npz, **out)
    print('Saved', args.out_npz, 'X.shape=', out['X'].shape, 'y_included=', 'y' in out)
