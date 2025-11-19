"""
Preprocess Sleep-EDF dataset into npz format for domain adaptation.
Output: data/sleepedf/source.npz, data/sleepedf/target.npz
Source = SC subjects, Target = ST subjects (ví dụ).
"""
import os
import numpy as np
import mne

RAW_DIR = "data/sleepedf_raw"
OUT_DIR = "data/sleepedf"
os.makedirs(OUT_DIR, exist_ok=True)

def load_subject(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    data = raw.get_data()  # (n_channels, n_times)
    return data.T  # (T, C)

def main():
    # Ví dụ giả sử: SC* là source, ST* là target
    sc_files = [f for f in os.listdir(RAW_DIR) if f.startswith("SC")]
    st_files = [f for f in os.listdir(RAW_DIR) if f.startswith("ST")]

    # Source
    Xs, Ys = [], []
    for f in sc_files:
        arr = load_subject(os.path.join(RAW_DIR, f))
        Xs.append(arr)
        Ys.append(np.zeros(len(arr)))  # TODO: thay bằng label thực
    Xs = np.concatenate(Xs); Ys = np.concatenate(Ys)

    # Target
    Xt, Yt = [], []
    for f in st_files:
        arr = load_subject(os.path.join(RAW_DIR, f))
        Xt.append(arr)
        Yt.append(np.zeros(len(arr)))  # TODO: thay bằng label thực
    Xt = np.concatenate(Xt); Yt = np.concatenate(Yt)

    np.savez(os.path.join(OUT_DIR, "source.npz"), X=Xs, y=Ys)
    np.savez(os.path.join(OUT_DIR, "target.npz"), X=Xt, y=Yt)
    print("Saved preprocessed Sleep-EDF to", OUT_DIR)

if __name__ == "__main__":
    main()
