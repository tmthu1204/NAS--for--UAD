# TS-TCC + AdaptNAS: Domain-Adaptive NAS for Time-Series

This repository implements our **combined pipeline**:
- **TS-TCC** (Eldele et al., IJCAI'21) for self-supervised time-series representation.
- **AdaptNAS** (Xu et al., KDD'23) for architecture search with domain adaptation.
- **Our contribution**: pseudo-label transfer + integrated NAS training loop.

---

## 📂 Project Structure
```

project_root/
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── ts_tcc.py
│   ├── optimizers/
│   │   ├── __init__.py
│   │   └── adaptnas.py
│   ├── trainers/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py
│       └── metrics.py
│
├── scripts/
│   ├── run_pretrain.sh
│   ├── run_adaptnas.sh
│   └── export_figures.py
│
├── figures/          # Thư mục xuất hình cho báo cáo
│
├── tests/
│   ├── test_models.py
│   ├── test_optimizers.py
│   └── test_trainers.py
│
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
└── LICENSE


````

---

## 📊 Supported Datasets

We support **public, standard datasets** (no synthetic data):
- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)  
- [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)  
- [Epilepsy EEG Dataset](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)

### Preprocessing
Convert dataset into `.npz`:
```bash
python scripts/preprocess.py \
  --in_npz raw/uci_har.npz \
  --out_npz data/uci_har/source.npz \
  --window 128
````

For target domain (unlabeled), preprocess similarly but without labels:

```bash
python scripts/preprocess.py \
  --in_npz raw/sleepedf.npz \
  --out_npz data/sleepedf/target.npz \
  --window 128
```

---

## 🚀 Run Pipeline

```bash
bash scripts/run_pipeline.sh data/uci_har/source.npz data/sleepedf/target.npz
```

This runs:

1. TS-TCC pretraining (unsupervised on source + target combined).
2. Pseudo-label transfer (Logistic Regression on source embeddings).
3. Random search over AdaptNAS candidate architectures.
4. Training best architecture with domain-adversarial objective.
5. Save metrics + ROC curves in `outputs/`.

---

## 📈 Outputs

* `outputs/results.json` → NAS search history, best architecture
* `outputs/figures/` → ROC, search curves

---

## 🔗 Citation & Credit

* **TS-TCC**: Eldele et al. *"Time-Series Representation Learning via Temporal and Contextual Contrasting."* IJCAI 2021. [GitHub](https://github.com/emadeldeen24/TS-TCC)
* **AdaptNAS**: Xu et al. *"Learning Strong Graph Neural Networks with Weak Information."* KDD 2023.
* If you use this repository, please cite our upcoming work and also the original TS-TCC paper.

---

## ⚖️ License

Our code: MIT License
TS-TCC (src/ts_tcc): MIT License (retained from original repo)

```