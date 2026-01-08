# TS-TCC + AdaptNAS for Unsupervised Anomaly Detection (UAD)

This repository implements a **domain-adaptive neural architecture search (NAS) pipeline for time-series anomaly detection**, combining self-supervised representation learning, reliability-aware target weighting, and bilevel NAS.

---

## 🔧 Core Components

The pipeline integrates three key components:

* **TS-TCC** (Eldele et al., IJCAI 2021)
  Self-supervised representation learning for time-series, used as a universal feature pretraining stage.

* **AdaptNAS** (Xu et al., KDD 2023)
  Bilevel neural architecture search framework with domain-adversarial training.

* **Our contributions**:

  * Reliability-aware **unsupervised target weighting** using **DeepSVDD (soft-boundary)**
  * Integration of anomaly-aware weighting into AdaptNAS (weakly supervised adaptation)
  * Architecture search space tailored for **time-series UAD**
  * Empirical comparison between **hand-crafted baselines** and **NAS-selected architectures**

---

## 📂 Project Structure

```
project_root/
│
├── src/
│   ├── pipeline.py              # Main end-to-end pipeline
│   │
│   ├── ts_tcc/                  # TS-TCC (IJCAI'21)
│   │   ├── models/
│   │   ├── trainer/
│   │   └── dataloader/
│   │
│   ├── adaptnas/                # AdaptNAS (KDD'23-style)
│   │   ├── search_space.py      # Architecture search space
│   │   ├── trainer.py           # Bilevel training
│   │   └── optimizer.py
│   │
│   ├── models/
│   │   ├── tscnn.py             # CNN encoder
│   │   ├── transformer.py       # Transformer sequence model
│   │   ├── classifier.py        # MLP head
│   │   ├── discriminator.py     # Domain discriminator
│   │   └── deepsvdd.py          # DeepSVDD (soft-boundary)
│   │
│   └── utils/
│       └── metrics.py           # AUROC, AP, POT, event-F1, delay
│
├── data/                        # Preprocessed datasets (.npz)
├── outputs/
│   ├── checkpoints/             # Saved model checkpoints
│   ├── baselines/               # Per-architecture results
│   ├── baselines_summary.json
│   └── results.json
│
├── README.md
└── requirements.txt
```

---

## 📊 Supported Datasets

The pipeline is designed for **real-world time-series anomaly detection** (no synthetic data).

Currently tested on:

* **SMD (Server Machine Dataset)**
* **UCI HAR**
* **Sleep-EDF**

### Dataset Format

Each dataset is stored as `.npz`:

```python
X: np.ndarray of shape (N, T, C)
y: np.ndarray of shape (N,)  # optional for target
```

* `source.npz`: labeled source domain
* `target.npz`: target domain (labels only used for evaluation)

---

## 🚀 Running the Pipeline

### Non-UAD Mode (Domain-Adaptive UAD)

```bash
python -m src.pipeline \
  --dataset_or_paths data/smd/machine-1-1/source.npz,data/smd/machine-1-1/target.npz \
  --epochs_pretrain 50 \
  --search_candidates 10 \
  --batch_size 64 \
  --device cuda
```

---

## 🧠 Full Pipeline Overview

### 1️⃣ TS-TCC Pretraining

* Self-supervised training on **source + target (or multi-machine data)**.
* Produces robust temporal representations shared across domains.

---

### 2️⃣ Architecture Search Space

Defined in `search_space.py`, including:

* CNN encoder (depth, kernel, stride, dilation)
* Sequence model:

  * GRU
  * TCN
  * Transformer
* Classifier depth & width
* Latent dimension

This space supports both **hand-crafted baselines** and **NAS-selected architectures**.

---

### 3️⃣ Reliability-Aware Target Weighting (Key Contribution)

For each candidate architecture:

1. Warm up the model on **source labels**
2. Freeze the model
3. Extract `forward_features()` on source
4. Train **DeepSVDD (soft-boundary)** on source features
5. Score **all target samples**
6. Convert anomaly scores into **continuous reliability weights** via robust sigmoid

This transforms UAD into a **weakly supervised adaptation problem**.

---

### 4️⃣ AdaptNAS Bilevel Optimization

* **Lower-level**:
  Train model parameters using:

  * Labeled source
  * Weighted (reliable) target samples

* **Upper-level**:
  Update architecture parameters using:

  * Source validation
  * Target validation (labels used only for evaluation & selection)

Best architecture is selected by:

* **Primary**: AUROC on target validation
* **Secondary**: validation accuracy

---

### 5️⃣ Final Training & Evaluation

We run **final-only training** for:

* Three hand-crafted baselines
* The best NAS-selected architecture

All models use the **same weighting strategy** for fair comparison.

---

## 🧪 Baselines

Baselines are constructed **directly from the same search space**:

| Baseline     | Encoder | Sequence Model |
| ------------ | ------- | -------------- |
| Base_CNN_GRU | CNN     | GRU            |
| Base_CNN_TCN | CNN     | TCN            |
| Base_CNN_TRF | CNN     | Transformer    |

They represent the **three major families of temporal modeling** supported by the search space.

---

## 📈 Evaluation Metrics

We report **UAD-standard metrics**:

* AUROC
* Average Precision (AP)
* F1 (best & POT)
* Event-level F1
* Detection delay (mean / median)

Classification accuracy (`target_top1_acc`) is reported **only for completeness**, not as the primary metric.

---

## 📂 Outputs

* `outputs/baselines/*.json`
  → Per-architecture metrics

* `outputs/baselines_summary.json`
  → Best baseline & NAS comparison

* `outputs/results.json`
  → NAS search history & final results

---

## 📌 Key Takeaways

* NAS-selected architectures **consistently outperform** hand-crafted baselines
* Reliability-aware weighting is **crucial** for stable adaptation
* The method bridges **unsupervised anomaly detection** and **weakly supervised NAS**

---

## 🔗 Citation & Credit

* **TS-TCC**
  Eldele et al., *Time-Series Representation Learning via Temporal and Contextual Contrasting*, IJCAI 2021.
  [https://github.com/emadeldeen24/TS-TCC](https://github.com/emadeldeen24/TS-TCC)

* **AdaptNAS**
  Xu et al., *AdaptNAS: Domain-Adaptive Neural Architecture Search*, KDD 2023.

If you use this code, please cite the above works and our upcoming paper.

---

## ⚖️ License

* Our code: MIT License
* TS-TCC components: MIT License (retained from original repository)