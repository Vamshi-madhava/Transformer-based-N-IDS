# Transformer-Based Network Intrusion Detection System (NIDS)

> A Transformer model for detecting malicious network activity using the CICIDS 2017 dataset. Built from scratch in PyTorch and trained using mixed-precision on Google Colab.

---

##  Overview

This project implements a **Transformer architecture** for binary classification of network traffic as either **benign** or **attack**. We leverage the [CICIDS 2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html), a modern benchmark for evaluating NIDS models.

---

## Results

| Metric         | Value   |
|----------------|---------|
| Train Accuracy | 98.89%  |
| Test Accuracy  | 99.12%  |
| Test Loss      | 0.0305  |

The model converges rapidly and generalizes well to unseen test data.

---

## Tech Stack

- Python 3.11
- PyTorch
- Google Colab (GPU)
- `torch.cuda.amp` for mixed precision training
- Matplotlib & Seaborn for visualization
- Scikit-learn for metrics

---

## Model Architecture

- **Input**: 77 tabular features
- **Feature Embedding**: `Linear(1 → d_model)`
- **Transformer Encoder Layers**: 2
- **Heads**: 4
- **FFN**: 128 hidden units
- **Pooling**: Mean pooling across feature tokens
- **Output**: Binary classifier (Attack vs Benign)

---

## Dataset

We used a preprocessed `.npz` version of the CICIDS dataset containing:
- `X_train`, `y_train`
- `X_test`, `y_test`

Each sample is treated **independently** (non-sequential), suitable for per-packet intrusion detection.

---

## Training

- **Optimizer**: Adam (`lr=1e-4`)
- **Loss**: CrossEntropyLoss
- **Epochs**: 10
- **Batch size**: 512

---

## Visualizations

We provide:
- Accuracy/Loss curves over epochs
- Confusion Matrix

---

## Evaluation Example

```bash
Test Accuracy: 99.12%
Test F1 Score: 0.99

---

## Future Work

- Extend to multi-class classification (different attack types)
- Add positional encoding for time-based sequencing
- Visualize attention heads on tabular features
- Evaluate with imbalanced data settings

---

## ✨ Acknowledgements

- [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/)
- PyTorch Team
- Google Colab (for free compute ❤️)

