# 🛡️ SENTINEL // MADE Anomaly Detector

> **Institutional-grade financial fraud detection engine powered by Masked Autoencoder Density Estimation (MADE) in PyTorch.**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38BDF8?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

---

## 📌 Overview

**SENTINEL-MADE** shifts fraud detection from brittle supervised binary classification to **unsupervised density estimation**. By learning the true joint probability distribution $p(\mathbf{x})$ of legitimate transactions, the system detects novel or unseen fraud tactics via explicit **Negative Log-Likelihood (NLL)** scoring.

---

## 🎯 Key Metrics & Performance

| Metric | Legitimate (Normal) | Fraudulent (Anomalous) |
| :--- | :--- | :--- |
| **Mean NLL Score** | `9.48` | `13,308.25` |
| **Decision Boundary** | `25.00` (95th %ile) | Exceeds boundary by up to **500x** |
| **Inference Speed** | `< 2 ms` per sample | Ideal for high-throughput payment gateways |

---

## 🛠️ Key Architectural Features

* **Strict Autoregressive Guarantee:** Uses custom `MaskedLinear` layers with lower-triangular weight masks to ensure $p(\mathbf{x}) = \prod p(x_i \mid \mathbf{x}_{<i})$.
* **Exact Likelihood Evaluation:** Directly outputs Gaussian parameters ($\mu, \sigma$) per continuous feature dimension.
* **Interactive Financial Telemetry UI:** Embedded single-page dark dashboard built with Tailwind CSS, Chart.js, and an IPython kernel bridge for real-time inference testing.
* **Feature-Level Surprise Breakdown:** Pinpoints exact vector dimensions driving anomaly scores for regulatory auditability.

---

## 🚀 Quickstart & Usage

### 1. Requirements & Setup
```bash
git clone [https://github.com/your-username/sentinel-made-fraud.git](https://github.com/your-username/sentinel-made-fraud.git)
cd sentinel-made-fraud
pip install torch pandas numpy matplotlib scikit-learn
