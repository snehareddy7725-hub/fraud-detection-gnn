
## 📌 Project Overview

Built a **Graph Neural Network (GNN)** for fraud detection on 590K+ transactions. Transactions are modeled as a graph where nodes represent transactions and edges connect transactions sharing cards, devices, or email domains.

**Key Results:**
- ✅ **72.91% Accuracy** on 88K test transactions
- ✅ **61.31% Fraud Recall** (caught 61% of fraudulent transactions)
- ✅ **AUC-ROC: 0.731**
- ✅ **PR-AUC: 0.114** (vs random baseline of 0.035)

## 🏗️ Architecture

### Graph Construction
- **Nodes**: 590,540 transactions
- **Edges**: 1,313,184 connections
- **Connections based on**: Card ID, Device Info, Email Domain

### GNN Model
GCNConv(18 → 64) → ReLU → Dropout(0.5)
GCNConv(64 → 32) → ReLU → Dropout(0.5)
GCNConv(32 → 2) → LogSoftmax

### Key Techniques
- **Class weighting**: Fraud class gets 27.6× more weight
- **Early stopping**: Patience of 15 epochs
- **Threshold tuning**: Optimized F1 score

## 📊 Results

| Metric | Value |
|--------|-------|
| Accuracy | 72.91% |
| F1-Score | 0.138 |
| AUC-ROC | 0.731 |
| PR-AUC | 0.114 |
| Fraud Recall | 61.31% |
| Fraud Precision | 7.77% |

### Feature Importance
Top fraud indicators:
1. Transaction Amount
2. Recipient Email Domain
3. Card Information
4. Device Type
5. Address Information

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/fraud-detection-gnn.git
cd fraud-detection-gnn
pip install -r requirements.txt
