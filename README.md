# Credit Card Fraud Detection

A comparative study of machine learning approaches for detecting fraudulent credit card transactions on a heavily imbalanced dataset.

---

## Problem Statement

Credit card fraud detection is a classic imbalanced classification problem — fraudulent transactions represent only **0.17%** of all transactions. Standard accuracy metrics are misleading in this setting, making model selection and evaluation non-trivial.

---

## Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud cases:** 492 (0.17%)
- **Features:** 28 PCA-transformed features (V1–V28) + Amount + Time
- **Target:** Class (0 = valid, 1 = fraud)

---

## Approach

Given the severe class imbalance, multiple modelling strategies were explored and compared to identify the most effective approach for minimizing missed fraud cases while maintaining acceptable false alarm rates in a real-world deployment setting :
1. **Deep Neural Network (DNN)** — baseline deep learning approach
2. **Random Forest** — ensemble method, no resampling
3. **Decision Tree** — interpretable baseline
4. **Undersampling** — reduce majority class to balance dataset
5. **SMOTE** — synthetic oversampling of minority class

Each model was evaluated using **Precision, Recall, and F1-Score on the fraud class**, as well as confusion matrices. Accuracy was deliberately avoided as the primary metric due to class imbalance.

---

## Results

| Model | Precision | Recall | F1-Score |
|---|---|---|---|
| Deep Neural Network | 84.0% | 81.5% | 82.7% |
| Decision Tree | 94.0% | 92.5% | 93.2% |
| **Random Forest** | **98.5%** | **93.1%** | **95.7%** |
| Undersampling | 49.6% | 88.8% | 63.6% |

> **Random Forest without resampling achieved the best overall performance**, balancing high precision (fewer false alarms) with strong recall (catching most fraud cases).

> **Undersampling** achieved high recall but at the cost of precision — nearly half of flagged transactions were legitimate, making it impractical for real-world deployment where false alarms have a cost.

> **SMOTE** was also tested but produced a degraded confusion matrix and was excluded from final comparison.

---

## Key Findings

- Resampling techniques (undersampling, SMOTE) did not improve performance on this dataset — the Random Forest handled imbalance implicitly through ensemble averaging
- Precision matters as much as recall in fraud detection — a model that flags too many legitimate transactions creates operational overhead
- DNN underperformed tree-based methods, likely due to the tabular nature of PCA-transformed features where ensemble methods have a natural advantage

---

## Tech Stack

- Python, NumPy, Pandas
- Scikit-learn (Random Forest, Decision Tree, evaluation metrics)
- TensorFlow/Keras (DNN)
- imbalanced-learn (SMOTE, undersampling)
- Matplotlib, Seaborn (EDA, confusion matrices)

---

## Project Structure

```
├── creditcard.csv          # Dataset (download from Kaggle)
├── fraud_detection.ipynb   # Main notebook with all models
└── README.md
```

---

## How to Run

```bash
# Install dependencies
pip install numpy pandas scikit-learn imbalanced-learn tensorflow matplotlib seaborn

# Download dataset from Kaggle and place creditcard.csv in root directory
# Open and run fraud_detection.ipynb top to bottom
```

---
