# Credit Card Fraud Detection with KNN + Streamlit

This repository contains a **K‑Nearest Neighbors (KNN)** model for detecting fraudulent credit‑card transactions, along with a **Streamlit** web app that serves the trained model.

The project uses the **Credit Card Fraud Detection** dataset from Kaggle and evaluates the model using **Accuracy, Precision, Recall, and F1‑Score**.  
The Streamlit app lets you upload transaction data and returns fraud predictions.

---

## 1. Dataset

**Kaggle:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  

- **Type:** Binary classification  
- **Target column:** `Class` (0 = normal, 1 = fraud)  
- **Notes:** The dataset is extremely imbalanced (frauds are very rare), so metrics like **Recall** and **F1‑Score** are more informative than Accuracy alone.

To (re)train the model yourself, download `creditcard.csv` from the Kaggle link above.

---

## 2. Repository Structure

Example structure (you can adjust to your repo):

```text
.
├── app.py                 # Streamlit frontend
├── requirements.txt       # Python dependencies
├── knn_model.pkl          # Trained KNN model (binary classifier)
├── scaler.pkl             # StandardScaler fitted on training data
├── feature_cols.pkl       # List of feature column names used for training
├── notebooks/
│   └── train_knn.ipynb    # (Optional) Notebook used to train/evaluate KNN
└── README.md
