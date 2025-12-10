import streamlit as st
import pandas as pd
import joblib

knn = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")

st.title("Credit Card Fraud Detection (KNN)")
st.write("Upload a CSV with the same columns as the Kaggle dataset (except 'Class').")

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    X = data[feature_cols]
    X_scaled = scaler.transform(X)

    preds = knn.predict(X_scaled)
    data["fraud_prediction"] = preds

    st.write("Preview of predictions:")
    st.write(data.head())

    st.write("Number of predicted frauds:", int((preds == 1).sum()))

    csv_out = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download results", csv_out, "predictions.csv", "text/csv")