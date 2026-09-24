# ML Lifecycle App - Developed by Siva Sarumiya

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="ML Lifecycle App - Siva Sarumiya")

st.title("ML Lifecycle App")
st.markdown("**Developed by Siva Sarumiya**")
st.markdown("---")

file = st.file_uploader("Upload your CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    st.subheader("1. Data Preview")
    st.dataframe(df.head())
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    st.subheader("2. Train Model")
    if st.button("Train"):
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)

        st.success("Model Trained!")
        st.metric("Accuracy", f"{acc*100:.2f}%")
        st.balloons()
else:
    st.info("Please upload a CSV file")

st.markdown("---")
st.write("© 2026 Developed by Siva Sarumiya")