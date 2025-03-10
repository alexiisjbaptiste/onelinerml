import streamlit as st
from .train import train
import pandas as pd

st.title("OneLinerML Model Monitoring Dashboard")
data_file = st.file_uploader("Upload CSV Data", type=["csv"])

if data_file is not None:
    target_column = st.text_input("Target Column", value="target")
    model_choice = st.selectbox("Select Model", ["linear_regression", "random_forest"])
    if st.button("Train Model"):
        try:
            data = pd.read_csv(data_file)
            model_instance, metrics = train(data, model=model_choice, target_column=target_column)
            st.write("Model Metrics:")
            st.json(metrics)
        except Exception as e:
            st.error(f"An error occurred: {e}")
