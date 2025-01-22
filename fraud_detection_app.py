# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:40:55 2024

@author: PAVITHRA
"""
import warnings
import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler

# Suppress specific Streamlit warnings (common in IDEs like Spyder)
warnings.filterwarnings("ignore", category=UserWarning, module="streamlit")

# Load the trained KNN model and scaler
model = load('fraud_detection_knn_model.joblib')
scaler = load('scaler.joblib')  # Load the scaler

# Create a Streamlit app
st.title("Fraud Detection Prediction App")

# Input fields for fraud detection
st.header("Enter Transaction Information")
step = st.number_input("Step", min_value=0, max_value=95, value=1)
transaction_type = st.selectbox("Transaction Type", ('TRANSFER', 'CASH_OUT', 'DEPOSIT', 'PAYMENT', 'OTHER'))
amount = st.number_input("Amount", min_value=0.0, value=100.0)
old_balance_org = st.number_input("Old Balance (Origin)", min_value=0.0, value=0.0)
new_balance_org = st.number_input("New Balance (Origin)", min_value=0.0, value=0.0)
old_balance_dest = st.number_input("Old Balance (Destination)", min_value=0.0, value=0.0)
new_balance_dest = st.number_input("New Balance (Destination)", min_value=0.0, value=0.0)

# Map input values to numeric using LabelEncoder
label_mapping_type = {
    'TRANSFER': 0,
    'CASH_OUT': 1,
    'DEPOSIT': 2,
    'PAYMENT': 3,
    'OTHER': 4,
}
transaction_type = label_mapping_type[transaction_type]

# Create a DataFrame with the entered input data
input_data = pd.DataFrame([[step, transaction_type, amount, old_balance_org, new_balance_org, old_balance_dest, new_balance_dest]], 
                          columns=['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])

# Feature scaling
input_data_scaled = scaler.transform(input_data)

# Make a prediction using the model
prediction = model.predict(input_data_scaled)

# Display the prediction result on the main screen
st.header("Prediction Result")
if prediction[0] == 0:
    st.success("This transaction is likely NOT fraudulent.")
else:
    st.error("This transaction is likely fraudulent.")