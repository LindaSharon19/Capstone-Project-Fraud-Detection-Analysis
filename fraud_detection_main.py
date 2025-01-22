# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:10:42 2024

@author: PAVITHRA
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import dump

# Load the dataset from the provided Excel file path
file_path = r"C:\Users\PAVITHRA\Downloads\Fraud_Analysis_Dataset.xlsx"
df = pd.read_excel(file_path)

# Data preprocessing
# Convert 'type' column to numeric using LabelEncoder
label_encoder = LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])

# Select features for the model
selected_features = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
X = df[selected_features]
y = df['isFraud']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the KNN model
model = KNeighborsClassifier(n_neighbors=5)  # Default number of neighbors is 5
model.fit(X_scaled, y)

# Save the trained model to a file
dump(model, 'fraud_detection_knn_model.joblib')
dump(scaler, 'scaler.joblib')  # Saving the scaler