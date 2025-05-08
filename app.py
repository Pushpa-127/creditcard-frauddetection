import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load data
st.title("Credit Card Fraud Detection using ML models")
st.write("This app uses machine learning models to predict credit card fraud.")

# Optionally upload a dataset
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    
    # Check if the 'Class' column exists
    if 'Class' not in df.columns:
        st.error("The dataset does not contain the required 'Class' column for prediction.")
    else:
        # Preprocess and train model
        # Handling missing values
        df = df.dropna()  # Optionally, fill missing values instead of dropping

        # Select features and target
        X = df.drop('Class', axis=1)  # Assuming 'Class' is the target column
        y = df['Class']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature Scaling (Optional, especially for algorithms sensitive to feature scales)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the model (example using RandomForest)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
