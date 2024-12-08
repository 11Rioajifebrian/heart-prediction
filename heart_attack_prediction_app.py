
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache
def load_data():
    return pd.read_csv("heart.csv")

# Application layout
st.title("Heart Attack Prediction App")
st.write("This app predicts the likelihood of a heart attack based on input parameters.")

# Load and display data
data = load_data()
st.write("Dataset Overview:")
st.dataframe(data.head())

# Splitting data
X = data.drop("output", axis=1)
y = data["output"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# User input for prediction
st.sidebar.header("Input Parameters")
def user_input_features():
    age = st.sidebar.slider("Age", int(data.age.min()), int(data.age.max()), int(data.age.mean()))
    sex = st.sidebar.selectbox("Sex (1=Male, 0=Female)", [0, 1])
    cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
    trtbps = st.sidebar.slider("Resting Blood Pressure", int(data.trtbps.min()), int(data.trtbps.max()), int(data.trtbps.mean()))
    chol = st.sidebar.slider("Cholesterol Level", int(data.chol.min()), int(data.chol.max()), int(data.chol.mean()))
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1=Yes, 0=No)", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalachh = st.sidebar.slider("Max Heart Rate Achieved", int(data.thalachh.min()), int(data.thalachh.max()), int(data.thalachh.mean()))
    exng = st.sidebar.selectbox("Exercise-Induced Angina (1=Yes, 0=No)", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression Induced", float(data.oldpeak.min()), float(data.oldpeak.max()), float(data.oldpeak.mean()))
    slp = st.sidebar.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
    caa = st.sidebar.slider("Major Vessels Colored (0-4)", 0, 4, 0)
    thall = st.sidebar.selectbox("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", [1, 2, 3])

    features = {
        "age": age, "sex": sex, "cp": cp, "trtbps": trtbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalachh": thalachh, "exng": exng,
        "oldpeak": oldpeak, "slp": slp, "caa": caa, "thall": thall
    }
    return pd.DataFrame(features, index=[0])

input_data = user_input_features()
st.write("User Input:")
st.write(input_data)

# Prediction
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

st.write("Prediction (1=Heart Attack, 0=No Heart Attack):", prediction[0])
st.write("Prediction Probability:", prediction_proba)
