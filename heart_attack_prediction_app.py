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

# Main function for Streamlit app
def main():
    # App title
    st.title("Heart Attack Prediction App")
    st.write("""
    This app predicts the likelihood of a heart attack based on input parameters.
    """)
    
    # Load data
    data = load_data()
    st.write("### Dataset Overview:")
    st.dataframe(data.head())

    # Prepare data for model training
    X = data.drop("output", axis=1)
    y = data["output"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Display model accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

    # Form for user input
    st.write("### Input Features")
    with st.form("user_input_form"):
        age = st.slider("Age", int(data.age.min()), int(data.age.max()), int(data.age.mean()))
        sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
        cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
        trtbps = st.slider("Resting Blood Pressure", int(data.trtbps.min()), int(data.trtbps.max()), int(data.trtbps.mean()))
        chol = st.slider("Cholesterol Level", int(data.chol.min()), int(data.chol.max()), int(data.chol.mean()))
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=Yes, 0=No)", [0, 1])
        restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
        thalachh = st.slider("Max Heart Rate Achieved", int(data.thalachh.min()), int(data.thalachh.max()), int(data.thalachh.mean()))
        exng = st.selectbox("Exercise-Induced Angina (1=Yes, 0=No)", [0, 1])
        oldpeak = st.slider("ST Depression Induced", float(data.oldpeak.min()), float(data.oldpeak.max()), float(data.oldpeak.mean()))
        slp = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
        caa = st.slider("Major Vessels Colored (0-4)", 0, 4, 0)
        thall = st.selectbox("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", [1, 2, 3])

        # Submit button
        submitted = st.form_submit_button("Submit")

    # Prediction after form submission
    if submitted:
        input_data = pd.DataFrame({
            "age": [age], "sex": [sex], "cp": [cp], "trtbps": [trtbps], "chol": [chol],
            "fbs": [fbs], "restecg": [restecg], "thalachh": [thalachh], "exng": [exng],
            "oldpeak": [oldpeak], "slp": [slp], "caa": [caa], "thall": [thall]
        })
        st.write("### User Input:")
        st.write(input_data)

        # Prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.write("### Prediction (1=Heart Attack, 0=No Heart Attack):", prediction[0])
        st.write("### Prediction Probability:", prediction_proba)

# Run the app
if __name__ == '__main__':
    main()
