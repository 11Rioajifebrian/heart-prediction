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
    st.title("Heart Attack Prediction App 🫀")
    st.write("""
    **Aplikasi ini membantu memprediksi kemungkinan serangan jantung berdasarkan data medis.**
    Masukkan informasi Anda di formulir sebelah kiri, lalu klik tombol **Submit** untuk melihat hasilnya.
    """)

    # Load data
    data = load_data()

    # Sidebar form for user input
    st.sidebar.header("Input Data Anda")
    with st.sidebar.form("user_input_form"):
        age = st.slider("Umur (Tahun)", int(data.age.min()), int(data.age.max()), int(data.age.mean()))
        sex = st.selectbox("Jenis Kelamin", options=["Perempuan (0)", "Laki-laki (1)"], index=1)
        cp = st.selectbox("Jenis Nyeri Dada", ["0: Tidak Nyeri", "1: Nyeri Ringan", "2: Nyeri Sedang", "3: Nyeri Berat"], index=1)
        trtbps = st.slider("Tekanan Darah Istirahat (mm Hg)", int(data.trtbps.min()), int(data.trtbps.max()), int(data.trtbps.mean()))
        chol = st.slider("Kadar Kolesterol (mg/dl)", int(data.chol.min()), int(data.chol.max()), int(data.chol.mean()))
        fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl", ["Tidak (0)", "Ya (1)"], index=0)
        restecg = st.selectbox("Hasil ECG (Elektrokardiografi)", ["0: Normal", "1: Abnormal", "2: Hipertrofi"], index=0)
        thalachh = st.slider("Detak Jantung Maksimal", int(data.thalachh.min()), int(data.thalachh.max()), int(data.thalachh.mean()))
        exng = st.selectbox("Angina Induksi Olahraga", ["Tidak (0)", "Ya (1)"], index=0)
        oldpeak = st.slider("Depresi ST", float(data.oldpeak.min()), float(data.oldpeak.max()), float(data.oldpeak.mean()))
        slp = st.selectbox("Kemiringan ST", ["0: Turun", "1: Datar", "2: Naik"], index=1)
        caa = st.slider("Jumlah Pembuluh Utama (0-4)", 0, 4, 0)
        thall = st.selectbox("Thalassemia", ["1: Normal", "2: Cacat Tetap", "3: Cacat Reversibel"], index=2)

        # Submit button
        submitted = st.form_submit_button("Submit")

    if submitted:
        # Process user inputs
        input_data = pd.DataFrame({
            "age": [age],
            "sex": [1 if "Laki-laki" in sex else 0],
            "cp": [int(cp[0])],
            "trtbps": [trtbps],
            "chol": [chol],
            "fbs": [1 if "Ya" in fbs else 0],
            "restecg": [int(restecg[0])],
            "thalachh": [thalachh],
            "exng": [1 if "Ya" in exng else 0],
            "oldpeak": [oldpeak],
            "slp": [int(slp[0])],
            "caa": [caa],
            "thall": [int(thall[0])]
        })

        # Train and predict
        X = data.drop("output", axis=1)
        y = data["output"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Display results
        st.write("## Hasil Prediksi")
        if prediction[0] == 1:
            st.success("⚠️ Anda berisiko terkena serangan jantung.")
            st.write(f"**Probabilitas risiko**: {prediction_proba[0][1] * 100:.2f}%")
        else:
            st.success("✅ Anda tidak berisiko terkena serangan jantung.")
            st.write(f"**Probabilitas aman**: {prediction_proba[0][0] * 100:.2f}%")

        st.write("### Data yang Anda Masukkan:")
        st.table(input_data)

if __name__ == '__main__':
    main()
