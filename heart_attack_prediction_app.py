import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi untuk memuat dataset
@st.cache  # Cache data untuk mengurangi waktu pemuatan ulang
def load_data():
    return pd.read_csv("heart.csv")

# Fungsi utama untuk aplikasi Streamlit
def main():
    # Judul aplikasi dengan styling HTML untuk tampilan menarik
    st.markdown("""
    <div style="background-color: #FF6F61; padding: 10px; border-radius: 10px;">
        <h1 style="color: white; text-align: center;">Heart Attack Prediction App 🫀</h1>
    </div>
    """, unsafe_allow_html=True)

    # Penjelasan singkat tentang aplikasi
    st.write("""
    <p style="text-align: center; font-size: 18px;">
    Aplikasi ini membantu memprediksi kemungkinan serangan jantung berdasarkan data medis.
    Masukkan informasi Anda di formulir di bawah ini, lalu klik tombol <strong>Submit</strong> untuk melihat hasilnya.
    </p>
    """, unsafe_allow_html=True)

    # Memuat data
    data = load_data()

    # **Fitur baru 1**: Sidebar untuk pengaturan threshold prediksi
    st.sidebar.header("Pengaturan Threshold")  
    threshold = st.sidebar.slider("Threshold Risiko (%)", 0, 100, 50) / 100  
    # Fitur ini memungkinkan pengguna untuk mengatur ambang batas (threshold) risiko sesuai kebutuhan mereka.

    # Membuat formulir input data pengguna
    with st.container():
        # Styling untuk pusatkan form
        st.markdown("""
        <div style="display: flex; justify-content: center;">
            <div style="width: 60%; background-color: #F8F9F9; padding: 20px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
        """, unsafe_allow_html=True)

        # Form input data pengguna
        with st.form("user_input_form"):
            st.markdown("<h3 style='text-align: center;'>Masukkan Data Anda</h3>", unsafe_allow_html=True)
            # Input untuk data medis
            age = st.slider("Umur (Tahun)", 20, 90, 50)
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

            # Tombol untuk submit input data
            submitted = st.form_submit_button("Submit")

        # Penutup styling untuk form
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Logika prediksi
    if submitted:
        # Data input pengguna diubah ke dalam bentuk DataFrame
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

        # Melatih model dan melakukan prediksi
        X = data.drop("output", axis=1)  # Memisahkan fitur (X) dari target (y)
        y = data["output"]  # Target (output)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)  # Menggunakan Logistic Regression sebagai model
        model.fit(X_train, y_train)  # Melatih model
        prediction_proba = model.predict_proba(input_data)[0][1]  # Probabilitas prediksi untuk "berisiko"

        # Menampilkan hasil prediksi berdasarkan threshold
        st.markdown("<hr>", unsafe_allow_html=True)
        if prediction_proba >= threshold:
            st.error(f"⚠️ **Anda berisiko terkena serangan jantung!**\n\nProbabilitas risiko: {prediction_proba * 100:.2f}%")
        else:
            st.success(f"✅ **Anda tidak berisiko terkena serangan jantung!**\n\nProbabilitas aman: {(1 - prediction_proba) * 100:.2f}%")

        # **Fitur baru 2**: Visualisasi input data pengguna dibandingkan dengan dataset
        st.markdown("<br><h4>Visualisasi Data Anda:</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # Membandingkan distribusi umur pengguna dengan dataset
        sns.histplot(data['age'], kde=True, bins=20, ax=ax[0], color='blue', label='Dataset')
        ax[0].axvline(age, color='red', linestyle='--', label='Input Anda')
        ax[0].set_title("Distribusi Umur")
        ax[0].legend()

        # Membandingkan distribusi kolesterol pengguna dengan dataset
        sns.histplot(data['chol'], kde=True, bins=20, ax=ax[1], color='green', label='Dataset')
        ax[1].axvline(chol, color='red', linestyle='--', label='Input Anda')
        ax[1].set_title("Distribusi Kolesterol")
        ax[1].legend()

        # Menampilkan plot ke dalam Streamlit
        st.pyplot(fig)

# Menjalankan aplikasi
if __name__ == '__main__':
    main()
