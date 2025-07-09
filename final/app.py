
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Analisis & Prediksi Penempatan Kerja", layout="wide")
st.title("🎓 Aplikasi Analisis & Prediksi Penempatan Kerja Mahasiswa")

# Load model
with open('rf_model.pkl') as file: model = pickle.load(file)
# Menu navigasi
menu = st.sidebar.selectbox("Pilih Halaman", ["📊 EDA", "🔍 Prediksi"])

# ------------------------------#
# 📊 HALAMAN EDA
# ------------------------------#
if menu == "📊 EDA":
    st.header("📊 Exploratory Data Analysis (EDA)")

    @st.cache_data
    def load_data():
        return pd.read_csv("Placement_Data_Full_Class.csv")

    df = load_data()

    st.subheader("📌 1. Preview Data")
    st.dataframe(df.head())

    st.subheader("📌 2. Statistik Deskriptif")
    st.write(df.describe(include='all'))

    st.subheader("📌 3. Distribusi Status Penempatan")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="status", ax=ax1)
    st.pyplot(fig1)

    st.subheader("📌 4. Gender vs Status Penempatan")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=df, x="gender", hue="status", ax=ax2)
    st.pyplot(fig2)

    st.subheader("📌 5. Korelasi Fitur Numerik")
    df_num = df.select_dtypes(include=["int64", "float64"])
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_num.corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

# ------------------------------#
# 🔍 HALAMAN PREDIKSI
# ------------------------------#
elif menu == "🔍 Prediksi":
    st.header("🔍 Prediksi Penempatan Kerja")

    st.markdown("Masukkan data berikut untuk memprediksi apakah mahasiswa akan **ditempatkan kerja (Placed)** atau tidak.")

    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    ssc_p = st.slider("Nilai SSC (Sekolah Menengah Pertama) [%]", 40, 100)
    hsc_p = st.slider("Nilai HSC (Sekolah Menengah Atas) [%]", 40, 100)
    hsc_s = st.selectbox("Jurusan SMA", ["Arts", "Commerce", "Science"])
    degree_p = st.slider("Nilai Sarjana (Degree) [%]", 40, 100)
    degree_t = st.selectbox("Bidang Studi", ["Sci&Tech", "Comm&Mgmt", "Others"])
    workex = st.selectbox("Pengalaman Kerja Sebelumnya", ["Yes", "No"])
    etest_p = st.slider("Nilai Tes Kecakapan (E-test) [%]", 0, 100)
    mba_p = st.slider("Nilai MBA [%]", 40, 100)
    specialisation = st.selectbox("Spesialisasi MBA", ["Mkt&HR", "Mkt&Fin"])

    # Manual Encoding (mengikuti hasil training)
    gender = 1 if gender == "Male" else 0
    hsc_s = {"Arts": 0, "Commerce": 1, "Science": 2}[hsc_s]
    degree_t = {"Comm&Mgmt": 0, "Others": 1, "Sci&Tech": 2}[degree_t]
    workex = 1 if workex == "Yes" else 0
    specialisation = 0 if specialisation == "Mkt&Fin" else 1

    input_data = np.array([[gender, ssc_p, hsc_p, hsc_s, degree_p, degree_t,
                            workex, etest_p, mba_p, specialisation]])

    if st.button("🔮 Prediksi Sekarang"):
        result = model.best_estimator_.predict(input_data)
        if result[0] == 1:
            st.success("✅ Mahasiswa kemungkinan BESAR akan **DITEMPATKAN kerja**.")
        else:
            st.error("⚠️ Mahasiswa kemungkinan **TIDAK ditempatkan kerja**.")

st.markdown("---")
st.caption("Final Project Data Science — By [Nama Kamu]")
