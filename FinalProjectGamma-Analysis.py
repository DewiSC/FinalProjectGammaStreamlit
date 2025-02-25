import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load Model yang sudah disimpan
model = pickle.load(open('Dataset/Model_final.pkl', 'rb'))

# Judul Aplikasi
st.title("Hotel Booking Cancellation Prediction")

# Form input untuk memasukkan fitur prediksi
st.sidebar.header("Masukkan Data Pemesanan:")

def user_input():
    lead_time = st.sidebar.slider("Lead Time (Hari)", 0, 500, 100)
    total_special_requests = st.sidebar.slider("Total Special Requests", 0, 5, 1)
    previous_cancellations = st.sidebar.slider("Previous Cancellations", 0, 10, 0)
    booking_changes = st.sidebar.slider("Booking Changes", 0, 10, 0)
    required_car_parking_spaces = st.sidebar.slider("Required Car Parking Spaces", 0, 5, 0)

    # Data dalam bentuk DataFrame
    data = {
        "lead_time": lead_time,
        "total_of_special_requests": total_special_requests,
        "previous_cancellations": previous_cancellations,
        "booking_changes": booking_changes,
        "required_car_parking_spaces": required_car_parking_spaces,
    }

    return pd.DataFrame([data])

# Ambil input dari user
input_df = user_input()

# Tampilkan input data
st.subheader("Data Pemesanan yang Dimasukkan:")
st.write(input_df)

# Prediksi dengan model
if st.button("Prediksi Pembatalan"):
    prediction = model.predict(input_df)

    # Interpretasi hasil
    if prediction[0] == 1:
        st.error("🚨 Booking kemungkinan besar akan DIBATALKAN!")
    else:
        st.success("✅ Booking kemungkinan besar akan DILANJUTKAN!")
