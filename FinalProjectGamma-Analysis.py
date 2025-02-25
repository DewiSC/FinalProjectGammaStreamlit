import streamlit as st
import pickle
import pandas as pd

# Load Model yang sudah disimpan
model = pickle.load(open('Dataset/Model_final.pkl', 'rb'))

# Mapping kategori ke angka (harus sesuai dengan training model)
meal_mapping = {"BB": 0, "HB": 1, "FB": 2, "SC": 3}
market_segment_mapping = {"Direct": 0, "Corporate": 1, "Online TA": 2, "Offline TA/TO": 3}
distribution_channel_mapping = {"Direct": 0, "TA/TO": 1, "Corporate": 2, "GDS": 3}
room_type_mapping = {"Standard": 0, "Superior": 1, "Deluxe": 2, "Suite": 3}
deposit_type_mapping = {"No Deposit": 0, "Refundable": 1, "Non Refundable": 2}
customer_type_mapping = {"Transient": 0, "Contract": 1, "Transient-Party": 2, "Group": 3}
stay_category_mapping = {"Short Stay": 0, "Medium Stay": 1, "Long Stay": 2}

# Judul Aplikasi
st.title("Hotel Booking Cancellation Prediction")

# Form input untuk memasukkan fitur prediksi
st.sidebar.header("Masukkan Data Pemesanan:")

def user_input():
    lead_time = st.sidebar.slider("Lead Time (Hari)", 0, 365, 100)
    stays_in_week_nights = st.sidebar.slider("Stays in Week Nights", 0, 14, 2)
    adults = st.sidebar.slider("Adults", 1, 5, 2)
    previous_cancellations = st.sidebar.slider("Previous Cancellations", 0, 10, 0)
    previous_bookings_not_canceled = st.sidebar.slider("Previous Bookings Not Canceled", 0, 10, 0)
    booking_changes = st.sidebar.slider("Booking Changes", 0, 10, 0)
    required_car_parking_spaces = st.sidebar.slider("Required Car Parking Spaces", 0, 5, 0)
    total_special_requests = st.sidebar.slider("Total Special Requests", 0, 5, 1)
    length_of_stay = st.sidebar.slider("Length of Stay", 1, 30, 3)

    # Input data kategorikal
    meal = st.sidebar.selectbox("Meal Type", list(meal_mapping.keys()))
    market_segment = st.sidebar.selectbox("Market Segment", list(market_segment_mapping.keys()))
    distribution_channel = st.sidebar.selectbox("Distribution Channel", list(distribution_channel_mapping.keys()))
    reserved_room_type = st.sidebar.selectbox("Reserved Room Type", list(room_type_mapping.keys()))
    assigned_room_type = st.sidebar.selectbox("Assigned Room Type", list(room_type_mapping.keys()))
    deposit_type = st.sidebar.selectbox("Deposit Type", list(deposit_type_mapping.keys()))
    customer_type = st.sidebar.selectbox("Customer Type", list(customer_type_mapping.keys()))
    stay_category = st.sidebar.selectbox("Stay Category", list(stay_category_mapping.keys()))
    is_repeated_guest = st.sidebar.radio("Is Repeated Guest?", [0, 1])  # Binary input

    # Konversi kategori ke numerik
    meal_encoded = meal_mapping[meal]
    market_segment_encoded = market_segment_mapping[market_segment]
    distribution_channel_encoded = distribution_channel_mapping[distribution_channel]
    reserved_room_type_encoded = room_type_mapping[reserved_room_type]
    assigned_room_type_encoded = room_type_mapping[assigned_room_type]
    deposit_type_encoded = deposit_type_mapping[deposit_type]
    customer_type_encoded = customer_type_mapping[customer_type]
    stay_category_encoded = stay_category_mapping[stay_category]

    # Data dalam bentuk DataFrame
    data = {
        "lead_time": lead_time,
        "stays_in_week_nights": stays_in_week_nights,
        "adults": adults,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "booking_changes": booking_changes,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_special_requests,
        "length_of_stay": length_of_stay,
        "meal": meal_encoded,
        "market_segment": market_segment_encoded,
        "distribution_channel": distribution_channel_encoded,
        "reserved_room_type": reserved_room_type_encoded,
        "assigned_room_type": assigned_room_type_encoded,
        "deposit_type": deposit_type_encoded,
        "customer_type": customer_type_encoded,
        "stay_category": stay_category_encoded,
        "is_repeated_guest": is_repeated_guest
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
