import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('src/xgb_model.pkl')
le_dict = joblib.load('src/label_encoders.pkl')

st.title("✈️ Flight Delay Risk Predictor")
st.write("Predict the likelihood of a flight delay based on route, airline, and schedule.")

# User inputs
airline = st.selectbox("Airline", options=le_dict['AIRLINE'].classes_)
origin = st.selectbox("Origin Airport", options=le_dict['ORIGIN_AIRPORT'].classes_)
destination = st.selectbox("Destination Airport", options=le_dict['DESTINATION_AIRPORT'].classes_)
month = st.slider("Month", 1, 12, 6)
day_of_week = st.slider("Day of Week (1=Mon, 7=Sun)", 1, 7, 3)
sched_departure = st.slider("Scheduled Departure (24hr, e.g. 1430 = 2:30pm)", 0, 2359, 900)
distance = st.number_input("Distance (miles)", min_value=50, max_value=5000, value=1000)

if st.button("Predict Delay Risk"):
    dep_hour = sched_departure // 100
    is_rush_hour = 1 if dep_hour in [7, 8, 17, 18, 19] else 0

    input_df = pd.DataFrame([{
        'MONTH': month,
        'DAY_OF_WEEK': day_of_week,
        'AIRLINE': le_dict['AIRLINE'].transform([airline])[0],
        'ORIGIN_AIRPORT': le_dict['ORIGIN_AIRPORT'].transform([origin])[0],
        'DESTINATION_AIRPORT': le_dict['DESTINATION_AIRPORT'].transform([destination])[0],
        'SCHEDULED_DEPARTURE': sched_departure,
        'DISTANCE': distance,
        'DEP_HOUR': dep_hour,
        'IS_RUSH_HOUR': is_rush_hour
    }])

    prob = model.predict_proba(input_df)[0][1]
    st.metric("Delay Risk", f"{prob*100:.1f}%")

    if prob > 0.3:
        st.warning("⚠️ Higher than average delay risk for this flight")
    else:
        st.success("✅ Lower than average delay risk for this flight")