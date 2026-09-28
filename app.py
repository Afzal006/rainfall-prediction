# ==========================================
# RAINFALL PREDICTION STREAMLIT APP
# ==========================================

import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧️",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("🌧️ Rainfall Prediction System")

st.write(
    "Enter the weather conditions below "
    "to predict whether rainfall will occur tomorrow."
)


# ==========================================
# LOAD MODEL
# ==========================================

try:

    model = joblib.load(
        "rainfall_model.pkl"
    )

    scaler = joblib.load(
        "rainfall_scaler.pkl"
    )

except FileNotFoundError:

    st.error(
        "Model files are not available. "
        "Please run train_models.py first."
    )

    st.stop()


# ==========================================
# USER INPUT
# ==========================================

st.subheader("Enter Weather Information")


min_temp = st.number_input(
    "Minimum Temperature (°C)",
    value=15.0
)

max_temp = st.number_input(
    "Maximum Temperature (°C)",
    value=25.0
)

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    value=0.0
)

evaporation = st.number_input(
    "Evaporation",
    min_value=0.0,
    value=5.0
)

sunshine = st.number_input(
    "Sunshine (hours)",
    min_value=0.0,
    value=7.0
)

wind_gust_speed = st.number_input(
    "Wind Gust Speed (km/h)",
    min_value=0.0,
    value=40.0
)

wind_9am = st.number_input(
    "Wind Speed at 9 AM (km/h)",
    min_value=0.0,
    value=15.0
)

wind_3pm = st.number_input(
    "Wind Speed at 3 PM (km/h)",
    min_value=0.0,
    value=20.0
)

humidity_9am = st.number_input(
    "Humidity at 9 AM (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

humidity_3pm = st.number_input(
    "Humidity at 3 PM (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

pressure_9am = st.number_input(
    "Pressure at 9 AM",
    value=1015.0
)

pressure_3pm = st.number_input(
    "Pressure at 3 PM",
    value=1010.0
)

cloud_9am = st.number_input(
    "Cloud 9 AM",
    min_value=0.0,
    max_value=8.0,
    value=4.0
)

cloud_3pm = st.number_input(
    "Cloud 3 PM",
    min_value=0.0,
    max_value=8.0,
    value=4.0
)

temp_9am = st.number_input(
    "Temperature at 9 AM (°C)",
    value=20.0
)

temp_3pm = st.number_input(
    "Temperature at 3 PM (°C)",
    value=24.0
)

rain_today = st.selectbox(
    "Did it rain today?",
    ["No", "Yes"]
)


# ==========================================
# PREDICTION
# ==========================================

if st.button("🌧️ Predict Rainfall"):

    # Convert RainToday
    rain_today_value = (
        1 if rain_today == "Yes" else 0
    )


    # ======================================
    # CREATE INPUT DATA
    # ======================================

    input_data = pd.DataFrame(
        [[
            min_temp,
            max_temp,
            rainfall,
            evaporation,
            sunshine,
            wind_gust_speed,
            wind_9am,
            wind_3pm,
            humidity_9am,
            humidity_3pm,
            pressure_9am,
            pressure_3pm,
            cloud_9am,
            cloud_3pm,
            temp_9am,
            temp_3pm,
            rain_today_value
        ]],

        columns=[
            "MinTemp",
            "MaxTemp",
            "Rainfall",
            "Evaporation",
            "Sunshine",
            "WindGustSpeed",
            "WindSpeed9am",
            "WindSpeed3pm",
            "Humidity9am",
            "Humidity3pm",
            "Pressure9am",
            "Pressure3pm",
            "Cloud9am",
            "Cloud3pm",
            "Temp9am",
            "Temp3pm",
            "RainToday"
        ]
    )


    # ======================================
    # SCALE INPUT
    # ======================================

    input_scaled = scaler.transform(
        input_data
    )


    # ======================================
    # MAKE PREDICTION
    # ======================================

    prediction = model.predict(
        input_scaled
    )[0]


    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "🌧️ Rain is likely tomorrow."
        )

    else:

        st.success(
            "☀️ Rain is not likely tomorrow."
        )


    # ======================================
    # PROBABILITY
    # ======================================

    if hasattr(
        model,
        "predict_proba"
    ):

        probability = model.predict_proba(
            input_scaled
        )[0]

        rain_probability = (
            probability[1] * 100
        )

        st.metric(
            "Rain Probability",
            f"{rain_probability:.2f}%"
        )

        st.progress(
            int(rain_probability)
        )


    # ======================================
    # SHOW INPUT DATA
    # ======================================

    with st.expander(
        "View Entered Weather Data"
    ):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Rainfall Prediction System | "
    "Machine Learning Project"
)