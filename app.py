import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="UAV Power Prediction",
    page_icon="✈️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("/content/uav_power_model_new.pkl")

model = load_model()

st.title("✈️ UAV Power Prediction System")
st.markdown("### AI-Based UAV Power Estimation and Route Comparison")
st.write("Predict UAV power consumption using environmental and flight parameters.")

st.divider()

st.subheader("Enter UAV and Flight Parameters")

col1, col2 = st.columns(2)

with col1:
    wind_station = st.number_input("Wind Speed (Station)", value=1.0)
    wind_test = st.number_input("Wind Speed (Test)", value=1.0)
    pressure_station = st.number_input("Air Pressure (Station)", value=1013.0)
    pressure_test = st.number_input("Air Pressure (Test)", value=1013.0)
    air_density = st.number_input("Air Density", value=1.225)

with col2:
    temperature = st.number_input("Temperature", value=25.0)
    route = st.selectbox(
        "Route",
        ["FAVS_R1", "FAVS_R2", "FAFS", "VAVS_R1", "VAVS_R2", "Manal Random"]
    )
    uav = st.selectbox("UAV Type", ["UAV1", "UAV2", "UAV3"])
    weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rainy"])
    battery = st.selectbox("Battery Name", ["Battery1", "Battery2"])

st.divider()

if st.button("Predict Power Consumption", type="primary", use_container_width=True):

    input_data = pd.DataFrame([{
        "WindSpeed_station": wind_station,
        "WindSpeed_test": wind_test,
        "AirPressure_station": pressure_station,
        "AirPressure_test": pressure_test,
        "Air Density": air_density,
        "Temperature": temperature,
        "Route": route,
        "Uav": uav,
        "Weather": weather,
        "BatteryName": battery
    }])

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed!")
    st.metric("Predicted UAV Power Consumption", f"{prediction:,.2f}")

st.divider()

st.subheader("Route Comparison")

if st.button("Compare All Routes", use_container_width=True):

    routes = [
        "FAVS_R1", "FAVS_R2", "FAFS",
        "VAVS_R1", "VAVS_R2", "Manal Random"
    ]

    comparison = []

    for candidate_route in routes:
        input_data = pd.DataFrame([{
            "WindSpeed_station": wind_station,
            "WindSpeed_test": wind_test,
            "AirPressure_station": pressure_station,
            "AirPressure_test": pressure_test,
            "Air Density": air_density,
            "Temperature": temperature,
            "Route": candidate_route,
            "Uav": uav,
            "Weather": weather,
            "BatteryName": battery
        }])

        predicted_power = model.predict(input_data)[0]

        comparison.append({
            "Route": candidate_route,
            "Predicted Power": predicted_power
        })

    result = pd.DataFrame(comparison).sort_values("Predicted Power")

    st.dataframe(result, use_container_width=True, hide_index=True)
    st.bar_chart(result.set_index("Route"))

    st.info(
        "Lowest predicted power among the compared routes: "
        + str(result.iloc[0]["Route"])
    )

st.caption("Academic prototype | Random Forest Regression")
