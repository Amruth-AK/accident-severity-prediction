import streamlit as st
import matplotlib.pyplot as plt
import datetime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.predictor import Predictor

st.set_page_config(page_title="Traffic Accident Severity Predictor", layout="wide")


@st.cache_resource
def load_predictor():
    return Predictor()


predictor = load_predictor()


# ─── SIDEBAR: Input Panel ────────────────────────────────────────────────────

with st.sidebar:
    st.title("Accident Conditions")
    st.markdown("---")

    st.markdown("**Weather**")
    temperature = st.slider("Temperature (°F)", -20, 120, 65)
    humidity    = st.slider("Humidity (%)", 0, 100, 60)
    pressure    = st.slider("Pressure (in)", 27.0, 32.0, 29.9, step=0.1)
    visibility  = st.slider("Visibility (mi)", 0.0, 10.0, 10.0, step=0.5)
    wind_speed  = st.slider("Wind Speed (mph)", 0, 80, 10)
    weather     = st.selectbox(
        "Weather Condition",
        ["Clear", "Light Rain", "Rain", "Heavy Rain", "Snow", "Fog",
         "Thunderstorm", "Overcast", "Partly Cloudy"],
    )
    wind_dir = st.selectbox("Wind Direction", ["CALM", "N", "NE", "E", "SE", "S", "SW", "W", "NW", "VAR"])

    st.markdown("---")
    st.markdown("**Accident Details**")
    distance = st.number_input("Distance Affected (mi)", min_value=0.0, value=0.5, step=0.1)

    st.markdown("**Time**")
    accident_date = st.date_input("Date", value=datetime.date.today())
    accident_time = st.time_input("Time", value=datetime.time(8, 0))

    st.markdown("---")
    st.markdown("**Road Features**")
    junction       = st.checkbox("Junction")
    crossing       = st.checkbox("Crossing")
    stop           = st.checkbox("Stop Sign")
    traffic_signal = st.checkbox("Traffic Signal")
    amenity        = st.checkbox("Amenity Nearby")

    st.markdown("---")
    st.markdown("**Location**")
    start_lat = st.number_input("Latitude",  value=37.0, format="%.4f")
    start_lng = st.number_input("Longitude", value=-95.0, format="%.4f")

    predict_btn = st.button("Predict Severity", type="primary", use_container_width=True)


# ─── MAIN PANEL ──────────────────────────────────────────────────────────────

st.title("US Traffic Accident Severity Predictor")
st.caption("Model: LightGBM + threshold tuning · Trained on 7.7M US accident records")

if predict_btn:
    start_dt = datetime.datetime.combine(accident_date, accident_time)
    end_dt   = start_dt + datetime.timedelta(minutes=30)

    civil_twilight = (
        "Night" if accident_time.hour < 6 or accident_time.hour >= 20 else "Day"
    )

    input_dict = {
        "Start_Time":        str(start_dt),
        "End_Time":          str(end_dt),
        "Start_Lat":         start_lat,
        "Start_Lng":         start_lng,
        "Temperature(F)":    temperature,
        "Humidity(%)":       humidity,
        "Pressure(in)":      pressure,
        "Visibility(mi)":    visibility,
        "Wind_Speed(mph)":   wind_speed,
        "Weather_Condition": weather,
        "Wind_Direction":    wind_dir,
        "Distance(mi)":      distance,
        "Civil_Twilight":    civil_twilight,
        "Amenity":           int(amenity),
        "Crossing":          int(crossing),
        "Give_Way":          0,
        "Junction":          int(junction),
        "No_Exit":           0,
        "Railway":           0,
        "Stop":              int(stop),
        "Traffic_Calming":   0,
        "Traffic_Signal":    int(traffic_signal),
    }

    with st.spinner("Running prediction..."):
        result = predictor.predict(input_dict)

    severity = result["severity"]
    color    = result["color"]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            f'<div style="background-color:{color}; padding:40px; border-radius:16px; '
            f'text-align:center; font-size:56px; font-weight:bold; color:white; '
            f'box-shadow: 0 4px 12px rgba(0,0,0,0.2);">'
            f'Severity {severity}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"<p style='margin-top:12px;'>{result['description']}</p>", unsafe_allow_html=True)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3))
        classes    = [f"Class {i}" for i in range(1, 5)]
        probs      = [result["probabilities"][i] for i in range(1, 5)]
        bar_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
        bars = ax.barh(classes, probs, color=bar_colors, edgecolor="white", height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Predicted Class Probabilities")
        ax.spines[["top", "right"]].set_visible(False)
        for bar, v in zip(bars, probs):
            ax.text(min(v + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                    f"{v:.1%}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

else:
    st.info("Set the accident conditions in the sidebar and click **Predict Severity**.")


# ─── ABOUT ───────────────────────────────────────────────────────────────────

with st.expander("About this project"):
    st.markdown("""
    Traffic accidents are one of the leading causes of injury and death on US roads.
    Being able to predict how severe an accident is, before emergency services arrive,
    can help prioritize response and allocate resources where they matter most.

    This project trains machine learning models on 7.7 million real US accident records
    to predict severity on a scale of 1 (minor) to 4 (critical), based on weather
    conditions, road features, and time of day. The best model is LightGBM, tuned with
    Optuna and combined with per-class threshold adjustment to better detect the rarest
    and most dangerous cases.

    [GitHub →](https://github.com/Amruth-AK/accident-severity-prediction)
    """)
