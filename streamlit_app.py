import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Load models safely
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model1 = load_model("model1.pkl")
model2 = load_model("model2.pkl")

# Streamlit app title
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>🌍 Global Climate Change Impact Predictor </h2>", unsafe_allow_html=True)

page = st.sidebar.selectbox("📌 Navigation ", ["Home", "Dataset", "About"])

if page == "Home":
    st.header("Input Features")
    feature1 = st.number_input("Temperature", value=0.0)
    feature2 = st.number_input("CO2 Emissions", value=0.0)
    feature3 = st.number_input("Sea Level Rise", value=0.0)
    feature4 = st.number_input("Precipitation", value=0.0)
    feature5 = st.number_input("Humidity", value=0.0)
    feature6 = st.number_input("Wind Speed", value=0.0)

    features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6]])

    if st.button("Predict"):
        if model1 and model2:
            prediction1 = model1.predict(features)[0]
            prediction2 = model2.predict(features)[0]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Climate Risk Index")
                st.write(f"### {prediction1}")

            with col2:
                st.subheader("Weather Severity Index")
                st.write(f"### {prediction2}")

            # Classification logic
            risk_message = "🔴 **Severe Risk:** Major instability, increasing severe weather events."
            if prediction1 <= 100:
                risk_message = "🟢 **Low Risk:** Climate impact is minimal."
            elif prediction1 <= 150:
                risk_message = "🟡 **Moderate Risk:** Some climate changes, mild disruptions."
            elif prediction1 <= 200:
                risk_message = "🟠 **High Risk:** Significant environmental shifts."

            severity_message = "🔴 **Very Severe:** High risk of flooding and damaging storms."
            if prediction2 <= 25:
                severity_message = "🟢 **Mild:** Calm weather, no significant risks."
            elif prediction2 <= 50:
                severity_message = "🟡 **Moderate:** Occasional storms, manageable conditions."
            elif prediction2 <= 75:
                severity_message = "🟠 **Severe:** Frequent storms, strong winds, possible disruptions."

            st.write(risk_message)
            st.write(severity_message)
        else:
            st.error("Model not loaded properly. Please check your model files.")

elif page == "Dataset":
    st.header("Climate Insights Dataset")

    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("climate_change_data.csv")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None

    df = load_data()
    if df is not None:
        st.write("### Dataset Preview")
        st.dataframe(df)

elif page == "About":
    st.header("ℹ️ About This App")
    st.write("""
    ### 🌍 Climate & Weather Risk Dashboard
    This interactive dashboard helps analyze climate and weather risks in real-time.
    
    - **Climate Risk Index**: Evaluates overall climate impact.
    - **Weather Severity Index**: Assesses the severity of weather conditions.
    
    🚀 **Future Goals**: Improve accuracy, add more indices, and include location-based insights.
    📞 **Contact**: contact@climateweather.com
    """)
