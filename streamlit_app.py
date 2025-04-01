import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load models
with open("/mount/src/climate-change-impact-score/model1.pkl", "rb") as file1:
            model1 = pickle.load(file1)
with open("/mount/src/climate-change-impact-score/model2.pkl", "rb") as file2:
            model2 = pickle.load(file2)

					
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

    # Convert input to array
    features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6]])

    # Predict button
    if st.button("Predict"):
        prediction1 = model1.predict(features)[0]
        prediction2 = model2.predict(features)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Climate Risk Index")
            st.write(f"### {prediction1}")
            
        with col2:
            st.subheader("Weather Severity Index")
            st.write(f"### {prediction2}")

            
    if prediction1 <= 100.00:
        st.write( "🟢 **Low Risk:** The climate impact is minimal, and environmental conditions are stable. No immediate concerns.")
    elif prediction1 <= 150:
        st.write ("🟡 **Moderate Risk:** Some climate changes are noticeable, but they have limited effects. Mild disruptions might occur.")
    elif prediction1 <= 200:
        st.write ("🟠 **High Risk:** Significant environmental shifts are happening. Extreme weather events are becoming more frequent.")
    else:
        st.write ("🔴 **Severe Risk:** Major instability, increasing severe weather events.")

    if prediction2 <= 25:
        st.write( "🟢 **Mild:** Calm weather, no significant risks.")
    elif prediction2 <= 50:
        st.write (".🟡 **Moderate:** Occasional storms, but manageable conditions.")
    elif prediction2 <= 75:
        st.write ("🟠 **Severe:** Frequent storms, strong winds, possible disruptions.")
    else:
        st.write ("🔴 **Very Severe:** High risk of flooding, damaging storms.")




elif page == "Dataset":
    st.header("Climate Insights Dataset")
    
# Read the CSV file directly
    df = pd.read_csv("climate_change_data.csv")

    # Display the dataframe
    st.write("### Dataset Preview")
    st.dataframe(df)

elif page == "About":
    st.header("ℹ️ About This App")

    st.write("""
    ### 🌍 Climate & Weather Risk Dashboard
    This interactive dashboard helps you analyze climate and weather risks in real time. It provides two key indices:

    1. **Climate Risk Index**:  
    This index measures the overall impact of climate change based on factors like **CO₂ emissions**, **sea level rise**, **precipitation**, and **temperature**. Higher values indicate greater environmental risks and the urgency to take action.

    2. **Weather Severity Index**:  
    This index evaluates the severity of weather conditions based on **humidity**, **wind speed**, and **precipitation**. It’s particularly useful for assessing storm risks and potential disaster management.

    ### 🔹 Features:
    - **Real-time calculations** based on input parameters.
    - **Interactive charts and visuals** for better data understanding.
    - **Risk classifications** for both climate and weather conditions.

    ### 🧑‍💻 About the Developer:
    This app was developed to help communities, researchers, and policymakers make informed decisions based on climate and weather data. It integrates data science models and climate research to provide a real-time view of environmental risks.

    ### 🌱 Future Goals:
    - **Add more environmental indices** to track additional factors.
    - **Enhance predictions** by integrating machine learning models.
    - **Add location-based features** to provide personalized risk assessments.

    ### 📞 Contact:
    For feedback or inquiries, please reach out to us at **contact@climateweather.com**.

    """)
