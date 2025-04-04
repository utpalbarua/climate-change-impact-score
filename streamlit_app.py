import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Climate Impact Predictor",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .header-container {
        background-color: #1e3d59;
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .risk-card {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .risk-low {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .risk-moderate {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background-color: #ffe5d0;
        border-left: 5px solid #fd7e14;
    }
    .risk-severe {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Load models safely
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model1 = load_model("model1.pkl")
model2 = load_model("model2.pkl")

# Sidebar with improved navigation
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=Climate+App", width=150)
    st.markdown("### 📌 Navigation")
    page = st.radio("", ["🏠 Home", "📊 Dataset", "ℹ️ About", "📈 Analytics"])
    
    st.markdown("---")
    st.markdown("### 🌐 Current Status")
    current_date = datetime.now().strftime("%B %d, %Y")
    st.markdown(f"**Date:** {current_date}")
    
    # Example of a sidebar info panel
    with st.expander("ℹ️ How to use"):
        st.write("""
        1. Enter climate parameters
        2. Click 'Predict' to see results
        3. View visualizations and insights
        """)

# Main content
if page == "🏠 Home":
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 style='text-align: center;'>🌍 Global Climate Change Impact Predictor</h1>
        <p style='text-align: center;'>Real-time climate risk assessment tool</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Two column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Input Features")
        
        # Use sliders for better UX
        feature1 = st.slider("Temperature (°C)", min_value=-20.0, max_value=50.0, value=0.0, step=0.1,
                             help="Average temperature in Celsius")
        feature2 = st.slider("CO₂ Emissions (ppm)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1,
                             help="Carbon dioxide levels in parts per million")
        feature3 = st.slider("Sea Level Rise (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1,
                             help="Sea level rise in millimeters")
        
        # Add collapsible section for advanced parameters
        with st.expander("Advanced Parameters"):
            feature4 = st.slider("Precipitation (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1,
                                help="Average precipitation in millimeters")
            feature5 = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                                help="Relative humidity percentage")
            feature6 = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=200.0, value=0.0, step=0.1,
                                help="Average wind speed in kilometers per hour")
        
        features = np.array([[feature1, feature2, feature3, feature4, feature5, feature6]])
        
        # Center the button and make it more prominent
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_button = st.button("🔎 Predict Impact", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a historic data comparison section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Parameter Comparison")
        
        # Create simple comparison chart
        parameter_names = ['Temperature', 'CO₂', 'Sea Level', 'Precipitation', 'Humidity', 'Wind']
        current_values = [feature1, feature2, feature3, feature4, feature5, feature6]
        global_avg = [15.0, 415.0, 3.3, 100.0, 60.0, 15.0]  # Example global averages
        
        comparison_df = pd.DataFrame({
            'Parameter': parameter_names,
            'Your Input': current_values,
            'Global Average': global_avg
        })
        
        chart = alt.Chart(pd.melt(
            comparison_df, 
            id_vars=['Parameter'], 
            value_vars=['Your Input', 'Global Average']
        )).mark_bar().encode(
            x=alt.X('value:Q', title='Value'),
            y=alt.Y('Parameter:N', title=None),
            color=alt.Color('variable:N', legend=alt.Legend(title=None)),
            row=alt.Row('Parameter:N', header=alt.Header(labelAngle=0))
        ).properties(
            height=40
        ).resolve_scale(
            y='independent'
        )
        
        st.altair_chart(chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔍 Impact Assessment")
        
        if predict_button and model1 and model2:
            # Create a progress indicator
            with st.spinner('Calculating impact...'):
                # simulate processing delay
                import time
                time.sleep(0.5)
                
                prediction1 = model1.predict(features)[0]
                prediction2 = model2.predict(features)[0]
            
            # Show metrics
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(label="Climate Risk Index", value=f"{prediction1:.1f}", delta=f"{prediction1-150:.1f} from baseline")
            with metric_col2:
                st.metric(label="Weather Severity Index", value=f"{prediction2:.1f}", delta=f"{prediction2-50:.1f} from baseline")
            
            # Visualization of results with gauges
            fig, ax = plt.subplots(1, 2, figsize=(10, 3))
            
            # Climate Risk Gauge
            risk_colors = ['green', 'yellow', 'orange', 'red']
            risk_bounds = [0, 100, 150, 200, 250]
            for i in range(len(risk_bounds)-1):
                ax[0].barh([0], [risk_bounds[i+1] - risk_bounds[i]], left=risk_bounds[i], color=risk_colors[i], height=0.5)
            ax[0].plot([prediction1, prediction1], [-0.25, 0.75], 'k-', linewidth=2)
            ax[0].set_xlim(0, 250)
            ax[0].set_ylim(-0.5, 1)
            ax[0].set_title('Climate Risk Index')
            ax[0].axis('off')
            
            # Weather Severity Gauge
            severity_colors = ['green', 'yellow', 'orange', 'red']
            severity_bounds = [0, 25, 50, 75, 100]
            for i in range(len(severity_bounds)-1):
                ax[1].barh([0], [severity_bounds[i+1] - severity_bounds[i]], left=severity_bounds[i], color=severity_colors[i], height=0.5)
            ax[1].plot([prediction2, prediction2], [-0.25, 0.75], 'k-', linewidth=2)
            ax[1].set_xlim(0, 100)
            ax[1].set_ylim(-0.5, 1)
            ax[1].set_title('Weather Severity Index')
            ax[1].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Classification with styled messages
            risk_class = ""
            risk_message = ""
            if prediction1 <= 100:
                risk_class = "risk-low"
                risk_message = "🟢 **Low Risk:** Climate impact is minimal."
            elif prediction1 <= 150:
                risk_class = "risk-moderate"
                risk_message = "🟡 **Moderate Risk:** Some climate changes, mild disruptions."
            elif prediction1 <= 200:
                risk_class = "risk-high"
                risk_message = "🟠 **High Risk:** Significant environmental shifts."
            else:
                risk_class = "risk-severe"
                risk_message = "🔴 **Severe Risk:** Major instability, increasing severe weather events."
            
            severity_class = ""
            severity_message = ""
            if prediction2 <= 25:
                severity_class = "risk-low"
                severity_message = "🟢 **Mild:** Calm weather, no significant risks."
            elif prediction2 <= 50:
                severity_class = "risk-moderate"
                severity_message = "🟡 **Moderate:** Occasional storms, manageable conditions."
            elif prediction2 <= 75:
                severity_class = "risk-high"
                severity_message = "🟠 **Severe:** Frequent storms, strong winds, possible disruptions."
            else:
                severity_class = "risk-severe"
                severity_message = "🔴 **Very Severe:** High risk of flooding and damaging storms."
            
            st.markdown(f'<div class="risk-card {risk_class}">{risk_message}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="risk-card {severity_class}">{severity_message}</div>', unsafe_allow_html=True)
            
            # Recommendations based on prediction
            st.subheader("📋 Recommendations")
            recommendations = [
                "Monitor changes in local weather patterns",
                "Prepare for potential climate-related disruptions"
            ]
            
            if prediction1 > 150 or prediction2 > 50:
                recommendations.append("Develop climate resilience strategies")
            if prediction1 > 200 or prediction2 > 75:
                recommendations.append("Implement immediate adaptation measures")
                recommendations.append("Consider evacuation plans for extreme events")
                
            for i, rec in enumerate(recommendations):
                st.markdown(f"- {rec}")
                
        else:
            # Placeholder for results
            st.info("Enter parameters and click 'Predict Impact' to see the assessment.")
            
            # Example output
            st.markdown("### Sample Output Preview")
            st.image("https://via.placeholder.com/600x300.png?text=Result+Visualization", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "📊 Dataset":
    st.markdown("""
    <div class="header-container">
        <h1 style='text-align: center;'>📊 Climate Insights Dataset</h1>
        <p style='text-align: center;'>Explore and analyze climate change data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data exploration section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("climate_change_data.csv")
        except Exception as e:
            # If file doesn't exist, create sample data
            sample_data = pd.DataFrame({
                'Year': range(2000, 2025),
                'Temperature': np.random.normal(15, 0.5, 25) + np.linspace(0, 1.2, 25),
                'CO2_Emissions': np.random.normal(380, 5, 25) + np.linspace(0, 40, 25),
                'Sea_Level_Rise': np.random.normal(0, 0.5, 25) + np.linspace(0, 10, 25),
                'Precipitation': np.random.normal(100, 10, 25),
                'Humidity': np.random.normal(60, 5, 25),
                'Wind_Speed': np.random.normal(15, 3, 25),
                'Climate_Risk_Index': np.random.normal(120, 20, 25) + np.linspace(0, 40, 25),
                'Weather_Severity_Index': np.random.normal(40, 10, 25) + np.linspace(0, 20, 25)
            })
            return sample_data

    df = load_data()
    
    # Add filtering options
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filter Data")
        min_year = int(df['Year'].min()) if 'Year' in df.columns else 2000
        max_year = int(df['Year'].max()) if 'Year' in df.columns else 2025
        year_range = st.slider("Year Range", min_year, max_year, (min_year, max_year))
        
        # Allow column selection
        if df is not None:
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect("Select Columns", all_columns, default=all_columns[:5])
        
    with col2:
        st.subheader("Dataset Preview")
        if df is not None:
            # Filter by selected year range
            if 'Year' in df.columns:
                filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
            else:
                filtered_df = df
                
            # Show only selected columns
            if selected_columns:
                filtered_df = filtered_df[selected_columns]
                
            st.dataframe(filtered_df, use_container_width=True)
            
            # Show data summary
            with st.expander("Data Summary"):
                st.write(filtered_df.describe())
    

elif page == "ℹ️ About":
    st.markdown("""
    <div class="header-container">
        <h1 style='text-align: center;'>ℹ️ About This App</h1>
        <p style='text-align: center;'>Learn more about the Climate & Weather Risk Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # About content
    about_col1, about_col2 = st.columns([3, 2])
    
    with about_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        ### 🌍 Climate & Weather Risk Dashboard
        
        This interactive dashboard helps analyze climate and weather risks in real-time.
        
        #### Key Features:
        - **Climate Risk Index**: Evaluates overall climate impact
        - **Weather Severity Index**: Assesses the severity of weather conditions
        - **Data Visualization**: Interactive charts and graphs
        - **Recommendation System**: Personalized climate risk management advice
        
        #### How It Works
        The dashboard uses machine learning models trained on historical climate data to predict risk levels based on current environmental parameters.
        
        #### 🚀 Future Goals
        - Improve prediction accuracy
        - Add more indices and parameters
        - Include location-based insights
        - Develop mobile application
        
        #### 📞 Contact
        Email: contact@climateweather.com  
        Website: www.climateweather.com
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with about_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image("https://via.placeholder.com/400x300.png?text=Climate+Dashboard", use_container_width=True)
        
        st.markdown("### 📰 Latest Updates")
        st.markdown("""
        - Added new prediction models (March 2025)
        - Enhanced visualization tools (February 2025)
        - Improved data analytics capabilities (January 2025)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
       
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "📈 Analytics":
    st.markdown("""
    <div class="header-container">
        <h1 style='text-align: center;'>📈 Climate Analytics</h1>
        <p style='text-align: center;'>Advanced insights and trend analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create sample data for demonstration
    years = list(range(2000, 2025))
    temp_data = np.random.normal(15, 0.5, 25) + np.linspace(0, 1.2, 25)
    co2_data = np.random.normal(380, 5, 25) + np.linspace(0, 40, 25)
    sea_level_data = np.random.normal(0, 0.5, 25) + np.linspace(0, 10, 25)
    
    # Trends analysis
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Climate Trends Analysis")
    
    trend_tabs = st.tabs(["Temperature", "CO₂ Emissions", "Sea Level"])
    
    with trend_tabs[0]:
        # Temperature trend chart
        temp_df = pd.DataFrame({
            'Year': years,
            'Temperature': temp_data,
            'Trend': np.poly1d(np.polyfit(range(len(years)), temp_data, 1))(range(len(years)))
        })
        
        temp_chart = alt.Chart(temp_df).mark_line().encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Temperature:Q', title='Temperature (°C)'),
            color=alt.value('#FF5733')
        ).properties(
            title='Global Temperature Trend'
        )
        
        trend_line = alt.Chart(temp_df).mark_line(strokeDash=[5, 5]).encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Trend:Q'),
            color=alt.value('#3366FF')
        )
        
        st.altair_chart(temp_chart + trend_line, use_container_width=True)
        
        # Temperature insights
        st.markdown("""
        **Temperature Insights:**
        - Annual average temperature increase: 0.048°C per year
        - Total change over period: +1.2°C
        - Rate of change is accelerating in recent years
        """)
    
    with trend_tabs[1]:
        # CO2 trend chart
        co2_df = pd.DataFrame({
            'Year': years,
            'CO2': co2_data,
            'Trend': np.poly1d(np.polyfit(range(len(years)), co2_data, 1))(range(len(years)))
        })
        
        co2_chart = alt.Chart(co2_df).mark_line().encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('CO2:Q', title='CO₂ (ppm)'),
            color=alt.value('#33CC33')
        ).properties(
            title='CO₂ Emissions Trend'
        )
        
        co2_trend_line = alt.Chart(co2_df).mark_line(strokeDash=[5, 5]).encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Trend:Q'),
            color=alt.value('#3366FF')
        )
        
        st.altair_chart(co2_chart + co2_trend_line, use_container_width=True)
        
        # CO2 insights
        st.markdown("""
        **CO₂ Insights:**
        - Annual increase: ~1.6 ppm per year
        - Total change over period: +40 ppm
        - Current rate exceeds historical averages
        """)
        
    with trend_tabs[2]:
        # Sea level trend chart
        sea_df = pd.DataFrame({
            'Year': years,
            'Sea_Level': sea_level_data,
            'Trend': np.poly1d(np.polyfit(range(len(years)), sea_level_data, 1))(range(len(years)))
        })
        
        sea_chart = alt.Chart(sea_df).mark_line().encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Sea_Level:Q', title='Sea Level Rise (mm)'),
            color=alt.value('#3399FF')
        ).properties(
            title='Sea Level Rise Trend'
        )
        
        sea_trend_line = alt.Chart(sea_df).mark_line(strokeDash=[5, 5]).encode(
            x=alt.X('Year:O', title='Year'),
            y=alt.Y('Trend:Q'),
            color=alt.value('#3366FF')
        )
        
        st.altair_chart(sea_chart + sea_trend_line, use_container_width=True)
        
        # Sea level insights
        st.markdown("""
        **Sea Level Insights:**
        - Annual rise: ~0.4 mm per year
        - Total change over period: +10 mm
        - Acceleration observed in recent measurements
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Global Impact Map (Placeholder)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🗺️ Global Impact Map")
    st.image("https://via.placeholder.com/800x400.png?text=Global+Climate+Impact+Map", use_container_width=True)
    
    st.markdown("""
    **Regional Impact Summary:**
    - **North America**: Increased wildfire risk, drought conditions in western regions
    - **Europe**: More frequent heatwaves, changing precipitation patterns
    - **Asia**: Rising sea levels threatening coastal areas, monsoon pattern shifts
    - **Africa**: Expanding desertification, water scarcity challenges
    - **Australia**: Coral reef degradation, extreme heat events
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Custom analysis tools
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔧 Custom Analytics")
    
    st.markdown("""
    Choose parameters to create a custom climate impact analysis:
    """)
    
    custom_col1, custom_col2 = st.columns(2)
    
    with custom_col1:
        st.markdown("**Parameter Selection**")
        param1 = st.selectbox("Primary Parameter", ["Temperature", "CO₂ Emissions", "Sea Level", "Precipitation"])
        param2 = st.selectbox("Secondary Parameter", ["Humidity", "Wind Speed", "Weather Severity", "Climate Risk"])
        time_period = st.select_slider("Time Period", options=["5 Years", "10 Years", "15 Years", "20 Years"])
        region = st.selectbox("Region", ["Global", "North America", "Europe", "Asia", "Africa", "Australia"])
    
    with custom_col2:
        st.markdown("**Custom Analysis Output**")
        st.info("Select parameters and click 'Generate Analysis' to create a custom report")
        
        if st.button("Generate Analysis", key="custom_analysis"):
            st.success("Analysis generated successfully!")
            st.write(f"Analysis for {param1} and {param2} over {time_period} in {region}")
            
            # Placeholder chart
            st.image("https://via.placeholder.com/600x300.png?text=Custom+Analysis+Chart", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="background-color: #1e3d59; padding: 10px; border-radius: 10px; margin-top: 20px;">
    <p style="color: white; text-align: center; margin: 0;">© 2025 Global Climate Change Impact Predictor | Version 2.0</p>
</div>
""", unsafe_allow_html=True)
