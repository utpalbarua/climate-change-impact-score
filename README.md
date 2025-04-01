
# 🌍 Global Climate Change Impact Predictor

This is an interactive web application built using Streamlit to analyze climate and weather risks in real-time. The app uses machine learning models to predict two key indices based on input features:  
1. **Climate Risk Index** - Evaluates the overall climate impact.
2. **Weather Severity Index** - Assesses the severity of weather conditions.

## Features
- **Input Features**: Users can input values for the following climate parameters:
    - Temperature
    - CO2 Emissions
    - Sea Level Rise
    - Precipitation
    - Humidity
    - Wind Speed
- **Predictions**: Based on the input, the app displays predictions for the Climate Risk Index and the Weather Severity Index.
- **Classification**: The app categorizes the risk and severity into different levels (Low, Moderate, High, Severe).
- **Dataset Preview**: View the dataset used for training the models, providing climate insights.
- **About Page**: Provides details about the app and its goals.

## Requirements
To run this project locally, ensure you have the following Python libraries installed:
- `streamlit`
- `joblib`
- `numpy`
- `pandas`

You can install them using pip:
```bash
pip install streamlit joblib numpy pandas
```

## How to Run
1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/climate-change-predictor.git
    ```
2. Navigate to the project directory:
    ```bash
    cd climate-change-predictor
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
OR, follow the link -**https://climate-change-impact-score-vygqpw3w6hufkzogupkw2w.streamlit.app/**

## Project Structure
- `app.py`: Main Streamlit app file.
- `model1.pkl` and `model2.pkl`: Pre-trained machine learning models for predicting the Climate Risk Index and Weather Severity Index.
- `climate_change_data.csv`: Dataset used for training the models (available in the `Dataset` section of the app).

## Pages
1. **Home**: Input your data and view predictions for climate risk and weather severity.
2. **Dataset**: Preview the climate change dataset used to train the models.
3. **About**: Learn more about the app and its future goals.

## Contact
For any inquiries or feedback, reach out to:
📧 **contact@climateweather.com**

---

### 🚀 Future Goals
- Improve model accuracy with more features.
- Add location-based climate insights.
- Extend to include more predictive indices.

```
