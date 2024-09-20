import streamlit as st
import pandas as pd
import joblib

model = None  # Initialize model as None
model_path = 'Carlifonia Housing Prediction Model.sav'

try:
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
except ModuleNotFoundError as e:
    st.error(f"Missing module: {e}. Please ensure all dependencies are installed.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")


# Page title and description
st.title('California Housing Price Prediction')
st.write("Enter the details below to predict the median house value based on California housing data.")

# Section for input features
st.subheader("Input Features")

# Create two columns for input fields
col1, col2 = st.columns(2)

# Input fields for the housing prediction model (with sliders for numerical inputs)
with col1:
    longitude = st.slider('Longitude', min_value=-125.0, max_value=-114.0, value=-120.0, step=0.1)
    latitude = st.slider('Latitude', min_value=32.0, max_value=42.0, value=36.0, step=0.1)
    housing_median_age = st.slider('Housing Median Age', min_value=0, max_value=100, value=30)
    total_rooms = st.slider('Total Rooms', min_value=1, max_value=10000, value=2000)
    population = st.slider('Population', min_value=1, max_value=50000, value=1000)

with col2:
    total_bedrooms = st.slider('Total Bedrooms', min_value=1, max_value=5000, value=300)
    households = st.slider('Households', min_value=1, max_value=5000, value=400)
    median_income = st.slider('Median Income (in tens of thousands)', min_value=0.0, max_value=20.0, value=3.0, step=0.1)

    # Dropdown for 'Ocean Proximity' feature
    ocean_proximity = st.selectbox('Ocean Proximity', options=['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'longitude': [longitude],
    'latitude': [latitude],
    'housing_median_age': [housing_median_age],
    'total_rooms': [total_rooms],
    'total_bedrooms': [total_bedrooms],
    'population': [population],
    'households': [households],
    'median_income': [median_income],
    'ocean_proximity': [ocean_proximity]
})

# Convert ocean_proximity to numerical values
input_data['ocean_proximity'] = input_data['ocean_proximity'].map({
    'NEAR BAY': 0, '<1H OCEAN': 1, 'INLAND': 2, 'NEAR OCEAN': 3, 'ISLAND': 4
})

# Prediction Button
if st.button('Predict House Value'):
    try:
        # Make the prediction
        prediction = model.predict(input_data)

        # Display the prediction in large font size and visually appealing color
        st.subheader('Predicted Median House Value:')
        st.markdown(f'<p style="font-size:24px; color:blue;">${prediction[0]:,.2f}</p>', unsafe_allow_html=True)

        # Provide some recommendations
        st.subheader("Considerations for House Value")
        st.write("""
        - **Location Matters**: Proximity to the ocean or major cities can impact house prices significantly.
        - **Median Income**: Higher household income can lead to higher home values.
        - **Population & Density**: More population and households tend to drive up housing demand and prices.
        """)

        # Visualization of input data
        st.subheader("Input Data Summary")
        st.write(input_data)

        # Simple bar chart to visualize the numerical inputs
        st.subheader("Visual Representation of Input Data")
        st.bar_chart(input_data.drop(columns=['ocean_proximity']))

    except Exception as e:
        st.error(f"Error: {e}")
