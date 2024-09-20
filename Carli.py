import streamlit as st
import pandas as pd
import joblib  # Use joblib instead of pickle for loading the model

# Load the saved model using joblib with error handling
model = None  # Initialize model as None
model_path = 'Car Prediction Model 1.sav'

try:
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# Display UI elements regardless of model loading status
st.title('Car Price Prediction Model')
st.write('Input Some Information to Predict the Price of Your Vehicle')

# Sidebar Input Fields
st.sidebar.title('Input the Features of Your Car')
car_name = st.sidebar.selectbox('Car_Name', options=list(range(0, 10)), index=0)
year = st.sidebar.slider('Year', min_value=2000, max_value=2023, value=2014)
present_price = st.sidebar.slider('Present Price (in thousands)', min_value=0.0, max_value=100.0, value=5.59)
kms_driven = st.sidebar.slider('Kms Driven', min_value=0, max_value=100000, value=27000)
fuel_type = st.sidebar.selectbox('Fuel Type', options=['Petrol', 'Diesel'])
seller_type = st.sidebar.selectbox('Seller Type', options=['Dealer', 'Individual'])
transmission = st.sidebar.selectbox('Transmission', options=['Manual', 'Automatic'])
owner = st.sidebar.selectbox('Owner', options=[0, 1, 2, 3])

# Convert categorical variables to numerical values
fuel_type_encoded = 1 if fuel_type == 'Diesel' else 0
seller_type_encoded = 1 if seller_type == 'Individual' else 0
transmission_encoded = 1 if transmission == 'Automatic' else 0

# Prepare the input data
input_data = pd.DataFrame({
    'Car_Name': [car_name],
    'Year': [year],
    'Present_Price': [present_price],
    'Kms_Driven': [kms_driven],
    'Fuel_Type': [fuel_type_encoded],
    'Seller_Type': [seller_type_encoded],
    'Transmission': [transmission_encoded],
    'Owner': [owner]
})

# Predict Button
if st.button('Predict'):
    if model is not None:
        try:
            # Make the prediction
            prediction = model.predict(input_data)
            predicted_price_million = prediction[0] * 1_000_000  # Convert to millions
            st.subheader('Prediction:')
            st.write(f"Predicted Selling Price: ${predicted_price_million:.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Model not loaded. Please resolve the issues mentioned above.")
