import streamlit as st
import pandas as pd

def load_model(file_path):
    """Load a trained model from a file."""
    import joblib
    return joblib.load(file_path)

st.write("""
# Housing Price Prediction Application
""")

# Sidebar title
st.sidebar.header('User Input Features')

def user_input_features():
    longitude = st.sidebar.slider('longitude', -180.0, 180.0, -122.23)
    latitude = st.sidebar.slider('latitude', -90.0, 90.0, 37.88)
    housing_median_age = st.sidebar.slider('housing_median_age', 1, 52, 29)
    total_rooms = st.sidebar.slider('total_rooms', 2, 39320, 2130)
    total_bedrooms = st.sidebar.slider('total_bedrooms', 1, 6445, 435)
    population = st.sidebar.slider('population', 3, 35682, 1425)
    households = st.sidebar.slider('households', 1, 6082, 430)
    median_income = st.sidebar.slider('median_income', 0.4999, 15.0001, 3.87)
    ocean_proximity = st.sidebar.selectbox('ocean_proximity', ('<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'))
    
    data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input features')
st.write(input_df.transpose())

# Load the trained model
model = load_model('trained_housing_model.pkl')

# Make prediction
prediction = model.predict(input_df)

st.subheader('Predicted Median House Value')
st.write(f"# ${prediction[0]:,.2f}")