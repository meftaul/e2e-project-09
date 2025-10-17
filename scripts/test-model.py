import joblib

def load_model(file_path):
    """Load a trained model from a file."""
    return joblib.load(file_path)

def get_sample_input():
    """Provide a sample input for testing the model."""
    import pandas as pd
    sample_data = {
        'longitude': [-122.23],
        'latitude': [37.88],
        'housing_median_age': [41],
        'total_rooms': [880],
        'total_bedrooms': [129],
        'population': [322],
        'households': [126],
        'median_income': [8.3252],
        'ocean_proximity': ['NEAR BAY']
    }
    return pd.DataFrame(sample_data)

def main():
    """Main function to test the model."""
    model = load_model('trained_housing_model.pkl')
    sample_input = get_sample_input()
    prediction = model.predict(sample_input)
    print(f"Predicted median house value: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()