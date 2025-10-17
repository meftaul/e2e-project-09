"""
Housing Price Prediction Model
A reusable pipeline for training and evaluating regression models on housing data.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


def load_data(file_path):
    """Load housing data and create income categories."""
    data = pd.read_csv(file_path)
    data['income_cat'] = pd.cut(
        data['median_income'],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )
    return data


def create_preprocessing_pipeline(num_features, cat_features):
    """Create preprocessing pipeline for numerical and categorical features."""
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    return ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])


def train_model(X_train, y_train, model, preprocessor):
    """Train the model with preprocessing."""
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', model)
    ])
    return pipeline.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    predictions = model.predict(X_test)
    return {
        'r2': r2_score(y_test, predictions),
        'rmse': root_mean_squared_error(y_test, predictions)
    }


def save_model(model, file_path):
    """Save the trained model to a file."""
    import joblib
    joblib.dump(model, file_path)

def main():
    """Main execution function."""
    # Configuration
    DATA_PATH = '/home/meftaul/Documents/e2e-project/dataset/housing.csv'
    TARGET_COLUMN = 'median_house_value'
    NUM_FEATURES = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                    'total_bedrooms', 'population', 'households', 'median_income']
    CAT_FEATURES = ['ocean_proximity']
    
    # Choose your model
    # model = LinearRegression()
    model = RandomForestRegressor(random_state=42)
    # model = DecisionTreeRegressor(random_state=42)
    
    # Load and prepare data
    housing = load_data(DATA_PATH)
    X = housing.drop([TARGET_COLUMN, 'income_cat'], axis=1)
    y = housing[TARGET_COLUMN]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=housing['income_cat']
    )
    
    # Create preprocessor and train model
    preprocessor = create_preprocessing_pipeline(NUM_FEATURES, CAT_FEATURES)
    trained_model = train_model(X_train, y_train, model, preprocessor)
    
    # Evaluate
    metrics = evaluate_model(trained_model, X_test, y_test)
    
    # Display results
    print("Model Evaluation Metrics:")
    print(f"R² Score: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.2f}")

    # Save the model
    save_model(trained_model, 'trained_housing_model.pkl')
    
    return trained_model, metrics


if __name__ == "__main__":
    model, metrics = main()