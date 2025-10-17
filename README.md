# End-to-End Machine Learning Project: Housing Price Prediction

A comprehensive project designed to teach students the complete machine learning workflow, from data exploration to model deployment.

## 📚 Project Overview

This project demonstrates the end-to-end machine learning pipeline for predicting median housing prices in California. Students will learn essential ML concepts including data preprocessing, model training, evaluation, and deployment through an interactive web application.

## 🎯 Learning Objectives

By completing this project, students will:

- Understand the complete ML project lifecycle
- Perform exploratory data analysis (EDA)
- Build preprocessing pipelines for numerical and categorical data
- Train and evaluate regression models
- Save and load trained models
- Deploy ML models using Streamlit
- Follow ML best practices and code organization

## 📁 Project Structure

```
e2e-project/
├── app/
│   └── app.py                          # Streamlit web application
├── dataset/
│   └── housing.csv                     # California housing dataset
├── documentation/
│   └── E2E-ML.excalidraw              # Project architecture diagram
├── notebooks/
│   ├── eda.ipynb                       # Exploratory data analysis
│   └── housing_data_profiling_report.html
├── scripts/
│   ├── main.py                         # Model training pipeline
│   └── test-model.py                   # Model testing script
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/meftaul/e2e-project-09.git
   cd e2e-project
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the dataset**
   
   Ensure `dataset/housing.csv` is present in the dataset directory.

## 📖 Step-by-Step Tutorial

### Step 1: Exploratory Data Analysis (EDA)

Start by understanding the data through visualization and statistical analysis.

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `notebooks/eda.ipynb`

3. Execute cells sequentially to:
   - Load and examine the dataset
   - Visualize feature distributions
   - Identify correlations
   - Detect missing values and outliers
   - Generate automated profiling report

**Key Concepts Covered:**
- Data distribution analysis
- Feature correlation heatmaps
- Missing value detection
- Outlier identification
- Data visualization techniques

### Step 2: Train the Model

Train a regression model using the preprocessing pipeline.

```bash
python scripts/main.py
```

**What happens during training:**
- Data loading with stratified sampling
- Feature preprocessing (imputation and scaling)
- Model training (Random Forest Regressor by default)
- Model evaluation (R² score and RMSE)
- Model serialization to `trained_housing_model.pkl`

**Expected Output:**
```
Loading dataset...
Creating preprocessing pipeline...
Training model...
Model Evaluation Metrics:
R² Score: 0.8XXX
RMSE: $XXXXX.XX
Model saved as 'trained_housing_model.pkl'
```

**Key Concepts Covered:**
- Train-test split with stratification
- Feature engineering (income categories)
- Preprocessing pipelines (ColumnTransformer)
- Model training and evaluation
- Model persistence with joblib

**Experiment with Different Models:**

Edit `scripts/main.py` to try different algorithms:
```python
# Linear Regression
model = LinearRegression()

# Random Forest (default)
model = RandomForestRegressor(random_state=42)

# Decision Tree
model = DecisionTreeRegressor(random_state=42)
```

### Step 3: Test the Trained Model

Verify the model works correctly with sample data.

```bash
python scripts/test-model.py
```

This script:
- Loads the saved model
- Creates sample input data
- Makes predictions
- Displays predicted house value

**Key Concepts Covered:**
- Model loading and deserialization
- Prediction on new data
- Data format consistency

### Step 4: Deploy with Streamlit

Launch the interactive web application for real-time predictions.

```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`

**Features:**
- Interactive sliders for all input features
- Real-time prediction updates
- User-friendly interface
- Formatted price output

**Key Concepts Covered:**
- Web application deployment
- Interactive UI components with Streamlit
- Model integration in production
- User input handling

## 🔍 Understanding the Components

### Dataset Features

**Numerical Features:**
- `longitude`, `latitude` - Geographic coordinates
- `housing_median_age` - Median age of houses in the block
- `total_rooms` - Total number of rooms in the block
- `total_bedrooms` - Total number of bedrooms in the block
- `population` - Total population in the block
- `households` - Total number of households in the block
- `median_income` - Median income for households (in tens of thousands)

**Categorical Feature:**
- `ocean_proximity` - Location category (<1H OCEAN, INLAND, ISLAND, NEAR BAY, NEAR OCEAN)

**Target Variable:**
- `median_house_value` - Median house value for households (in USD)

### Data Preprocessing Pipeline

The preprocessing pipeline handles different feature types:

**Numerical Features:**
1. **Imputation**: Fill missing values with mean
2. **Scaling**: Standardize features using StandardScaler

**Categorical Features:**
1. **Imputation**: Fill missing values with most frequent value
2. **Encoding**: One-hot encoding for categorical variables

### Model Evaluation Metrics

- **R² Score (Coefficient of Determination)**: Measures how well the model explains variance in the data (range: 0-1, higher is better)
- **RMSE (Root Mean Squared Error)**: Average prediction error in dollars (lower is better)

## 🛠️ Advanced Customization

Students can extend this project by:

### 1. Feature Engineering
```python
# Add new features
data['rooms_per_household'] = data['total_rooms'] / data['households']
data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
data['population_per_household'] = data['population'] / data['households']
```

### 2. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)
```

### 3. Try Advanced Models
- Gradient Boosting (XGBoost, LightGBM)
- Neural Networks (MLPRegressor)
- Ensemble methods (Stacking, Voting)

### 4. Enhanced Web Application
- Add model performance metrics display
- Include feature importance visualization
- Show prediction confidence intervals
- Add data validation and error handling

## 📊 Expected Results

After successful training, you should see:

- **R² Score**: ~0.80-0.85 (80-85% variance explained)
- **RMSE**: ~$40,000-$50,000 (average prediction error)

These metrics indicate good model performance for housing price prediction.

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Install dependencies: `pip install -r requirements.txt` |
| `FileNotFoundError: trained_housing_model.pkl` | Run `python scripts/main.py` first to train and save the model |
| Streamlit won't start | Check if port 8501 is available, or use `streamlit run app/app.py --server.port 8502` |
| Poor model performance | Check data quality, try different models, or tune hyperparameters |
| Memory errors | Reduce dataset size or use simpler models |

## 📚 Additional Resources

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)

### Learning Materials
- [Machine Learning Course by Andrew Ng](https://www.coursera.org/learn/machine-learning)
- [Hands-On Machine Learning with Scikit-Learn](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

## 🎓 Learning Checkpoints

Complete these milestones to master the project:

- [ ] Successfully load and explore the dataset
- [ ] Complete EDA and interpret visualizations
- [ ] Understand the preprocessing pipeline components
- [ ] Train the model and achieve R² > 0.75
- [ ] Test the model with custom input data
- [ ] Deploy and interact with the Streamlit application
- [ ] Experiment with at least 2 different models
- [ ] Create new features and measure impact
- [ ] Document your findings and model comparisons

## 💡 Project Ideas for Further Learning

1. **Classification Version**: Predict price categories (Low, Medium, High)
2. **Time Series**: Add temporal data and predict price trends
3. **Deployment**: Deploy to cloud platforms (Heroku, AWS, GCP)
4. **API Development**: Create REST API using FastAPI
5. **Model Monitoring**: Add performance tracking and retraining logic

## 🤝 Contributing

This is an educational project. Feel free to fork and experiment!

## 📝 License

This project is for educational purposes.

---

**Happy Learning! 🚀**

For questions or feedback, please open an issue on GitHub.
