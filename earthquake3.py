import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer

# 1. Load and Prepare Data
def load_and_preprocess_data():
    # Load dataset
    data = pd.read_csv('usgs_main.csv')
    
    # Clean data
    data = data[data['mag'] >= 0]  # Remove negative magnitudes
    data = data.drop(columns=['nst', 'gap', 'dmin', 'horizontalError', 'magError', 'magNst'])
    data = data.dropna(subset=['mag'])
    
    return data

# 2. Preprocessing Pipeline
def preprocess_data(data):
    # Split features and target
    X = data[['latitude', 'longitude', 'depth', 'rms', 'magType']]
    y = data['mag']
    
    # Split data before preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define features
    numerical_features = ['latitude', 'longitude', 'depth', 'rms']
    categorical_features = ['magType']
    
    # Impute missing values
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    X_train[numerical_features] = num_imputer.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = num_imputer.transform(X_test[numerical_features])
    
    X_train[categorical_features] = cat_imputer.fit_transform(X_train[categorical_features])
    X_test[categorical_features] = cat_imputer.transform(X_test[categorical_features])
    
    # One-hot encoding
    X_train = pd.get_dummies(X_train, columns=['magType'], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=['magType'], drop_first=True)
    
    # Save important components
    joblib.dump(num_imputer, 'numerical_imputer.pkl')
    joblib.dump(cat_imputer, 'categorical_imputer.pkl')
    joblib.dump(X_train.columns, 'feature_columns.pkl')
    
    # Align test columns with train
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    return X_train, X_test, y_train, y_test

# 3. Model Training
def train_model(X_train, y_train):
    # Hyperparameter tuning
    param_dist = {
        'n_estimators': np.arange(100, 300, 50),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=15,
        cv=3,
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X_train, y_train)
    return search.best_estimator_

# 4. Evaluation and Saving
def evaluate_and_save(model, X_test, y_test):
    # Make predictions
    y_pred = np.maximum(0, model.predict(X_test))  # Ensure non-negative
    
    # Calculate metrics
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    
    # Save model
    joblib.dump(model, 'earthquake_magnitude_predictor.pkl')
    
    # Save metrics and sample predictions
    with open('model_performance.txt', 'w') as f:
        f.write("Model Performance Metrics:\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        
        f.write("\nSample Predictions:\n")
        sample_comparison = pd.DataFrame({
            'Actual': y_test[:10],
            'Predicted': y_pred[:10]
        })
        f.write(sample_comparison.to_string())
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Magnitude')
    plt.ylabel('Predicted Magnitude')
    plt.title('Actual vs Predicted Magnitudes')
    plt.savefig('actual_vs_predicted.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    data = load_and_preprocess_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate and save
    evaluate_and_save(model, X_test, y_test)
    
    print("Model training and evaluation complete!")
    print("Saved files:")
    print("- earthquake_magnitude_predictor.pkl (trained model)")
    print("- numerical_imputer.pkl, categorical_imputer.pkl (data processors)")
    print("- feature_columns.pkl (required feature structure)")
    print("- model_performance.txt (performance metrics)")
    print("- actual_vs_predicted.png (visualization)")

