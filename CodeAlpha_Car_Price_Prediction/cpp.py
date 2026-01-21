from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

# Load model and columns
rf_model = pickle.load(open("rf_model.pkl", "rb"))
lr_model = pickle.load(open("lr_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Ensure static folder exists
if not os.path.exists('static'):
    os.makedirs('static')
    
# Helper function to generate visualizations
def generate_visualizations(y_test, y_pred_lr, y_pred_rf, X_train, model, columns):
    # Model Comparison Visualization (MAE, RMSE, R²)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_lr = r2_score(y_test, y_pred_lr)
    r2_rf = r2_score(y_test, y_pred_rf)

    models = ['Linear Regression', 'Random Forest']
    colors = ['crimson', 'skyblue']

    # MAE
    plt.figure(figsize=(6,4))
    plt.bar(models, [mae_lr, mae_rf], color=colors)
    plt.title("MAE Comparison")
    plt.ylabel("MAE")
    plt.savefig("static/MAE.png")
    plt.close()

    # RMSE
    plt.figure(figsize=(6,4))
    plt.bar(models, [rmse_lr, rmse_rf], color=colors)
    plt.title("RMSE Comparison")
    plt.ylabel("RMSE")
    plt.savefig("static/RMSE.png")
    plt.close()

    # R²
    plt.figure(figsize=(6,4))
    plt.bar(models, [r2_lr, r2_rf], color=colors)
    plt.title("R² Score Comparison")
    plt.ylabel("R² Score")
    plt.savefig("static/R2.png")
    plt.close()

    # Feature Importance
    importances = model.feature_importances_
    plt.figure(figsize=(8,6))
    plt.barh(columns, importances, color='skyblue')
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.savefig("static/impact.png")
    plt.close()
    
    # Model Performance Visualization
    # Actual vs Predicted (Linear Regression)
    plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label="Actual Price")
    plt.plot(y_pred_lr, label="Predicted Price")
    plt.xlabel("Test Samples")
    plt.ylabel("Car Price")
    plt.title("Actual vs Predicted Prices (Linear Regression)")
    plt.legend()
    plt.savefig("static/Lr_model_perf.png")
    plt.close()
    
    # Actual vs Predicted (Random Forest)
    plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label="Actual Price")
    plt.plot(y_pred_rf, label="Predicted Price")
    plt.xlabel("Test Samples")
    plt.ylabel("Car Price")
    plt.title("Actual vs Predicted Prices (Random Forest)")
    plt.legend()
    plt.savefig("static/rf_model_perf.png")
    plt.close()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        'Present_Price_log': np.log1p(float(request.form['present_price'])),
        'Car_Age': int(request.form['car_age']),
        'Driven_kms_log': np.log1p(float(request.form['driven_kms'])),
        'Mileage_per_Year': float(request.form['mileage_year']),
        'Fuel_Type': int(request.form['fuel_type']),
        'Selling_type': int(request.form['selling_type']),
        'Transmission': int(request.form['transmission']),
        'Owner': int(request.form['owner']),
        'Horsepower': float(request.form['horsepower']),
        'Mileage': float(request.form['mileage'])
    }

    input_cpp = pd.DataFrame([data])

    # Add missing columns
    for col in columns:
        if col not in input_cpp.columns:
            input_cpp[col] = 0

    input_cpp = input_cpp[columns]

    # Predict with both models
    rf_prediction = round(rf_model.predict(input_cpp)[0], 2)
    lr_prediction = round(lr_model.predict(input_cpp)[0], 2)
    
    # For visualization: recreate test set
    cpp = pd.read_csv("car data.csv")
    cpp.drop('Car_Name', axis=1, inplace=True)
    
    current_year = pd.Timestamp.now().year
    cpp['Car_Age'] = current_year - cpp['Year']
    cpp.drop('Year', axis=1, inplace=True)
    
    cpp['Mileage_per_Year'] = cpp['Driven_kms'] / (cpp['Car_Age'] + 1)
    cpp['Driven_kms_log'] = np.log1p(cpp['Driven_kms'])
    cpp['Present_Price_log'] = np.log1p(cpp['Present_Price'])
    cpp.drop('Driven_kms', axis=1, inplace=True)
    
    cat_cols = cpp.select_dtypes(include='object').columns
    for col in cat_cols:
        cpp[col] = pd.factorize(cpp[col])[0]

    X = cpp.drop("Selling_Price", axis=1)
    y = cpp["Selling_Price"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred_lr_test = lr_model.predict(X_test)
    y_pred_rf_test = rf_model.predict(X_test)
    
    # Generate visualization plots
    generate_visualizations(y_test, y_pred_lr_test, y_pred_rf_test, X_train, rf_model, columns)
    
    return render_template("after.html", prediction=rf_prediction)

if __name__ == "__main__":
    app.run(debug=True)
