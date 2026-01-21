import pandas as pd
import numpy as np
import pickle
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
cpp = pd.read_csv("car data.csv")

# Drop unnecessary column
cpp.drop('Car_Name', axis=1, inplace=True)

# Feature Engineering
current_year = datetime.now().year
cpp['Car_Age'] = current_year - cpp['Year']
cpp.drop('Year', axis=1, inplace=True)

cpp['Mileage_per_Year'] = cpp['Driven_kms'] / (cpp['Car_Age'] + 1)
cpp['Driven_kms_log'] = np.log1p(cpp['Driven_kms'])
cpp['Present_Price_log'] = np.log1p(cpp['Present_Price'])
cpp.drop('Driven_kms', axis=1, inplace=True)

# Encode categorical features
le = LabelEncoder()
for col in cpp.select_dtypes(include='object').columns:
    cpp[col] = le.fit_transform(cpp[col])

# Features & Target
X = cpp.drop("Selling_Price", axis=1)
y = cpp["Selling_Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Save Random Forest Model
pickle.dump(rf_model, open("rf_model.pkl", "wb"))

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Save Random Forest Model
pickle.dump(lr_model, open("lr_model.pkl", "wb"))

# Save column order
pickle.dump(X_train.columns.tolist(), open("columns.pkl", "wb"))

print("Random Forest and Linear Regression models and columns saved successfully!")
