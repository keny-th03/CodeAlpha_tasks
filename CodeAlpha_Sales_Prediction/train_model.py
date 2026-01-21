import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle 
import os

# Load Dataset
sales = pd.read_csv("Advertising.csv")

#Drop unnecessary column if exists
if 'Unnamed: 0' in sales.columns:
    sales.drop('Unnamed: 0', axis=1, inplace=True)
    
# Features & target
X = sales[['TV', 'Radio', 'Newspaper']]
y = sales['Sales']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create folder to save model
os.makedirs("models", exist_ok=True)

# Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Save model
with open("models/linear_regression_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

print("Model trained and saved successfully!")