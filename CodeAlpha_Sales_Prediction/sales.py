#Import Required Libraries

import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns 
import pickle

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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

#Load trained models
with open("models/linear_regression_model.pkl", "rb") as f:
    lr_model = pickle.load(f)
    
# Streamlit UI
st.title("📊 Sales Prediction")
st.write("Enter the details below to predict sales:")

# User Inputs
tv_spend = st.slider(
    "TV Advertising ($)",
    int(X['TV'].mean()),
    int(X['TV'].max()),
    int(X['TV'].min())
    )

radio_spend = st.slider(
    "Radio Advertising ($)",
    int(X['TV'].mean()),
    int(X['TV'].max()),
    int(X['TV'].min())
    )

newspaper_spend = st.slider(
    "Newspaper Advertising ($)",
    int(X['TV'].mean()),
    int(X['TV'].max()),
    int(X['TV'].min())
    )

# Predict Button
if st.button("Predict Sales"):
    # Make a DataFrame from user input
    user_input = pd.DataFrame(
        [[tv_spend, radio_spend, newspaper_spend]],
        columns=["TV", "Radio", "Newspaper"]
    )
    
    # Predict using Linear Regression model
    predicted_sales = lr_model.predict(user_input)[0]
    
    st.success(f"💵 Predicted Sales: **${predicted_sales:,.2f}**")

# Show EDA plots

st.markdown("---")
st.subheader("Exploratory Data Analysis")

if st.checkbox("Show Sales Distribution"):
    st.write("📊 Distribution of target variable (Sales)")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    sns.histplot(sales['Sales'], 
                 bins=10, 
                 kde=True, 
                 color='navy',
                 ax=ax
    )
    
    ax.set_title("Distribution of Sales")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)
    
if st.checkbox("Show Advertising Spend Distributions"):
    st.write("📊 Distribution of Advertising Spends")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    colors = ['crimson', 'blue', 'orange']
    cols = ['TV', 'Radio', 'Newspaper']

    for i, col in enumerate(cols):
        sns.histplot(
            sales[col],
            bins=10,
            kde=True,
            color=colors[i],
            ax=axes[i]
        )
        axes[i].set_title(f'Distribution of {col}')

    plt.tight_layout()
    st.pyplot(fig)
    
if st.checkbox("Show Pairplot (Sales Level)"):
    st.write("🔗 Relationship between Advertising Channels and Sales")

    # Create Sales categories
    sales['Sales_Level'] = pd.cut(
        sales['Sales'],
        bins=3,
        labels=['Low', 'Medium', 'High']
    )

    fig = sns.pairplot(
        sales,
        hue='Sales_Level',
        palette='Set1',
        diag_kind='kde'
    )

    st.pyplot(fig)

if st.checkbox("Show Correlation Heatmap"):
    st.write("Correlation between Advertising Channels and Sales")

    # Select only numeric columns
    numeric_sales = sales.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        numeric_sales.corr(),
        annot=True,
        cmap='coolwarm',
        linewidths=0.5,
        ax=ax
    )

    ax.set_title("Correlation between Features")
    st.pyplot(fig)
    
# Model Evaluation
st.markdown("---")
st.subheader("Model Evaluation")

y_pred_lr = lr_model.predict(X_test)

if st.checkbox("Show Model Evaluation (Linear Regression)"):

    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    st.subheader("📈 Linear Regression Evaluation Metrics")

    eval_df = pd.DataFrame({
        "Metric": ["MAE", "MSE", "RMSE", "R² Score"],
        "Value": [mae_lr, mse_lr, rmse_lr, r2_lr]
    })

    st.table(eval_df)

#Impact of Advertising on Sales
st.markdown("---")
st.subheader("📊 Features Impact Analysis")
if st.checkbox("Show Impact of Advertising on Sales"):

    # Create coefficient dataframe
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Impact': lr_model.coef_
    })

    # Show table
    st.table(coefficients)

    # Visualization
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        coefficients['Feature'],
        coefficients['Impact'],
        color=['blue', 'green', 'red']
    )

    ax.set_title("Impact of Advertising Channels on Sales")
    ax.set_ylabel("Coefficient (Impact)")

    st.pyplot(fig)
  