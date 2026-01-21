# Import Required Libraries

import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load Dataset
iris = pd.read_csv("Iris.csv")
iris.drop("Id", axis=1, inplace=True)

# Features and target
X = iris.drop("Species", axis=1)
y = iris["Species"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Load all trained models
with open("models/random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models/logistic_regression_model.pkl", "rb") as f:
    log_model = pickle.load(f)

with open("models/knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("models/decision_tree_model.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("models/svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

# Streamlit UI
st.title("🌸 Iris Flower Classification")
st.write("Enter the measurements of the flower to predict its species:")

model_name = st.selectbox(
    "Choose a model",
    [
        "Random Forest",
        "Logistic Regression",
        "K-Nearest Neighbors",
        "Decision Tree",
        "Support Vector Machine"
    ]
)

# Map selection to the correct model
if model_name == "Random Forest":
    selected_model = rf_model
elif model_name == "Logistic Regression":
    selected_model = log_model
elif model_name == "K-Nearest Neighbors":
    selected_model = knn_model
elif model_name == "Decision Tree":
    selected_model = dt_model
else:
    selected_model = svm_model

# User input sliders
sepal_length = st.slider(
    "Sepal Length (cm)", 
    float(X['SepalLengthCm'].min()), 
    float(X['SepalLengthCm'].max()), 
    float(X['SepalLengthCm'].mean()))

sepal_width = st.slider(
    "Sepal Width (cm)", 
    float(X['SepalWidthCm'].min()), 
    float(X['SepalWidthCm'].max()), 
    float(X['SepalWidthCm'].mean()))

petal_length = st.slider(
    "Petal Length (cm)", 
    float(X['PetalLengthCm'].min()), 
    float(X['PetalLengthCm'].max()), 
    float(X['PetalLengthCm'].mean()))

petal_width = st.slider(
    "Petal Width (cm)", 
    float(X['PetalWidthCm'].min()), 
    float(X['PetalWidthCm'].max()), 
    float(X['PetalWidthCm'].mean())
)

# Predict button
if st.button("Predict Flower Species"):
    user_flower = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=X.columns
    )
    
    prediction = selected_model.predict(user_flower)
    species = le.inverse_transform(prediction)[0]
    st.success(f"🌼 Predicted Flower Species: **{species}** using **{model_name}**")
    
# show EDA plots

st.markdown("---")
st.subheader("Exploratory Data Analysis")

if st.checkbox("Show Pairplot"):
    st.write("Pairplot of Iris dataset")
    fig = sns.pairplot(iris, hue="Species")
    st.pyplot(fig)

if st.checkbox("Show Correlation Heatmap"):
    st.write("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(
        iris.select_dtypes(include='number').corr(), 
        annot=True, 
        cmap="coolwarm", 
        ax=ax
    )
    st.pyplot(fig)
    
# Accuracy comparison bar chart
st.markdown("---")
st.subheader("Comparison of All Models' Accuracy")

if st.checkbox("Show Model Accuracy Comparison"):
    
    log_accuracy = log_model.score(X_test, y_test)
    knn_accuracy = knn_model.score(X_test, y_test)
    dt_accuracy = dt_model.score(X_test, y_test)
    rf_accuracy = rf_model.score(X_test, y_test)
    svm_accuracy = svm_model.score(X_test, y_test)
    
    results = pd.DataFrame({
        "Model": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"],
        "Accuracy": [log_accuracy, knn_accuracy, dt_accuracy, rf_accuracy, svm_accuracy]
    })
    # Sort by accuracy
    results = results.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    st.write(results)
    # Plot bar chart
    plt.figure(figsize=(6,4))
    plt.bar(results["Model"], results["Accuracy"], color=["skyblue", "lightgreen", "salmon", "purple", "orange"])
    plt.ylim(0, 1)
    plt.title("Model Accuracy Comparison", pad=15, fontsize=14)
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    
    # Add accuracy values on top of bars
    for i, v in enumerate(results["Accuracy"]):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(plt)

#Model Evaluation
st.markdown("---")
st.subheader("Confusion Matrix & Classification Report (All Models)")
    
if st.checkbox("Show Confusion Matrices and Classification Reports"):
    all_models = {
        "Logistic Regression": log_model,
        "KNN": knn_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
        "SVM": svm_model
    }
    st.write("Confusion Matrices & Classification Reports")
    for name, mdl in all_models.items():
        st.markdown(f"{name}")
        
        # Predict on test set
        y_pred = mdl.predict(X_test)
    
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', 
                    xticklabels=le.classes_, 
                    yticklabels=le.classes_, 
                    ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(f"{name} Confusion Matrix")
        st.pyplot(fig)
    
        # Classification Report
        report = classification_report(y_test, y_pred, target_names=le.classes_)
        st.text("Classification Report:")
        st.text(report)
        
