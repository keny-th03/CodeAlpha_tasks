import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
import os

# Load Dataset
iris = pd.read_csv("Iris.csv")
iris.drop("Id", axis=1, inplace=True)

# Features and target
X = iris.drop("Species", axis=1)
y = iris["Species"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create folder to save models
os.makedirs("models", exist_ok=True)

# Train and save Logistic Regression
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
with open("models/logistic_regression_model.pkl", "wb") as f:
    pickle.dump(log_model, f)

# Train and save KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
with open("models/knn_model.pkl", "wb") as f:
    pickle.dump(knn_model, f)

# Train and save Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
with open("models/decision_tree_model.pkl", "wb") as f:
    pickle.dump(dt_model, f)

# Train and save Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Train and save SVM
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

print("✅ All models trained and saved successfully!")
