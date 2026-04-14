"""
Animal Disease Prediction - Model Training Script
Converted from animal_disease_prediction.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. Load Dataset
# ============================================================
df = pd.read_csv("cleaned_animal_disease_prediction.csv")
print("Dataset loaded successfully")
print(df.head())
print(f"\nOriginal shape: {df.shape}")

# ============================================================
# 2. Keep Top 10 Diseases Only
# ============================================================
top_diseases = df['Disease_Prediction'].value_counts().nlargest(10).index
df = df[df['Disease_Prediction'].isin(top_diseases)]
print(f"\nNew shape: {df.shape}")
print(df['Disease_Prediction'].value_counts())

# ============================================================
# 3. Check & Handle Missing Values
# ============================================================
print("\n--- Missing Values ---")
print(df.isnull().sum())

for col in df.columns:
    if col != 'Disease_Prediction':
        df[col] = df[col].fillna("Unknown")

# ============================================================
# 4. Label Encoding (save each encoder for the API)
# ============================================================
label_encoders = {}
column_classes = {}

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        column_classes[col] = list(le.classes_)

# Save label encoder classes for the frontend/API
with open("label_encoder_classes.json", "w") as f:
    json.dump(column_classes, f, indent=2)

print("\nLabel encoders saved.")
print("\n--- Dataset Info ---")
print(df.shape)
df.info()
print(df.describe())

# ============================================================
# 5. Split Input / Output
# ============================================================
X = df.drop("Disease_Prediction", axis=1)
y = df["Disease_Prediction"]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape:   {y.shape}")

# Save feature column names
feature_columns = list(X.columns)
with open("feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)

# ============================================================
# 6. Train-Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 7. Train Multiple Models
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=15,
        min_samples_split=5, random_state=42
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc}")

# ============================================================
# 8. Select Best Model
# ============================================================
best_model_name = max(results, key=results.get)
print(f"\nBEST MODEL: {best_model_name}")

# Classification report for best model
model = models[best_model_name]
y_pred = model.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# ============================================================
# 9. Visualize Model Comparison
# ============================================================
plt.figure(figsize=(10, 6))
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
plt.bar(results.keys(), results.values(), color=colors)
plt.title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
print("Chart saved as model_comparison.png")

# ============================================================
# 10. Save Best Model & Encoders
# ============================================================
pickle.dump(models[best_model_name], open("animal_disease_model.pkl", "wb"))
print("Model saved successfully as animal_disease_model.pkl")

# Save all label encoders as pickle for the API
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))
print("Label encoders saved as label_encoders.pkl")

# Save disease name mapping
if 'Disease_Prediction' in label_encoders:
    disease_le = label_encoders['Disease_Prediction']
    disease_mapping = {int(i): name for i, name in enumerate(disease_le.classes_)}
    with open("disease_mapping.json", "w") as f:
        json.dump(disease_mapping, f, indent=2)
    print("Disease mapping saved as disease_mapping.json")

print("\n[DONE] Training complete! All artifacts saved.")
print(f"   Best model: {best_model_name} ({results[best_model_name]:.2%} accuracy)")
