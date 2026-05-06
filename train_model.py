"""
DiabTrace Analytics — Model Training Script
B.Tech Final Year Project | SUIIT Burla | CSE 2026

Run this script to retrain the Random Forest model using the dataset.
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

print("=" * 60)
print("DiabTrace Analytics — Random Forest Training")
print("=" * 60)

# Load dataset
DATASET_PATH = os.path.join('dataset', 'fingerprint_diabetes_dataset_3000.xlsx')
print(f"\nLoading dataset from: {DATASET_PATH}")
df = pd.read_excel(DATASET_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['Diabetes_Risk'].value_counts()}")

# Feature engineering
df['Fingerprint_Type_enc'] = df['Fingerprint_Type'].map({'arch': 0, 'loop': 1, 'whorl': 2})
df['Family_History_enc'] = df['Family_History'].map({'no': 0, 'yes': 1})
df['Ridge_Density'] = df['Ridge_Count'] / 10.0  # derived feature

FEATURES = ['Fingerprint_Type_enc', 'Ridge_Count', 'Ridge_Density', 'Age', 'BMI', 'Family_History_enc']
X = df[FEATURES]
y = df['Diabetes_Risk']

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# Random Forest — hyperparameters from thesis Table 4.1
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)

# 5-fold cross-validation on training set
print("\nRunning 5-fold cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Final training on full training set
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Set Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=['low', 'medium', 'high']))

# Feature importance
print("\nFeature Importance:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    bar = '█' * int(imp * 40)
    print(f"  {feat:30s} {bar} {imp:.4f}")

# Save model
os.makedirs('models', exist_ok=True)
MODEL_PATH = os.path.join('models', 'model.pkl')
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"\nModel saved to: {MODEL_PATH}")
print("Training complete!")
