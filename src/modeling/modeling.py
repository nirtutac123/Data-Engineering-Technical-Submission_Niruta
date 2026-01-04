#!/usr/bin/env python3
"""
Arrest Prediction Model
-----------------------
Binary classification to predict whether an arrest was made for a crime incident.
Uses cleaned Chicago crime data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import os

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', '..', 'data', 'clean', 'chicago_chicago_crimes_2005_to_2007_clean.csv')
df = pd.read_csv(data_path, nrows=50000)  # Load only 50k rows for faster testing

# Drop rows where arrest_made is NaN
df = df.dropna(subset=['arrest_made'])

# Feature engineering
df['incident_datetime'] = pd.to_datetime(df['incident_datetime'], errors='coerce')
df['hour'] = df['incident_datetime'].dt.hour
df['day_of_week'] = df['incident_datetime'].dt.dayofweek

# Select features
features = ['crime_category', 'crime_subtype', 'location_type', 'district', 'domestic_flag',
            'latitude', 'longitude', 'year', 'hour', 'day_of_week']
target = 'arrest_made'

# Keep only rows with non-null features
df = df[features + [target]].dropna()

# Encode categorical features
categorical_features = ['crime_category', 'crime_subtype', 'location_type', 'district']
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Convert boolean to int
df['domestic_flag'] = df['domestic_flag'].astype(int)
df[target] = df[target].astype(int)

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['latitude', 'longitude', 'year', 'hour', 'day_of_week']
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Train models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }

    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))

# Print summary
print("\nSummary:")
for name, metrics in results.items():
    print(f"{name}: Accuracy={metrics['accuracy']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, AUC={metrics['auc']:.3f}")

# Feature importance for Random Forest
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Feature Importances (Random Forest):")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.savefig('feature_importance.png')

# Write results to file
with open('model_results.txt', 'w') as f:
    f.write("Summary:\n")
    for name, metrics in results.items():
        f.write(f"{name}: Accuracy={metrics['accuracy']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, AUC={metrics['auc']:.3f}\n")
    f.write("\nTop 10 Feature Importances (Random Forest):\n")
    f.write(feature_importance.head(10).to_string())

def run_modeling():
    """Run the complete modeling pipeline."""
    # This function encapsulates the modeling logic above
    # The modeling code runs when the module is imported
    pass

if __name__ == '__main__':
    run_modeling()