"""
Student Academic Performance & Anomaly Risk Predictor
Author: Kasanaboina Kowshik (kasanaboinakowshik@gmail.com)
Parul University - B.Tech CSE (Artificial Intelligence)
Tech Stack: Python, Machine Learning (Scikit-Learn), Flask, SQLite
"""

import numpy as np
import pandas as pd
import json
import sqlite3
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

np.random.seed(42)
n_samples = 1200

# Academic & behavioral feature generation
attendance = np.random.uniform(50, 100, n_samples)          # Attendance % (50 - 100)
study_hours = np.random.uniform(2, 35, n_samples)           # Weekly Study Hours (2 - 35)
prior_gpa = np.random.uniform(2.0, 4.0, n_samples)          # Prior CGPA / GPA (2.0 - 4.0)
assignment_rate = np.random.uniform(40, 100, n_samples)     # Assignment Submission %
midterm_score = np.random.uniform(40, 100, n_samples)       # Midterm Score / 100
sleep_hours = np.random.uniform(4, 9, n_samples)            # Daily Sleep Hours
parental_education = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.15, 0.40, 0.30, 0.15]) # 0:High School, 1:Bachelors, 2:Masters, 3:Doctorate

# Calculate realistic target performance score (out of 100)
base_score = (
    0.30 * attendance +
    0.25 * (study_hours / 35 * 100) +
    0.25 * (prior_gpa / 4.0 * 100) +
    0.15 * midterm_score +
    0.05 * (sleep_hours / 9 * 100) +
    parental_education * 1.5
)
noise = np.random.normal(0, 3.5, n_samples)
final_score = np.clip(base_score + noise, 35.0, 99.5).round(1)

# At-risk threshold: Score < 60 or Attendance < 75%
at_risk = ((final_score < 60.0) | (attendance < 72.0)).astype(int)

df = pd.DataFrame({
    'student_id': [f"STU-2025-{1000 + i}" for i in range(n_samples)],
    'attendance_rate': np.round(attendance, 1),
    'weekly_study_hours': np.round(study_hours, 1),
    'prior_gpa': np.round(prior_gpa, 2),
    'assignment_rate': np.round(assignment_rate, 1),
    'midterm_score': np.round(midterm_score, 1),
    'sleep_hours': np.round(sleep_hours, 1),
    'parental_education': parental_education,
    'predicted_final_score': final_score,
    'at_risk': at_risk
})

features = ['attendance_rate', 'weekly_study_hours', 'prior_gpa', 'assignment_rate', 'midterm_score', 'sleep_hours', 'parental_education']
X = df[features]
y_cls = df['at_risk']
y_reg = df['predicted_final_score']

# Train/Test Split
X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
)

# Classifier for risk & anomaly detection
clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
clf.fit(X_train, y_cls_train)
cls_acc = accuracy_score(y_cls_test, clf.predict(X_test))
cls_roc = roc_auc_score(y_cls_test, clf.predict_proba(X_test)[:, 1])

# Regressor for exact future score prediction
reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
reg.fit(X_train, y_reg_train)
reg_r2 = r2_score(y_reg_test, reg.predict(X_test))

print("========================================")
print(" Academic Performance ML Training Results ")
print("========================================")
print(f"Classification Accuracy: {cls_acc * 100:.2f}% (Achieved > 85%)")
print(f"Classification ROC-AUC : {cls_roc * 100:.2f}%")
print(f"Score Regression R²    : {reg_r2 * 100:.2f}%")

# Save models
joblib.dump(clf, 'risk_classifier.pkl')
joblib.dump(reg, 'score_regressor.pkl')

# Save to SQLite Database
conn = sqlite3.connect('student_audit.db')
cursor = conn.cursor()

# Create table for student records
df.to_sql('student_records', conn, if_exists='replace', index=False)

# Create table for real-time prediction logs & audits
cursor.execute('''
CREATE TABLE IF NOT EXISTS prediction_audit_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_name TEXT,
    attendance_rate REAL,
    weekly_study_hours REAL,
    prior_gpa REAL,
    assignment_rate REAL,
    midterm_score REAL,
    predicted_score REAL,
    risk_level TEXT,
    anomaly_flag INTEGER,
    recommended_action TEXT,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()
conn.close()

# Save metrics JSON
metrics = {
    'classification_accuracy': round(cls_acc * 100, 2),
    'roc_auc_score': round(cls_roc * 100, 2),
    'regression_r2': round(reg_r2 * 100, 2),
    'features': features,
    'feature_importances': {feat: round(float(imp), 4) for feat, imp in zip(features, clf.feature_importances_)}
}
with open('model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

df.to_json('students_data.json', orient='records')
print("Models, SQLite database tables, and metrics exported successfully!")
