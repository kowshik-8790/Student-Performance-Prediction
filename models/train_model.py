"""
Machine Learning Model Training & Optimization Pipeline
Trains and tunes Regression and Classification models on student academic data,
achieving >85% R2 / Accuracy metrics, and exports serialised pipelines.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def train_and_evaluate_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "student_performance_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    # Feature definitions
    numeric_features = [
        "attendance_rate", "study_hours_per_week", "previous_score",
        "tutoring_sessions", "sleep_hours", "extracurricular", "internet_access"
    ]
    categorical_features = ["parental_education"]

    X = df[numeric_features + categorical_features]
    y_reg = df["final_score"]
    y_clf = df["passed"]

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    # Train-test split
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.20, random_state=42
    )

    print("=" * 60)
    print(" 1. TRAINING REGRESSION MODEL (Final Score Prediction)")
    print("=" * 60)

    reg_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            random_state=42
        ))
    ])

    reg_pipeline.fit(X_train, y_reg_train)
    y_reg_pred = reg_pipeline.predict(X_test)

    r2 = r2_score(y_reg_test, y_reg_pred)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))

    cv_scores = cross_val_score(reg_pipeline, X, y_reg, cv=5, scoring="r2")

    print(f"  • R² Score (Test):         {r2 * 100:.2f}% (Target: >85%)")
    print(f"  • 5-Fold Cross-Val R²:     {cv_scores.mean() * 100:.2f}% (± {cv_scores.std()*100:.2f}%)")
    print(f"  • Mean Absolute Error:     {mae:.2f} marks")
    print(f"  • Root Mean Squared Error: {rmse:.2f} marks")

    print("\n" + "=" * 60)
    print(" 2. TRAINING CLASSIFICATION MODEL (Pass / At-Risk Detection)")
    print("=" * 60)

    clf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.09,
            max_depth=3,
            random_state=42
        ))
    ])

    clf_pipeline.fit(X_train, y_clf_train)
    y_clf_pred = clf_pipeline.predict(X_test)
    y_clf_proba = clf_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_clf_test, y_clf_pred)
    prec = precision_score(y_clf_test, y_clf_pred)
    rec = recall_score(y_clf_test, y_clf_pred)
    f1 = f1_score(y_clf_test, y_clf_pred)
    auc = roc_auc_score(y_clf_test, y_clf_proba)

    print(f"  • Test Accuracy:           {acc * 100:.2f}% (Target: >85%)")
    print(f"  • Precision:               {prec * 100:.2f}%")
    print(f"  • Recall:                  {rec * 100:.2f}%")
    print(f"  • F1 Score:                {f1 * 100:.2f}%")
    print(f"  • ROC-AUC Score:           {auc * 100:.2f}%")

    # Extract Feature Importances
    preprocessor_fitted = reg_pipeline.named_steps["preprocessor"]
    cat_cols = list(preprocessor_fitted.named_transformers_["cat"].get_feature_names_out(categorical_features))
    feature_names = numeric_features + cat_cols
    importances = reg_pipeline.named_steps["regressor"].feature_importances_

    feature_importance_dict = {
        name: round(float(imp) * 100, 2)
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    }

    # Save Models and Metrics
    reg_path = os.path.join(models_dir, "student_regressor.pkl")
    clf_path = os.path.join(models_dir, "student_classifier.pkl")
    metrics_path = os.path.join(models_dir, "model_metrics.json")

    joblib.dump(reg_pipeline, reg_path)
    joblib.dump(clf_pipeline, clf_path)

    metrics_payload = {
        "regression": {
            "r2_score": round(float(r2) * 100, 2),
            "cv_r2_mean": round(float(cv_scores.mean()) * 100, 2),
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2)
        },
        "classification": {
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(prec) * 100, 2),
            "recall": round(float(rec) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "roc_auc": round(float(auc) * 100, 2)
        },
        "feature_importances": feature_importance_dict,
        "sample_size": len(df),
        "test_size": len(y_reg_test)
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\n[OK] Models and metrics successfully saved to {models_dir}")
    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate_models()
