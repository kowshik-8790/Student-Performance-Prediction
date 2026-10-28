"""
Flask Web Application for Student Performance Prediction
Integrates Machine Learning pipelines, SQLite database storage,
real-time anomaly alerts, and interactive analytics.
"""

import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

from src.predictor import get_predictor
from src.db_helper import save_prediction, get_prediction_history, get_dashboard_summary

app = Flask(__name__)
app.secret_key = "student_performance_prediction_secret_key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_metrics():
    metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    input_values = None

    if request.method == "POST":
        try:
            full_name = request.form.get("full_name", "Anonymous Student").strip()
            student_id = request.form.get("student_id", f"STU-{os.urandom(2).hex().upper()}").strip()

            input_data = {
                "student_id": student_id,
                "full_name": full_name,
                "gender": request.form.get("gender", "Female"),
                "age": int(request.form.get("age", 17)),
                "attendance_rate": float(request.form.get("attendance_rate", 85.0)),
                "study_hours_per_week": float(request.form.get("study_hours_per_week", 10.0)),
                "previous_score": float(request.form.get("previous_score", 75.0)),
                "tutoring_sessions": int(request.form.get("tutoring_sessions", 0)),
                "sleep_hours": float(request.form.get("sleep_hours", 7.0)),
                "extracurricular": int(request.form.get("extracurricular", 0)),
                "internet_access": int(request.form.get("internet_access", 1)),
                "parental_education": request.form.get("parental_education", "Bachelor")
            }

            predictor = get_predictor()
            pred_output = predictor.predict_single(input_data)

            # Save record to SQLite
            full_record = {**input_data, **pred_output}
            save_prediction(full_record)

            prediction_result = full_record
            input_values = input_data
            flash("Prediction generated and saved successfully to SQLite database!", "success")

        except Exception as e:
            flash(f"Error generating prediction: {str(e)}", "danger")

    return render_template("index.html", result=prediction_result, inputs=input_values)


@app.route("/dashboard")
def dashboard():
    summary = get_dashboard_summary()
    metrics = load_metrics()
    return render_template("dashboard.html", summary=summary, metrics=metrics)


@app.route("/history")
def history():
    records = get_prediction_history(limit=100)
    return render_template("history.html", records=records)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    batch_results = None
    if request.method == "POST":
        file = request.files.get("csv_file")
        if not file or file.filename == "":
            flash("Please upload a valid CSV file.", "warning")
            return redirect(request.url)

        try:
            df = pd.read_csv(file)
            required_cols = ["attendance_rate", "study_hours_per_week", "previous_score"]
            for col in required_cols:
                if col not in df.columns:
                    flash(f"Missing required CSV column: {col}", "danger")
                    return redirect(request.url)

            # Fill optional defaults
            if "student_id" not in df.columns:
                df["student_id"] = [f"STU-{1000+i}" for i in range(len(df))]
            if "full_name" not in df.columns:
                df["full_name"] = [f"Student {i+1}" for i in range(len(df))]
            if "parental_education" not in df.columns:
                df["parental_education"] = "Bachelor"
            if "tutoring_sessions" not in df.columns:
                df["tutoring_sessions"] = 0
            if "sleep_hours" not in df.columns:
                df["sleep_hours"] = 7.0
            if "extracurricular" not in df.columns:
                df["extracurricular"] = 0
            if "internet_access" not in df.columns:
                df["internet_access"] = 1

            predictor = get_predictor()
            results_df = predictor.predict_batch(df)

            # Persist to SQLite
            for _, row in results_df.iterrows():
                save_prediction(row.to_dict())

            batch_results = results_df.to_dict(orient="records")
            flash(f"Successfully processed {len(batch_results)} student records!", "success")

        except Exception as e:
            flash(f"Error processing CSV: {str(e)}", "danger")

    return render_template("batch.html", results=batch_results)


# ----------------------------------------------------------------------------
# REST API ENDPOINTS
# ----------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No input payload provided"}), 400

    try:
        predictor = get_predictor()
        res = predictor.predict_single(data)
        return jsonify({"status": "success", "prediction": res}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    metrics = load_metrics()
    return jsonify(metrics), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
