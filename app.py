"""
Student Performance Prediction & Academic Early Warning System
Flask Backend Application
Author: Kasanaboina Kowshik (kasanaboinakowshik@gmail.com)
Tech Stack: Python, Machine Learning (Scikit-Learn), Flask, SQLite, HTML, CSS
"""

import os
import sqlite3
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, static_folder='.', static_url_path='')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'student_audit.db')
CLS_MODEL_PATH = os.path.join(BASE_DIR, 'risk_classifier.pkl')
REG_MODEL_PATH = os.path.join(BASE_DIR, 'score_regressor.pkl')

# Load trained ML models
clf_model = None
reg_model = None

if os.path.exists(CLS_MODEL_PATH):
    clf_model = joblib.load(CLS_MODEL_PATH)
if os.path.exists(REG_MODEL_PATH):
    reg_model = joblib.load(REG_MODEL_PATH)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    metrics_path = os.path.join(BASE_DIR, 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics not found"}), 404

@app.route('/api/students', methods=['GET'])
def get_students():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM student_records LIMIT 100")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify({"students": rows, "count": len(rows)})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        student_name = data.get('student_name', 'Anonymous Student')
        attendance = float(data.get('attendance_rate', 75.0))
        study_hours = float(data.get('weekly_study_hours', 15.0))
        prior_gpa = float(data.get('prior_gpa', 3.0))
        assignment_rate = float(data.get('assignment_rate', 70.0))
        midterm_score = float(data.get('midterm_score', 65.0))
        sleep_hours = float(data.get('sleep_hours', 7.0))
        parental_edu = int(data.get('parental_education', 1))

        input_features = np.array([[
            attendance, study_hours, prior_gpa, assignment_rate, midterm_score, sleep_hours, parental_edu
        ]])

        # 1. Regress predicted final exam score
        if reg_model:
            pred_score = round(float(reg_model.predict(input_features)[0]), 1)
        else:
            pred_score = round(0.35 * attendance + 0.35 * midterm_score + 0.30 * (prior_gpa / 4.0 * 100), 1)

        # 2. Classify risk level
        if clf_model:
            risk_prob = float(clf_model.predict_proba(input_features)[0][1])
            is_risk = int(clf_model.predict(input_features)[0])
        else:
            risk_prob = 0.8 if (pred_score < 60 or attendance < 75) else 0.2
            is_risk = 1 if risk_prob > 0.5 else 0

        # 3. Real-time Anomaly Detection
        anomalies = []
        if attendance < 75.0:
            anomalies.append(f"University Attendance Shortage Warning: {attendance}% is below mandatory 75% threshold.")
        if prior_gpa >= 3.4 and midterm_score < 50.0:
            anomalies.append("Severe Academic Drop Anomaly: Student maintains high historical GPA but scored exceptionally low in midterm.")
        if study_hours > 28.0 and midterm_score < 55.0:
            anomalies.append("Effort-Result Disparity Anomaly: High self-study hours reported with low assessment outcome; potential comprehension bottleneck.")

        # Determine Grade and Recommended Action
        if pred_score >= 80:
            grade_category = "Distinction / First Class with Distinction"
            action = "Nominate for honors projects, research assistantships, or peer mentoring."
        elif pred_score >= 65:
            grade_category = "First Class"
            action = "Maintain current momentum and emphasize lab problem-solving exercises."
        elif pred_score >= 50:
            grade_category = "Second Class (Pass)"
            action = "Prescribe targeted weekly tutorial hours and doubt-clearing sessions."
        else:
            grade_category = "Critical Academic Risk / Potential Backlog"
            action = "Immediate faculty advisor intervention, parent advisory notice, and mandatory remedial classes."

        # Log to SQLite Database
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO prediction_audit_logs (
                student_name, attendance_rate, weekly_study_hours, prior_gpa,
                assignment_rate, midterm_score, predicted_score, risk_level,
                anomaly_flag, recommended_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            student_name, attendance, study_hours, prior_gpa, assignment_rate,
            midterm_score, pred_score, "High Risk" if is_risk else "Normal",
            1 if len(anomalies) > 0 else 0, action
        ))
        conn.commit()
        log_id = cursor.lastrowid
        conn.close()

        return jsonify({
            "status": "success",
            "log_id": log_id,
            "student_name": student_name,
            "predicted_score": pred_score,
            "grade_category": grade_category,
            "at_risk": bool(is_risk),
            "risk_probability": round(risk_prob * 100, 2),
            "anomaly_detected": len(anomalies) > 0,
            "anomalies": anomalies,
            "recommended_action": action
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/audit-logs', methods=['GET'])
def get_audit_logs():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM prediction_audit_logs ORDER BY logged_at DESC LIMIT 50")
    logs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify({"logs": logs, "count": len(logs)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Student Performance Prediction Flask Server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
