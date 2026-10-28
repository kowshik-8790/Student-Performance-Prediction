"""
SQLite Database Helper Module
Manages student records, prediction history, analytics aggregates, and system logs.
"""

import os
import sqlite3
import pandas as pd


def get_db_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "database", "student_records.db")


def get_schema_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "database", "schema.sql")


def init_database():
    """Initialise SQLite database with tables and sample records."""
    db_path = get_db_path()
    schema_path = get_schema_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    with open(schema_path, "r") as f:
        conn.executescript(f.read())

    # Pre-populate with first 60 rows from generated dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "data", "student_performance_dataset.csv")

    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path).head(60)
        cur = conn.cursor()

        for _, row in df.iterrows():
            cur.execute("""
                INSERT OR IGNORE INTO students 
                (student_id, full_name, gender, age, parental_education, internet_access)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                row["student_id"], row["full_name"], row["gender"],
                int(row["age"]), row["parental_education"], int(row["internet_access"])
            ))

            status = "Passing" if row["passed"] == 1 else "At Risk"
            cur.execute("""
                INSERT INTO performance_predictions
                (student_id, attendance_rate, study_hours_per_week, previous_score,
                 tutoring_sessions, sleep_hours, extracurricular, predicted_score,
                 predicted_grade, predicted_status, confidence_score, is_anomaly,
                 anomaly_reason, intervention_plan)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["student_id"], float(row["attendance_rate"]), float(row["study_hours_per_week"]),
                float(row["previous_score"]), int(row["tutoring_sessions"]), float(row["sleep_hours"]),
                int(row["extracurricular"]), float(row["final_score"]), row["grade"], status,
                round(0.85 + (float(row["final_score"]) % 10) * 0.01, 2),
                int(row["is_anomaly"]), row["anomaly_reason"],
                "Standard academic monitoring" if status == "Passing" else "Prescribe peer tutoring and weekly check-in."
            ))

        conn.commit()

    conn.close()
    print(f"[OK] Database initialized at {db_path}")


def save_prediction(data):
    """Save a real-time student prediction to SQLite database."""
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()

    # Ensure student profile exists
    cur.execute("""
        INSERT OR IGNORE INTO students 
        (student_id, full_name, gender, age, parental_education, internet_access)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        data["student_id"], data["full_name"], data.get("gender", "Unspecified"),
        int(data.get("age", 17)), data["parental_education"], int(data["internet_access"])
    ))

    cur.execute("""
        INSERT INTO performance_predictions
        (student_id, attendance_rate, study_hours_per_week, previous_score,
         tutoring_sessions, sleep_hours, extracurricular, predicted_score,
         predicted_grade, predicted_status, confidence_score, is_anomaly,
         anomaly_reason, intervention_plan)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["student_id"], float(data["attendance_rate"]), float(data["study_hours_per_week"]),
        float(data["previous_score"]), int(data["tutoring_sessions"]), float(data["sleep_hours"]),
        int(data["extracurricular"]), float(data["predicted_score"]), data["predicted_grade"],
        data["predicted_status"], float(data["confidence_score"]), int(data["is_anomaly"]),
        data["anomaly_reason"], data["intervention_plan"]
    ))

    pred_id = cur.lastrowid
    conn.commit()
    conn.close()
    return pred_id


def get_prediction_history(limit=50):
    """Retrieve historical prediction logs with student profiles."""
    conn = sqlite3.connect(get_db_path())
    query = """
        SELECT 
            p.id, p.student_id, s.full_name, p.attendance_rate, p.study_hours_per_week,
            p.previous_score, p.predicted_score, p.predicted_grade, p.predicted_status,
            p.confidence_score, p.is_anomaly, p.anomaly_reason, p.intervention_plan, p.created_at
        FROM performance_predictions p
        JOIN students s ON p.student_id = s.student_id
        ORDER BY p.id DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()
    return df.to_dict(orient="records")


def get_dashboard_summary():
    """Retrieve high-level metrics for dashboard cards and charts."""
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM performance_predictions")
    total_predictions = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM performance_predictions WHERE predicted_status = 'Passing'")
    passing_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM performance_predictions WHERE predicted_status = 'At Risk'")
    risk_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM performance_predictions WHERE is_anomaly = 1")
    anomaly_count = cur.fetchone()[0]

    cur.execute("SELECT AVG(predicted_score) FROM performance_predictions")
    avg_score = round(cur.fetchone()[0] or 0.0, 1)

    cur.execute("SELECT AVG(attendance_rate) FROM performance_predictions")
    avg_attendance = round(cur.fetchone()[0] or 0.0, 1)

    # Grade distribution
    cur.execute("""
        SELECT predicted_grade, COUNT(*) 
        FROM performance_predictions 
        GROUP BY predicted_grade 
        ORDER BY predicted_grade
    """)
    grade_distribution = dict(cur.fetchall())

    conn.close()

    pass_rate = round((passing_count / total_predictions) * 100, 1) if total_predictions > 0 else 0.0

    return {
        "total_predictions": total_predictions,
        "passing_count": passing_count,
        "risk_count": risk_count,
        "anomaly_count": anomaly_count,
        "pass_rate": pass_rate,
        "avg_score": avg_score,
        "avg_attendance": avg_attendance,
        "grade_distribution": grade_distribution
    }


if __name__ == "__main__":
    init_database()
