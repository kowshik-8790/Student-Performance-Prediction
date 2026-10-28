-- SQLite Database Schema for Student Performance Prediction

DROP TABLE IF EXISTS performance_predictions;
DROP TABLE IF EXISTS students;
DROP TABLE IF EXISTS system_logs;

CREATE TABLE students (
    student_id          VARCHAR(20) PRIMARY KEY,
    full_name           VARCHAR(100) NOT NULL,
    gender              VARCHAR(10) NOT NULL,
    age                 INTEGER NOT NULL,
    parental_education  VARCHAR(50) NOT NULL,
    internet_access     INTEGER NOT NULL DEFAULT 1,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE performance_predictions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id              VARCHAR(20) NOT NULL,
    attendance_rate         REAL NOT NULL,
    study_hours_per_week    REAL NOT NULL,
    previous_score          REAL NOT NULL,
    tutoring_sessions       INTEGER NOT NULL DEFAULT 0,
    sleep_hours             REAL NOT NULL,
    extracurricular         INTEGER NOT NULL DEFAULT 0,
    predicted_score         REAL NOT NULL,
    predicted_grade         VARCHAR(5) NOT NULL,
    predicted_status        VARCHAR(20) NOT NULL,
    confidence_score        REAL NOT NULL,
    is_anomaly              INTEGER NOT NULL DEFAULT 0,
    anomaly_reason          TEXT,
    intervention_plan       TEXT,
    created_at              DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

CREATE TABLE system_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  VARCHAR(50) NOT NULL,
    message     TEXT NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_prediction_status ON performance_predictions(predicted_status);
CREATE INDEX idx_prediction_student ON performance_predictions(student_id);
