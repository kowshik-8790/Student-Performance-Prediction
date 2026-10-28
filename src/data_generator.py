"""
Synthetic Student Performance Dataset Generator
Generates realistic multi-attribute student academic records with attendance,
study habits, demographics, and previous academic scores.
"""

import os
import random
import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)

NUM_STUDENTS = 2500

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Riley",
    "Casey", "Jamie", "Avery", "Logan", "Peyton", "Cameron", "Dakota",
    "Reese", "Skyler", "Rowan", "Hayden", "Kai", "Quinn", "Jesse", "Kendall",
    "Harper", "Rowan", "Sawyer", "Finley", "Emerson", "River", "Amari"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
]

PARENT_EDU_LEVELS = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
PARENT_EDU_WEIGHTS = [0.30, 0.25, 0.30, 0.12, 0.03]


def generate_student_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    records = []
    for i in range(1, NUM_STUDENTS + 1):
        student_id = f"STU-{10000 + i}"
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        full_name = f"{first_name} {last_name}"
        gender = random.choice(["Male", "Female"])
        age = random.randint(15, 19)

        parent_edu = np.random.choice(PARENT_EDU_LEVELS, p=PARENT_EDU_WEIGHTS)
        internet_access = 1 if random.random() < 0.90 else 0
        extracurricular = 1 if random.random() < 0.55 else 0

        # Study habits & previous scores
        study_hours = round(np.random.gamma(shape=3.5, scale=2.5), 1)  # Mean ~8.7 hrs
        study_hours = float(np.clip(study_hours, 1.0, 32.0))

        attendance = round(np.random.beta(a=8, b=2) * 100, 1)  # Mean ~80%
        attendance = float(np.clip(attendance, 35.0, 100.0))

        tutoring = int(np.random.poisson(lam=1.5))
        tutoring = int(np.clip(tutoring, 0, 8))

        sleep_hours = round(np.random.normal(loc=7.0, scale=1.1), 1)
        sleep_hours = float(np.clip(sleep_hours, 4.0, 10.0))

        # Previous score correlated with attendance & study habits
        base_ability = (attendance * 0.4) + (study_hours * 1.5) + np.random.normal(0, 8)
        previous_score = round(float(np.clip(base_ability, 30.0, 98.0)), 1)

        # Parent education impact weight
        edu_bonus = {
            "High School": 0, "Associate": 2, "Bachelor": 4, "Master": 6, "Doctorate": 8
        }[parent_edu]

        # Calculate final academic score (Realistic regression target)
        raw_final_score = (
            (previous_score * 0.45) +
            (attendance * 0.30) +
            (study_hours * 1.1) +
            (tutoring * 1.4) +
            (1.5 if extracurricular == 1 else 0.0) +
            (2.0 if internet_access == 1 else 0.0) +
            (1.5 if 6.5 <= sleep_hours <= 8.5 else -2.0) +
            edu_bonus +
            np.random.normal(0, 4.0)
        )
        final_score = round(float(np.clip(raw_final_score, 20.0, 100.0)), 1)

        # Pass / Fail classification threshold (60.0)
        passed = 1 if final_score >= 60.0 else 0

        # Grade scale
        if final_score >= 90.0:
            grade = "A"
        elif final_score >= 80.0:
            grade = "B"
        elif final_score >= 70.0:
            grade = "C"
        elif final_score >= 60.0:
            grade = "D"
        else:
            grade = "F"

        # Anomaly detection flag (e.g. high attendance + high study hours but failed, or high previous score with severe drop)
        is_anomaly = 0
        anomaly_reason = "Normal Pattern"
        if previous_score >= 80 and final_score < 60:
            is_anomaly = 1
            anomaly_reason = "Sudden High-Performing Drop (Acute Risk)"
        elif attendance < 55 and final_score >= 75:
            is_anomaly = 1
            anomaly_reason = "Unusual High Grade with Severe Absenteeism"

        records.append({
            "student_id": student_id,
            "full_name": full_name,
            "gender": gender,
            "age": age,
            "parental_education": parent_edu,
            "internet_access": internet_access,
            "attendance_rate": attendance,
            "study_hours_per_week": study_hours,
            "previous_score": previous_score,
            "tutoring_sessions": tutoring,
            "sleep_hours": sleep_hours,
            "extracurricular": extracurricular,
            "final_score": final_score,
            "grade": grade,
            "passed": passed,
            "is_anomaly": is_anomaly,
            "anomaly_reason": anomaly_reason
        })

    df = pd.DataFrame(records)
    dataset_path = os.path.join(data_dir, "student_performance_dataset.csv")
    df.to_csv(dataset_path, index=False)
    print(f"Dataset generated at {dataset_path} ({len(df)} records)")

    # Also generate a smaller sample batch for UI CSV testing
    sample_batch = df.sample(n=25, random_state=42)[
        ["student_id", "full_name", "attendance_rate", "study_hours_per_week", 
         "previous_score", "tutoring_sessions", "sleep_hours", "extracurricular", 
         "parental_education", "internet_access"]
    ]
    sample_batch_path = os.path.join(data_dir, "sample_batch_test.csv")
    sample_batch.to_csv(sample_batch_path, index=False)
    print(f"Sample test batch generated at {sample_batch_path}")

    return dataset_path


if __name__ == "__main__":
    generate_student_dataset()
