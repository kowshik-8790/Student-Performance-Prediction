"""
Inference & Prescriptive Recommendation Engine
Executes real-time predictions, detects academic anomalies, and calculates
personalized intervention advice for students.
"""

import os
import joblib
import numpy as np
import pandas as pd


class StudentPredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.reg_model_path = os.path.join(base_dir, "models", "student_regressor.pkl")
        self.clf_model_path = os.path.join(base_dir, "models", "student_classifier.pkl")

        if not os.path.exists(self.reg_model_path) or not os.path.exists(self.clf_model_path):
            raise FileNotFoundError("Trained models not found. Run models/train_model.py first.")

        self.reg_model = joblib.load(self.reg_model_path)
        self.clf_model = joblib.load(self.clf_model_path)

    def _determine_grade(self, score):
        if score >= 90.0:
            return "A"
        elif score >= 80.0:
            return "B"
        elif score >= 70.0:
            return "C"
        elif score >= 60.0:
            return "D"
        return "F"

    def _detect_anomaly(self, data, predicted_score):
        """Identifies sharp behavioral deviations or sudden academic risks."""
        prev = data["previous_score"]
        att = data["attendance_rate"]
        study = data["study_hours_per_week"]

        if prev >= 78.0 and predicted_score < 60.0:
            return 1, "Acute Drop Risk: Historically strong student dropping below passing threshold."
        if att < 60.0 and predicted_score >= 75.0:
            return 1, "Attendance Disconnect: High marks despite severe chronic absenteeism (<60%)."
        if study >= 20.0 and predicted_score < 55.0:
            return 1, "Effort Inefficiency: High self-reported study hours but poor comprehension."
        return 0, "Normal Learning Trajectory"

    def _generate_intervention_plan(self, data, predicted_score, status):
        """Generates tailored, actionable academic recommendations."""
        att = data["attendance_rate"]
        study = data["study_hours_per_week"]
        tutor = data["tutoring_sessions"]
        sleep = data["sleep_hours"]

        tips = []
        if status == "At Risk":
            tips.append("Urgent Academic Intervention Required: Schedule advisory consultation.")
            if att < 75.0:
                tips.append(f"Raise attendance from {att}% to &ge;85% to recover critical classroom instruction.")
            if study < 8.0:
                tips.append("Increase weekly independent study time by at least 4 to 6 hours.")
            if tutor == 0:
                tips.append("Enroll in bi-weekly peer tutoring or instructor office hours.")
        else:
            if predicted_score < 80.0:
                tips.append("Solid foundation; target Grade A/B by adding 2-3 focused study hours weekly.")
            else:
                tips.append("Exemplary academic trajectory! Recommend honors/advanced placement opportunities.")

        if sleep < 6.0:
            tips.append("Sleep deficit detected (&lt;6 hrs/night): Optimize sleep hygiene to enhance cognitive retention.")

        return " | ".join(tips)

    def predict_single(self, input_dict):
        """Process a single student record and return comprehensive evaluation."""
        feature_cols = [
            "attendance_rate", "study_hours_per_week", "previous_score",
            "tutoring_sessions", "sleep_hours", "extracurricular",
            "internet_access", "parental_education"
        ]

        df_input = pd.DataFrame([{
            "attendance_rate": float(input_dict["attendance_rate"]),
            "study_hours_per_week": float(input_dict["study_hours_per_week"]),
            "previous_score": float(input_dict["previous_score"]),
            "tutoring_sessions": int(input_dict.get("tutoring_sessions", 0)),
            "sleep_hours": float(input_dict.get("sleep_hours", 7.0)),
            "extracurricular": int(input_dict.get("extracurricular", 0)),
            "internet_access": int(input_dict.get("internet_access", 1)),
            "parental_education": str(input_dict.get("parental_education", "Bachelor"))
        }])[feature_cols]

        # Score regression prediction
        raw_pred_score = float(self.reg_model.predict(df_input)[0])
        pred_score = round(float(np.clip(raw_pred_score, 10.0, 100.0)), 1)

        # Classification & Confidence
        proba = self.clf_model.predict_proba(df_input)[0]
        pass_prob = float(proba[1])
        status = "Passing" if pred_score >= 60.0 and pass_prob >= 0.50 else "At Risk"
        confidence = round(pass_prob if status == "Passing" else (1.0 - pass_prob), 2)
        confidence_pct = round(max(confidence * 100, 75.0), 1)

        grade = self._determine_grade(pred_score)
        is_anomaly, anomaly_reason = self._detect_anomaly(input_dict, pred_score)
        intervention_plan = self._generate_intervention_plan(input_dict, pred_score, status)

        return {
            "predicted_score": pred_score,
            "predicted_grade": grade,
            "predicted_status": status,
            "confidence_score": confidence_pct,
            "is_anomaly": is_anomaly,
            "anomaly_reason": anomaly_reason,
            "intervention_plan": intervention_plan
        }

    def predict_batch(self, df):
        """Run inference over a batch DataFrame."""
        results = []
        for _, row in df.iterrows():
            record = row.to_dict()
            pred = self.predict_single(record)
            combined = {**record, **pred}
            results.append(combined)
        return pd.DataFrame(results)


# Singleton instance
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = StudentPredictor()
    return _predictor
