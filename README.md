# 🎓 Student Performance Prediction & Early Warning System

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-000000?style=for-the-badge&logo=flask&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-Frontend-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-Styling-1572B6?style=for-the-badge&logo=css3&logoColor=white)

---

## 📌 Executive Overview
An end-to-end machine learning web application built with **Flask**, **Scikit-Learn**, and **SQLite** to predict student academic performance and detect at-risk students before final examinations. 

By analyzing multi-dimensional behavioral and academic features—including **attendance percentage**, **weekly study habits**, **prior CGPA**, **midterm scores**, and **assignment completion**—the application achieves **96.25% classification accuracy** (achieved > 85% benchmark) in real-time academic risk and anomaly detection.

---

## 🚀 Key Features
- **🔮 Real-Time Score Prediction & Grade Classification**:
  - Predicts final examination percentage and grade tier (Distinction, First Class, Pass, Critical Risk).
  - Uses Random Forest Regressor & Classifier trained on 1,200 verified academic records.
- **🚨 Real-Time Anomaly & Deficit Detection**:
  - Flags students falling below university mandatory attendance thresholds (< 75%).
  - Detects grade drop anomalies (high historical CGPA vs poor midterm performance).
  - Alerts faculty on effort-to-outcome disparities.
- **🗄️ Relational SQLite Database Logging**:
  - Automatically logs every prediction into `prediction_audit_logs` table in `student_audit.db`.
  - Enables audit trails, longitudinal cohort analysis, and automated parent warning notifications.
- **🌐 Interactive Web Dashboard**:
  - Clean HTML/CSS/JavaScript interface with interactive gauge charts, batch roster inspection, and 1-click filter for weak students.

---

## 📊 Model Evaluation & Metrics
| Metric | Score Achieved | Benchmark |
| :--- | :--- | :--- |
| **Risk Classification Accuracy** | **96.25%** | > 85.0% Target |
| **ROC-AUC Score** | **98.89%** | Exceptional Discrimination |
| **Score Regression R² Score** | **83.73%** | High Predictive Power |

---

## 📂 Repository Structure
```
student-performance-prediction/
├── app.py                      # Flask Backend REST API & Server
├── train_model.py              # ML Model Training & SQLite Database Init
├── requirements.txt            # Python Dependencies
├── risk_classifier.pkl         # Trained Random Forest Risk Classifier
├── score_regressor.pkl         # Trained Random Forest Score Regressor
├── student_audit.db            # SQLite Relational Database
├── model_metrics.json          # Exported evaluation metrics & feature importances
├── students_data.json          # Class roster records JSON
├── index.html                  # Responsive Web Dashboard UI
├── app.js                      # Client-side dynamic interaction logic
└── README.md                   # Project Documentation
```

---

## 💻 Installation & Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Kasanaboina-Kowshik/student-performance-prediction.git
cd student-performance-prediction
pip install -r requirements.txt
```

### 2. (Optional) Re-train Machine Learning Models
```bash
python train_model.py
```

### 3. Launch the Flask Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 👤 Author
- **Kasanaboina Kowshik**
- **Education:** B.Tech in Computer Science (Artificial Intelligence), Parul University
- **Email:** [kasanaboinakowshik@gmail.com](mailto:kasanaboinakowshik@gmail.com)
- **Phone:** +91 8790211519
