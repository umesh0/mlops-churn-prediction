# 🚀 End-to-End MLOps Churn Prediction Project

This repository contains an end-to-end MLOps project where a churn prediction
model is built, versioned, tracked, and reproduced using industry best practices.

Anyone can clone this repository and reproduce the results.

---

## 📌 Problem Statement
Predict whether a bank customer will churn based on historical customer data.

---

## 🏗️ Architecture Overview

- **AWS EC2** – Model training environment  
- **IAM Role** – Secure AWS access (no hardcoded credentials)  
- **DVC + S3** – Data versioning and remote storage  
- **MLflow** – Experiment tracking and model artifacts  
- **Scikit-learn** – Model training with preprocessing pipelines  

---


---

## ⚙️ End-to-End Setup (Exact Commands)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/umesh0/mlops-churn-prediction.git
cd mlops-churn-prediction

2️⃣ Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Install dependencies
pip install -r requirements.txt
If required:
pip install pandas scikit-learn mlflow dvc[s3]

📦 Data Versioning with DVC
The dataset is versioned using DVC and stored in Amazon S3.
dvc pull
This pulls the actual dataset from remote storage.

Model Training Pipeline
Run the training script:
python src/train.py

What happens:
Categorical features are encoded
Numerical features are passed through
Logistic Regression model is trained
Metrics and artifacts are logged to MLflow

Experiment Tracking with MLflow
Start MLflow UI:
mlflow ui --host 0.0.0.0 --port 5000 --allowed-hosts '*'

Open in browser:http://<EC2-PUBLIC-IP>:5000


Tracked in MLflow:

Metric: ROC-AUC

Parameters: model type

Artifacts: full preprocessing + model pipeline

📈 Results

ROC-AUC: ~0.76

Model pipeline saved as MLflow artifact

🔐 Security & Best Practices
IAM Roles used instead of AWS access keys
No secrets committed to GitHub
Local/generated files excluded using .gitignore
Dataset tracked using DVC pointers, not raw files

🧠 What I Learned
Building a complete end-to-end MLOps pipeline
Versioning large datasets using DVC + S3
Tracking experiments and models using MLflow
Handling categorical features using preprocessing pipelines
Debugging real-world MLflow issues (host headers, backend mismatch)
Writing clean, reproducible, and production-ready ML code
Structuring GitHub repositories for real-world usage

🏆 Key Takeaways
MLOps is about reproducibility and reliability
Source code should be separated from generated artifacts
Proper documentation is as important as correct code

👤 Author
Umesh Jaswal


## 📂 Project Structure

