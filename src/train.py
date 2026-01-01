import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set MLflow experiment
mlflow.set_experiment("churn-prediction")

# Load data
df = pd.read_csv("data/raw/churn.csv")

# Drop non-useful columns
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

X = df.drop(columns=["Exited"])
y = df["Exited"]

# Categorical & numerical columns
cat_cols = ["Geography", "Gender"]
num_cols = [c for c in X.columns if c not in cat_cols]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# Model
model = LogisticRegression(max_iter=500)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow run
with mlflow.start_run():
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    mlflow.log_param("model", "logistic_regression")
    mlflow.log_metric("roc_auc", auc)
    mlflow.sklearn.log_model(pipeline, "model")

    print("ROC AUC:", auc)

