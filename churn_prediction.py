"""
Customer Churn Prediction
=========================
End-to-end machine-learning pipeline that predicts whether a telecom
customer will churn (cancel their subscription).

Dataset : IBM Telco Customer Churn (IBM sample data, 7,043 customers)
Source  : https://github.com/IBM/telco-customer-churn-on-icp4d
          (file: data/Telco-Customer-Churn.csv)

Pipeline steps
--------------
1. Load and clean the data (TotalCharges -> numeric, drop id column).
2. Quick exploratory summary (shape, churn rate, missing values).
3. Preprocessing: one-hot encoding for categoricals, scaling for numerics
   (sklearn ColumnTransformer inside a Pipeline, so no leakage).
4. Stratified train/test split (80/20).
5. Train three models: Logistic Regression, Random Forest, XGBoost.
6. Evaluate each on the held-out test set
   (accuracy, precision, recall, F1, ROC-AUC, confusion matrix).
7. Select the best model by F1-score, save it with joblib and write
   all metrics to results/metrics.json.

Run:  python churn_prediction.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # XGBoost is optional
    HAS_XGB = False

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "Telco-Customer-Churn.csv")
DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RESULTS_PATH = os.path.join("results", "metrics.json")
MODEL_PATH = os.path.join("models", "best_churn_model.pkl")


# ----------------------------------------------------------------------
# 1. Load & clean
# ----------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load the Telco churn CSV (download it first if it is missing)."""
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found locally - downloading from:\n  {DATA_URL}")
        os.makedirs("data", exist_ok=True)
        downloaded = pd.read_csv(DATA_URL)
        downloaded.to_csv(DATA_PATH, index=False)
    df = pd.read_csv(DATA_PATH)

    # TotalCharges arrives as text; a handful of brand-new customers
    # (tenure = 0) have a blank value there -> coerce to NaN, then drop.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["TotalCharges"])
    print(f"Dropped {before - len(df)} rows with missing TotalCharges")

    # customerID is an identifier, not a feature.
    df = df.drop(columns=["customerID"])
    # Binary target: Yes -> 1, No -> 0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


# ----------------------------------------------------------------------
# 2. EDA summary
# ----------------------------------------------------------------------
def eda_summary(df: pd.DataFrame) -> dict:
    churn_rate = df["Churn"].mean()
    print("\n--- EDA summary ---")
    print(f"Shape                 : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Churned (1)           : {int(df['Churn'].sum())} customers")
    print(f"Stayed  (0)           : {int((df['Churn'] == 0).sum())} customers")
    print(f"Churn rate            : {churn_rate:.2%}")
    print(f"Tenure (months)       : min {df['tenure'].min()}, "
          f"median {df['tenure'].median()}, max {df['tenure'].max()}")
    print(f"MonthlyCharges (avg)  : churners "
          f"{df.loc[df.Churn == 1, 'MonthlyCharges'].mean():.2f} vs "
          f"stayers {df.loc[df.Churn == 0, 'MonthlyCharges'].mean():.2f}")
    print(f"Missing values total  : {int(df.isna().sum().sum())}")
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "churn_rate": round(float(churn_rate), 4),
        "churned_customers": int(df["Churn"].sum()),
    }


# ----------------------------------------------------------------------
# 3-5. Preprocessing + model training
# ----------------------------------------------------------------------
def build_pipeline(model, numeric_features, categorical_features) -> Pipeline:
    """Preprocessing (impute/encode/scale) + classifier in one pipeline."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate(model, X_test, y_test) -> dict:
    """Compute the full metric set for one trained model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def main() -> None:
    df = load_data()
    dataset_info = eda_summary(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical_features = [c for c in X.columns if c not in numeric_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # Churn is imbalanced (~26.5% positive), so class weights / scale
    # factors are used to keep the models honest about the minority class.
    neg, pos = np.bincount(y_train)
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=5,
            scale_pos_weight=neg / pos,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    results = {}
    trained = {}
    print("\n--- Model results (test set) ---")
    for name, model in models.items():
        pipe = build_pipeline(model, numeric_features, categorical_features)
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        results[name] = metrics
        trained[name] = pipe
        print(
            f"{name:20s} acc={metrics['accuracy']:.4f}  "
            f"prec={metrics['precision']:.4f}  rec={metrics['recall']:.4f}  "
            f"f1={metrics['f1_score']:.4f}  roc_auc={metrics['roc_auc']:.4f}"
        )

    # Best model = highest F1 on the test set (a balanced summary for an
    # imbalanced problem, where raw accuracy would be misleading).
    best_name = max(results, key=lambda n: results[n]["f1_score"])
    print(f"\nBest model by F1-score: {best_name} "
          f"(F1 = {results[best_name]['f1_score']:.4f})")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: "
          f"{results[best_name]['confusion_matrix']}")

    # Top features from the Random Forest, useful for a churn story.
    rf = trained["Random Forest"]
    feature_names = rf.named_steps["preprocessor"].get_feature_names_out()
    importances = rf.named_steps["model"].feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    top_features = {
        str(feature_names[i]): round(float(importances[i]), 4) for i in top_idx
    }
    print("\nTop 10 Random Forest features:")
    for feat, imp in top_features.items():
        print(f"  {feat:45s} {imp:.4f}")

    # ------------------------------------------------------------------
    # 6. Persist results + model artifact
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    report = {
        "project": "Customer Churn Prediction",
        "dataset": {
            "name": "IBM Telco Customer Churn",
            "source_url": DATA_URL,
            **dataset_info,
        },
        "split": {
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "test_fraction": 0.20,
            "random_state": RANDOM_STATE,
        },
        "models": results,
        "best_model": best_name,
        "best_model_metrics": results[best_name],
        "top_features_random_forest": top_features,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    joblib.dump(trained[best_name], MODEL_PATH)
    print(f"\nSaved metrics -> {RESULTS_PATH}")
    print(f"Saved best model ({best_name}) -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
