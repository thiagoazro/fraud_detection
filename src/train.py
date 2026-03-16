import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, confusion_matrix
)

import xgboost as xgb
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-svc:5000")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "fraud-xgb-mlops")
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "fraud-xgb")

# CSV existente (montado no container/pod)
DATA_PATH = os.environ.get("DATA_PATH", "/data/transactions_1M.csv")

SEED = int(os.environ.get("SEED", "42"))
TARGET_PRECISION = float(os.environ.get("TARGET_PRECISION", "0.90"))

# opcional: treinar só com uma amostra do CSV (para acelerar)
SAMPLE_N = int(os.environ.get("SAMPLE_N", "0"))  # 0 = usa tudo

# =========================
# HELPERS
# =========================
def pick_threshold_for_precision(y_true, y_prob, target_precision: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    best_thr = 0.5
    best_recall = -1.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= target_precision and r > best_recall:
            best_recall = r
            best_thr = float(t)
    return best_thr

def plot_cm(cm, path: str):
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV não encontrado em {csv_path}. "
            f"Monte o arquivo no container/pod e defina DATA_PATH, se necessário."
        )

    df = pd.read_csv(csv_path)

    required = {
        "amount","hour","dow","channel","international","new_merchant",
        "acct_age_days","txn_count_24h","txn_amount_24h","distance_km",
        "device_change","ip_risk","is_fraud"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV está sem colunas obrigatórias: {sorted(missing)}")

    if SAMPLE_N and SAMPLE_N > 0 and SAMPLE_N < len(df):
        df = df.sample(n=SAMPLE_N, random_state=SEED)

    return df

# =========================
# MAIN
# =========================
def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = load_dataset(DATA_PATH)

    y = df["is_fraud"].values
    X = df.drop(columns=["is_fraud"])

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    pos = max(1, int(ytr.sum()))
    neg = max(1, int((ytr == 0).sum()))
    scale_pos_weight = neg / pos

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": int(os.environ.get("MAX_DEPTH", "6")),
        "eta": float(os.environ.get("ETA", "0.08")),
        "subsample": float(os.environ.get("SUBSAMPLE", "0.9")),
        "colsample_bytree": float(os.environ.get("COLSAMPLE", "0.9")),
        "min_child_weight": float(os.environ.get("MIN_CHILD_WEIGHT", "1.0")),
        "lambda": float(os.environ.get("LAMBDA", "1.0")),
        "alpha": float(os.environ.get("ALPHA", "0.0")),
        "scale_pos_weight": float(scale_pos_weight),
        "seed": SEED,
    }

    with mlflow.start_run() as run:
        mlflow.log_param("data_path", DATA_PATH)
        mlflow.log_param("rows_used", len(df))
        mlflow.log_param("target_precision", TARGET_PRECISION)
        mlflow.log_param("sample_n", SAMPLE_N)

        for k, v in params.items():
            mlflow.log_param(k, v)

        dtr = xgb.DMatrix(Xtr, label=ytr)
        dte = xgb.DMatrix(Xte, label=yte)

        booster = xgb.train(
            params=params,
            dtrain=dtr,
            num_boost_round=int(os.environ.get("NUM_BOOST_ROUND", "300")),
        )

        yprob = booster.predict(dte)
        auprc = float(average_precision_score(yte, yprob))
        auroc = float(roc_auc_score(yte, yprob))

        thr = pick_threshold_for_precision(yte, yprob, TARGET_PRECISION)
        yhat = (yprob >= thr).astype(int)

        cm = confusion_matrix(yte, yhat)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))

        mlflow.log_metric("auprc", auprc)
        mlflow.log_metric("auroc", auroc)
        mlflow.log_metric("precision_at_thr", float(precision))
        mlflow.log_metric("recall_at_thr", float(recall))
        mlflow.log_metric("threshold", float(thr))

        os.makedirs("artifacts", exist_ok=True)
        cm_path = "artifacts/confusion_matrix.png"
        plot_cm(cm, cm_path)
        mlflow.log_artifact(cm_path)

        meta = {
            "features": list(X.columns),
            "rows_used": int(len(df)),
            "class_balance_train": {"pos": int(pos), "neg": int(neg)},
        }
        with open("artifacts/metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        mlflow.log_artifact("artifacts/metadata.json")

        mlflow.xgboost.log_model(booster, artifact_path="model")

        client = MlflowClient()
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=mv.version,
        )

        print("run_id:", run.info.run_id)
        print("registered:", MODEL_NAME, "version:", mv.version, "alias: champion")

if __name__ == "__main__":
    main()
