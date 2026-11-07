"""Train 5 classifiers on the Bank Marketing dataset and save models + metrics.

Usage: python train.py --data path/to/bank-full.csv
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
DROP = ["duration"]  # known only after the call ends -> leakage
TARGET = "y"


def build_pipe(model, num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("prep", pre), ("clf", model)])


def score(pipe, X, y):
    pred = pipe.predict(X)
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        s = pipe.predict_proba(X)[:, 1]
    else:
        s = pipe.decision_function(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, s),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }


def main(path):
    df = pd.read_csv(path, sep=None, engine="python")  # auto-detects ';'
    df = df.drop(columns=DROP, errors="ignore")
    y = (df[TARGET] == "yes").astype(int)
    X = df.drop(columns=[TARGET])
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    print(f"{len(df)} rows, {X.shape[1]} features, positive rate {y.mean():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # SVM and kNN scale badly with rows -> SVM is fit on a stratified subsample
    X_svm, _, y_svm, _ = train_test_split(
        X_tr, y_tr, train_size=min(10000, len(X_tr) // 2), stratify=y_tr, random_state=SEED)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "kNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", C=1.0, random_state=SEED),
    }

    rows = {}
    for name, m in models.items():
        pipe = build_pipe(m, num_cols, cat_cols)
        if name == "SVM":
            pipe.fit(X_svm, y_svm)
        else:
            pipe.fit(X_tr, y_tr)
        rows[name] = score(pipe, X_te, y_te)
        joblib.dump(pipe, f"model/{name.lower().replace(' ', '_')}.joblib")
        print(name, {k: round(v, 4) for k, v in rows[name].items()})

    pd.DataFrame(rows).T.round(4).to_csv("model/metrics.csv", index_label="Model")

    # small upload file for the Streamlit app (raw columns + target)
    test = X_te.copy()
    test[TARGET] = np.where(y_te == 1, "yes", "no")
    test.sample(n=min(1000, len(test)), random_state=SEED).to_csv("test_data.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    main(ap.parse_args().data)
