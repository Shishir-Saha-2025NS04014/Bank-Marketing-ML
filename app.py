import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Marketing Classifier", layout="wide")
st.title("Term Deposit Subscription Predictor")
st.caption("Upload test data, pick a model, and see how it performs.")

MODELS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib",
}

choice = st.selectbox("Choose a model", list(MODELS))
file = st.file_uploader("Upload CSV (same columns as bank-full, including `y`)", type="csv")


@st.cache_resource
def load(path):
    return joblib.load(path)


if file is not None:
    df = pd.read_csv(file, sep=None, engine="python").drop(columns=["duration"], errors="ignore")
    if "y" not in df.columns:
        st.error("The CSV needs the target column `y` (yes/no) to compute metrics.")
        st.stop()
    y = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["y"])
    pipe = load(MODELS[choice])
    pred = pipe.predict(X)
    clf = pipe.named_steps["clf"]
    score = pipe.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else pipe.decision_function(X)

    st.subheader(f"{choice}: evaluation on {len(df)} rows")
    c = st.columns(6)
    vals = [
        ("Accuracy", accuracy_score(y, pred)),
        ("AUC", roc_auc_score(y, score) if y.nunique() > 1 else float("nan")),
        ("Precision", precision_score(y, pred, zero_division=0)),
        ("Recall", recall_score(y, pred, zero_division=0)),
        ("F1", f1_score(y, pred, zero_division=0)),
        ("MCC", matthews_corrcoef(y, pred)),
    ]
    for col, (k, v) in zip(c, vals):
        col.metric(k, f"{v:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(confusion_matrix(y, pred), cmap="Blues")
        for (i, j), n in pd.DataFrame(confusion_matrix(y, pred)).stack().items():
            ax.text(j, i, n, ha="center", va="center")
        ax.set_xticks([0, 1], ["no", "yes"]); ax.set_yticks([0, 1], ["no", "yes"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        st.dataframe(pd.DataFrame(classification_report(
            y, pred, target_names=["no", "yes"], output_dict=True, zero_division=0)).T.round(3))
else:
    st.info("Upload `test_data.csv` to get started.")
    try:
        st.markdown("**Held-out test results for all models**")
        st.dataframe(pd.read_csv("model/metrics.csv"), hide_index=True)
    except FileNotFoundError:
        pass
