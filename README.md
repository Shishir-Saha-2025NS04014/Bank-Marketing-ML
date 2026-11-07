# Bank Marketing: Term Deposit Subscription Prediction

ML Assignment 2 (M.Tech DSE, BITS Pilani)
Name: Shishir Saha
BITS ID: 2025NS04014

## a. Problem statement

A Portuguese bank runs phone marketing campaigns to sell term deposits. Calling every client is expensive, so the goal is to predict, from client and campaign details, whether a client will subscribe to a term deposit (`y` = yes/no). This is a binary classification problem. Five classifiers are trained on the same data, compared on six evaluation metrics, and served through a Streamlit app.

## b. Dataset description

- **Source:** UCI Machine Learning Repository, *Bank Marketing* (file `bank-full.csv`), from Moro, Cortez and Rita (2014).
- **Size:** 45,211 rows, 16 input features plus the target `y`.
- **Target:** `y` (yes/no). Only 11.7% of clients said yes, so the classes are imbalanced.
- **Numeric features (7):** `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`.
- **Categorical features (9):** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`.
- **Preprocessing:**
  - `duration` is dropped. Call length is only known after the call ends, so it leaks the outcome and would not be available when deciding whom to call. That leaves 15 features for modelling.
  - Numeric columns are standardised (`StandardScaler`) and categorical columns are one-hot encoded (`OneHotEncoder`), inside a scikit-learn `Pipeline` so the saved models accept raw CSV rows.
  - `"unknown"` values are kept as their own category.
- **Split:** 80/20 stratified train/test split (`random_state=42`), giving 9,043 test rows. All metrics below are on this held-out test set.
- **Note:** a majority-class predictor ("always no") scores about 88.3% accuracy on this data, so accuracy alone is misleading. Recall, F1, MCC and AUC matter more.

## c. GitHub repository link

<>

Streamlit app: https://bank-marketing-ml-shishir.streamlit.app/

## d. Models used

All models use the same preprocessing and the same train/test split. Metrics are computed at the default 0.5 decision threshold (AUC uses predicted probabilities, or the SVM decision function).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8933 | 0.7717 | 0.6632 | 0.1786 | 0.2815 | 0.3065 |
| Decision Tree (max_depth=6) | 0.8919 | 0.7136 | 0.6258 | 0.1881 | 0.2892 | 0.3022 |
| kNN (k=15) | 0.8921 | 0.7349 | 0.6475 | 0.1701 | 0.2695 | 0.2939 |
| Naive Bayes (Gaussian) | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine (RBF) | 0.8936 | 0.7251 | 0.6678 | 0.1805 | 0.2842 | 0.3097 |

Note: the SVM is trained on a stratified 10,000-row subsample of the training set because an RBF SVM is slow on 36,000 rows. All other models use the full training set.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.772), so it ranks likely subscribers better than any other model, and its accuracy is among the highest. At the 0.5 threshold it is cautious: precision is decent (0.66) but recall is only 0.18, so it misses most subscribers. Fast to train and easy to interpret. |
| Decision Tree | Limiting depth to 6 prevents overfitting, and it has the highest recall (0.19) of the four cautious models. Its AUC (0.714) is the lowest of the five, because a shallow tree produces only a few distinct probability values, which gives a coarse ranking. |
| kNN | Similar accuracy to the other models but the lowest recall (0.17) and F1. One-hot encoded categorical features make distances less meaningful, and the minority class gets outvoted by the majority among the 15 neighbours. AUC is middling (0.735). |
| Naive Bayes | Behaves very differently from the rest. Lowest accuracy (0.845) and precision (0.37), but by far the highest recall (0.46), F1 (0.41) and MCC (0.327). Its independence assumption is violated by correlated one-hot features, which makes it over-predict "yes". That hurts accuracy but catches more real subscribers. |
| Support Vector Machine | Highest accuracy and precision (0.894, 0.668) even though it was trained on only 10,000 rows, but recall is low (0.18) like the other cautious models. Its AUC (0.725) is below Logistic Regression's. MCC is the second-best (0.310). |
| **Overall, winner for this dataset?** | **Logistic Regression** is the best all-round choice: highest AUC, near-top accuracy, and the simplest and fastest model. If the business goal is to find as many subscribers as possible and wasted calls are cheap, **Naive Bayes** is better, since it has the best recall, F1 and MCC. Four of the five models are nearly tied on accuracy and MCC, and their low recall comes from the class imbalance and the 0.5 threshold. Lowering the threshold or using class weights would probably improve recall for all of them. |

## Running the project

```
pip install -r requirements.txt
python model/train.py --data bank-full.csv   # trains the models, writes model/*.joblib and test_data.csv
streamlit run app.py
```

`test_data.csv` is a 1,000-row random sample of the held-out test set, so the metrics shown in the app for it will differ slightly from the table above, which uses all 9,043 test rows.

## Project structure

```
app.py               Streamlit app (CSV upload, model dropdown, metrics, confusion matrix, report)
requirements.txt
README.md
test_data.csv        sample of the test set for uploading to the app
model/
    train.py         training and evaluation code
    *.joblib         saved models (all 5)
    metrics.csv      metrics table from the training run
```
