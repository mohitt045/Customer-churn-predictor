# Customer Churn Prediction

A machine learning system that flags customers likely to cancel their service, so retention teams can intervene proactively instead of reactively. The project runs end to end: raw CSV → EDA → model selection → a deployed Streamlit app that returns an instant churn prediction from four inputs.

**Final model:** linear Support Vector Classifier (`SVC(C=0.01, kernel='linear')`) — **90.5% accuracy** on a held-out test set.

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Dataset](#dataset)
- [Exploratory Findings](#exploratory-findings)
- [Modeling](#modeling)
- [Final Model](#final-model)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

Churn — the loss of a paying customer — is one of the most expensive problems a subscription business faces, since acquiring a new customer costs far more than keeping an existing one. This project answers one operational question:

> *Given basic information we already hold about a customer, can we predict — before they leave — whether they are at risk of churning?*

The pipeline covers data cleaning, exploratory analysis, feature engineering, training and tuning five candidate classifiers, and packaging the winner behind a Streamlit UI that a non-technical user can operate.

---

## Demo

The Streamlit app (`app.py`) takes four inputs and returns a prediction instantly:

| Input | Widget | Range |
|---|---|---|
| Age | Number input | 18–100 |
| Tenure (months) | Number input | 0–130 |
| Monthly Charges | Number input | ≥ $30 |
| Gender | Dropdown | Male / Female |

Click **Predict Churn** → the app encodes gender, applies the saved `StandardScaler`, and prints **"Churn"** or **"Not Churn"**.

---

## Dataset

`customer_churn_data.csv` — 1,000 customer records across 10 columns.

| Column | Type | Description |
|---|---|---|
| CustomerID | Integer | Unique identifier |
| Age | Integer | Customer age in years (12–83) |
| Gender | Categorical | Male / Female |
| Tenure | Integer | Months the account has been held (0–122) |
| MonthlyCharges | Float | Recurring monthly bill ($30.00–$119.96) |
| ContractType | Categorical | Month-to-Month / One-Year / Two-Year |
| InternetService | Categorical | DSL / Fiber Optic (297 missing) |
| TotalCharges | Float | Cumulative amount billed to date |
| TechSupport | Categorical | Subscribes to tech support (Yes/No) |
| **Churn** | Categorical | **Target** — Yes / No |

**Data quality notes**

- **Class balance:** 883 records (88.3%) are `Churn = Yes`, 117 (11.7%) are `Churn = No` — materially imbalanced (see [Limitations](#limitations)).
- **Missing values:** `InternetService` missing for 297 of 1,000 rows (29.7%), filled with an empty placeholder rather than dropped to preserve all records. All other columns complete.
- **Duplicates:** none found.
- **Encoding:** `Gender` label-encoded (Female = 1, Male = 0); `Churn` label-encoded (Yes = 1, No = 0).

---

## Exploratory Findings

| Metric | Churn = No | Churn = Yes |
|---|---|---|
| Average Monthly Charges | $62.55 | $75.96 |
| Average Tenure (months) | 30.3 | 17.5 |
| Average Age (years) | 43.5 | 44.8 |

- **Tenure is the strongest visible signal** — churned customers averaged 17.5 months versus 30.3 for retained ones. Newer customers leave at a noticeably higher rate.
- **Price sensitivity** — churned customers paid roughly $13 more per month on average.
- **Age barely differentiates** — 44.8 vs. 43.5 years.
- **Contract mix** — 511 Month-to-Month, 289 One-Year, 200 Two-Year. Average monthly charges fall with longer terms ($75.91 → $73.82 → $71.33), suggesting a modest loyalty discount.
- **Correlation** — `TotalCharges` correlates strongly with `Tenure` (r = 0.89), as expected. No other strong linear correlations, so multicollinearity is not a concern for the chosen feature set.

---

## Modeling

**Feature set:** `Age`, `Gender`, `Tenure`, `MonthlyCharges` — chosen for being available at the moment a churn score is needed and simple enough for a first production model.

**Preprocessing:** 80/20 train-test split (`train_test_split`), then `StandardScaler` on all four features so `MonthlyCharges` doesn't dominate margin- and distance-based algorithms. The fitted scaler is exported as `scaler.pkl`.

**Candidates:** five classifiers on the same split, four tuned with 5-fold `GridSearchCV`, scored with `accuracy_score` on the held-out set.

| Algorithm | Tuning | Test Accuracy |
|---|---|---|
| **Logistic Regression** | Baseline, default params | **90.5%** |
| **Support Vector Machine** | `C: [0.01, 0.1, 0.5, 1]`, `kernel: [linear, rbf, poly]` → best `C=0.01, kernel=linear` | **90.5%** |
| K-Nearest Neighbors | `n_neighbors: [3,5,7,9]`, `weights: [uniform, distance]` → best `n_neighbors=5, weights=distance` | 89.5% |
| Random Forest | `n_estimators: [32,64,128,256]`, `max_features`, `bootstrap` → best `n_estimators=128, max_features=sqrt, bootstrap=True` | 89.5% |
| Decision Tree | `criterion`, `splitter`, `max_depth`, `min_samples_split/leaf` | 84.0% |

Logistic Regression and the tuned linear SVM tied at the top. The single Decision Tree trailed at 84.0%, consistent with overfitting on a modest four-feature dataset.

---

## Final Model

```python
from sklearn.svm import SVC
model = SVC(C=0.01, kernel='linear')   # GridSearchCV best_estimator_
```

Chosen over the equally accurate Logistic Regression baseline because:

- Strong regularization (low `C`) favors a wide, simple margin, which generalizes better on 1,000 records.
- A linear kernel keeps the decision boundary interpretable — risk rises smoothly with a weighted combination of inputs.
- It matched every other candidate on accuracy, so the simpler, more stable option won.

**Inference contract**

- Input: feature vector in the exact order `[Age, Gender, Tenure, MonthlyCharges]`
- `Gender` encoded as Female = 1, Male = 0
- Must be transformed with `scaler.pkl` before prediction — the model was trained on standardized features only
- Output: `1` = predicted churn, `0` = predicted retention

`model.pkl` and `scaler.pkl` are a **matched pair**. Retraining requires re-exporting both together.

---

## Project Structure

```
.
├── app.py                      # Streamlit UI + inference logic
├── notebook.ipynb              # EDA, preprocessing, training, tuning
├── customer_churn_data.csv     # Source dataset (1,000 records)
├── model.pkl                   # Trained linear SVM
├── scaler.pkl                  # Fitted StandardScaler
├── requirements.txt
└── README.md
```

---

## Getting Started

**Prerequisites:** Python 3.9+

```bash
# Clone the repo
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt`:

```
streamlit
scikit-learn
pandas
numpy
joblib
matplotlib
seaborn
```

---

## Usage

**Run the web app**

```bash
streamlit run app.py
```

Then open `http://localhost:8501`.

**Reproduce the analysis**

```bash
jupyter notebook notebook.ipynb
```

**Predict from Python**

```python
import joblib
import numpy as np

model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# [Age, Gender (F=1/M=0), Tenure (months), MonthlyCharges]
x = np.array([[42, 1, 8, 89.50]])
pred = model.predict(scaler.transform(x))[0]

print("Churn" if pred == 1 else "Not Churn")
```

---

## Limitations

Read these before using the model for real retention decisions.

- **Severe class imbalance.** 88.3% of the training data is labeled churned. A model that predicted "churn" for everyone would score ~88% accuracy, so 90.5% is a weaker result than it appears. Accuracy is the wrong headline metric here.
- **Accuracy-only evaluation.** No precision, recall, F1, ROC-AUC, or confusion matrix was recorded. The false-negative rate — churners the model misses — is unknown, and that is the number retention teams actually care about.
- **Narrow feature set.** `ContractType`, `InternetService`, `TechSupport`, and `TotalCharges` were explored during EDA but excluded from the model. Contract type in particular is a well-known churn driver and is likely leaving signal on the table.
- **Small dataset.** 1,000 records, 200 of which form the test set. Metric estimates carry wide confidence intervals.
- **Data-quality flags.** `Age` includes values as low as 12, which is implausible for an account holder and suggests entry errors. Nearly 30% of `InternetService` values are missing.
- **No monitoring.** There is no drift detection or scheduled re-evaluation once the model is in use.

---

## Roadmap

- [ ] Re-evaluate with precision, recall, F1, ROC-AUC and a confusion matrix, and set the decision threshold from business cost rather than the 0.5 default
- [ ] Handle imbalance — class weights, SMOTE, or stratified resampling — and re-benchmark all five candidates
- [ ] Add `ContractType`, `TechSupport`, and `InternetService` as one-hot features and measure the lift
- [ ] Wrap preprocessing and the estimator in a single `sklearn.Pipeline` so the scaler can never drift out of sync with the model
- [ ] Return a churn *probability* alongside the label so teams can rank outreach by risk
- [ ] Validate input ranges and clean implausible `Age` values
- [ ] Deploy to Streamlit Community Cloud and add a monitoring/retraining schedule

---

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn` · `joblib` · `Streamlit` · `Jupyter`

---

## License

Released under the MIT License. See `LICENSE` for details.
