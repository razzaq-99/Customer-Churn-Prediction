# Customer-Churn-Prediction

**Predict which customers are likely to churn and provide actionable business insight to reduce churn**.
This repository contains the dataset, analysis notebook, trained model, and saved encoders used to build and evaluate the churn prediction solution.

---

# Project Overview

Telco customers who leave (churn) are expensive to replace. This project analyzes customer usage and billing data to predict churn, so that the business can proactively target at-risk customers with retention offers. The model is trained and tuned to maximize business-relevant metrics (e.g., recall and ROC-AUC), and the repo includes the notebook used for EDA, preprocessing and model training.

---

# Files in this Repo

* `Telco-Customer-Churn.csv` — Source dataset (raw).
* `customer_churn.ipynb` — Analysis & model-building notebook (run this to reproduce everything).
* `customer_churn_model.pkl` — Trained classifier (pickled model).
* `encoders.pkl` — Pickled encoders / preprocessing objects (used to transform categorical columns).
* `Corr_between_numerical_columns.png` — Correlation heatmap between numeric features (attached).
* `README.md` — (this file)

> Note: See `customer_churn.ipynb` for exact model type, preprocessing steps, and hyperparameters used (the notebook contains the full pipeline and training logs).

---

# Dataset

This Telco-style dataset includes key customer features such as:

* `tenure` — how long the customer has been with the company (months)
* `MonthlyCharges` — monthly bill amount
* `TotalCharges` — total bill since joining
* plus categorical features (contract type, payment method, services subscribed, etc.) and the target `Churn`.

---

# Exploratory Data Analysis — Key Findings (from `Corr_between_numerical_columns.png`)

The correlation heatmap (saved as `Corr_between_numerical_columns.png`) shows:

* **TotalCharges vs. tenure: high positive correlation (≈ 0.83)**
  → Customers with longer tenure have accumulated much higher total charges (expected: TotalCharges is roughly MonthlyCharges × tenure).
* **TotalCharges vs. MonthlyCharges: moderate positive correlation (≈ 0.65)**
  → Customers with higher monthly bills tend to have larger total charges (again intuitive).
* **tenure vs. MonthlyCharges: low correlation (≈ 0.25)**
  → Monthly billing amount does not strongly depend on how long a customer has been with the company.

**Implications**

* `TotalCharges` is heavily related to `tenure` and `MonthlyCharges` — check for multi-collinearity and correct datatype (sometimes `TotalCharges` is read as string in raw CSV).
* Low correlation between `tenure` and `MonthlyCharges` suggests churn drivers are not solely linked to pricing across tenure; pricing and tenure likely affect churn independently.

---

# Modeling & Pipeline (high level)

The repo follows a reproducible pipeline (see the notebook for exact code):

1. **Data cleaning**

   * Convert `TotalCharges` to numeric (handle spaces/missing values).
   * Impute or drop missing values as required.
2. **Feature engineering**

   * Derive / transform features (e.g., bin tenure, create interaction if needed).
3. **Categorical encoding**

   * Encode categorical features (encoders saved to `encoders.pkl`).
4. **Train / test split**

   * Stratified split to preserve churn ratio.
5. **Model selection & tuning**

   * A supervised classifier was trained and **hyperparameter tuning** was performed to improve performance (see notebook for the exact algorithm and hyperparameters).
6. **Evaluation**

   * Use multiple metrics: **Precision, Recall, F1-score, ROC-AUC**. For churn problems, **Recall (true positive rate)** and **ROC-AUC** are often prioritized because missing an at-risk customer (false negative) is costly.
7. **Save artifacts**

   * Final model was exported to `customer_churn_model.pkl`. Preprocessing encoders were exported to `encoders.pkl`.

---

# How to reproduce / run locally

```bash
# 1. clone repo
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# 2. create virtualenv (optional but recommended)
python -m venv venv
# on Windows:
venv\Scripts\activate
# on macOS / Linux:
source venv/bin/activate

# 3. install required packages
pip install -r requirements.txt
# If requirements.txt is not present, install at least:
# pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyterlab imbalanced-learn

```

---

# Conclusion & Business Recommendations:

**Model conclusion (summary):**

* A robust classifier has been trained and improved via hyperparameter tuning; artifacts are saved in `customer_churn_model.pkl` and `encoders.pkl`. Evaluation in the notebook uses business-focused metrics (ROC-AUC, precision/recall) and cross-validation to ensure stability.
* The EDA indicates `TotalCharges` is strongly tied to `tenure`, while `MonthlyCharges` plays an independent role. This helps interpret churn drivers: long-tenure customers have higher cumulative spend but may be retained differently than newer customers with high monthly bills.

**Business actions informed by the model & EDA:**

1. **Target short-tenure, high-risk customers** with onboarding campaigns — early churn is often preventable.
2. **Offer tailored discounts or payment flexibility** for customers with high `MonthlyCharges` who show at-risk signals — reducing the monthly burden can lower churn.
3. **Reward long-tenure customers** (loyalty perks) to keep lifetime value high — long-tenure customers already represent significant TotalCharges.
4. **Monitor feature drift** (billing changes, product mix) and retrain model regularly.

**Caveats & responsible use:**

* Evaluate model fairness on subgroups (e.g., geography, plan type) to avoid biased treatment.
* Don’t rely on a single metric — prefer a combination that reflects business cost of false positives vs false negatives.
* Deploy interventions A/B-tested to ensure retention tactics are effective and cost-efficient.

---

# Next steps / Improvements

* Add **SHAP** or LIME explanations for per-customer interpretability and to build trust with stakeholders.
* Create a **monitoring job**: track model performance and data drift in production; alert when retraining is needed.
* Test **ensemble methods** and calibration (Platt scaling / isotonic) if predicted probabilities are used to set offer thresholds.
* Explore usage / service features (e.g., number of support tickets, contract tenure) to increase predictive power.
* Build a small UI or API endpoint to serve predictions for the CRM or retention team.


