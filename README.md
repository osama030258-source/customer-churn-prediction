# Customer Churn Prediction and Retention Analytics System

A machine learning solution that predicts customer churn using the IBM Telco Customer Churn dataset, with an interactive Streamlit dashboard for live predictions and business insights.

## 🎯 Project Overview

This project identifies customers at risk of churning and surfaces the key drivers behind that risk, enabling data-driven retention strategies. It covers the full ML lifecycle: data cleaning, EDA, feature engineering, model training, evaluation, and deployment.

## 📊 Dataset

- **Source:** [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)
- **Size:** 7,043 customers, 21 features
- **Target:** Churn (Yes/No) — 26.5% churn rate

## 🔧 Key Steps

### 1. Data Cleaning
- Fixed `TotalCharges` column (stored as string with blank values for 11 zero-tenure customers)
- Verified no duplicate records

### 2. Exploratory Data Analysis
- Churn is heavily concentrated among month-to-month contract customers, fiber optic internet users, and electronic check payers
- Churned customers have significantly lower tenure and higher monthly charges

### 3. Feature Engineering
Four custom features were built to improve predictive power:
| Feature | Description |
|---|---|
| `TenureSegment` | New / Regular / Loyal customer buckets |
| `SpendCategory` | Low / Medium / High monthly spend |
| `ServiceCount` | Total number of subscribed services |
| `ContractRiskScore` | Numeric risk score based on contract type (1–3) |

`ContractRiskScore` and `TenureSegment` both ranked in the top 7 most important features across models.

### 4. Modeling

Three classifiers were trained and evaluated:

| Model | Accuracy | Recall (Churn) | Precision (Churn) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.80 | 0.53 | 0.65 | 0.8414 |
| Random Forest | 0.76 | **0.78** | 0.53 | 0.8406 |
| XGBoost | 0.76 | 0.76 | 0.53 | 0.8362 |

All models were validated with 5-fold cross-validation (ROC-AUC consistently ~0.83–0.85).

**Key insight:** All three models have nearly identical discriminative power (ROC-AUC within 0.005 of each other). The performance differences come from class-weighting choices, not model quality. Random Forest and XGBoost use `class_weight`/`scale_pos_weight` to prioritize catching churners (higher recall), which better suits a retention use case where missing an at-risk customer is costlier than a wasted outreach.

### 5. Deployment

A Streamlit dashboard with three pages:
- **Churn Prediction** — live prediction form with model selection
- **Customer Insights** — interactive EDA visualizations
- **Model Performance** — side-by-side metrics comparison

## 🚀 Running the Project

```bash
# Clone the repo
git clone https://github.com/osama030258-source/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Mac/Linux

# Install dependencies
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost streamlit jupyter openpyxl

# Run the notebook
jupyter notebook notebooks/01_data_cleaning_eda.ipynb

# Run the dashboard
streamlit run app.py
```

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Deployment:** Streamlit
- **Environment:** Jupyter Notebook

## 📁 Project Structure
customer-churn-prediction/
├── app.py                          # Streamlit dashboard
├── data/
│   └── telco_churn.csv             # Raw dataset
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── notebooks/
│   └── 01_data_cleaning_eda.ipynb  # Full analysis pipeline
└── README.md
## 💡 Business Recommendations

- **Target month-to-month customers early** — they churn at ~43% vs. under 3% for two-year contracts. Incentivizing longer-term contracts (discounts, loyalty perks) could meaningfully reduce churn.
- **Watch new customers closely** — churn is highest (47%) in the first 12 months. An onboarding/engagement program in this window could improve retention.
- **Review fiber optic pricing/service quality** — this segment churns disproportionately despite being a premium offering.
- **Investigate the electronic check payment friction** — this payment method correlates with higher churn, possibly indicating a less-engaged or price-sensitive customer segment.

## 👤 Author

Osama Khan
- GitHub: [@osama030258-source](https://github.com/osama030258-source)