# Customer Churn Risk & Retention Analytics

## Project Overview
This project is the foundational deliverable for a business analytics initiative aimed at predicting and preventing customer churn for a European retail bank.

The primary goal is to move from a reactive to a proactive customer retention strategy by building a machine learning model that identifies at-risk customers. The final solution will be a predictive dashboard integrated into the bank's CRM, enabling targeted retention campaigns.

This repository contains the code, documentation, and project management artifacts for the entire analytics lifecycle.

## Business Problem
The bank is experiencing an 8% year-over-year increase in customer churn, leading to significant revenue loss and high customer acquisition costs. The core challenge is the inability to proactively identify and engage customers who are likely to leave.

## Project Goals
- **Predictive Accuracy:** Develop a churn prediction model with an ROC-AUC score > 0.85.
- **Business Impact:** Reduce overall customer churn by 5% through targeted interventions.
- **Operational Efficiency:** Increase the response rate of retention campaigns by 20%.
- **Model Interpretability:** Ensure model outputs are understandable to business stakeholders using SHAP/LIME.

## Dataset
- **Source:** `Bank_churn_modelling.csv` (Kaggle)
- **Description:** 10,000 records with 13 features, including demographics, account details, and behavioral indicators. The target variable `Exited` indicates customer churn.
- **Suitability:** Realistic, adequately sized for robust machine learning, and ideal for a banking analytics project.

## Repository Structure
```
churn-analytics-project/
├── .github/
├── .gitignore
├── README.md
├── LICENSE
├── requirements.txt # Python dependencies
├── environment.yml # For Conda environment
├── data/
│ ├── raw/ # Immutable raw data (e.g., bank_churn_data.csv)
│ ├── processed/ # Cleaned data ready for modeling (tracked by DVC)
│
├── src/ # Core source code
│
└── docs/ # Project documentation
```


## Technical Stack

| Phase | Tool |
| :--- | :--- |
| **Cloud Platform** | AWS or Azure |
| **Data Pipeline & Storage** | AWS S3, AWS Glue/Spark, Snowflake/Redshift |
| **Modeling & Analysis** | Python (Scikit-learn, XGBoost, PyTorch), Jupyter Notebooks |
| **Model Monitoring & MLOps** | MLflow, Seldon Core |
| **Dashboard & Visualization** | Tableau, Power BI, or Streamlit |
| **Deployment Mechanism** | Docker, Kubernetes, or Serverless (AWS Lambda/Azure Functions) |

## Getting Started

### Prerequisites
- Python 3.8+
- Conda or Pipenv (recommended)

### Installation

1.  **Clone the repository:**
 ```
git clone https://github.com/Ebenezer-ebu/churn-analytics-project.git
cd churn-analytics-project
```

```
pip install -r requirements.txt
```

### Authors
- Ebenezer Ifezulike - Initial work - [Ebenezer-ebu]

### Acknowledgments
- Kaggle for the Bank_churn_modelling.csv dataset.
- NXU for the project framework and guidance.
- Open-source libraries: Scikit-learn, XGBoost, Pandas, Matplotlib, SHAP.
