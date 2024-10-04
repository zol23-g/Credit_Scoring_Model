# Credit Scoring Model for Bati Bank

## Overview
This project aims to develop a Credit Scoring Model for Bati Bank in collaboration with an eCommerce platform. The model will assess the creditworthiness of potential borrowers using transaction data.

## Business Need
Bati Bank seeks to enable a buy-now-pay-later service, allowing customers to purchase products on credit. The project involves creating a model to categorize users as high risk or low risk based on their transaction history.

## Key Objectives
- Define a proxy variable for risk categorization.
- Select observable features that predict default likelihood.
- Develop models to:
  - Assign risk probabilities.
  - Generate credit scores.
  - Predict optimal loan amounts and durations.

## Data Description
The dataset includes:
- **TransactionId**: Unique transaction identifier.
- **AccountId**: Unique customer identifier.
- **Amount**: Transaction value.
- **FraudResult**: Fraud status of the transaction.

## Learning Outcomes
- Advanced skills in scikit-learn and feature engineering.
- Knowledge of credit risk analysis and predictive modeling.
- Experience with CI/CD deployment of ML models.

## Installation
To set up the project, clone the repository and install the required packages:
```bash
git clone https://github.com/zol23-g/Credit_Scoring_Model.git
cd Credit_Scoring_Model
pip install -r requirements.txt
