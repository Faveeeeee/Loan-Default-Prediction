# Loan Default Prediction

This project uses real-world lending data from LendingClub (2007–2010) to predict whether a borrower will fully repay a loan.
The key business goal is to help lenders reduce false positives — i.e., avoid approving loans to borrowers likely to default.

## Problem Statement

Lending institutions lose heavily when loans are issued to borrowers who don’t repay.
Our challenge: predict default risk and build a model that prioritizes precision, ensuring fewer risky borrowers are mistakenly approved.

## Approach

## Exploratory Data Analysis (EDA):

Found that low FICO scores (<700), high interest rates, and small business loans are the strongest risk indicators.

Outliers in revolving balance showed 2x higher default rates, highlighting financial strain signals.

## Preprocessing:

One-hot encoding for categorical features (purpose), scaling for numerical features.

Outlier treatment (log transform on skewed features).

## Class Imbalance Handling:

Compared class weights vs. SMOTE oversampling.

## Final choice: 
class weights for Logistic/SVM, SMOTE for tree-based models.

## Models Compared:

Logistic Regression, Support Vector Machine, Random Forest, XGBoost.

## Focus metric: Precision (to minimize false positives).

## Results

XGBoost (tuned) gave the best balance with ~0.55 precision, outperforming Logistic Regression (~0.40) and Random Forest (~0.33).

## Key insights for lenders:

Borrowers with FICO < 700 or small business loans had 2–3x higher chance of default.

High interest rates and multiple credit inquiries further increased risk.

# Tech Stack

Python (pandas, NumPy, matplotlib, seaborn)

Scikit-learn (Logistic Regression, SVM, Random Forest, pipelines, preprocessing)

XGBoost

imblearn (SMOTE)

## Future Work

Ensemble methods (stacking/blending) to push precision higher.

More advanced feature engineering (interaction terms, time-based features).

Evaluation on more recent datasets for generalization.
