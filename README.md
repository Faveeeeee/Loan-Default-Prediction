# Loan Default Prediction

This project uses real-world LendingClub data (2007–2010) to predict whether a borrower will fully repay a loan. The core business objective is to minimize false negatives — avoiding situations where a risky borrower is mistakenly approved.
Because this is the most costly type of error, the project prioritizes Recall as the main evaluation metric.

**Problem Statement**

Lending institutions incur significant losses when borrowers fail to repay their loans. Our task is to predict default risk and build a model that can identify as many true defaulters as possible, preventing risky approvals.

##Approach

**Exploratory Data Analysis (EDA):**
Revealed that low FICO scores, high interest rates, recent credit inquiries, and small-business loan purposes were strong indicators of default. Revolving utilization and installment size also showed meaningful patterns.

**Preprocessing:**

One-hot encoding for loan purpose

Scaling for numeric features

Outlier treatment (log transform for skewed variables)

**Class Imbalance Handling:**
Compared SMOTE oversampling vs. class weights.
Class weights performed best for linear models such as Logistic Regression.

**Models Compared:**

Logistic Regression

Support Vector Machine

Random Forest

XGBoost

**Focus Metric:**
Recall — to minimize false negatives (approving borrowers who later default).

**Results**

Logistic Regression outperformed all other models on Recall, achieving ~0.57.
This made it the best option for the business goal of capturing as many defaulters as possible.

## Key Insights

Borrowers with FICO < 700 or small business loan purposes had 2–3× higher default risk.

High interest rates and multiple recent inquiries were strong risk indicators.

Income, installment size, and revolving utilization added additional predictive power.

Loan purpose plays a major role — especially small business loans.

## Tech Stack

Python (pandas, NumPy, matplotlib, seaborn)

scikit-learn (Logistic Regression, SVM, Random Forest, pipelines, preprocessing)

imbalanced-learn (class weights, SMOTE)

XGBoost (model comparison)

## Future Work

Improve Recall further through advanced feature engineering

Try ensemble methods (stacking, blending)

Conduct threshold tuning for better risk segmentation

Test on more recent datasets for generalization


