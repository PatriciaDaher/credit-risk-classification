# credit-risk-classification - README
# Module 20 Supervised Machine Learning 
## Patricia Daher


# Overview
This project aims to build a machine learning model to assess the creditworthiness of borrowers using historical lending data. The goal is to classify loans into two categories:

Healthy loans (0) - Low risk of default.
High-risk loans (1) - Likely to default.

Using logistic regression, we analyze loan features to predict risk, helping lenders make informed decisions.

Repository Structure
credit-risk-classification/  
│  
├── Credit_Risk/  
│   ├── credit_risk_classification.ipynb  # Jupyter Notebook
│   ├── Credit Risk Analysis Report.md  # Jupyter Notebook
│   └── Resources
│       └── lending_data.csv                 # Dataset     
├── README.md              # Project summary & results  
└── LICENSE.md 

# Steps Performed

1. Data Preparation
Dataset: lending_data.csv (historical loan data).
Features (X): Loan size, interest rate, borrower income, debt-to-income ratio, etc.
Target (y): loan_status (0 = healthy, 1 = high-risk).
Data Split:
75% training, 25% testing (train_test_split).

2. Logistic Regression Model

- Trained on original data.
- Evaluated using:
Confusion Matrix
Classification Report

# Results
Model Performance Metrics
Accuracy: 99%
Healthy Loans (0):
Precision: 1.00 (All predicted healthy loans were correct).
Recall: 0.99 (Captured 99% of actual healthy loans).
High-Risk Loans (1):
Precision: 0.84 (84% of predicted high-risk loans were correct).
Recall: 0.94 (Identified 94% of true high-risk loans).

# Technologies Methods and Algorithms: 

Python libraries: 
1- numpy 
2- pandas 
3- pathlib: Path 
4- sklearn.metrics: confusion_matrix 
5- sklearn.model_selection: train_test_split  
6- sklearn.linear_model: LogisticRegression 

1- For data loading used Pandas read_csv() to Read CSV into DataFrame. 
2- For data splitting used train_test_split to Split data into train/test sets. 
3- For classification model used Logistic Regression to Predict loan status (healthy/high-risk). 
4- For evaluation used Confusion Matrix to Quantify model errors. 
5- for evaluation used classification Report to summarize precision, recall, F1, support. 


# Key Findings 
With an accuracy of 99%, The model correctly predicts loan status (healthy/high risk) for 99% of the cases. 

The logistic regression model performed best in identifying healthy loans (Class 0) with high precision (1.00), meaning that all healthy loans were correctly predicted with no false positives and recall (0.99) meaning that the model captures 99% of actual healthy loans missing only 1%.  

On the other Hand, the model performed decently in identifying high risk loans. High risk loans (Class 1) with precision 0.84 means that 84% of predicted high risk loans were correct while 16% were false positive. Regarding recall, 94% of true high risk loans are identified, missing only 6% 

In conclusion, Although the model performs well for both classes, the model is slightly balanced towards predicting healthy loans.  

Performance does depend on the problem we are trying to solve.  
Business context matters. If our priority is minimizing risk, then catching all high-risk loans is important. If the priority is reducing false alarms, and avoiding all unnecessary loan denials, the precision of 0.84 could be improved.  

# License
General Public License