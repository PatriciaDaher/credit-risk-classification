# Module 20 Report 
# Supervised Machine Learning 

## Overview of the Analysis 
The purpose of this analysis is to build a machine learning model to assess the credit risk for borrowers using dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.  

Specifically, the goal is to predict whether a loan was "healthy" (represented by a 0) or "high-risk" (represented by a 1) based on the following financial variables: 
1- loan_size: The size of the loan requested  
2- interest_rate: THe interest rate charged  
3- borrower_income: THe borrower's annual income 
4- dept_to_income: The ratio of the borrower's debt to income 
5- num_of_accounts: Number of Open Credit Accounts 
6- derogatory_marks: Number of negative marks on a borrower's credit 
7- total_debt: Total current debt of the borrower 
8- loan_status: The target variable indicating loan status (0= healthy, 1= high-risk) 

The first inspection of the loan_status variable showed a class imbalance with significantly more 0s (healthy loans) than 1s (high risk loans. 

 

### Machine learning process steps: 

### Part 1: Split the Data into Training and Testing Sets 

- Step 1: We upload, read and explored the `lending_data.csv` data. 

- Step 2: We separate the labels set (`y`) from the “loan_status” column, and then create the features (`X`) DataFrame from the remaining columns. 

- Step 3: Split the data into training and testing datasets by using `train_test_split`. 

### Part 2: Created a Logistic Regression Model with the Original Data 

- Step 1: Fit a logistic regression model by using the training data (`X_train` and `y_train`). 

- Step 2: Saved the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model. 

- Step 3: Evaluated the model’s performance by doing the following: 
a- Generating a confusion matrix. 
b- Printing the classification report. 

### Conclusion:  

We evaluated the model's performance using an Accuracy Score, Precision and recall scores, and Confusion Matrix and Classification report. 

### Methods and Algorithms : 

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

4- For evaluation used Confusion Matrix to Quantify model errors (FP, FN, TP, TN). 

5- for evaluation used classification Report to summarize precision, recall, F1, support. 

## Results 

Machine Learning Model 1: Logistic Regression  

1- Accuracy - 99% 

2- Class 0 Healthy Loan: 
- Precision - (Class 0 - Healthy Loan) - 1.00 
- Recall - (Class 0 - Healthy Loan) - 0.99 

3- Class 1 High-Risk Loan 
- Precision (Class 1-High Risk Loan) - 0.84 
- Recall - (Class 1-High Risk Loan) - 0.94 

## Summary 

With an accuracy of 99%, The model correctly predicts loan status (healthy/high risk) for 99% of the cases. 

The logistic regression model performed best in identifying healthy loans (Class 0) with high precision (1.00), meaning that all healthy loans were correctly predicted with no false positives and recall (0.99) meaning that the model captures 99% of actual healthy loans missing only 1%.  

On the other Hand, the model performed decently in identifying high risk loans. High risk loans (Class 1) with precision 0.84 means that 84% of predicted high risk loans were correct while 16% were false positive. Regarding recall, 94% of true high risk loans are identified, missing only 6% 

In conclusion, Although the model performs well for both classes, the model is slightly balanced towards predicting healthy loans.  

Performance does depend on the problem we are trying to solve.  
Business context matters. If our priority is minimizing risk, then catching all high-risk loans is important. If the priority is reducing false alarms, and avoiding all unnecessary loan denials, the precision of 0.84 could be improved.  

Recommendations: 

Use the logistic regression model because it achieves near perfect performance for healthy loans and strong detection of high risk loans. 

Optional improvements for higher precision on class 1 would be adjusting the classification threshold (require higher probability to label a loan as high- risk). 

regardless, this model is still very useful and well recommended due to its simplicity and efficiency. 

 