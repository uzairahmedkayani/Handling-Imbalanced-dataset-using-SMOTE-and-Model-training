from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Base.csv")

# Display few rows of the table
print(df.head())

# Display Total number of records along with No of columns
print(df.shape)

# Display "outcome" column's each values' count
print(df.fraud_bool.value_counts())

# Check for missing values
# print(df.isnull().sum())

# Separating the target variable from independent variables (Selected ones only)
X = df.iloc[:, [1, 2, 4, 5]]
y = df.fraud_bool
print(X.head())
print(y.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Training the Decision Tree model on imbalanced data
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Evaluate Decision Tree model
print('Decision Tree Model:')
print('Accuracy:', accuracy_score(y_test, y_pred_dt))
print('Precision:', precision_score(y_test, y_pred_dt))
print('Recall:', recall_score(y_test, y_pred_dt))
print('F1 Score:', f1_score(y_test, y_pred_dt))

# Training the Logistic Regression model on imbalanced data
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluate Logistic Regression model
print('Logistic Regression Model:')
print('Accuracy:', accuracy_score(y_test, y_pred_lr))
print('Precision:', precision_score(y_test, y_pred_lr, zero_division=1))  # Set zero_division=1
print('Recall:', recall_score(y_test, y_pred_lr))
print('F1 Score:', f1_score(y_test, y_pred_lr))

# Using SMOTE technique
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Before SMOTE:", Counter(y_train))
print("After SMOTE:", Counter(y_train_smote))

# Training the Decision Tree model after applying SMOTE
dt_model_smote = DecisionTreeClassifier()
dt_model_smote.fit(X_train_smote, y_train_smote)
y_pred_dt_smote = dt_model_smote.predict(X_test)

# Evaluate Decision Tree model after applying SMOTE
print('Decision Tree Model (SMOTE):')
print('Accuracy:', accuracy_score(y_test, y_pred_dt_smote))
print('Precision:', precision_score(y_test, y_pred_dt_smote))
print('Recall:', recall_score(y_test, y_pred_dt_smote))
print('F1 Score:', f1_score(y_test, y_pred_dt_smote))

# Training the Logistic Regression model after applying SMOTE
lr_model_smote = LogisticRegression()
lr_model_smote.fit(X_train_smote, y_train_smote)
y_pred_lr_smote = lr_model_smote.predict(X_test)

# Evaluate Logistic Regression model after applying SMOTE
print('Logistic Regression Model (SMOTE):')
print('Accuracy:', accuracy_score(y_test, y_pred_lr_smote))
print('Precision:', precision_score(y_test, y_pred_lr_smote, zero_division=1))  # Set zero_division=1
print('Recall:', recall_score(y_test, y_pred_lr_smote))
print('F1 Score:', f1_score(y_test, y_pred_lr_smote))

# Generate AUC ROC curve for Decision Tree model
y_pred_prob_dt = dt_model.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_prob_dt)
auc_dt = roc_auc_score(y_test, y_pred_prob_dt)

# Generate AUC ROC curve for Logistic Regression model
y_pred_prob_lr = lr_model.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob_lr)
auc_lr = roc_auc_score(y_test, y_pred_prob_lr)

# Compare models with cross-validation
dt_scores = cross_val_score(dt_model_smote, X, y, cv=5, scoring='accuracy')
lr_scores = cross_val_score(lr_model_smote, X, y, cv=5, scoring='accuracy')

print('Cross-validation scores (Decision Tree):', dt_scores)
print('Average Accuracy (Decision Tree):', dt_scores.mean())

print('Cross-validation scores (Logistic Regression):', lr_scores)
print('Average Accuracy (Logistic Regression):', lr_scores.mean())

# Plot the AUC ROC curves
plt.figure()
plt.plot(fpr_dt, tpr_dt, label='Decision Tree (AUC = {:.2f})'.format(auc_dt))
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = {:.2f})'.format(auc_lr))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC ROC Curve')
plt.legend(loc='lower right')
plt.show()
