# Phase-3-Project
Phase 3 Project
### Project Overview
Telecoms Customer Churn Prediction.

### Objective
To predict customer churn for a telecoms company.

### Data
The data contains information about customers such as their account information, call details, and demographic information.
### Business Problem
The telecoms company wants to predict customer churn so as to retain customers and improve profitability.


Business Understanding

A telecommunications company is facing a significant challenge with customer churn. The company is losing valuable customers to competitors, which is negatively impacting revenue and market share. The business needs to identify customers who are likely to churn in the near future so that they can proactively implement targeted retention strategies. By predicting potential churners, the company aims to reduce customer attrition, increase customer lifetime value, and ultimately improve profitability.


### Objectives 
1.Develop a predictive model to identify customers at high risk of churning.

2.Understand the key factors contributing to customer churn.

3.Provide actionable insights to the business for developing effective retention strategies.

### 1. Loading the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


### 2. Loading and inspecting the data|
### Load and  Inspect Data

file_path = ('Telcoms dataset.csv')
data = pd.read_csv(file_path)
data
data.head(10)
data.isnull().sum()
### 3. Data Preprocessing.

Converting categorical variables to binary form and normalizing the data.
Convert target variable('churn) to binary form.
Drop irrelevant columns.


from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split

# Converting categoriacl variables to binary form
data[['international plan', 'voice mail plan']] = data[['international plan', 'voice mail plan']].apply(LabelEncoder().fit_transform)

## Encoding the state variable using label encoder
label_encoder = LabelEncoder()
data['state'] = label_encoder.fit_transform(data['state'])

### Converting target variable "churn" to binary(0=False,1=True)
data['churn'] = data['churn'].astype(int)


data.head()
### Dropping irrelevant columns 
data.drop(['area code', 'phone number'], axis=1, inplace=True)
### Splitting the data into training and testing sets
# Features 
X = data.drop('churn', axis=1)

## Target Variable
y = data['churn']

## Train test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.5, random_state=42)

### Normalizing Numerical features
scaler= StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Displaying the shapes of the datasets for confirmation

X_test_scaled.shape , X_train_scaled.shape, y_train.shape, y_test.shape 

### Logistic Regression.

I chose to implement logistic regression for prediction of customer churn. 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

# Training the logistic regression model
log_regress = LogisticRegression(random_state=42, max_iter= 1000)
log_regress.fit(X_train_scaled,y_train)
## Making predictions 
y_pred = log_regress.predict(X_test_scaled)

### Evaluating the model


accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy: .2f}')
print('Classifiaction Report:  ')
print(class_report)
print('Confusion Matrix:')
print(conf_matrix)


# Refit the best model on the entire training set
best_model = LogisticRegression(C=0.01, penalty='l2', solver='liblinear', max_iter=1000, random_state=42)
best_model.fit(X_train_scaled, y_train)

# Evaluate the best model on the test set
y_pred_best = best_model.predict(X_test_scaled)

# Calculate evaluation metrics
best_accuracy = accuracy_score(y_test, y_pred_best)
best_report = classification_report(y_test, y_pred_best)
best_conf_matrix = confusion_matrix(y_test, y_pred_best)

# Print the evaluation results
print("Best Model Accuracy:", best_accuracy)
print("\nBest Model Classification Report:\n", best_report)
print("\nBest Model Confusion Matrix:\n", best_conf_matrix)

### 4. Model Improvement
### Model Comparison
from sklearn.ensemble import RandomForestClassifier

# Initailize Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train_scaled, y_train)

# Make predictions
y_pred_rf = rf_clf.predict(X_test_scaled)

## Evaluate Random Forest Classifier

accuracy_rf = accuracy_score(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

print(f'Accuracy: {accuracy_rf: .2f}')
print('Classifiaction Report:  ')
print(class_report_rf)
print('Confusion Matrix:')
print(conf_matrix_rf)






### Handling Class Imbalance

from imblearn.over_sampling import SMOTE

## Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

## Train a new logistic regression model on the SMOTE-balanced data
log_regress_smote = LogisticRegression(random_state=42, max_iter=1000)
log_regress_smote.fit(X_train_smote, y_train_smote)

### Evaluate the model
y_pred_smote = log_regress_smote.predict(X_test_scaled)

accuracy_smote = accuracy_score(y_test, y_pred_smote)
class_report_smote = classification_report(y_test, y_pred_smote)
conf_matrix_smote = confusion_matrix(y_test, y_pred_smote)

print('Balanced Model Accuracy:', accuracy_score(y_test, y_pred_smote))
print('Balanced Model Classification Report:', classification_report(y_test, y_pred_smote))
print(class_report_smote)
print('Balanced Model Confusion Matrix:')
print(conf_matrix_smote)




### Cross-Validation
from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())
### ROC Curve and AUC Score
### ROC Curve and AUC Score|
from sklearn.metrics import roc_curve,auc 
import matplotlib.pyplot as plt 

# Compute ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


The ROC curve plots the true positive rate against false positive rate at various classification thresholds. the curve starts at the bottom  left(0,0) and ends at the top right(1,1) , a perfect classifier would have had a curve that goes straight up the y-axis and then cross at the top ,touching the (0,1) point.

An AUC of 0.89 indicates that your model has good discriminative ability. It correctly ranks a random positive instance higher than a random negative instance 89% of the time.

This is considered a strong predictive performance. In the context of churn prediction, it means your model is quite good at distinguishing between customers who are likely to churn and those who are not.

The curve is well above the diagonal line (which represents random guessing), confirming that the model performs much better than chance.


### Feature Engineering
 Feature engineering is a crucial step in improving model performance. It involves creating new features or transforming existing ones to better capture the underlying patterns in the data. Here are some feature engineering techniques you can apply to your churn prediction model:

 1. Interaction Terms: Create interaction terms between features to capture the combined effect of two features on the target variable.

## Interaction between total day minutes and customer service calls
data['day_minutes_service_calls'] = data['total day minutes'] * data['customer service calls']

## interaction between account length and total charges
data['account_length_total_charges'] = data['account length'] * data['total day charge'] + data['total eve charge'] + data['total night charge'] + data['total intl charge']

## interaction between total day minutes and total night minutes
data['day_minutes_night_minutes'] = data['total day minutes'] * data['total night minutes']

## interaction between total day charge and total eve charge
data['day_charge_eve_charge'] = data['total day charge'] * data['total eve charge']

print(data[['total day minutes', 'day_minutes_service_calls']].head(10))
print(data[['account length', 'account_length_total_charges']].head(10))
print(data[['total day minutes', 'day_minutes_night_minutes']].head(10))
print(data[['total day charge', 'day_charge_eve_charge']].head(10))




2. Binning Continous Variables:

Convert Continous variables into categorical variables by binning them into different ranges.


## Binning total day minutes
data['day_minutes_bin'] = pd.cut(data['total day minutes'], bins=5, labels=['0', '1', '2', '3', '4'])

## Binning total eve minutes
data['eve_minutes_bin'] = pd.cut(data['total eve minutes'], bins=5, labels=['0', '1', '2', '3', '4'])

## Binning total night minutes
data['night_minutes_bin'] = pd.cut(data['total night minutes'], bins=5, labels=['0', '1', '2', '3', '4'])

## Binning total intl minutes
data['intl_minutes_bin'] = pd.cut(data['total intl minutes'], bins=5, labels=['0', '1', '2', '3', '4'])
# 0 = low, 1 = medium, 2 = high, 3 = very high, 4 = extremely high

print(data[['total day minutes', 'day_minutes_bin']].head(10))
print(data[['total eve minutes', 'eve_minutes_bin']].head(10))
print(data[['total night minutes', 'night_minutes_bin']].head(10))
print(data[['total intl minutes', 'intl_minutes_bin']].head(10))
3. Aggregate Features:

Create aggregate features such as average call duration for each customer.

# Total call duration
data['total_call_duration'] = data['total day minutes'] + data['total eve minutes'] + data['total night minutes'] + data['total intl minutes']

# Total number of calls
data['total_calls'] = data['total day calls'] + data['total eve calls'] + data['total night calls'] + data['total intl calls']

# Average call duration
data['avg_call_duration'] = data['total_call_duration'] / data['total_calls']

print(data[['total_call_duration', 'total_calls', 'avg_call_duration']].head(10))
print(data[['total day minutes', 'total eve minutes', 'total night minutes', 'total intl minutes', 'total_call_duration', 'total_calls', 'avg_call_duration']].head(10))




4.Ratio Features

Create ratio features such as the ratio of day minutes to night minutes.


# Ratio of international to total call duration
data['intl_to_total_call_duration'] = data['total intl minutes'] / data['total_call_duration']

# Ratio of day to night minutes
data['day_to_night_minutes'] = data['total day minutes'] / data['total night minutes']

print(data[['total intl minutes', 'total_call_duration', 'intl_to_total_call_duration']].head(10))
print(data[['total day minutes', 'total night minutes', 'day_to_night_minutes']].head(10))
5. Time Based Features

If you have data on the time when customers churned, you can create time-based features.

### Assuming you have a 'account length' column

data['account_age'] = (pd.Timestamp.now() - pd.to_datetime(data['account length'])).dt.days


print(data[['account length', 'account_age']].head(10))

6. Polynormial Features:

Create polynomial features to capture non-linear relationships between features.


from sklearn.preprocessing import PolynomialFeatures

## Select numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns

## Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
data_poly = poly.fit_transform(data[numerical_cols])

## Convert back to DataFrame
data_poly = pd.DataFrame(data_poly, columns=poly.get_feature_names_out(numerical_cols))

print(data_poly.head(10))
7. Domain-specific-Features:

Based on the domain knowledge, you can create features that are specific to the telecom industry.

# Flag for high usage across all times of day
data['high_usage_flag'] = ((data['total day minutes'] > data['total day minutes'].mean()) & 
                           (data['total eve minutes'] > data['total eve minutes'].mean()) & 
                           (data['total night minutes'] > data['total night minutes'].mean())).astype(int)

# Difference between day and night usage
data['day_night_usage_diff'] = data['total day minutes'] - data['total night minutes']

print(data[['total day minutes', 'total night minutes', 'high_usage_flag', 'day_night_usage_diff']].head(10))

### Feature Importance analysis
import matplotlib.pyplot as plt 
### Feature Importance analysis
importance_features = best_model.coef_[0]
feature_names = X.columns

### Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_names, importance_features)

plt.xticks(rotation=90)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()




### Check for multicollinearity
1. Variance Inflation Factor
VIF quantifies the extent to which the variance of an estimated regression coefficient is increased due to multicollinearity. It is calculated as the ratio of the variance of the estimated coefficient to the variance of the original data.

### Generally, VIF values:1-5: Moderate correlation, 5-10: High correlation, >10: Very high correlation

from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Assuming 'X' is your feature matrix
vif_results = calculate_vif(X)
print(vif_results.sort_values('VIF', ascending=False))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    return {
        'model': model.__class__.__name__,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

# Evaluate Logistic Regression
lr_results = evaluate_model(best_model, X_train_scaled, y_train, X_test_scaled, y_test)

# Evaluate Random Forest
rf_results = evaluate_model(rf_clf, X_train_scaled, y_train, X_test_scaled, y_test)

# If you have other models, evaluate them here

# Combine results
all_results = [lr_results, rf_results]  # Add other model results here if available

# Print results
for result in all_results:
    print(f"\nModel: {result['model']}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1 Score: {result['f1_score']:.4f}")
    print(f"AUC: {result['auc']:.4f}")
    print(f"Cross-validation Mean: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")

# Determine the best model
best_model = max(all_results, key=lambda x: x['auc'])
print(f"\nBest Model: {best_model['model']} with AUC: {best_model['auc']:.4f}")
### Summary
While Logistic Regression offered more straightforward interpretability, the significant performance improvements and additional insights provided by the Random Forest Classifier made it the superior choice for this customer churn prediction task.

Why I chose Random Forest Classifier:
Superior Performance, Robustness to overfitting, Handling Non-Linear Relationships, Feature Importance Insights, Handling Mixed Data Types, Resilience To Outliers, Perfromance With High Dimensional Dta, Balance of Bias and Variance. 

