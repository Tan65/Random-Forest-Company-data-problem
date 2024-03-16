# Random-Forest-Company-data-problem
Problem Statement: A cloth manufacturing company is interested to know about the segment or attributes causes high sale.  Approach - A Random Forest can be built with target variable Sales (we will first convert it in categorical variable) &amp; all other variable will be independent in the analysis.  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import lightgbm as lgb
import xgboost as xgb

# Step 1: Read the CSV file into a DataFrame
data = pd.read_csv('Company_Data.csv')

# Step 2: Convert Sales into a categorical variable
data['Sales'] = data['Sales'].apply(lambda x: 'High' if x > 8 else 'Low')

# Step 3: Convert categorical variables into numeric form
data = pd.get_dummies(data, columns=['ShelveLoc', 'Urban', 'US'])

# Step 4: Exploratory Data Analysis (EDA)
# Univariate Analysis
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
num_plots = len(num_cols)
num_rows = (num_plots // 4) + (1 if num_plots % 4 != 0 else 0)  # Calculate number of rows needed
plt.figure(figsize=(20, num_rows * 5))
for i, col in enumerate(num_cols):
    plt.subplot(num_rows, 4, i+1)
    sns.histplot(data[col], kde=True)
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()

# Bivariate Analysis
plt.figure(figsize=(15, 10))
sns.pairplot(data, hue='Sales', corner=True)
plt.suptitle('Pairplot of Variables by Sales Category', y=1.02)
plt.show()

# Multivariate Analysis (Correlation Heatmap)
plt.figure(figsize=(12, 8))
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Step 5: Split the data into features (X) and the target variable (y)
X = data.drop('Sales', axis=1)
y = data['Sales']

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Build a Random Forest model
rf_model = RandomForestClassifier()

# Step 8: Fit the model to the training data
rf_model.fit(X_train, y_train)

# Step 9: Evaluate the model
rf_accuracy = rf_model.score(X_test, y_test)
print('Random Forest Accuracy:', rf_accuracy)

# Step 10: Analyze feature importances
importances = rf_model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print(feature_importances)

# Step 11: Bagging and Boosting (Decision Tree as base estimator)
base_estimator = DecisionTreeClassifier()
bagging_model = BaggingClassifier(base_estimator=base_estimator, n_estimators=100, random_state=42)
bagging_model.fit(X_train, y_train)
bagging_accuracy = bagging_model.score(X_test, y_test)
print('Bagging Accuracy:', bagging_accuracy)

boosting_model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100, random_state=42)
boosting_model.fit(X_train, y_train)
boosting_accuracy = boosting_model.score(X_test, y_test)
print('Boosting Accuracy:', boosting_accuracy)

# Step 12: Entropy and Gini Criterion
def calculate_entropy(data):
    labels = data['Sales']
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_gini(data):
    labels = data['Sales']
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    gini = 1 - np.sum(probabilities ** 2)
    return gini

entropy_full = calculate_entropy(data)
gini_full = calculate_gini(data)
print("Entropy of the full dataset:", entropy_full)
print("Gini impurity of the full dataset:", gini_full)

# Step 13: LightGBM and XGBoost
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)
lgb_accuracy = lgb_model.score(X_test, y_test)
print('LightGBM Accuracy:', lgb_accuracy)

from sklearn.preprocessing import LabelEncoder

# Step 5: Encode the target variable into numerical form
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Step 6: Train the XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train_encoded)

# Step 7: Evaluate the XGBoost model
xgb_accuracy = xgb_model.score(X_test, y_test_encoded)
print('XGBoost Accuracy:', xgb_accuracy)
