import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Load dataset (replace this path 
data = pd.read_csv('feature_selection.csv')

# Define independent and dependent variables
X = data.drop(columns=['two_year_recid'])  # Assuming 'two_year_recid' is the target
y = data['two_year_recid']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize base model for RFE
model = LogisticRegression(max_iter=1000)

# Recursive Feature Elimination: Choose the top N features
desired_features = 5  # Change this number based on your experiment
rfe = RFE(model, n_features_to_select=desired_features)
rfe.fit(X_train, y_train)

# Summarize selected features
selected_columns = X_train.columns[rfe.support_]
print("Selected Features:", list(selected_columns))

# Train a new model with the selected features
X_train_selected = X_train[selected_columns]
X_test_selected = X_test[selected_columns]

model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)

# Evaluate performance
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
