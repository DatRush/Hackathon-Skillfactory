import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

# Function to compute meta features using cross-validation
def compute_meta_feature(clf, X_train, X_test, y_train, cv):
    X_meta_train = np.zeros_like(y_train, dtype=np.float32)  # Placeholder for train meta-features
    # Loop over each fold defined by the cross-validator
    for train_fold_index, predict_fold_index in cv.split(X_train, y_train):  
        X_fold_train, X_fold_predict = X_train[train_fold_index], X_train[predict_fold_index]
        y_fold_train = y_train[train_fold_index]

        folded_clf = clone(clf)  # Clone the classifier for this fold
        folded_clf.fit(X_fold_train, y_fold_train)  # Fit on the training fold
        X_meta_train[predict_fold_index] = folded_clf.predict_proba(X_fold_predict)[:, 1]  # Predict on the validation fold

    meta_clf = clone(clf)
    meta_clf.fit(X_train, y_train)  # Fit a new classifier on the full training data

    X_meta_test = meta_clf.predict_proba(X_test)[:, 1]  # Predict on the test set

    return X_meta_train, X_meta_test

# Function to generate meta features for all classifiers
def generate_meta_features(classifiers, X_train, X_test, y_train, cv):
    features = [
        compute_meta_feature(clf, X_train, X_test, y_train, cv)
        for clf in tqdm(classifiers)  # Use tqdm to show progress
    ]

    stacked_features_train = np.vstack([
        features_train for features_train, features_test in features
    ]).T  # Stack features for training data

    stacked_features_test = np.vstack([
        features_test for features_train, features_test in features
    ]).T  # Stack features for test data

    return stacked_features_train, stacked_features_test

# Function to load data from a folder and combine it into a single DataFrame
def load_and_combine_data(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    combined_data = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    return combined_data


data = load_and_combine_data('train')

# Data preprocessing
data.drop(['ID', 'IT', '1D'], axis=1, inplace=True)  # Remove unnecessary columns
data.fillna(data.mean(), inplace=True)  # Fill missing values with mean

features = data.drop('Category', axis=1).columns
target = 'Category'

# Split data into training and test sets
cover_train, cover_test = train_test_split(data, test_size=0.2)

cover_X_train, cover_y_train = cover_train[features], cover_train[target]
cover_X_test, cover_y_test = cover_test[features], cover_test[target]

scaler = StandardScaler()  # Initialize a standard scaler
cover_X_train = scaler.fit_transform(cover_X_train)  # Scale training data
cover_X_test = scaler.transform(cover_X_test)  # Scale test data


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # Define a stratified K-Fold cross-validator

# Generate stacked features using defined classifiers
stacked_features_train, stacked_features_test = generate_meta_features([
    LogisticRegression(C=1, penalty='l1', solver='saga', max_iter=4000),
    LogisticRegression(C=0.1, penalty='l2', solver='liblinear', max_iter=1000),  
    RandomForestClassifier(n_estimators=300, max_depth=8, max_features='sqrt'),
    GradientBoostingClassifier(n_estimators=300, criterion='squared_error', loss='exponential', max_depth=4, max_features='sqrt')
], cover_X_train, cover_X_test, cover_y_train.values, cv)

# Combine original and meta features
total_features_train = np.hstack([cover_X_train, stacked_features_train])
total_features_test = np.hstack([cover_X_test, stacked_features_test])

np.random.seed(42)
clf = LogisticRegression(solver='lbfgs')
clf.fit(stacked_features_train, cover_y_train)  # Train a logistic regression model on the training meta-features
print(accuracy_score(clf.predict(stacked_features_test), cover_y_test))  # Print accuracy of the model

# Load test data, process it similarly to training data
test_data = load_and_combine_data('test')

test_ids = test_data['ID'].copy()

# Renaming columns to match the training data columns
test_data = test_data.rename(columns={'texture_max': 'texture_max', 'p_mean': 'perimeter_mean', 'p_std': 'perimeter_std', 'p_max': 'perimeter_max', 'area_mean': 'area_mean', 'conc_points_mean': 'concave_points_mean', 'conc_points_std': 'concave_points_std', 'conc_points_max': 'concave_points_max', 'symmetry_mean': 'symmetry_mean', 'symmetry_std': 'symmetry_std', 'symmetry_max': 'symmetry_max'})
test_data.fillna(test_data.mean(), inplace=True)

for col in features:
    if col not in test_data.columns:
        test_data[col] = 0  # Add missing columns as zeros

test_data = test_data[list(features)]

test_data = scaler.transform(test_data)  # Scale the test data

_, stacked_features_test = generate_meta_features([
    LogisticRegression(C=1, penalty='l1', solver='saga', max_iter=4000),
    LogisticRegression(C=0.1, penalty='l2', solver='liblinear', max_iter=1000),  
    RandomForestClassifier(n_estimators=300, max_depth=8, max_features='sqrt'),
    GradientBoostingClassifier(n_estimators=300, criterion='squared_error', loss='exponential', max_depth=4, max_features='sqrt')
], cover_X_train, test_data, cover_y_train.values, cv)  

predictions = clf.predict(stacked_features_test)  # Make predictions on the test data

# Create a DataFrame for submission
submission = pd.DataFrame({
    'ID': test_ids,
    'Category': predictions
})
submission.drop_duplicates(subset='ID', inplace=True)  # Drop any duplicates
submission.to_csv('submission.csv', index=False)  # Save submission to CSV
