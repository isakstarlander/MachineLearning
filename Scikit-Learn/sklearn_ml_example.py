### SCIKIT-LEARN PRACTICAL WALKTHROUGH ###
# source: https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn

## 1. Set up environment, i.e. install Python, numpy, pandas and scikit-learn ##

## 2. Import libraries and modules ##
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

## 3. Load data ##
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

print(data.head())
print(data.shape)
print(data.describe())

## 4. Split data into training and test sets ##
y = data.quality
X = data.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size=0.2,
    random_state=123,
    stratify=y)

## 5. Declare data preprocessing steps ##
# Standardize data (a.k.a. mean normalization) - subtract with mean and divide by standard deviations
# Instead of invoking scale function, Transformer API allows you to fit a preprocessing step using
# the training data and then use the same transformation on future data sets. Here are the actual steps:
# 1. Fit transformer to the training set (saving means and standard deviations)
# 2. Apply the transformer to the training set (scaling the training data)
# 3. Apply the transformer to the test set (using the same mean and std)
scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
# print(X_train_scaled.mean(axis=0))
# print(X_train_scaled.std(axis=0))

X_test_scaled = scaler.transform(X_test)
# print(X_test_scaled.mean(axis=0))
# print(X_test_scaled.std(axis=0))

pipeline = make_pipeline(preprocessing.StandardScaler(), 
    RandomForestRegressor(n_estimators=100))

## 6. Declare hyperparameters to tune ##
# There are two types of parameters to worry about, model parameters and hyperparameters.
# Model parameters can be learned directly from the data, i.e. regression coefficients.
# Hyperparameters express "higher-level" structural information about the model and are typically
# set before training the model.

# print(pipeline.get_params())
hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
    'randomforestregressor__max_depth': [None, 5, 3, 1]}

## 7. Tune model using a cross-validation pipeline ##
# Cross-validation is a process for reliably estimating the performance of a method for 
# building a model by training and evaluating your model multiple times using the same method.
# These are the steps for cross-validation:
# 1. Split your data into k equal parts, or "folds" (typically k=10)
# 2. Train your model on k-1 folds (e.g. the first 9 folds)
# 3. Evaluate it on the remaining "hold-out" fold (e.g. the 10th fold)
# 4. Perform steps (2) and (3) k times, each time holding out a different fold
# 5. Aggreggate the performance accross all k folds. This is your performance metric

# Best practice when performing CV is to include your data preprocessing inside
# the cross-validation loop. Here's the CV pipeline after including preprocessing steps:
# 1. Split your data into k equal parts, or "folds" (typically k=10)
# 2. Preprocess k-1 training folds
# 3. Train your model on the same k-1 folds
# 4. Preprocess the hold-out fold using the same transformations from step (2)
# 5. Evaluate your model on the same hold-out fold
# 6. Perform steps (2) - (5) k times, each time holding out a different fold
# 7. Aggreggate the performance accross all k folds. This is your performance metric
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

clf.fit(X_train, y_train)
print(clf.best_params_)

## 8. Refit on the entire training set ##
# You generally get a small performance improvement by refitting on the entire training set.
# GridSearchCV does this automatically. Now you can hence use the clf object as your model 
# when applying it to other sets of data.

## 9. Evaluate model pipeline on test data ##
y_pred = clf.predict(X_test)

print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

# Rule of thumb is that your first model likely isn't the best possible model. A combination of
# three strategies can be used to decide if you are satisfied with your model performance:
# 1. Start with the goal of the model. If the model is tied to a business problem, have you
# successfully solved the problem?
# 2. Look in academic research/literature to get a sense of the current performance benchmarks
# for specific types of data
# 3. Try to find low-hanging fruit in terms of ways to improve your model

## 9. Save model for future use ##
joblib.dump(clf, 'rf_regressor.pkl')

# To load previously saved model:
# clf2 = joblib.load('rf_regressor.pkl')
# clf2.predict(X_test)