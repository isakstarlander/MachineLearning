### Predicting the price of avocado ###
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## 1. Import libraries and modules ##
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline

## 2. Load data ##
dataset_path = 'avocado.csv'
data = pd.read_csv(dataset_path, sep=',')
data = data.drop('Index', axis=1)
data = data.drop('region', axis=1)
data = data.drop('Date', axis=1)

seed = 123

# print(data.head())
# print(data.shape)
# print(data.describe())

## 3. Split data into training and test sets ##
y = data.AveragePrice
X = data.drop('AveragePrice', axis=1)

# Label encode binary string values, i.e. 'conventional' or 'organic' types
X.type = LabelEncoder().fit_transform(X.type)

# One hot encode dummy string values, i.e. countries
# X.region = LabelBinarizer().fit_transform(X.region.values)
# print(X.region.values)
# print(X.head())

# DO NOT SPLIT INTO TRAIN/TEST WHEN USING KFOLD CROSS VALIDATION
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#     test_size=0.2,
#     random_state=seed,
#     stratify=y)


## 4. Build baseline model for the Keras NN ##
def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

## 5. Evaluate the baseline model ##
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)
print('Results: %.2f (%.2f) MSE' % (results.mean(), results.std()))