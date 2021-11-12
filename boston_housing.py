"""
https://www.kaggle.com/bextuychiev/lasso-regression-with-pipelines-tutorial
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

houses = pd.read_csv('data/train.csv').select_dtypes(include=np.number)
# houses.select_dtypes(include=np.number)
# print(houses.head())

X = houses.values[:, :-1]
y = houses.values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)


# Preprocessing. Scale Data and fill in missing values
numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
])

# categorical_pipeline = Pipeline(steps=[
#     ('impute', SimpleImputer(strategy='most_frequent')),
#     ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
# ])


# Models below
lin_reg = LinearRegression()

LR_pipeline = Pipeline(steps=[
    ('preprocess', numeric_pipeline),
    ('model', lin_reg)
])

LR_pipeline.fit(X_train, y_train)

print("Normal Least Squars")
print(f"Test Score: {LR_pipeline.score(X_train, y_train)}")
print(f"Train Score: {LR_pipeline.score(X_test, y_test)}")
print("\n")
lasso = Lasso(alpha=100)

lasso_pipeline = Pipeline(steps=[
    ('preprocess', numeric_pipeline),
    ('model', lasso)
])

lasso_pipeline.fit(X_train, y_train)

print("Lasso Regularization")
print(f"Test Score: {lasso_pipeline.score(X_train, y_train)}")
print(f"Train Score: {lasso_pipeline.score(X_test, y_test)}")


# ridge = RidgeCV(alphas=np.arange(1, 100, 5), scoring='r2', cv=10)
# _ = ridge.fit(X, y)
