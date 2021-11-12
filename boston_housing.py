"""
https://www.kaggle.com/bextuychiev/lasso-regression-with-pipelines-tutorial
https://towardsdatascience.com/intro-to-regularization-with-ridge-and-lasso-regression-with-sklearn-edcf4c117b7a

Dataset from
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data?select=train.csv

NOTE: if "real" data set not going according to plan, try

from sklearn.datasets import make_regression

This will create a mock dataset with specified noise.
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (Lasso, LassoCV, LinearRegression, Ridge,
                                  RidgeCV)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

houses = pd.read_csv('data/train.csv').select_dtypes(include=np.number)
# houses.select_dtypes(include=np.number)
# print(houses.head())

X = houses.values[:, :-1]
# X = X + (100*np.random.randn(X.shape[1]))
y = houses.values[:, -1]

# RANDOM STATE 0 SHOWS A GOOD EXAMPLE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.88, random_state=0)


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

print("Normal Least Squares")
print(f"Test Score: {LR_pipeline.score(X_train, y_train)}")
print(f"Train Score: {LR_pipeline.score(X_test, y_test)}")
print("\n")

ridge = RidgeCV(alphas=np.linspace(0.00000001, 10, 1000), cv=3)
ridge_pipeline = Pipeline(steps=[
    ('preprocess', numeric_pipeline),
    ('model', ridge)
])

ridge_pipeline.fit(X_train, y_train)
print(f"Ridge Regularization, alpha={ridge.alpha_}")
print(f"Test Score: {ridge_pipeline.score(X_train, y_train)}")
print(f"Train Score: {ridge_pipeline.score(X_test, y_test)}")
print("\n")


lasso = LassoCV(alphas=np.linspace(1, 500, 1000), cv=2)
lasso_pipeline = Pipeline(steps=[
    ('preprocess', numeric_pipeline),
    ('model', lasso)
])

lasso_pipeline.fit(X_train, y_train)
print(f"Lasso Regularization {lasso.alpha_}")
print(f"Test Score: {lasso_pipeline.score(X_train, y_train)}")
print(f"Train Score: {lasso_pipeline.score(X_test, y_test)}")


# ridge = RidgeCV(alphas=np.arange(1, 100, 5), scoring='r2', cv=10)
# _ = ridge.fit(X, y)
