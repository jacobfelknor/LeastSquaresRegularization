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
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

houses = pd.read_csv("data/train.csv").select_dtypes(include=np.number)
# print(houses.shape)
# print(houses.head())

X = houses.values[:, 1:-1]  # remove the cost of the house and the "ID" field
y = houses.values[:, -1]  # extract housing costs from test data

# RANDOM STATES [0,12312] SHOW A GOOD EXAMPLE
# STATE 12313 shows an extreme case
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=0)


# Preprocessing. Scale Data and fill in missing values
numeric_pipeline = Pipeline(steps=[("impute", SimpleImputer(strategy="mean")), ("scale", MinMaxScaler())])

# categorical_pipeline = Pipeline(steps=[
#     ('impute', SimpleImputer(strategy='most_frequent')),
#     ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
# ])


# Models below
lin_reg = LinearRegression()

LR_pipeline = Pipeline(steps=[("preprocess", numeric_pipeline), ("model", lin_reg)])
LR_pipeline.fit(X_train, y_train)

print(f"Normal Least Squares, Features Used: {X_train.shape[1]}")
print(f"Train Score: {LR_pipeline.score(X_train, y_train)}")
print(f"Test Score: {LR_pipeline.score(X_test, y_test)}")
print("\n")

ridge = RidgeCV(alphas=np.linspace(0.00000001, 10, 1000), cv=3)
ridge_pipeline = Pipeline(steps=[("preprocess", numeric_pipeline), ("model", ridge)])

ridge_pipeline.fit(X_train, y_train)
nonzero_coeff = np.count_nonzero(ridge.coef_)
print(f"Ridge Regularization, alpha={ridge.alpha_}, Features Used: {nonzero_coeff}")
print(f"Train Score: {ridge_pipeline.score(X_train, y_train)}")
print(f"Test Score: {ridge_pipeline.score(X_test, y_test)}")
print("\n")


lasso = LassoCV(alphas=np.linspace(500, 1500, 1000), cv=2)  # for extreme case
# lasso = LassoCV(alphas=np.linspace(1, 500, 1000), cv=2)  # for other states
lasso_pipeline = Pipeline(steps=[("preprocess", numeric_pipeline), ("model", lasso)])

lasso_pipeline.fit(X_train, y_train)
nonzero_coeff = np.count_nonzero(lasso.coef_)
print(f"Lasso Regularization, alpha={lasso.alpha_}, Features Used: {nonzero_coeff}")
print(f"Train Score: {lasso_pipeline.score(X_train, y_train)}")
print(f"Test Score: {lasso_pipeline.score(X_test, y_test)}")


# ridge = RidgeCV(alphas=np.arange(1, 100, 5), scoring='r2', cv=10)
# _ = ridge.fit(X, y)
