import numpy as np
from scipy.sparse import dia
from scipy.sparse.construct import rand
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split

diabetes = load_diabetes()
# print(diabetes["DESCR"])

# only BMI, may be a usefull one to plot
# X = diabetes["data"][:, 2:3]
X = diabetes["data"]
y = diabetes["target"]


# "good" random states to show off
# -> 10, 42,
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

print(f"Normal Least Squares: {lin_reg.score(X_test, y_test)}")

best_alpha = 0
best_ridge_score = 0
for alpha in np.linspace(0.00001, 10, 100):
    ridge_reg = Ridge(alpha=alpha)
    ridge_reg.fit(X_train, y_train)
    score = ridge_reg.score(X_test, y_test)
    if score > best_ridge_score:
        best_ridge_score = score
        best_alpha = alpha

print(f"Ridge, alpha={best_alpha}: {best_ridge_score}")


best_alpha = 0
best_lasso_score = 0
for alpha in np.linspace(0.00001, 10, 100):
    lasso_reg = Lasso(alpha=alpha)
    lasso_reg.fit(X_train, y_train)
    score = lasso_reg.score(X_test, y_test)
    if score > best_lasso_score:
        best_lasso_score = score
        best_alpha = alpha

print(f"Lasso, alpha={best_alpha}: {best_lasso_score}")
