import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self):
        self.coeffs = None
    def fit(self,X,y):
        #y = mx + b
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        mx = np.column_stack((np.ones(len(X)), X))
        self.coeffs = np.linalg.inv(mx.T @ mx) @ mx.T @ y
        return self.coeffs
    def pred(self,X):
        X_in = np.column_stack((np.ones(len(X)),X))
        return X_in @ self.coeffs

