import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self):
        self.coeffs = np.array([])  # Initialize coeffs as an empty array

    def fit(self, X,y):
        if isinstance(X,pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        X_with_intercept = np.column_stack((np.ones(len(X)), X))
        y = np.append(y, np.zeros(1))
        self.coeffs = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        return self.coeffs

    def predict(self, new_X):
        if len(self.coeffs) == 0:
            raise ValueError("Model coefficients are not initialized. Please call fit() first.")
        X_in = np.column_stack((np.ones(len(new_X)), new_X))
        return X_in @ self.coeffs.ravel()
