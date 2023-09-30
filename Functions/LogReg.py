import pandas as pd
import numpy as np

class LogR:
    def __init__(self):
        self.coeffs = None

    def fit(self, X,y,lr,iter):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        n_samples, n_features = X.shape
        self.coeffs = np.zeros(n_features)
        for _ in range(iter):
            linear_op = np.dot(X , self.coeffs)
            preds = self.sigmoid(linear_op)
            err = y - preds
            grad = (X.T @ err) / n_samples
            self.coeffs += lr*grad

    def pred_probs(self,X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        linear_op = X @ self.coeffs
        return self.sigmoid(linear_op)
    
    def preds(self,X, thresh=0.5):
        if isinstance(X,pd.DataFrame):
            X = X.values
        linear_op = X @ self.coeffs
        pred = self.sigmoid(linear_op)
        return (pred >= thresh).astype(int)
    
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))




    
    

