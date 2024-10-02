import pandas as pd
import numpy as np
from Functions.Animator import update_plot

class LogR:
    def __init__(self):
        self.coeffs = None

    def fit(self, X,y,lr,iter,batch_size=100):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        n_samples, n_features = X.shape
        self.coeffs = np.zeros(n_features)

        for _ in range(iter):
            for i in range(0, n_samples, batch_size):
                grad = self.gradient(X[i:i + batch_size], y[i:i + batch_size],n_samples)

                self.coeffs += lr * grad

                update_plot(i, X, y, self, batch_size)
            linear_op = np.dot(X , self.coeffs)
            preds = self.sigmoid(linear_op)
            err = y - preds
            grad = (X.T @ err) / n_samples
            self.coeffs += lr*grad
    def gradient(self,X,y,n):
        # Calculate the gradients for the given batch of data
        linear_op = X @ self.coeffs
        preds = self.sigmoid(linear_op)
        err = y - preds
        gradients = (X.T @ err) / n

        return gradients
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




    
    

