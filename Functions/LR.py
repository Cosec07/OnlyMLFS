import numpy as np

class LinearRegression:
    def __init__(self):
        self.coeffs = (None,None)
    def fit(self,X,y):
        #y = mx + b
        self.m = ((X*y).mean() - (X.mean() * y.mean()) ) / ( (X**2).mean() - (X.mean())**2)
        self.b = y.mean() - (self.m*X.mean())
        self.coeffs = (self.m,self.b)
        return self.coeffs
    def pred(self,X):
        return (self.m * X) + self.b

