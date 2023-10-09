import numpy as np
import pandas as pd

class SVC:
    def __init__(self):
        self.kernel = None
        self.w = None
        self.b = None
    
    def fit(self,X,y,lr):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        lr = 0.01
        while True:
            loss = 0
            for i in range(X[0]):
                


