import numpy as np
from Functions import LR

X = np.array([1,3,5])
y = np.array([4.8, 12.4, 15.5])
n = np.size(X)

obj = LR.LinearRegression()
print(obj.fit(X,y))
print(obj.pred(1.3))
