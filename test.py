import numpy as np
from Functions import LogReg

#Dataset Sampling code
np.random.seed(69)

n_samples = 100
X = np.random.randn(n_samples,2)
y = (2 * X[:,0] - 3 * X[:,1] + np.random.randn(n_samples)) > 0
X_test = np.random.randn(n_samples,2)

obj = LogReg.LogR()
print(obj.fit(X,y,0.001,100))
print(obj.pred_probs(X_test))
print(obj.preds(X_test))