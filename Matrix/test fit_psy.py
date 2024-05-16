#%%
import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from fit_psyche.psychometric_curve import PsychometricCurve

import matplotlib.pyplot as plt

#%%

def func(x, A, B):
    for xx in x:
        y=1/ (1 + 10**( -A(xx-B)))
    return y

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

x = np.linspace(start=-16, stop=16, num=66)
y = (x > x.mean()).astype(float)
# y[2] = y[2] + np.abs(np.random.rand())
# y[3] = y[3] - np.abs(np.random.rand())

y=sigmoid(x)

y=y+np.random.uniform(-0.4, 0.4, size=len(x))
# y=y+(0.9*x)
# y=y/max(y)

# plt.figure()
# plt.plot(x,y)
pc = PsychometricCurve(model='wh').fit(x, y)
pc.plot(x, y)
print(pc.score(x, y))
print(pc.coefs_)

# grid = RandomizedSearchCV(PsychometricCurve(), n_jobs=3,
#                           param_distributions={'model': ['wh', 'logit'],
#                                                'guess_rate_lims': [(0.01, 0.05), (0.01, 0.03), (0.03, 0.04)],
#                                                'lapse_rate_lims': [(0.01, 0.05), (0.01, 0.03), (0.03, 0.04)]})
# grid.fit(x, y)

# print(grid.best_estimator_.get_params())
# print(grid.best_estimator_.coefs_)