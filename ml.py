
import numpy as np
# true labels
y_true = np.array([1, 0, 1, 0, 1])
y_pred = np.array([1, 1, 0, 0, 1])
misclassified = (y_true != y_pred).astype(int)
# misclassification rate
L_theta = misclassified.mean()
print("Misclassification rate:", L_theta)
# this helps u createa a misclassification rate that helps that understand the current Ml learning better or not also if the number is bigger it is worse 
### this is the next place that it doesnt allow that a errorr would be place  and it helps to create a set of data that doesnt allow a error 

# it calls asymmetric loss function ℓ(y, yˆ) 
import numpy as np
def asymmetric_loss(y_true, y_pred, c_fn=5, c_fp=1):
    losses = []
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            losses.append(0)
        elif yt == 1 and yp == 0:  # false negative
            losses.append(c_fn)
        elif yt == 0 and yp == 1:  # false positive
            losses.append(c_fp)
    return np.mean(losses)
# Example: true vs predictions
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0, 0, 1, 0, 1])
print("Asymmetric loss:", asymmetric_loss(y_true, y_pred, c_fn=5, c_fp=1))
### we also got ERM with asymmetric loss we are using for false positive or false negative it is usefull since we are going to need make sure we are going to do the most rewarding one for our use case rather than be too strict
## basic examples 
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
# toy dataset
X, y = make_classification(n_samples=200, n_features=2, random_state=42)
# train logistic regression
clf = LogisticRegression().fit(X, y)
y_pred = clf.predict(X)
# asymmetric loss
def asymmetric_loss(y_true, y_pred, c_fn=5, c_fp=1):
    return np.mean([
        0 if yt == yp else (c_fn if yt == 1 else c_fp)
        for yt, yp in zip(y_true, y_pred)
    ])
# empirical risk with asymmetric loss
risk = asymmetric_loss(y, y_pred)
print("Empirical risk (asymmetric):", risk)
