
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
# true vs predictions
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
import numpy as np

def l2_loss(y, y_hat):
    return (y - y_hat) ** 2

def mse(y_true, y_pred):
    """
    y_true: list or numpy array of true values
    y_pred: list or numpy array of predicted values
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)
#showcase
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

print("Single loss example (y=3, y_hat=2.5):", l2_loss(3, 2.5))
print("MSE for dataset:", mse(y_true, y_pred))
import numpy as np
import matplotlib.pyplot as plt

# Gaussian distribution function
def gaussian(y, mu=0, sigma=1):
    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-0.5 * ((y - mu)**2) / sigma**2)

y_values = np.linspace(-5, 5, 200)
mu = 0   # mean
sigma = 1.0  # standard deviation

pdf = gaussian(y_values, mu, sigma)

plt.plot(y_values, pdf)
plt.title(f"Gaussian Distribution (μ={mu}, σ²={sigma**2})")
plt.xlabel("y")
plt.ylabel("p(y | μ, σ²)")
plt.grid(True)
plt.show()
# now Negative Log-Likelihood Loss version 
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Create toy data ---
np.random.seed(0)
X = np.linspace(-3, 3, 100)
true_mu = np.sin(X)             # true function
true_sigma = 0.3 + 0.5 * (X**2) # variance grows with |x|
y = true_mu + np.random.randn(*X.shape) * true_sigma

plt.scatter(X, y, s=15, label="data", alpha=0.6)
plt.plot(X, true_mu, 'r', label="true mean")
plt.title("Noisy data with input-dependent variance")
plt.legend()
plt.show()

# We'll use a linear model for both mean and log-variance
# mu(x) = a1*x + b1
# log_sigma2(x) = a2*x + b2
a1, b1 = 0.0, 0.0
a2, b2 = 0.0, np.log(0.5)  # start with some guess

lr = 0.01  # learning rate

for epoch in range(3000):
    mu_pred = a1*X + b1
    log_sigma2_pred = a2*X + b2
    sigma2_pred = np.exp(log_sigma2_pred)

    # NLL loss
    loss = 0.5*np.mean(log_sigma2_pred + (y - mu_pred)**2 / sigma2_pred)

    # Gradients (hand derived)
    d_mu = (mu_pred - y) / sigma2_pred
    d_a1 = np.mean(d_mu * X)
    d_b1 = np.mean(d_mu)

    d_log_sigma2 = 0.5 - 0.5*(y - mu_pred)**2 / sigma2_pred
    d_a2 = np.mean(d_log_sigma2 * X)
    d_b2 = np.mean(d_log_sigma2)

    # Gradient step
    a1 -= lr * d_a1
    b1 -= lr * d_b1
    a2 -= lr * d_a2
    b2 -= lr * d_b2

mu_learned = a1*X + b1
sigma_learned = np.sqrt(np.exp(a2*X + b2))

plt.scatter(X, y, s=15, alpha=0.4, label="data")
plt.plot(X, true_mu, 'r', label="true mean")
plt.plot(X, mu_learned, 'b', label="learned mean")

# Plot uncertainty bands (±2σ)
plt.fill_between(X, mu_learned - 2*sigma_learned, mu_learned + 2*sigma_learned,
                 color='blue', alpha=0.2, label='±2σ band')

plt.title("Heteroscedastic Regression: Mean + Uncertainty")
plt.legend()
plt.show()

import torch

sigma2 = 0.5

# Example predictions and targets
y_true = torch.tensor([2.0, 1.0, 3.0])
y_pred = torch.tensor([2.5, 0.8, 2.7])

# MSE
mse_loss = torch.mean((y_true - y_pred)**2)

# NLL
nll_loss = (1/(2*sigma2)) * mse_loss + 0.5*torch.log(2*torch.pi*sigma2)

print("MSE:", mse_loss.item())
print("NLL:", nll_loss.item())

