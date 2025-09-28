# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

X_np = np.linspace(-3, 3, 200)
true_mu = np.sin(X_np)
true_sigma = 0.3 + 0.5 * (X_np**2)
y_np = true_mu + np.random.randn(*X_np.shape) * true_sigma
X = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

class HeteroscedasticRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 2)   
        )

    def forward(self, x):
        out = self.net(x)
        mu = out[:, 0:1]
        log_sigma2 = out[:, 1:2]
        return mu, log_sigma2

def gaussian_nll(y_true, mu, log_sigma2):
    sigma2 = torch.exp(log_sigma2)
    return 0.5 * torch.mean(log_sigma2 + (y_true - mu)**2 / sigma2)
  model = HeteroscedasticRegressor()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 3000
for epoch in range(epochs):
    optimizer.zero_grad()
    mu_pred, log_sigma2_pred = model(X)
    loss = gaussian_nll(y, mu_pred, log_sigma2_pred)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

with torch.no_grad():
    mu_pred, log_sigma2_pred = model(X)
    sigma_pred = torch.sqrt(torch.exp(log_sigma2_pred))

mu_pred_np = mu_pred.squeeze().numpy()
sigma_pred_np = sigma_pred.squeeze().numpy()

plt.scatter(X_np, y_np, s=10, alpha=0.4, label="data")
plt.plot(X_np, true_mu, 'r', label="true mean")
plt.plot(X_np, mu_pred_np, 'b', label="learned mean")
plt.fill_between(X_np,
                 mu_pred_np - 2*sigma_pred_np,
                 mu_pred_np + 2*sigma_pred_np,
                 color='blue', alpha=0.2, label='±2σ band')
plt.legend()
plt.title" PyTorch")
plt.show()

