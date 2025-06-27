import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ExplicitEu import ExplicitEu
from ImplicitEu import ImplicitEu
from ImplicitAm import ImplicitAmBer

# Set seed for reproducibility
torch.manual_seed(123)

# -----------------------------
# Model Definition
# -----------------------------
class BlackScholesMertonModel1(nn.Module):
    def __init__(self):
        super(BlackScholesMertonModel1, self).__init__()
        layers = []
        layers.append(nn.BatchNorm1d(2))
        for _ in range(9):
            layers.append(nn.Linear(50 if _ > 0 else 2, 50))
            layers.append(nn.BatchNorm1d(50))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(50, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Parameters
# -----------------------------
S0 = 100
K = 100
sigma = 0.4
r = 0.03
dividend = 0.00
tau = 3
M = 500
N = 600
Smax = 500
is_call = True
N_b = 100
N_exp = 1000
N_f = 10000
lb = [0, 0]
ub = [500, tau]

# -----------------------------
# Generate Reference Solution
# -----------------------------
option = ImplicitAmBer(S0, K, r, tau, sigma, Smax, M, N, is_call)
option.price()
option_fde_prices = option.grid

# -----------------------------
# Data Preparation
# -----------------------------
def initialize_data(N_b, N_exp, N_f, lb, ub, K, tau):
    stock_price_collocation = torch.randint(0, ub[0] + 1, (N_f, 1)).float()
    time_collocation = torch.randint(0, 100 * ub[1] + 1, (N_f, 1)).float() / 100
    stock_price_mean = stock_price_collocation.mean()
    stock_price_std = stock_price_collocation.std()
    time_mean = time_collocation.mean()
    time_std = time_collocation.std()
    X_f = torch.cat((time_collocation, stock_price_collocation), dim=1)
    X_f_norm = torch.cat(((time_collocation - time_mean) / time_std,
                          (stock_price_collocation - stock_price_mean) / stock_price_std), dim=1)
    
    time_boundary = torch.randint(1, 100 * ub[1] + 1, (N_b, 1)).float() / 100
    X_b = torch.cat((time_boundary, torch.zeros_like(time_boundary)), dim=1)

    stock_price_exp = torch.randint(0, ub[0] + 1, (N_exp, 1)).float()
    option_price_exp = stock_price_exp - K
    u_exp = torch.maximum(option_price_exp, torch.zeros_like(option_price_exp))
    X_exp = torch.cat((tau * torch.ones_like(stock_price_exp), stock_price_exp), dim=1)

    return X_f, X_f_norm, X_b, X_exp, u_exp

X_f, X_f_norm, X_b, X_exp, u_exp = initialize_data(N_b, N_exp, N_f, lb, ub, K, tau)

# Collocation target from FDE
u_collocation = []
for instance in X_f:
    t = int(instance[0].item() * 200)
    s = int(instance[1].item())
    u_val = np.round(option_fde_prices[s, t], 3)
    u_collocation.append([u_val])
u_collocation = torch.tensor(u_collocation).float()

# -----------------------------
# Model Initialization
# -----------------------------
model = BlackScholesMertonModel1()
for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)

# -----------------------------
# Move to Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
X_f = X_f.to(device)
X_b = X_b.to(device)
X_exp = X_exp.to(device)
u_exp = u_exp.to(device)
u_collocation = u_collocation.to(device)
f_collocation = torch.zeros(N_f, 1).to(device)
u_boundary = torch.zeros(N_b, 1).to(device)

# -----------------------------
# Training
# -----------------------------
def train_model(model, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = []

    X_f.requires_grad = True

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward prediction
        u_b_pred = model(X_b)
        u_exp_pred = model(X_exp)
        u_pred = model(X_f)

        # Derivatives
        grads = torch.autograd.grad(u_pred.sum(), X_f, create_graph=True)[0]
        u_t = grads[:, 0:1]
        u_s = grads[:, 1:2]
        u_ss = torch.autograd.grad(u_s.sum(), X_f, create_graph=True)[0][:, 1:2]

        S = X_f[:, 1:2]
        f_pred = u_t + 0.5 * sigma**2 * S**2 * u_ss + (r - dividend) * S * u_s - r * u_pred

        # Losses
        loss_f = 100 * loss_fn(f_pred, f_collocation)
        loss_exp = loss_fn(u_exp_pred, u_exp)
        loss_b = loss_fn(u_b_pred, u_boundary)
        loss = loss_f + loss_exp + loss_b
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:5d} | Residual: {loss_f.item():.5f} | BC: {loss_b.item():.5f} | IC: {loss_exp.item():.5f} | Total: {loss.item():.5f}")
        history.append(loss.item())

    return history

# Train in two phases
print("Training Phase 1")
history1 = train_model(model, epochs=710, lr=8e-3)
print("Training Phase 2")
history2 = train_model(model, epochs=4700, lr=1e-3)

# -----------------------------
# Plotting Losses
# -----------------------------
plt.plot(history1 + history2)
plt.yscale("log")
plt.title("PINN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.grid(True)
plt.show()
