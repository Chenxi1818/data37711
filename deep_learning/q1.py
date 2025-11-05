#%%
import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#%%
data = pd.read_parquet("taxi_trips.parquet")

data.info()

y = data['trip_duration']

X = data.drop('trip_duration', axis='columns')

# splits
seed = 99

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed)

# standardize features
scaler = StandardScaler().fit(X_train)

X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

#%%
# baseline model that outputs conditional mean and variance
# simple linear regression baseline
lr_simple = LinearRegression()
lr_simple.fit(X_train_s, y_train)

# %%

#%% ------------------------------------------------------------
# PyTorch MLP model that outputs conditional mean and variance
#---------------------------------------------------------------

# build PyTorch datasets
train_ds = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32),
                         torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1))
val_ds = TensorDataset(torch.tensor(X_val_s, dtype=torch.float32),
                       torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1))
test_ds = TensorDataset(torch.tensor(X_test_s, dtype=torch.float32),
                        torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1))

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

#%% ------------------------------------------------------------
# Define MLP model (μ and logσ² heads)
#---------------------------------------------------------------

class MLPProb(nn.Module):
    def __init__(self, in_dim, hidden=(128, 64), dropout=0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        self.feature = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last, 1)
        self.logvar_head = nn.Linear(last, 1)
        nn.init.constant_(self.logvar_head.bias, -1.0)  # start with reasonable log-variance

    def forward(self, x):
        h = self.feature(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


# Gaussian NLL loss
def gaussian_nll_loss(mu, logvar, y):
    # equivalent to 0.5*(log σ² + (y-μ)² / σ²)
    return torch.mean(0.5 * (logvar + (y - mu)**2 * torch.exp(-logvar)))

#%% ------------------------------------------------------------
# Training loop
#---------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPProb(in_dim=X_train_s.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 100

best_val = float('inf')
best_state = None

# --- New variables for early stopping ---
patience = 10 
patience_counter = 0

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        mu, logvar = model(xb)
        loss = gaussian_nll_loss(mu, logvar, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    total_loss /= len(train_loader.dataset)

    # validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            mu, logvar = model(xb)
            loss = gaussian_nll_loss(mu, logvar, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

# --- Modified early stopping logic ---
    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        patience_counter = 0  # Reset patience because we found a better model
    else:
        patience_counter += 1  # Increment patience
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val:.4f}")
        break  # <-- This is the "stopping" part
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

# restore best weights
model.load_state_dict(best_state)
print("\nTraining complete. Best Val Loss:", best_val)

#%% ------------------------------------------------------------
# Evaluate model on test set
#---------------------------------------------------------------
# NLL
def gaussian_nll(y_true, mu_pred, sigma2_pred):
    """Compute mean Gaussian negative log-likelihood."""
    return np.mean(0.5 * (np.log(sigma2_pred) + ((y_true - mu_pred) ** 2) / sigma2_pred))


# ----------- baseline

test_pred = lr_simple.predict(X_test_s)
residuals_pred = y_test - test_pred
sigma2_pred = np.mean(residuals_pred**2)
mu_pred = np.mean(test_pred)

nll_lr_simple = gaussian_nll(y_test, mu_pred, sigma2_pred)
mse = mean_squared_error(y_test, test_pred)
mae = mean_absolute_error(y_test, test_pred)

print("\n=== Simple linear regression on Test Performance ===")
print(f"Baseline | mse: {mse:.3f} | mae: {mae:.3f} | nll: {nll_lr_simple:.3f}")


# %%
# evaluation on MLP
model.eval()
y_true, y_pred_mu, y_pred_std = [], [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        mu, logvar = model(xb)
        std = torch.exp(0.5 * logvar)
        y_true.append(yb.numpy())
        y_pred_mu.append(mu.cpu().numpy())
        y_pred_std.append(std.cpu().numpy())

y_true = np.vstack(y_true)
y_pred_mu = np.vstack(y_pred_mu)
y_pred_std = np.vstack(y_pred_std)
y_pred_sigma2 = y_pred_std ** 2

mse = mean_squared_error(y_true, y_pred_mu)
mae = mean_absolute_error(y_true, y_pred_mu)
nll = gaussian_nll(y_true, y_pred_mu, y_pred_sigma2)

print("\n=== MLP on Test Performance ===")
print(f"MLP (mean+variance) | MSE: {mse:.3f} | MAE: {mae:.3f} | NLL: {nll:.3f}")
#%%
y_pred_mu

#%%
from scipy.stats import norm

# threshold (45 minutes)
threshold = 45.0

# compute probabilities
prob_less_than_45 = norm.cdf(threshold, y_pred_mu, y_pred_std)

print("Predicted probability (Y < 45 min):")
print(prob_less_than_45)

# %%
# ensure both are flat 1D arrays
y_true_binary = np.array((y_test < 45).astype(int)).flatten()
y_pred_binary = np.array((prob_less_than_45 >= 0.5).astype(int)).flatten()

# now compute accuracy
accuracy = np.mean(y_true_binary == y_pred_binary)
print(f"Classification accuracy (threshold=0.5): {accuracy:.3f}")