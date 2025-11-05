#%%
import numpy as np
import pandas as pd

#%%
df = pd.read_csv('train.csv')

# %%
print(df.head())
print(df.describe())
# %%
# ---- plot the grid 
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))

sc = plt.scatter(df['x1'], df['x2'], c=df['y'], cmap='viridis', s=100)
plt.colorbar(sc, label='y value')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('2D Color Map of y over (x1, x2)')
plt.tight_layout()
plt.savefig('true.png', dpi=300, bbox_inches='tight')  # ✅ Save before show
plt.show()


# %%
# ---- RKS with Ridge regression


# %%
# --- 1. prepare data
from dataloader import HW2Dataset

ds = HW2Dataset.from_csv("train.csv", "val.csv", "test.csv")
print(ds.train.X.shape, ds.train.y.shape)
print(ds.val.X.shape, ds.val.y.shape)
print(ds.test.X.shape, ds.test.y.shape)

X_train = ds.train.X
y_train = ds.train.y

X_val = ds.val.X
y_val = ds.val.y

X_test = ds.test.X
y_test = ds.test.y

#%%
# --- 2. train RKS with Ridge regression
from models.modelv2 import sweep_rks_model

sweep_h = [10, 50, 100, 200, 400]
best_model, best_val_mse, best_h = sweep_rks_model(
    sweep_h, X_train, y_train, X_val, y_val, sigma=1.0, ridge_lambda=1e-3
)

# %%
# --- 3. train one-hidden layer MLP
import torch

from models.modelv2 import mlp_sweep

#sweep_hidden = [i for i in range(1,6)]

sweep_hidden = [64, 128, 256, 512, 768]

best_model, best_val_mse, best_hidden = mlp_sweep(
    sweep_values=sweep_hidden,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    lr=1e-3, batch_size=32, num_epochs=2000, patience=20,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# %%
# --- comparison
# -----------------------------
# RKS
from models.modelv2 import RandomKitchenSinks

RKSmodel = RandomKitchenSinks(h=best_h, sigma=1.0, ridge_lambda=1e-3)

RKSmodel.fit(X_train, y_train)

y_test_pred = RKSmodel.predict(X_test)

test_mse_rks = np.mean((y_test - y_test_pred) ** 2)

# %%
from models.modelv2 import MLP
import numpy as np

MLPmodel = MLP(d_in=X_test.shape[1], d_hidden=best_hidden, lr=1e-3,
                    batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu")

MLPmodel.fit(X_train, y_train, X_val=X_val, y_val=y_val, num_epochs=2000, patience=20)

y_test_pred = MLPmodel.predict(X_test)

test_mse_mlp = np.mean((y_test - y_test_pred) ** 2)

print(f"\nBest RKS h = {best_h}")
print(f"Best MLP hidden size = {best_hidden}")
print(f"Test MSE (RKS): {test_mse_rks:.4f}")
print(f"Test MSE (MLP): {test_mse_mlp:.4f}")

# %%
# --- plot
from plotting import plot_surface_2d
# %%
plot_surface_2d(RKSmodel.predict, (-3, 3), (-3, 3), title="RKS surface", filename="RKS")

plot_surface_2d(MLPmodel.predict, (-3, 3), (-3, 3), title="MLP surface", filename="MLP")
