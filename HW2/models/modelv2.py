import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

class RandomKitchenSinks:
    """
    Random Kitchen Sinks (RKS) featurization with ridge regression.
    Implements both sine and cosine random features per the RBF kernel approximation.
    """
    def __init__(self, h: int = 100, sigma: float = 1.0, ridge_lambda: float = 1e-3, random_state: int = 42):
        """
        h: number of random feature pairs (sin, cos)
        sigma: kernel scale parameter controlling feature spread
        ridge_lambda: ridge regression regularization strength
        """
        self.h = h
        self.sigma = sigma
        self.ridge_lambda = ridge_lambda
        self.random_state = random_state
        np.random.seed(random_state)
        self.W = None
        self.b = None
        self.coef_ = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        """
        Compute random Fourier features using both sin and cos.
        z_i = [sin(Wx + b), cos(Wx + b)]
        """
        if self.W is None:
            d = X.shape[1]
            self.W = np.random.normal(0, 1.0 / self.sigma, size=(self.h, d))  # w_j ~ N(0, I_d / σ²)
            self.b = np.random.uniform(0, 2 * np.pi, size=self.h)             # b_j ~ U[0, 2π]

        projection = X @ self.W.T + self.b
        phi_sin = np.sin(projection)
        phi_cos = np.cos(projection)
        phi = np.concatenate([phi_sin, phi_cos], axis=1)  # shape (n, 2h)

        # Optional bias term (intercept)
        phi = np.concatenate([np.ones((phi.shape[0], 1)), phi], axis=1)
        return phi

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit ridge regression model using closed-form solution:
        w = (ΦᵀΦ + λI)⁻¹ Φᵀy
        """
        phi = self._features(X)
        n_features = phi.shape[1]

        A = phi.T @ phi + self.ridge_lambda * np.eye(n_features)
        b = phi.T @ y
        self.coef_ = np.linalg.solve(A, b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using learned coefficients.
        """
        phi = self._features(X)
        return phi @ self.coef_


def sweep_rks_model(sweep_h: List[int], X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    sigma=1.0, ridge_lambda=1e-3) -> Tuple:
    """
    Sweep over number of random feature pairs h and return best model and losses.
    """
    train_losses, val_losses = [], []
    best_model, best_val_mse, best_h = None, float('inf'), None

    for h in sweep_h:
        model = RandomKitchenSinks(h=h, sigma=sigma, ridge_lambda=ridge_lambda)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_mse = np.mean((y_train - y_train_pred) ** 2)
        val_mse = np.mean((y_val - y_val_pred) ** 2)

        train_losses.append(train_mse)
        val_losses.append(val_mse)

        print(f"h={h:4d} | Train MSE={train_mse:.5f} | Val MSE={val_mse:.5f}")

        if val_mse < best_val_mse:
            best_model, best_val_mse, best_h = model, val_mse, h

    # Plot training and validation MSE vs. h
    plt.figure(figsize=(8, 5))
    plt.plot(sweep_h, train_losses, marker='o', label='Training MSE')
    plt.plot(sweep_h, val_losses, marker='o', label='Validation MSE')
    plt.xlabel('Number of Random Feature Pairs (h)')
    plt.ylabel('Mean Squared Error')
    plt.title('RKS Ridge Regression Hyperparameter Sweep')
    plt.legend()
    plt.grid(True)
    plt.savefig('RKS_Sweep.png', dpi=300)
    plt.show()

    print(f"\nBest h = {best_h}, Validation MSE = {best_val_mse:.5f}")
    return best_model, best_val_mse, best_h

"""
For 1-hidden-layer MLP (MLP)
"""
class MLP(nn.Module):
    """1-hidden-layer MLP for regression with early stopping."""
    def __init__(self, d_in: int, d_hidden: int = 128, d_out: int = 1,
                 lr: float = 1e-3, batch_size: int = 128,
                 activation: str = "relu", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size

        # Choose activation
        if activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "sigmoid":
            act_fn = nn.Sigmoid()
        else:
            act_fn = nn.ReLU()

        # Build network
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_out)
        ).to(self.device)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            num_epochs: int = 300, patience: int = 20, min_delta: float = 1e-4):
        """
        Fit the MLP with early stopping.
        - patience: number of epochs with no improvement before stopping
        - min_delta: minimum change in validation loss to qualify as improvement
        """
        # Convert to torch tensors
        X_train = torch.from_numpy(X_train).float().to(self.device)
        y_train = torch.from_numpy(y_train).float().view(-1, 1).to(self.device)
        if X_val is not None:
            X_val = torch.from_numpy(X_val).float().to(self.device)
            y_val = torch.from_numpy(y_val).float().view(-1, 1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        best_val_loss = float('inf')
        epochs_no_improve = 0

        best_state = self.net.state_dict()

        self.net.train()
        for epoch in range(num_epochs):
            # ---- Training ----
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self.net(X_batch)
                loss = loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(loader)

            # ---- Validation ----
            val_loss = None
            if X_val is not None:
                self.net.eval()
                with torch.no_grad():
                    preds_val = self.net(X_val)
                    val_loss = loss_fn(preds_val, y_val).item()
                self.net.train()

                # Early stopping check
                if val_loss + min_delta < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = self.net.state_dict()
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1} | Best Val Loss: {best_val_loss:.5f}")
                    self.net.load_state_dict(best_state)
                    break

            # ---- Logging ----
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.5f}" +
                      (f" | Val Loss: {val_loss:.5f}" if val_loss else ""))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return model predictions as NumPy array."""
        self.net.eval()
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            return self.forward(X).cpu().numpy()


def mlp_sweep(sweep_values: List[int], X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              lr: float = 1e-3, batch_size: int = 128,
              num_epochs: int = 2000, device: str = "cpu", patience: int = 20) -> Tuple:
    """
    Sweep over hidden layer dimension and plot train/val MSE.
    """
    train_losses, val_losses = [], []
    best_model, best_val_mse, best_hidden = None, float('inf'), None

    for d_hidden in sweep_values:
        model = MLP(d_in=X_train.shape[1], d_hidden=d_hidden, lr=lr,
                    batch_size=batch_size, device=device)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, num_epochs=num_epochs, patience=patience)


        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_mse = np.mean((y_train - y_train_pred) ** 2)
        val_mse = np.mean((y_val - y_val_pred) ** 2)

        train_losses.append(train_mse)
        val_losses.append(val_mse)

        print(f"d_hidden={d_hidden:4d} | Train MSE={train_mse:.5f} | Val MSE={val_mse:.5f}")

        if val_mse < best_val_mse:
            best_model, best_val_mse, best_hidden = model, val_mse, d_hidden

    # Plot training & validation losses
    plt.figure(figsize=(8, 5))
    plt.plot(sweep_values, train_losses, marker='o', label='Training MSE')
    plt.plot(sweep_values, val_losses, marker='o', label='Validation MSE')
    plt.xlabel('Hidden Layer Dimension (d_hidden)')
    plt.ylabel('Mean Squared Error')
    plt.title('MLP Hyperparameter Sweep')
    plt.legend()
    plt.grid(True)
    plt.savefig('MLP_Sweep.png', dpi=300)  # ✅ Save before show
    plt.show()

    print(f"\nBest hidden size = {best_hidden}, Validation MSE = {best_val_mse:.5f}")
    return best_model, best_val_mse, best_hidden
