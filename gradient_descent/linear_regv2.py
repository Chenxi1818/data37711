import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# --- load, split, and scale data ---

def get_data(random_state=99):
    """Loads, splits, and scales the breast cancer data."""
    data = pd.read_csv('mcs_ds_edited_iter_shuffled.csv')

    X = data.iloc[:, :4]
    y = data['ale']


    print(f"The sample size is {X.shape[0]}")

    # Split data using the same random_state as your script
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Scale data for SGD
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return y_train, y_test, X_train_scaled, X_test_scaled

y_train, y_test, X_train_scaled, X_test_scaled = get_data(random_state=99)

# --- Baseline: Sklearn Linear Regression ---
def run_sklearn_model(X_train_scaled, y_train, X_test_scaled, y_test):
    """Trains and evaluates the baseline sklearn LinearRegression."""
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_baseline = lr.predict(X_test_scaled)

    # evaluate the performance using Mean Absolute Error (MAE)

    mae_baseline = mean_absolute_error(y_test, y_pred_baseline)

    return mae_baseline

mae_baseline = run_sklearn_model(X_train_scaled, y_train, X_test_scaled, y_test)

# --- JAX Linear Regression (SGD: squared error loss conditional on normal distribution of residuals) ---

# convert data to JAX arrays
X_train_jnp = jnp.array(X_train_scaled)
y_train_jnp = jnp.array(y_train)
X_test_jnp = jnp.array(X_test_scaled)
y_test_jnp = jnp.array(y_test)

n_features = X_train_jnp.shape[1]

def predict(params, x):
    """Computes f(x) = w^T * x + b"""
    w, b = params
    return jnp.dot(x, w) + b

# Define the Mean Squared Error (MSE) loss function
def loss_fn_mse(params, x, y):
    """Computes L = (y - f(x))^2"""
    y_hat = predict(params, x)
    # The prompt specifies (y - f(x))^2, so we return the mean of this.
    return jnp.mean((y - y_hat) ** 2)

# train the JAX Linear Regression
learning_rate = 0.01
num_epochs = 2000
key = jax.random.key(42)

# Initialize parameters
params = (
    jax.random.normal(key, (n_features,)),  # w
    0.0                                   # b
)

# Set up optimizer
optimizer = optax.sgd(learning_rate)
opt_state = optimizer.init(params)

# Create a jitted train step for speed
@jax.jit
def train_step(params, opt_state, x, y):
    # Get gradients using automatic differentiation
    grads = jax.grad(loss_fn_mse)(params, x, y)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Training loop
for epoch in range(num_epochs):
    params, opt_state = train_step(params, opt_state, X_train_jnp, y_train_jnp)

    # Optional: Print loss
    if (epoch + 1) % 200 == 0:
        loss_val = loss_fn_mse(params, X_train_jnp, y_train_jnp)
        print(f"Epoch {epoch+1}/{num_epochs}, MSE Loss: {loss_val:.4f}")

# Evaluate the JAX model on the test set
y_pred_jax_mse = predict(params, X_test_jnp)

# Use the same metric (MAE) as the baseline for a fair comparison
mae_jax_mse = jnp.mean(jnp.abs(y_test_jnp - y_pred_jax_mse))

print(f"JAX Model (trained on MSE loss function) MAE: {mae_jax_mse:.4f}")
print("-" * 30)

# --- JAX Linear Regression (SGD: loss function conditional on t distribution of residuals) ---

# Define the loss function conditional on t distribution of residuals
def loss_t_fn(params, x, y):
    y_hat = predict(params, x)

    residuals = y - y_hat 

    log_likelihood = jax.scipy.stats.t.logpdf(x=residuals, df=5) # return a logpdf

    return -jnp.mean(log_likelihood)

# need to reinitialize parameters
# train the JAX Linear Regression
learning_rate = 0.01
num_epochs = 2000
key_t = jax.random.key(43) # new key to initialize parameters

# Initialize parameters
params = (
    jax.random.normal(key_t, (n_features,)),  # w
    0.0                                   # b
)

# Set up optimizer
optimizer = optax.sgd(learning_rate=learning_rate)
opt_state = optimizer.init(params)

# Create a jitted train step that uses the passed loss_function
@jax.jit
def train_t_step(params, opt_state, x, y):
    # Get gradients using automatic differentiation
    grads = jax.grad(loss_t_fn)(params, x, y)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Training loop
for epoch in range(num_epochs):
    params, opt_state = train_t_step(params, opt_state, X_train_jnp, y_train_jnp)

    # Optional: Print loss
    if (epoch + 1) % 200 == 0:
        loss_val = loss_t_fn(params, X_train_jnp, y_train_jnp)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss with t-distribution: {loss_val:.4f}")

# Evaluate the JAX model on the test set
y_pred_jax_t = predict(params, X_test_jnp)

# Use the same metric (MAE) as the baseline for a fair comparison
mae_jax_t = jnp.mean(jnp.abs(y_test_jnp - y_pred_jax_t))

print(f"JAX Model (trained on loss function conditional on t distribution of residuals) MAE: {mae_jax_t:.4f}")
print("-" * 30)

# Final comparison
print("\n--- 📊 Final Comparison (Mean Absolute Error) ---")
print(f"Sklearn Baseline MAE: {mae_baseline:.4f}")
print(f"JAX Model (norm-dis) MAE:        {mae_jax_mse:.4f}")
print(f"JAX Model (t-dis) MAE:        {mae_jax_t:.4f}")

