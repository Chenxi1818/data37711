import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
import optax
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def get_data(random_state=99):
    """Loads, splits, and scales the breast cancer data."""
    data = load_breast_cancer()
    X = data.data
    y = data.target

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

# baseline from the sklearn’s implementation
def run_sklearn_model(X_train_scaled, y_train, X_test_scaled, y_test):
    """Trains and evaluates the baseline sklearn LogisticRegression."""
    clf = LogisticRegression(random_state=0, max_iter=10000)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    return accuracy_score(y_test, y_pred)

# logit model use sigmoid function to convert the logits into probability.
# probit model use norm cdf 
# A small value to prevent log(0)
EPSILON = 1e-7

def run_jax_model(X_train_jnp, y_train_jnp, X_test_jnp, y_test_jnp, 
                  squashing_fn, key, epochs=2000, lr=0.01):
    """
    A generic function to train and evaluate a JAX model (Logistic or Probit).
    
    Args:
        squashing_fn: The function to convert logits to probabilities 
                      (e.g., jax.nn.sigmoid or jax.scipy.stats.norm.cdf).
    """
    n_features = X_train_jnp.shape[1]

    # --- 1. Define Model & Loss ---
    def predict_logits(params, x):
        w, b = params
        return jnp.dot(x, w) + b

    def predict_probs(params, x):
        logits = predict_logits(params, x)
        return squashing_fn(logits) # This is the only part that changes!

    def loss_fn(params, x, y):
        probs = predict_probs(params, x)
        probs_clipped = jnp.clip(probs, EPSILON, 1.0 - EPSILON)
        bce = y * jnp.log(probs_clipped) + (1 - y) * jnp.log(1.0 - probs_clipped)
        return -jnp.mean(bce)

    # --- 2. Initialize ---
    params = (
        jax.random.normal(key, (n_features,)) * 0.01,  # w
        0.0                                          # b
    )
    optimizer = optax.sgd(learning_rate=lr)
    opt_state = optimizer.init(params)

    # --- 3. Training Loop ---
    @jax.jit
    def train_step(params, opt_state, x, y):
        grads = jax.grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    for epoch in range(epochs):
        params, opt_state = train_step(params, opt_state, X_train_jnp, y_train_jnp)

    # --- 4. Evaluate ---
    test_probs = predict_probs(params, X_test_jnp)
    predicted_classes = (test_probs > 0.5).astype(jnp.int32)
    accuracy = jnp.mean(predicted_classes == y_test_jnp)
    return accuracy

def main():
    """Main function to run all three models and compare results."""
    
    # Store all results here
    results = {}
    
    # --- 1. Get and Prepare Data ---
    RANDOM_STATE = 99
    y_train, y_test, X_train_scaled, X_test_scaled = get_data(random_state=RANDOM_STATE)

    # --- 2. Model 1: Sklearn (Baseline) ---
    print("Running Sklearn Logistic Regression...")
    acc_sklearn = run_sklearn_model(X_train_scaled, y_train, X_test_scaled, y_test)
    results['Sklearn Logistic'] = acc_sklearn
    print(f"Accuracy: {acc_sklearn:.4f}\n")

    # --- 3. Prepare Data for JAX ---
    X_train_jnp = jnp.array(X_train_scaled)
    y_train_jnp = jnp.array(y_train)
    X_test_jnp = jnp.array(X_test_scaled)
    y_test_jnp = jnp.array(y_test)
    key = jax.random.key(RANDOM_STATE) # Use the same random key

    # --- 4. Model 2: JAX Logistic ---
    print("Running JAX Logistic Regression (SGD)...")
    acc_jax_logistic = run_jax_model(
        X_train_jnp, y_train_jnp, X_test_jnp, y_test_jnp,
        squashing_fn=jax.nn.sigmoid,  # Pass sigmoid
        key=key,
        epochs=10000 # Using original epoch count
    )
    results['JAX Logistic'] = acc_jax_logistic
    print(f"Accuracy: {acc_jax_logistic:.4f}\n")

    # --- 5. Model 3: JAX Probit ---
    print("Running JAX Probit Regression (SGD)...")
    acc_jax_probit = run_jax_model(
        X_train_jnp, y_train_jnp, X_test_jnp, y_test_jnp,
        squashing_fn=jax.scipy.stats.norm.cdf,        # Pass jax.scipy.stats.norm.cdf
        key=key,
        epochs=10000 # Using original epoch count
    )
    results['JAX Probit'] = acc_jax_probit
    print(f"Accuracy: {acc_jax_probit:.4f}\n")
    
    # --- 6. Final Comparison ---
    print("--- 📊 Final Model Comparison ---")
    for model_name, accuracy in results.items():
        print(f"{model_name:<20}: {accuracy:.4f}")

if __name__ == "__main__":
    main()