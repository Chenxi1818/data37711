#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
#%%
# --------------------------------------------------------
# 1. Load the wine dataset
# --------------------------------------------------------
df = pd.read_csv(r"data\winequality-white.csv", sep=";")

X = df.drop(columns=["quality"]).values
y = df["quality"].values

# --------------------------------------------------------
# 2. Split data: 70% train, 10% validation, 20% test
# --------------------------------------------------------
seed = 99
np.random.seed(seed)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=seed
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2/3, random_state=seed
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))

# --------------------------------------------------------
# 3. Choose two model classes & define grids
# --------------------------------------------------------
# Random Forest hyperparameter grid
rf_param_grid = {
    "n_estimators": [20, 50, 100, 150, 200, 300, 400, 500, 600, 800],
    "max_depth": [None, 5, 10, 20, 30, 40, 50, 60, 70, 80],
    "min_samples_split": [2, 4, 6, 8, 10]
}

# KNN hyperparameter grid
knn_param_grid = {
    "n_neighbors": [i for i in range(1, 51)],
    "algorithm": ["auto", "ball_tree", "kd_tree"],
    "leaf_size": [30, 60],
    "weights": ["uniform", "distance"]
}

# --------------------------------------------------------
# 4. Grid search on validation set
# --------------------------------------------------------


def grid_search(model_class, param_grid):
    best_params = None
    best_val_mse = float("inf")

    val_mses = []
    test_mses = []

    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        model = model_class(**params)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_mse = mean_squared_error(y_val, val_pred)
        test_mse = mean_squared_error(y_test, test_pred)

        val_mses.append(val_mse)
        test_mses.append(test_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_params = params

    return best_params, best_val_mse, np.array(val_mses), np.array(test_mses)



# --------- Random Forest search ---------
rf_best_params, rf_best_val_mse, rf_val_mses, rf_test_mses = grid_search(RandomForestRegressor, rf_param_grid)
rf_best_params, rf_best_val_mse

# Fit best RF and evaluate on test set
rf_best = RandomForestRegressor(**rf_best_params)
rf_best.fit(X_train, y_train)
rf_test_pred = rf_best.predict(X_test)
rf_test_mse = mean_squared_error(y_test, rf_test_pred)

# --------- KNN search ---------
knn_best_params, knn_best_val_mse, knn_val_mses, knn_test_mses = grid_search(KNeighborsRegressor, knn_param_grid)

knn_best = KNeighborsRegressor(**knn_best_params)
knn_best.fit(X_train, y_train)
knn_test_pred = knn_best.predict(X_test)
knn_test_mse = mean_squared_error(y_test, knn_test_pred)

#%%
# --------------------------------------------------------
# 5. Print results
# --------------------------------------------------------

print("=== Random Forest Results ===")
print("Best hyperparameters:", rf_best_params)
print("Validation MSE:", rf_best_val_mse)
print("Test MSE:", rf_test_mse)

print("\n=== KNN Results ===")
print("Best hyperparameters:", knn_best_params)
print("Validation MSE:", knn_best_val_mse)
print("Test MSE:", knn_test_mse)

#%%
# --------------------------------------------------------
# 5. Plot results
# --------------------------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))

# Random Forest points
plt.scatter(rf_val_mses, rf_test_mses, alpha=0.5, label="Random Forest")

# KNN points (optional)
plt.scatter(knn_val_mses, knn_test_mses, alpha=0.5, label="KNN")

plt.xlabel("Validation MSE")
plt.ylabel("Test MSE")
plt.title("Validation Risk vs Test Risk Across Hyperparameters")

# Correct identity line
global_min = min(
    rf_val_mses.min(), rf_test_mses.min(),
    knn_val_mses.min(), knn_test_mses.min()
)

global_max = max(
    rf_val_mses.max(), rf_test_mses.max(),
    knn_val_mses.max(), knn_test_mses.max()
)

plt.plot([global_min, global_max],
         [global_min, global_max],
         linestyle="--", color="black")

plt.legend()
plt.show()