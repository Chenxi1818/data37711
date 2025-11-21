#%%
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#%%
# ---------------------------
# Run the experiment
# ---------------------------
def run_gb_experiment(csv_path, dataset_name):
    print(f"\n==============================")
    print(f" Results for {dataset_name}")
    print(f"==============================")

    data = pd.read_csv(csv_path, header=None)
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=99
    )

    n_estimators_list = [1, 10, 100, 300, 1000]
    train_errs = []
    test_errs = []

    for n in n_estimators_list:
        model = GradientBoostingClassifier(
            n_estimators=n,
            learning_rate=0.1,
            max_depth=3,
            random_state=1
        )
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_err = 1 - accuracy_score(y_train, train_pred)
        test_err = 1 - accuracy_score(y_test, test_pred)

        train_errs.append(train_err)
        test_errs.append(test_err)

        print(f"{n} trees → train error: {train_err:.4f}, test error: {test_err:.4f}")

    return np.array(train_errs), np.array(test_errs)



dset0_train, dset0_test = run_gb_experiment('data/dset0.csv', "dset0")
dset1_train, dset1_test = run_gb_experiment('data/dset1.csv', "dset1")



# ---------------------------
# Plotting
# ---------------------------
import matplotlib.pyplot as plt
def plot_train_test(train_errs, test_errs, dset_name: str):
    steps = [1, 10, 100, 300, 1000]
    plt.figure(figsize=(8, 6))

    plt.plot(steps, train_errs, marker="o", label="Train Error")
    plt.plot(steps, test_errs, marker="o", label="Test Error")

    plt.xlabel("Number of Trees")
    plt.ylabel("Error Rate")
    plt.title(f"Train vs Test Error ({dset_name})")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()


plot_train_test(dset0_train, dset0_test, "dataset 0")
plot_train_test(dset1_train, dset1_test, "dataset 1")

# %%
data0 = pd.read_csv('data/dset0.csv', header=None)
data1 = pd.read_csv('data/dset1.csv', header=None)
# %%
data0.head()
data0

#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def compute_train_error(data):
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    clf = DecisionTreeClassifier(max_depth=None, random_state=0)
    clf.fit(X, y)
    pred = clf.predict(X)
    return 1 - accuracy_score(y, pred)

err0 = compute_train_error(data0)
err1 = compute_train_error(data1)

print("dset0 train error:", err0)
print("dset1 train error:", err1)

# %%
