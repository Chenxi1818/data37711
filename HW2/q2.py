#%%
import pandas as pd

# Load data
df = pd.read_csv("retail.csv")

# Basic overview
print(df.info())

# %%
print(df.head())

# %%
df["StockCode"] = df["StockCode"].astype("category")
df["CustomerID"] = df["CustomerID"].astype("category")
# %%
df.info()
# %%
# Number of unique values in each categorical column
print("Unique StockCodes:", df['StockCode'].nunique())
print("Unique CustomerIDs:", df['CustomerID'].nunique())

# %%
# ------ target encoding --------
# -------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from category_encoders import TargetEncoder
from sklearn.ensemble import GradientBoostingRegressor

# %%
# --- 1. prepare data
# Features and target
X = df[['StockCode', 'CustomerID']]
y = df['Quantity']

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# target encoding for features
encoder = TargetEncoder(cols=['StockCode', 'CustomerID'])
encoder.fit(X_train, y_train)

# Apply the learned encoding
X_train_enc = encoder.transform(X_train)
X_val_enc = encoder.transform(X_val)
X_test_enc = encoder.transform(X_test)


# %%
# --- 2. build xgboost
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=1e-3,
    max_depth=6,
    random_state=42,
    early_stopping_rounds=10,  # Still fine here
    eval_metric="rmse"         # ✅ Move here
)

#%%
# --- 3. train xgboost

xgb.fit(
    X_train_enc, y_train,
    eval_set=[(X_val_enc, y_val)],
    verbose=False
)


# %%
# --- 4. test xgboost
y_pred_test = xgb.predict(X_test_enc)

# Evaluation metrics
def evaluate(y_true, y_pred, name="Set"):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} — MSE: {mse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

evaluate(y_test, y_pred_test, "Test")
# %%
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred_test, alpha=0.3)
plt.xlabel("Actual Quantity")
plt.ylabel("Predicted Quantity")
plt.title("Predicted vs. Actual Quantity")
plt.show()

# %%
# --- representation-learning approach using embeddings
# ----------------------------
# %%
# --- 1. prepare data
# Encode as integer IDs
import numpy as np

df["StockCode_cat"] = df["StockCode"].cat.codes
df["CustomerID_cat"] = df["CustomerID"].cat.codes

df[["StockCode", "StockCode_cat", "CustomerID", "CustomerID_cat"]].head()

df["CustomerID_cat"].nunique()

# Features and target
X = df[['StockCode_cat', 'CustomerID_cat']].values
y = df['Quantity'].values

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


#  choose embedding size

n_stock = df['StockCode'].nunique() 
n_customer = df['CustomerID'].nunique()  

# a simpler, practical heuristic is: 
# embed_dim = int(np.ceil(num_categories ** 0.25))
stock_embed_dim = int(np.ceil(n_stock ** 0.25))  
customer_embed_dim = int(np.ceil(n_customer ** 0.25)) 

# %%
# --- 2. Build the neural network

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# Inputs
stock_in = Input(shape=(1,), name="StockCode")
cust_in = Input(shape=(1,), name="CustomerID")

# Embeddings
stock_emb = layers.Embedding(input_dim=n_stock+1, output_dim=stock_embed_dim, name="stock_emb")(stock_in)
cust_emb = layers.Embedding(input_dim=n_customer+1, output_dim=customer_embed_dim, name="cust_emb")(cust_in)

# Flatten embeddings
stock_vec = layers.Flatten()(stock_emb)
cust_vec = layers.Flatten()(cust_emb)

# Concatenate
x = layers.Concatenate()([stock_vec, cust_vec])

# Dense layers
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(1)(x)  # Regression output

# Model
model = Model(inputs=[stock_in, cust_in], outputs=x)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# %%
# -- 3. Train the model (with early stopping)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    [X_train[:, 0], X_train[:, 1]], y_train,
    validation_data=([X_val[:, 0], X_val[:, 1]], y_val),
    epochs=100,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

# %%
# --- 4. test nn
y_pred_test = model.predict([X_test[:, 0], X_test[:, 1]]).flatten()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Test — MSE: {mse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

# %%
# -----------------------
# --- hybrid modeling
# --- -------------------

#%%
# --- 1. Prepare the data

# a. Label-encoded features (for embeddings)
df["StockCode"] = df["StockCode"].astype("category")
df["CustomerID"] = df["CustomerID"].astype("category")

df["StockCode_cat"] = df["StockCode"].cat.codes
df["CustomerID_cat"] = df["CustomerID"].cat.codes

# b. Target-encoded features (for numeric input)
X = df[["StockCode", "CustomerID"]]
y = df["Quantity"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

encoder = TargetEncoder(cols=["StockCode", "CustomerID"])
encoder.fit(X_train, y_train)

X_train_enc = encoder.transform(X_train)
X_val_enc = encoder.transform(X_val)
X_test_enc = encoder.transform(X_test)

# c. Add categorical integer IDs (for embedding input)
# Add categorical codes to each split (using original df categories)
X_train_enc["StockCode_cat"] = X_train["StockCode"].astype("category").cat.codes
X_train_enc["CustomerID_cat"] = X_train["CustomerID"].astype("category").cat.codes

X_val_enc["StockCode_cat"] = X_val["StockCode"].astype("category").cat.codes
X_val_enc["CustomerID_cat"] = X_val["CustomerID"].astype("category").cat.codes

X_test_enc["StockCode_cat"] = X_test["StockCode"].astype("category").cat.codes
X_test_enc["CustomerID_cat"] = X_test["CustomerID"].astype("category").cat.codes

#%%
# --- 2. Build the hybrid neural network
# a. get the n of dimensions

n_stock = df["StockCode"].nunique()    # 3665
n_customer = df["CustomerID"].nunique()  # 4339
stock_embed_dim = int(np.ceil(n_stock ** 0.25))   
customer_embed_dim = int(np.ceil(n_customer ** 0.25))  


# b. build hybrid neural network
# Inputs
stock_in = Input(shape=(1,), name="StockCode_cat")
cust_in = Input(shape=(1,), name="CustomerID_cat")
numeric_in = Input(shape=(2,), name="TargetEncoded")  # 2 numeric features

# Embeddings
stock_emb = layers.Embedding(input_dim=n_stock+1, output_dim=stock_embed_dim)(stock_in)
cust_emb = layers.Embedding(input_dim=n_customer+1, output_dim=customer_embed_dim)(cust_in)

# Flatten embeddings
stock_vec = layers.Flatten()(stock_emb)
cust_vec = layers.Flatten()(cust_emb)

# Concatenate everything (horizontally)
x = layers.Concatenate()([stock_vec, cust_vec, numeric_in])

# Dense layers
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation="relu")(x)
out = layers.Dense(1)(x)

# Model
model = Model(inputs=[stock_in, cust_in, numeric_in], outputs=out)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

# %%
# --- 3. Train the model

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    [X_train_enc["StockCode_cat"], X_train_enc["CustomerID_cat"], 
     X_train_enc[["StockCode", "CustomerID"]]],
    y_train,
    validation_data=(
        [X_val_enc["StockCode_cat"], X_val_enc["CustomerID_cat"], 
         X_val_enc[["StockCode", "CustomerID"]]],
        y_val
    ),
    epochs=100,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

# %%
# --- 4. test model
y_pred_test = model.predict([
    X_test_enc["StockCode_cat"],
    X_test_enc["CustomerID_cat"],
    X_test_enc[["StockCode", "CustomerID"]]
]).flatten()

mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Test — MSE: {mse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

# %%
