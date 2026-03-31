import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
train = pd.read_csv("stock_2_testing/stock_2_train.csv")

# Split features and target
X = train.drop(columns=["target"])
y = train["target"]

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- MODELS ----
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# ---- TRAIN + EVALUATE ----
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    mse = mean_squared_error(y_val, preds)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    print(f"\n{name}")
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2:", r2)