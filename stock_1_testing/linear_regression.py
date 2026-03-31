import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------
# LOAD DATA
# ------------------------
train = pd.read_csv("stock_1_testing/stock_1_train.csv")
test = pd.read_csv("stock_1_testing/stock_1_test.csv")

X = train.drop(columns=["target"])
y = train["target"]

# ------------------------
# TRAIN / VALIDATION SPLIT
# ------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------
# MODELS
# ------------------------
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# ------------------------
# TRAIN + EVALUATE + PICK BEST
# ------------------------
best_model = None
best_mae = float("inf")

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

    # Select best model based on MAE
    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_name = name

print(f"\n✅ Best model: {best_name} (MAE={best_mae})")

# ------------------------
# RETRAIN BEST MODEL ON FULL DATA
# ------------------------
best_model.fit(X, y)

# ------------------------
# PREDICT ON TEST DATA
# ------------------------
test_preds = best_model.predict(test)

# ------------------------
# SAVE SUBMISSION
# ------------------------
submission = pd.DataFrame({
    "target": test_preds
})

submission.to_csv("stock_2_testing/stock_1_predictions.csv", index=False)

print("\n📁 Predictions saved to stock_1_predictions.csv")
