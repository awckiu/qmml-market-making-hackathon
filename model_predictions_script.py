import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    LeaveOneOut,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    HuberRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ==========================================
# CONFIG
# ==========================================
N_STOCKS = 9

SUMMARY_OUTPUT = "summary.csv"
RMSE_OUTPUT = "model_rmse_results.csv"

RANDOM_STATE = 42

# 🔥 NEW: base GitHub URL
BASE_URL = "https://raw.githubusercontent.com/Queen-Mary-Machine-Learning-Society/QMML/main/Hackathons/MarketMaking/hackathon_data/"

# ==========================================
# STOCK RULE TABLE
# ==========================================
STOCK_RULES = {
    1: {"group": "large", "spread_mult": 1.00},
    2: {"group": "small", "spread_mult": 1.35},
    3: {"group": "tiny",  "spread_mult": 1.90},
    4: {"group": "large", "spread_mult": 1.00},
    5: {"group": "small", "spread_mult": 1.35},
    6: {"group": "tiny",  "spread_mult": 1.75},
    7: {"group": "large", "spread_mult": 1.00},
    8: {"group": "small", "spread_mult": 1.25},
    9: {"group": "tiny",  "spread_mult": 1.85},
}

# ==========================================
# HELPERS
# ==========================================
def make_scaled_model(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

def make_tree_model(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model)
    ])

def choose_cv(stock_i, n_rows):
    group = STOCK_RULES[stock_i]["group"]
    if group == "tiny":
        return LeaveOneOut()
    elif group == "small":
        return RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    else:
        return KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def rmse_cv(model, X, y, cv):
    scores = cross_val_score(
        model, X, y,
        cv=cv,
        scoring="neg_root_mean_squared_error"
    )
    return -scores.mean()

def confidence_from_rmse(adjusted_rmse, fair_value):
    denom = max(abs(fair_value), 1e-8)
    rel_error = adjusted_rmse / denom
    return round(float(1 / (1 + rel_error)), 4)

def format_model_price(model_name, price):
    return f"{model_name}: {price:.6f}"

# ==========================================
# MODEL SELECTION
# ==========================================
def get_models_for_stock(stock_i, n_rows):
    group = STOCK_RULES[stock_i]["group"]

    if group == "tiny":
        return {
            "Ridge": make_scaled_model(Ridge()),
            "Lasso": make_scaled_model(Lasso(alpha=0.01, max_iter=20000)),
            "ElasticNet": make_scaled_model(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000)),
            "BayesianRidge": make_scaled_model(BayesianRidge()),
            "HuberRegressor": make_scaled_model(HuberRegressor()),
            "LinearRegression": make_scaled_model(LinearRegression()),
            "SVR": make_scaled_model(SVR()),
            "KNN": make_scaled_model(KNeighborsRegressor(n_neighbors=2))
        }

    elif group == "small":
        return {
            "LinearRegression": make_scaled_model(LinearRegression()),
            "Ridge": make_scaled_model(Ridge()),
            "Lasso": make_scaled_model(Lasso(alpha=0.01, max_iter=20000)),
            "ElasticNet": make_scaled_model(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000)),
            "RandomForest": make_tree_model(RandomForestRegressor(n_estimators=100)),
            "ExtraTrees": make_tree_model(ExtraTreesRegressor(n_estimators=100))
        }

    else:
        return {
            "LinearRegression": make_scaled_model(LinearRegression()),
            "RandomForest": make_tree_model(RandomForestRegressor(n_estimators=200)),
            "XGBoost": make_tree_model(XGBRegressor(n_estimators=200, objective="reg:squarederror")),
            "LightGBM": make_tree_model(LGBMRegressor(n_estimators=200))
        }

# ==========================================
# MAIN
# ==========================================
summary_rows = []
rmse_rows = []

for stock_i in range(1, N_STOCKS + 1):
    print(f"\nProcessing Stock {stock_i}...")

    train_url = f"{BASE_URL}stock_{stock_i}_train.csv"
    test_url = f"{BASE_URL}stock_{stock_i}_test.csv"

    try:
        train_df = pd.read_csv(train_url)
        test_df = pd.read_csv(test_url)
    except Exception as e:
        print(f"Skipping Stock {stock_i}: {e}")
        continue

    if "target" not in train_df.columns:
        continue

    X = train_df.drop(columns=["target"])
    y = train_df["target"]
    X_test = test_df.copy()

    n_rows = len(train_df)
    group = STOCK_RULES[stock_i]["group"]
    spread_mult = STOCK_RULES[stock_i]["spread_mult"]

    cv = choose_cv(stock_i, n_rows)
    models = get_models_for_stock(stock_i, n_rows)

    results = []

    for name, model in models.items():
        try:
            rmse = rmse_cv(model, X, y, cv)
            results.append((name, rmse))
            print(f"{name}: RMSE={rmse:.4f}")
        except:
            continue

    results = sorted(results, key=lambda x: x[1])[:3]

    preds = []
    rmses = []

    for name, rmse in results:
        model = models[name]
        model.fit(X, y)
        pred = float(model.predict(X_test)[0])
        preds.append((name, pred))
        rmses.append(rmse)

    fair_value = np.mean([p for _, p in preds])
    avg_rmse = np.mean(rmses)

    adjusted_rmse = avg_rmse * spread_mult

    bid = fair_value - adjusted_rmse
    ask = fair_value + adjusted_rmse

    summary_rows.append({
        "stock": f"Stock {stock_i}",
        "best1": format_model_price(preds[0][0], preds[0][1]),
        "best2": format_model_price(preds[1][0], preds[1][1]),
        "best3": format_model_price(preds[2][0], preds[2][1]),
        "average": fair_value,
        "bid": bid,
        "ask": ask,
        "spread": ask - bid,
        "confidence": confidence_from_rmse(adjusted_rmse, fair_value)
    })

# save
pd.DataFrame(summary_rows).to_csv(SUMMARY_OUTPUT, index=False)

print("\nDone.") 