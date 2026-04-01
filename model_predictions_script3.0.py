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
STOCK_I = 4
TRAIN_FILE = "cleaned_training_data/stock_4_train_cleaned.csv"
TEST_FILE = "cleaned_training_data/stock_4_test.csv"
CURRENT_SUMMARY_FILE = "summary.csv"

UPDATED_SUMMARY_FILE = "summary_updated.csv"
CLEANED_SUMMARY_FILE = "summary_cleaned.csv"
RMSE_OUTPUT = "stock_4_model_rmse_results.csv"

RANDOM_STATE = 42

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

SAFE_SIMPLE_MODELS = {
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "BayesianRidge",
    "HuberRegressor",
    "SVR_Linear"
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


def choose_cv(stock_i):
    group = STOCK_RULES[stock_i]["group"]

    if group == "tiny":
        return LeaveOneOut()
    elif group == "small":
        return RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    else:
        return KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def rmse_cv(model, X, y, cv):
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=None
    )
    return -scores.mean()


def raw_confidence_from_rmse(adjusted_rmse, fair_value):
    denom = max(abs(fair_value), 1e-8)
    rel_error = adjusted_rmse / denom
    return float(1 / (1 + rel_error))


def format_model_price(model_name, price):
    return f"{model_name}: {price:.6f}"


def prediction_agreement_multiplier(predictions, fair_value):
    pred_values = np.array(predictions, dtype=float)
    pred_std = float(np.std(pred_values))
    denom = max(abs(fair_value), 1e-8)
    rel_dispersion = pred_std / denom

    if rel_dispersion < 0.01:
        return 1.00
    elif rel_dispersion < 0.025:
        return 1.10
    elif rel_dispersion < 0.05:
        return 1.25
    else:
        return 1.50


def weighted_average_prediction(predictions, rmses):
    eps = 1e-8
    weights = np.array([1 / np.sqrt(max(r, eps)) for r in rmses], dtype=float)
    weights = weights / weights.sum()

    pred_values = np.array(predictions, dtype=float)
    weighted_avg = float(np.sum(weights * pred_values))

    return weighted_avg, weights.tolist()


def confidence_adjustment_multiplier(confidence):
    if confidence >= 0.90:
        return 0.90
    elif confidence >= 0.80:
        return 1.00
    elif confidence >= 0.70:
        return 1.10
    elif confidence >= 0.60:
        return 1.20
    else:
        return 1.35


def asymmetric_side_multipliers(predictions, fair_value):
    preds = np.array(predictions, dtype=float)
    denom = max(abs(fair_value), 1e-8)

    signed_mean = float(np.mean(preds - fair_value))
    skew_score = signed_mean / denom

    capped_skew = max(min(skew_score, 0.05), -0.05)
    tilt_strength = 3.0 * abs(capped_skew)

    if capped_skew > 0:
        ask_mult = 1.0 + tilt_strength
        bid_mult = max(0.85, 1.0 - tilt_strength)
    elif capped_skew < 0:
        bid_mult = 1.0 + tilt_strength
        ask_mult = max(0.85, 1.0 - tilt_strength)
    else:
        bid_mult = 1.0
        ask_mult = 1.0

    return bid_mult, ask_mult, skew_score


def rel_prediction_range(predictions, fair_value):
    preds = np.array(predictions, dtype=float)
    denom = max(abs(fair_value), 1e-8)
    return float((preds.max() - preds.min()) / denom)


def count_safe_simple_models(model_names):
    return sum(1 for name in model_names if name in SAFE_SIMPLE_MODELS)


def select_top3_models(stock_i, results, fitted_predictions):
    group = STOCK_RULES[stock_i]["group"]
    results_sorted = sorted(results, key=lambda x: x[1])

    if group != "tiny":
        return results_sorted[:3], "default_top3"

    candidate = results_sorted[:3]
    candidate_names = [name for name, _ in candidate]
    candidate_preds = [fitted_predictions[name] for name in candidate_names]
    fair_value = float(np.mean(candidate_preds))
    candidate_rel_range = rel_prediction_range(candidate_preds, fair_value)
    candidate_safe_count = count_safe_simple_models(candidate_names)

    if candidate_safe_count >= 2 and candidate_rel_range <= 0.08:
        return candidate, "default_top3"

    simple_results = [item for item in results_sorted if item[0] in SAFE_SIMPLE_MODELS]
    other_results = [item for item in results_sorted if item[0] not in SAFE_SIMPLE_MODELS]

    safer_candidate = []
    safer_candidate.extend(simple_results[:2])

    used = {name for name, _ in safer_candidate}
    for item in simple_results[2:] + other_results:
        if item[0] not in used:
            safer_candidate.append(item)
            used.add(item[0])
        if len(safer_candidate) == 3:
            break

    if len(safer_candidate) < 3:
        safer_candidate = results_sorted[:3]

    return safer_candidate, "safe_fallback"


def get_models_for_stock(stock_i, n_rows):
    group = STOCK_RULES[stock_i]["group"]

    if group == "tiny":
        return {
            "LinearRegression": make_scaled_model(LinearRegression()),
            "Ridge": make_scaled_model(Ridge(alpha=1.0)),
            "Lasso": make_scaled_model(Lasso(alpha=0.01, max_iter=20000)),
            "ElasticNet": make_scaled_model(
                ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000)
            ),
            "BayesianRidge": make_scaled_model(BayesianRidge()),
            "HuberRegressor": make_scaled_model(HuberRegressor()),
            "SVR_Linear": make_scaled_model(SVR(kernel="linear", C=0.5, epsilon=0.1)),
            "SVR_RBF": make_scaled_model(SVR(kernel="rbf", C=0.5, epsilon=0.1)),
            "KNN": make_scaled_model(KNeighborsRegressor(n_neighbors=2)),
            "RandomForest": make_tree_model(
                RandomForestRegressor(
                    n_estimators=50,
                    max_depth=3,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        }

    elif group == "small":
        return {
            "LinearRegression": make_scaled_model(LinearRegression()),
            "Ridge": make_scaled_model(Ridge(alpha=1.0)),
            "Lasso": make_scaled_model(Lasso(alpha=0.01, max_iter=20000)),
            "ElasticNet": make_scaled_model(
                ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000)
            ),
            "BayesianRidge": make_scaled_model(BayesianRidge()),
            "HuberRegressor": make_scaled_model(HuberRegressor()),
            "SVR_Linear": make_scaled_model(SVR(kernel="linear", C=1.0, epsilon=0.1)),
            "SVR_RBF": make_scaled_model(SVR(kernel="rbf", C=1.0, epsilon=0.1)),
            "KNN": make_scaled_model(KNeighborsRegressor(n_neighbors=3)),
            "RandomForest": make_tree_model(
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            ),
            "ExtraTrees": make_tree_model(
                ExtraTreesRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            ),
            "GradientBoosting": make_tree_model(
                GradientBoostingRegressor(
                    n_estimators=80,
                    learning_rate=0.05,
                    max_depth=2,
                    random_state=RANDOM_STATE
                )
            )
        }

    else:
        return {
            "LinearRegression": make_scaled_model(LinearRegression()),
            "Ridge": make_scaled_model(Ridge(alpha=1.0)),
            "Lasso": make_scaled_model(Lasso(alpha=0.01, max_iter=20000)),
            "ElasticNet": make_scaled_model(
                ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000)
            ),
            "BayesianRidge": make_scaled_model(BayesianRidge()),
            "HuberRegressor": make_scaled_model(HuberRegressor()),
            "KNN": make_scaled_model(
                KNeighborsRegressor(n_neighbors=max(3, min(7, int(np.sqrt(n_rows)))))
            ),
            "SVR_RBF": make_scaled_model(SVR(kernel="rbf", C=1.0, epsilon=0.1)),
            "RandomForest": make_tree_model(
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            ),
            "ExtraTrees": make_tree_model(
                ExtraTreesRegressor(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            ),
            "GradientBoosting": make_tree_model(
                GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=RANDOM_STATE
                )
            ),
            "HistGradientBoosting": make_tree_model(
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=RANDOM_STATE
                )
            ),
            "AdaBoost": make_tree_model(
                AdaBoostRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    random_state=RANDOM_STATE
                )
            ),
            "XGBoost": make_tree_model(
                XGBRegressor(
                    n_estimators=200,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=0
                )
            ),
            "LightGBM": make_tree_model(
                LGBMRegressor(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    verbose=-1
                )
            )
        }

# ==========================================
# MAIN
# ==========================================
print(f"\nProcessing Stock {STOCK_I}...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)
summary_df = pd.read_csv(CURRENT_SUMMARY_FILE)

if "target" not in train_df.columns:
    raise ValueError("Training file must contain a 'target' column.")

X = train_df.drop(columns=["target"])
y = train_df["target"]
X_test = test_df.copy()
X_test = X_test.reindex(columns=X.columns, fill_value=np.nan)

n_rows = len(train_df)
group = STOCK_RULES[STOCK_I]["group"]
spread_mult = STOCK_RULES[STOCK_I]["spread_mult"]

cv = choose_cv(STOCK_I)
models = get_models_for_stock(STOCK_I, n_rows)

results = []
fitted_predictions = {}
rmse_rows = []

for name, model in models.items():
    try:
        rmse = rmse_cv(model, X, y, cv)
        model.fit(X, y)
        pred = float(model.predict(X_test)[0])

        results.append((name, rmse))
        fitted_predictions[name] = pred

        rmse_rows.append({
            "stock": f"Stock {STOCK_I}",
            "group": group,
            "model": name,
            "rmse": rmse,
            "prediction": pred
        })

        print(f"{name}: RMSE = {rmse:.6f}, Pred = {pred:.6f}")

    except Exception as e:
        print(f"{name} failed: {e}")

top3, selection_method = select_top3_models(STOCK_I, results, fitted_predictions)

pred_names = []
pred_values = []
pred_rmses = []

for name, rmse in top3:
    pred_names.append(name)
    pred_values.append(fitted_predictions[name])
    pred_rmses.append(rmse)

fair_value, weights = weighted_average_prediction(pred_values, pred_rmses)
simple_average = float(np.mean(pred_values))
avg_rmse = float(np.mean(pred_rmses))

adjusted_rmse = avg_rmse * spread_mult
agreement_mult = prediction_agreement_multiplier(pred_values, fair_value)
uncertainty_after_agreement = adjusted_rmse * agreement_mult

raw_confidence = raw_confidence_from_rmse(uncertainty_after_agreement, fair_value)
confidence_mult = confidence_adjustment_multiplier(raw_confidence)
final_uncertainty = uncertainty_after_agreement * confidence_mult
final_confidence = raw_confidence_from_rmse(final_uncertainty, fair_value)

bid_side_mult, ask_side_mult, skew_score = asymmetric_side_multipliers(pred_values, fair_value)

bid_uncertainty = final_uncertainty * bid_side_mult
ask_uncertainty = final_uncertainty * ask_side_mult

bid = fair_value - bid_uncertainty
ask = fair_value + ask_uncertainty
spread = ask - bid

new_stock4_row = {
    "stock": f"Stock {STOCK_I}",
    "group": group,
    "best1": format_model_price(pred_names[0], pred_values[0]),
    "best2": format_model_price(pred_names[1], pred_values[1]),
    "best3": format_model_price(pred_names[2], pred_values[2]),
    "best1_weight": round(weights[0], 4),
    "best2_weight": round(weights[1], 4),
    "best3_weight": round(weights[2], 4),
    "weighted_average": round(fair_value, 6),
    "simple_average": round(simple_average, 6),
    "bid": round(bid, 6),
    "ask": round(ask, 6),
    "spread": round(spread, 6),
    "avg_top3_rmse": round(avg_rmse, 6),
    "confidence": round(final_confidence, 4)
}

new_stock4_df = pd.DataFrame([new_stock4_row])

summary_df = summary_df[summary_df["stock"] != f"Stock {STOCK_I}"].copy()
summary_df = pd.concat([summary_df, new_stock4_df], ignore_index=True)

summary_df["stock_num"] = summary_df["stock"].str.extract(r"(\d+)").astype(int)
summary_df = summary_df.sort_values("stock_num").drop(columns=["stock_num"]).reset_index(drop=True)

summary_df.to_csv(UPDATED_SUMMARY_FILE, index=False)

rmse_df = pd.DataFrame(rmse_rows)
rmse_df.to_csv(RMSE_OUTPUT, index=False)

cleaned_summary_df = summary_df.copy()
cleaned_summary_df.to_csv(CLEANED_SUMMARY_FILE, index=False)

print("\nDone.")
print(f"Saved {UPDATED_SUMMARY_FILE}")
print(f"Saved {CLEANED_SUMMARY_FILE}")
print(f"Saved {RMSE_OUTPUT}")