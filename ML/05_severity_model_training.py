# CLAIM SEVERITY ESTIMATION
# Run this AFTER Step 03 or 04

# IMPORT LIBRARIES

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, KFold, RepeatedKFold, StratifiedKFold,
    RepeatedStratifiedKFold, GridSearchCV, cross_validate,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, accuracy_score,
)
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
'''
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
'''
os.makedirs("plots", exist_ok=True)

RANDOM_STATE = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 3   # regression folds are cheap, so we afford a few more repeats
INNER_SPLITS = 3

# LOAD THE UNSCALED FEATURES AND REMOVE LEAKY COLUMNS

#   injury_claim + property_claim + vehicle_claim  ==  total_claim_amount
#   claim_to_premium_ratio  =  total_claim_amount / policy_annual_premium
#   claim_severity_tier     =  a quartile bucketing of total_claim_amount

# If these stay in X, the model is not predicting severity from the claim's
# underlying circumstances (incident type, policy, customer profile) but
# reading the answer directly from columns derived from the target itself.
# This is TARGET LEAKAGE, and it quietly produces meaningless scores
# that fall apart on a real unseen claim

X_full = pd.read_csv("X_features_unscaled.csv")
y_amount = pd.read_csv("y_severity_regression.csv").iloc[:, 0]
y_tier = pd.read_csv("y_severity_class.csv").iloc[:, 0]

LEAKY_SEVERITY_COLS = [
    "injury_claim", "property_claim", "vehicle_claim",
    "injury_claim_ratio", "property_claim_ratio", "vehicle_claim_ratio",
    "claim_to_premium_ratio",
]
X = X_full.drop(columns=[c for c in LEAKY_SEVERITY_COLS if c in X_full.columns])

print("X_full shape:", X_full.shape, "-> after removing leaky columns:", X.shape)
print("y_amount range:", y_amount.min(), "-", y_amount.max())
print("y_tier distribution:\n", y_tier.value_counts())

# HOLD OUT THE TEST SET AND LEAVE IT ALONE
# Severity is a continuous regression target, so stratify isn't used for the
# amount split. We additionally carry y_tier through the SAME split (by
# splitting on index) so the regression and tier-classification test sets
# refer to the exact same 200 claims

X_train, X_test, y_amount_train, y_amount_test, y_tier_train, y_tier_test = train_test_split(
    X, y_amount, y_tier, test_size=0.2, random_state=RANDOM_STATE
)
print("Train:", X_train.shape, "| Test:", X_test.shape, "(locked until 5.8)")

# PREPROCESSING

binary_cols = [c for c in X_train.columns if set(X_train[c].unique()) <= {0, 1}]
continuous_cols = [c for c in X_train.columns if c not in binary_cols]
feature_order = continuous_cols + binary_cols
print(f"{len(continuous_cols)} continuous columns, {len(binary_cols)} binary (0/1) columns")


def make_preprocessor():
    return ColumnTransformer(
        [("scale", StandardScaler(), continuous_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

# REGRESSION:
# Ridge (regularised linear regression) replaces plain Linear Regression as
# the baseline here, because it has a tuning knob (alpha) & plain Linear
# Regression has nothing to tune, which would make the "equal tuning effort"
# comparison uneven in its favour (zero settings to overfit on).

REG_CANDIDATES = {
    "Ridge Regression": {
        "make": lambda: Ridge(random_state=RANDOM_STATE),
        "grid": {"reg__alpha": [0.1, 1, 10, 50, 100, 300]},
    },
    "Random Forest": {
        "make": lambda: RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
        "grid": {"reg__max_depth": [4, 8, None], "reg__min_samples_leaf": [1, 3]},
    },
    "XGBoost": {
        "make": lambda: XGBRegressor(n_estimators=200, subsample=0.8, colsample_bytree=0.8,
                                     random_state=RANDOM_STATE, n_jobs=1),
        "grid": {"reg__max_depth": [2, 3, 5], "reg__learning_rate": [0.05, 0.1]},
    },
}


def build_reg_pipeline(name):
    return Pipeline([("prep", make_preprocessor()), ("reg", REG_CANDIDATES[name]["make"]())])


def make_tuned_reg(name, inner_splits=INNER_SPLITS):
    return GridSearchCV(
        build_reg_pipeline(name), REG_CANDIDATES[name]["grid"],
        scoring="neg_mean_absolute_error",
        cv=KFold(inner_splits, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1, refit=True,
    )


REG_SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}
outer_cv_reg = RepeatedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=RANDOM_STATE)

# RUN THE REGRESSION COMPARISON (NESTED, REPEATED K-FOLD)
# Same nested CV idea : an inner loop tunes each model's
# settings, an OUTER loop scores the tuned model on unseen rows
# All three models are judged on the same outer folds.

reg_rows, reg_fold_mae = [], {}

dummy_reg = cross_validate(DummyRegressor(strategy="mean"), X_train, y_amount_train,
                           cv=outer_cv_reg, scoring=REG_SCORING)
reg_rows.append({"model": "Predict the mean (reference)",
                 **{f"{m}_mean": dummy_reg[f"test_{m}"].mean() for m in REG_SCORING},
                 **{f"{m}_std": dummy_reg[f"test_{m}"].std() for m in REG_SCORING}})

for name in REG_CANDIDATES:
    t0 = time.time()
    cv_out = cross_validate(make_tuned_reg(name), X_train, y_amount_train,
                            cv=outer_cv_reg, scoring=REG_SCORING, n_jobs=1)
    reg_fold_mae[name] = -cv_out["test_mae"]  # flip sign: sklearn reports NEGATIVE MAE
    reg_rows.append({"model": name,
                     **{f"{m}_mean": cv_out[f"test_{m}"].mean() for m in REG_SCORING},
                     **{f"{m}_std": cv_out[f"test_{m}"].std() for m in REG_SCORING}})
    print(f"{name:18s} | MAE {-cv_out['test_mae'].mean():8.0f} +/- {cv_out['test_mae'].std():6.0f} "
          f"| R2 {cv_out['test_r2'].mean():.3f} | {time.time()-t0:.0f}s")

reg_comparison = pd.DataFrame(reg_rows)

# MAE/RMSE were scored as negative numbers by sklearn convention (so that
# "higher is better" holds for every metric during tuning); flip them back
# to normal positive dollar amounts for reporting.

for col in ["mae_mean", "mae_std", "rmse_mean", "rmse_std"]:
    reg_comparison[col] = reg_comparison[col].abs()

ranked_reg = reg_comparison[reg_comparison["model"] != "Predict the mean (reference)"] \
    .sort_values("mae_mean").reset_index(drop=True)
print("\n", ranked_reg[["model", "mae_mean", "mae_std", "rmse_mean", "r2_mean"]].round(2).to_string(index=False))

best_reg_name = ranked_reg.iloc[0]["model"]
best_reg_row = ranked_reg.iloc[0]
print(f"\nBest regressor by MAE: {best_reg_name} "
      f"(MAE ${best_reg_row['mae_mean']:,.0f} +/- ${best_reg_row['mae_std']:,.0f}, "
      f"R2 {best_reg_row['r2_mean']:.3f})")

# REGRESSION COMPARISON CHART

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
models_order = list(REG_CANDIDATES.keys())
mae_vals = [ranked_reg.set_index("model").loc[m, "mae_mean"] for m in models_order]
mae_errs = [ranked_reg.set_index("model").loc[m, "mae_std"] for m in models_order]
axes[0].bar(models_order, mae_vals, yerr=mae_errs, capsize=4, color="#4c72b0")
axes[0].axhline(reg_comparison.loc[reg_comparison["model"] == "Predict the mean (reference)", "mae_mean"].iloc[0],
                color="grey", linestyle="--", label="predict the mean")
axes[0].set_ylabel("MAE ($, lower is better)")
axes[0].set_title("Mean Absolute Error by model")
axes[0].legend()

r2_vals = [ranked_reg.set_index("model").loc[m, "r2_mean"] for m in models_order]
r2_errs = [ranked_reg.set_index("model").loc[m, "r2_std"] for m in models_order]
axes[1].bar(models_order, r2_vals, yerr=r2_errs, capsize=4, color="#55a868")
axes[1].axhline(0, color="grey", linestyle="--", label="predict the mean (R2=0)")
axes[1].set_ylabel("R2 (higher is better)")
axes[1].set_title("R2 by model")
axes[1].legend()
plt.suptitle("Fair regression comparison: repeated nested K-fold CV (mean +/- std)")
plt.tight_layout()
plt.savefig("plots/11_severity_model_comparison.png", dpi=120)
plt.show()

# FINAL REGRESSOR: TUNE ON ALL TRAINING DATA, OPEN THE TEST SET ONCE

final_reg_search = make_tuned_reg(best_reg_name, inner_splits=5)
final_reg_search.fit(X_train, y_amount_train)
print("Best settings found:", final_reg_search.best_params_)

final_reg_pipe = final_reg_search.best_estimator_
amount_pred = final_reg_pipe.predict(X_test)

test_reg_metrics = {
    "mae": mean_absolute_error(y_amount_test, amount_pred),
    "rmse": np.sqrt(mean_squared_error(y_amount_test, amount_pred)),
    "r2": r2_score(y_amount_test, amount_pred),
}
print({k: round(v, 3) for k, v in test_reg_metrics.items()})

# Predicted vs actual plot

plt.figure(figsize=(7, 6))
plt.scatter(y_amount_test, amount_pred, alpha=0.5, color="#4c72b0")
plt.plot([y_amount_test.min(), y_amount_test.max()],
         [y_amount_test.min(), y_amount_test.max()], "r--", label="Perfect prediction")
plt.xlabel("Actual claim amount ($)"); plt.ylabel("Predicted claim amount ($)")
plt.title(f"Predicted vs actual severity -- {best_reg_name} (test set)")
plt.legend(); plt.tight_layout()
plt.savefig("plots/12_severity_predicted_vs_actual.png", dpi=120)
plt.show()

print("\nNote on R2: a modest score here (0.2-0.5) is expected and HONEST -- the")
print("leaky claim-breakdown columns were deliberately removed (Step 5.2), so the")
print("model predicts severity purely from incident/policy/customer context, which")
print("is a genuinely hard problem. A low leak-free R2 is far more defensible than")
print("an inflated one from a model secretly given the answer.")

# SEVERITY TIER CLASSIFICATION COMPARISON

CLF_CANDIDATES = {
    "Logistic Regression": {
        "make": lambda: LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "grid": {"clf__C": [0.01, 0.03, 0.1, 1, 10]},
    },
    "Random Forest": {
        "make": lambda: RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
        "grid": {"clf__max_depth": [4, 8, None], "clf__min_samples_leaf": [1, 3]},
    },
    "XGBoost": {
        "make": lambda: XGBClassifier(n_estimators=200, subsample=0.8, colsample_bytree=0.8,
                                      eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=1),
        "grid": {"clf__max_depth": [2, 3, 5], "clf__learning_rate": [0.05, 0.1]},
    },
}


def build_clf_pipeline(name):
    return Pipeline([("prep", make_preprocessor()), ("clf", CLF_CANDIDATES[name]["make"]())])


def make_tuned_clf(name, inner_splits=INNER_SPLITS):
    return GridSearchCV(
        build_clf_pipeline(name), CLF_CANDIDATES[name]["grid"],
        scoring="f1_macro",
        cv=StratifiedKFold(inner_splits, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1, refit=True,
    )


CLF_SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
outer_cv_clf = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=RANDOM_STATE)

y_tier_encoded = y_tier_train.astype("category").cat.codes  # XGBoost needs 0..k-1 integer labels
tier_categories = y_tier_train.astype("category").cat.categories

clf_rows = []
dummy_clf = cross_validate(DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
                           X_train, y_tier_encoded, cv=outer_cv_clf, scoring=CLF_SCORING)
clf_rows.append({"model": "Random guessing (reference)",
                 **{f"{m}_mean": dummy_clf[f"test_{m}"].mean() for m in CLF_SCORING},
                 **{f"{m}_std": dummy_clf[f"test_{m}"].std() for m in CLF_SCORING}})

for name in CLF_CANDIDATES:
    t0 = time.time()
    cv_out = cross_validate(make_tuned_clf(name), X_train, y_tier_encoded,
                            cv=outer_cv_clf, scoring=CLF_SCORING, n_jobs=1)
    clf_rows.append({"model": name,
                     **{f"{m}_mean": cv_out[f"test_{m}"].mean() for m in CLF_SCORING},
                     **{f"{m}_std": cv_out[f"test_{m}"].std() for m in CLF_SCORING}})
    print(f"{name:18s} | Accuracy {cv_out['test_accuracy'].mean():.3f} +/- {cv_out['test_accuracy'].std():.3f} "
          f"| Macro-F1 {cv_out['test_f1_macro'].mean():.3f} | {time.time()-t0:.0f}s")

clf_comparison = pd.DataFrame(clf_rows)
ranked_clf = clf_comparison[clf_comparison["model"] != "Random guessing (reference)"] \
    .sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
print("\n", ranked_clf.round(3).to_string(index=False))
print("\nReference (random guessing, 4 balanced classes -> ~0.25 accuracy expected):")
print(clf_comparison[clf_comparison["model"] == "Random guessing (reference)"].round(3).to_string(index=False))

best_clf_name = ranked_clf.iloc[0]["model"]
print(f"\nBest tier classifier by macro-F1: {best_clf_name}")

# FINAL TIER CLASSIFIER: TUNE ON ALL TRAINING DATA, OPEN THE TEST SET ONCE

y_tier_test_encoded = y_tier_test.astype("category").cat.codes

final_clf_search = make_tuned_clf(best_clf_name, inner_splits=5)
final_clf_search.fit(X_train, y_tier_encoded)
print("Best settings found:", final_clf_search.best_params_)

final_clf_pipe = final_clf_search.best_estimator_
tier_pred_encoded = final_clf_pipe.predict(X_test)

print(classification_report(y_tier_test_encoded, tier_pred_encoded,
                            target_names=list(tier_categories)))

cm = confusion_matrix(y_tier_test_encoded, tier_pred_encoded)
ConfusionMatrixDisplay(cm, display_labels=list(tier_categories)).plot(cmap="Purples")
plt.title(f"Severity tier classification -- {best_clf_name} (test set)")
plt.tight_layout()
plt.savefig("plots/13_severity_tier_confusion_matrix.png", dpi=120)
plt.show()

test_clf_metrics = {
    "accuracy": accuracy_score(y_tier_test_encoded, tier_pred_encoded),
    "f1_macro": f1_score(y_tier_test_encoded, tier_pred_encoded, average="macro"),
}

# FEATURE IMPORTANCE FOR THE CHOSEN REGRESSOR (SANITY CHECK BEFORE SHAP)

final_reg_model = final_reg_pipe.named_steps["reg"]
if hasattr(final_reg_model, "feature_importances_"):
    imp = pd.Series(final_reg_model.feature_importances_, index=feature_order)
elif hasattr(final_reg_model, "coef_"):
    imp = pd.Series(np.abs(final_reg_model.coef_), index=feature_order)
else:
    imp = None

if imp is not None:
    top15 = imp.sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=top15.values, y=top15.index, color="#55a868")
    plt.title(f"Top 15 built-in importances: {best_reg_name} (severity)")
    plt.tight_layout()
    plt.savefig("plots/14_severity_feature_importance.png", dpi=120)
    plt.show()
    print(top15.round(4))

# SAVE EVERYTHING
joblib.dump(final_reg_model, "models/severity_regressor.joblib")
joblib.dump(final_reg_pipe.named_steps["prep"], "models/severity_regressor_preprocessor.joblib")
joblib.dump(final_clf_pipe.named_steps["clf"], "models/severity_tier_classifier.joblib")
joblib.dump(final_clf_pipe.named_steps["prep"], "models/severity_tier_preprocessor.joblib")
joblib.dump(feature_order, "models/severity_feature_names.joblib")
joblib.dump(list(tier_categories), "models/severity_tier_categories.joblib")

metadata = {
    "regressor": {"model": best_reg_name, "best_params": final_reg_search.best_params_,
                  "cv_mae_mean": float(best_reg_row["mae_mean"]), "cv_mae_std": float(best_reg_row["mae_std"]),
                  "cv_r2_mean": float(best_reg_row["r2_mean"]),
                  "test_metrics": {k: float(v) for k, v in test_reg_metrics.items()}},
    "tier_classifier": {"model": best_clf_name, "best_params": final_clf_search.best_params_,
                        "test_metrics": {k: float(v) for k, v in test_clf_metrics.items()}},
}
with open("models/severity_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)

reg_comparison.to_csv("results/severity_regression_comparison.csv", index=False)
clf_comparison.to_csv("results/severity_tier_comparison.csv", index=False)
pd.DataFrame([{"model": best_reg_name, **test_reg_metrics}]).to_csv(
    "results/severity_regression_final_test_metrics.csv", index=False)
pd.DataFrame([{"model": best_clf_name, **test_clf_metrics}]).to_csv(
    "results/severity_tier_final_test_metrics.csv", index=False)

print("Saved models/: severity_regressor.joblib, severity_regressor_preprocessor.joblib,")
print("  severity_tier_classifier.joblib, severity_tier_preprocessor.joblib,")
print("  severity_feature_names.joblib, severity_tier_categories.joblib, severity_model_metadata.json")
print("Saved results/: severity_regression_comparison.csv, severity_tier_comparison.csv,")
print("  severity_regression_final_test_metrics.csv, severity_tier_final_test_metrics.csv")
print("\nNote: the regressor and the tier classifier each got their OWN tuned model and")
print("their OWN fitted preprocessor -- they are not guaranteed to pick the same winning")
print("model type, so Step 6 (SHAP) must load and treat them independently.")
