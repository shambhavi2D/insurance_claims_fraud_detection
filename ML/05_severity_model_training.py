## MODEL TRAINING for CLAIM SEVERITY ESTIMATION
"""
Run this after 03_feature_extraction.py
GOAL: predict how SEVERE/costly a claim will be
We build 2 versions of the severity model:

   (a) REGRESSION      : predict the exact total_claim_amount (a number)

   (b) CLASSIFICATION  : predict a severity Tier (Low/Medium/High/Severe), which is often more directly useful for claim
    PRIORITIZATION in a decision-support dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42

"""
# creating output folders (already provided by me)

import os
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)
"""

# LOAD DATA

X_full = pd.read_csv("data/extracted/X_features.csv")
y_amount = pd.read_csv("data/extracted/y_severity_regression.csv").iloc[:, 0]   # total_claim_amount
y_tier = pd.read_csv("data/extracted/y_severity_class.csv").iloc[:, 0]          # Low/Medium/High/Severe

print("X_full shape:", X_full.shape)
print("y_amount range:", y_amount.min(), "-", y_amount.max())
print("y_tier distribution:\n", y_tier.value_counts())

"""### REMOVE LEAKY FEATURES BEFORE MODELING SEVERITY

 In Step 3 we engineered injury_claim_ratio / property_claim_ratio /
 vehicle_claim_ratio and claim_to_premium_ratio to avoid feeding the fraud
 model four perfectly collinear raw claim columns.

 That reasoning was fine for fraud detection, because "how is this claim broken down" is a useful fraud signal.

 For SEVERITY estimation, the scenario is different :

   injury_claim + property_claim + vehicle_claim  =  total_claim_amount

   claim_to_premium_ratio  =  total_claim_amount / policy_annual_premium

   claim_severity_tier     =  quartile-bucketed total_claim_amount

 If we leave these columns in X, we are just letting the model read the answer directly from columns that are mathematically derived from the target. This is
 called TARGET LEAKAGE, and it's the most common way a model silently produces 99% accuracy results that are meaningless and fall apart when it meets real, unseen data
"""

LEAKY_SEVERITY_COLS = [
    "injury_claim", "property_claim", "vehicle_claim",
    "injury_claim_ratio", "property_claim_ratio", "vehicle_claim_ratio",
    "claim_to_premium_ratio",
]

X_severity = X_full.drop(columns=LEAKY_SEVERITY_COLS)
print("Shape after removing leaky columns:", X_severity.shape)
print("Removed:", LEAKY_SEVERITY_COLS)

"""### TRAIN / TEST SPLIT
Severity is a regression-friendly continuous target, so we don't
stratify here (stratifying is for classification class balance).

For the classification version below, the tiers are already perfectly
balanced (qcut into quartiles), so random splitting keeps that balance.
"""

X_train, X_test, y_amount_train, y_amount_test, y_tier_train, y_tier_test = train_test_split(
    X_severity, y_amount, y_tier, test_size=0.2, random_state=RANDOM_STATE
)

print("Train/test sizes:", X_train.shape, X_test.shape)

"""### REGRESSION: PREDICT THE EXACT CLAIM AMOUNT"""

# ---- Baseline: Linear Regression -------------------------------------------
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_amount_train)
lin_pred = lin_reg.predict(X_test)

# ---- Random Forest Regressor ------------------------------------------------
rf_reg = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
rf_reg.fit(X_train, y_amount_train)
rf_pred = rf_reg.predict(X_test)

# ---- XGBoost Regressor -------------------------------------------------------
xgb_reg = XGBRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1
)
xgb_reg.fit(X_train, y_amount_train)
xgb_pred = xgb_reg.predict(X_test)

"""#### REGRESSION EVALUATION
MAE  (Mean Absolute Error) : average dollar error, easy to explain to a
                                  non-technical claims officer

RMSE (Root Mean Squared Error) : penalizes big misses more heavily than
                                  MAE; useful because a severity model being
                                  wildly wrong on one huge claim is worse
                                  than being slightly off on many small ones
                                  
R^2  : proportion of variance in claim amount the model explains (1.0 =
                                  perfect, 0.0 = no better than predicting
                                  the mean every time)
"""

reg_results = {}
for name, pred in [("Linear Regression", lin_pred), ("Random Forest", rf_pred), ("XGBoost", xgb_pred)]:
    mae = mean_absolute_error(y_amount_test, pred)
    rmse = np.sqrt(mean_squared_error(y_amount_test, pred))
    r2 = r2_score(y_amount_test, pred)
    reg_results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"{name:20s} | MAE: {mae:8.0f} | RMSE: {rmse:8.0f} | R2: {r2:.3f}")

reg_summary = pd.DataFrame(reg_results).T.round(3)
print("\n", reg_summary)

# PREDICTED VS ACTUAL PLOT (BEST REGRESSOR)
best_reg_name = reg_summary["R2"].idxmax()
best_pred = {"Linear Regression": lin_pred, "Random Forest": rf_pred, "XGBoost": xgb_pred}[best_reg_name]

plt.figure(figsize=(7, 6))
plt.scatter(y_amount_test, best_pred, alpha=0.5, color="#4c72b0")
plt.plot([y_amount_test.min(), y_amount_test.max()],
         [y_amount_test.min(), y_amount_test.max()], "r--", label="Perfect prediction")
plt.xlabel("Actual Claim Amount ($)")
plt.ylabel("Predicted Claim Amount ($)")
plt.title(f"Predicted vs Actual Claim Severity -- {best_reg_name}")
plt.legend()
plt.tight_layout()
plt.savefig("plots/11_severity_predicted_vs_actual.png", dpi=120)
plt.show()

"""### CLASSIFICATION: PREDICT THE SEVERITY TIER
A claims officer dashboard does not need "the
claim will cost exactly $xx,xxx". It needs "High severity
claim, prioritize."
A 4-class tier prediction is simpler to act on and easier to display as a priority indicator in a UI.
"""

rf_clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
rf_clf.fit(X_train, y_tier_train)
tier_pred = rf_clf.predict(X_test)

print(classification_report(y_tier_test, tier_pred))

cm = confusion_matrix(y_tier_test, tier_pred, labels=["Low", "Medium", "High", "Severe"])
ConfusionMatrixDisplay(cm, display_labels=["Low", "Medium", "High", "Severe"]).plot(cmap="Purples")
plt.title("Severity Tier Classification -- Confusion Matrix")
plt.tight_layout()
plt.savefig("plots/12_severity_tier_confusion_matrix.png", dpi=120)
plt.show()

# FEATURE IMPORTANCE FOR SEVERITY (SANITY CHECK)
importances = pd.Series(rf_reg.feature_importances_, index=X_severity.columns)
top15 = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 6))
sns.barplot(x=top15.values, y=top15.index, color="#55a868")
plt.title("Top 15 Feature Importances -- Severity (Random Forest Regressor)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("plots/13_severity_feature_importance.png", dpi=120)
plt.show()
print(top15)

# SAVE MODELS FOR THE EXPLAINABILITY + DASHBOARD STAGES
joblib.dump(rf_reg, "models/severity_regressor.joblib")
joblib.dump(rf_clf, "models/severity_tier_classifier.joblib")
joblib.dump(list(X_severity.columns), "models/severity_feature_names.joblib")

print("Saved: models/severity_regressor.joblib, models/severity_tier_classifier.joblib")
print("Note: the severity models use a DIFFERENT (smaller) feature set than the")
print("fraud model, since the claim-amount columns were removed to prevent leakage.")
print("Keep this in mind when building the explainability step and the final")
print("integrated dashboard -- fraud and severity each need their own feature list.")

""" 
#  SAVE THE RESULTS (Already provided)

reg_summary["Best Model"] = reg_summary.index == best_reg_name
reg_summary.to_csv("results/severity_regression_comparison.csv")

tier_report = classification_report(y_tier_test, tier_pred, output_dict=True)
pd.DataFrame(tier_report).T.round(3).to_csv("results/severity_tier_classification_report.csv")

print("Saved: results/severity_regression_comparison.csv")
print("Saved: results/severity_tier_classification_report.csv")
"""
