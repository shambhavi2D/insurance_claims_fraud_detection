###EXPLAINABILITY : SHAP FOR FRAUD, SEVERITY
"""
Reference papers: [3] Transformer + SHAP, [6] XAI survey, [8] our closest
reference (explainable fraud detection), which explicitly does NOT cover
severity, so explaining both the fraud AND severity (amount + tier) models
with SHAP goes a step beyond what [8] delivers.
"""

# IMPORT LIBRARIES

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

os.makedirs("plots", exist_ok=True)
"""
os.makedirs("results", exist_ok=True)
"""

try:
    shap.initjs()
except Exception:
    pass  # harmless outside a real Jupyter kernel -- static plots still work fine

# LOAD THE SAVED MODELS, PREPROCESSORS AND FEATURE LISTS
'''
WHY LOAD RATHER THAN RETRAIN: SHAP explanations are only meaningful for the
EXACT trained model object that was evaluated and would be deployed, not a
freshly retrained copy that might land on slightly different weights.
'''
fraud_model = joblib.load("data/models/best_fraud_model.joblib")
fraud_preprocessor = joblib.load("data/models/fraud_preprocessor.joblib")
fraud_feature_names = joblib.load("data/models/fraud_feature_names.joblib")

severity_regressor = joblib.load("data/models/severity_regressor.joblib")
severity_regressor_preprocessor = joblib.load("data/models/severity_regressor_preprocessor.joblib")
severity_tier_classifier = joblib.load("data/models/severity_tier_classifier.joblib")
severity_tier_preprocessor = joblib.load("data/models/severity_tier_preprocessor.joblib")
severity_feature_names = joblib.load("data/models/severity_feature_names.joblib")
severity_tier_categories = joblib.load("data/models/severity_tier_categories.joblib")

print("Fraud model:", type(fraud_model).__name__)
print("Severity regressor:", type(severity_regressor).__name__)
print("Severity tier classifier:", type(severity_tier_classifier).__name__)
print("Tier categories (in encoded order):", list(severity_tier_categories))

# REBUILD THE SAME TRAIN/TEST SPLITS USED BEFORE, THEN PREPROCESS

y_fraud = pd.read_csv("data/extracted/y_fraud.csv").iloc[:, 0]
X_fraud_unscaled_full = pd.read_csv("data/extracted/X_features_unscaled.csv")
X_fraud_raw_train, X_fraud_raw_test, y_fraud_train, y_fraud_test = train_test_split(
    X_fraud_unscaled_full, y_fraud, test_size=0.2, random_state=RANDOM_STATE, stratify=y_fraud
)
X_fraud_train = pd.DataFrame(fraud_preprocessor.transform(X_fraud_raw_train),
                             columns=fraud_feature_names, index=X_fraud_raw_train.index)
X_fraud_test = pd.DataFrame(fraud_preprocessor.transform(X_fraud_raw_test),
                            columns=fraud_feature_names, index=X_fraud_raw_test.index)

# Severity (drop leaky cols, then one 3-way split)
LEAKY_SEVERITY_COLS = [
    "injury_claim", "property_claim", "vehicle_claim",
    "injury_claim_ratio", "property_claim_ratio", "vehicle_claim_ratio",
    "claim_to_premium_ratio",
]
X_severity_unscaled_full = X_fraud_unscaled_full.drop(
    columns=[c for c in LEAKY_SEVERITY_COLS if c in X_fraud_unscaled_full.columns]
)
y_amount = pd.read_csv("data/extracted/y_severity_regression.csv").iloc[:, 0]
y_tier = pd.read_csv("data/extracted/y_severity_class.csv").iloc[:, 0]

# do the exact same 3-way # split here to land on the identical test rows.
X_sev_raw_train, X_sev_raw_test, y_amount_train, y_amount_test, y_tier_train, y_tier_test = \
    train_test_split(X_severity_unscaled_full, y_amount, y_tier,
                     test_size=0.2, random_state=RANDOM_STATE)

X_sev_train_reg = pd.DataFrame(severity_regressor_preprocessor.transform(X_sev_raw_train),
                               columns=severity_feature_names, index=X_sev_raw_train.index)
X_sev_test_reg = pd.DataFrame(severity_regressor_preprocessor.transform(X_sev_raw_test),
                              columns=severity_feature_names, index=X_sev_raw_test.index)

X_sev_train_tier = pd.DataFrame(severity_tier_preprocessor.transform(X_sev_raw_train),
                                columns=severity_feature_names, index=X_sev_raw_train.index)
X_sev_test_tier = pd.DataFrame(severity_tier_preprocessor.transform(X_sev_raw_test),
                               columns=severity_feature_names, index=X_sev_raw_test.index)

y_tier_test_encoded = y_tier_test.astype("category").cat.codes

print("Fraud test set:", X_fraud_test.shape)
print("Severity test set:", X_sev_test_reg.shape)

"""### FRAUD MODEL EXPLAINABILITY

BUILD THE RIGHT SHAP EXPLAINER FOR THE MODEL TYPE:
LinearExplainer is exact and fast for linear models; TreeExplainer is exact and fast for tree ensembles (Random Forest, XGBoost); a generic, slower, model-agnostic explainer
"""

fraud_model_name = type(fraud_model).__name__

if hasattr(fraud_model, "coef_"):
    explainer_fraud = shap.LinearExplainer(fraud_model, X_fraud_train)
    shap_values_fraud = explainer_fraud(X_fraud_test)
    print(f"Used LinearExplainer for {fraud_model_name}")
elif hasattr(fraud_model, "estimators_") or hasattr(fraud_model, "get_booster"):
    # Random Forest exposes .estimators_, XGBoost exposes .get_booster()
    explainer_fraud = shap.TreeExplainer(fraud_model)
    shap_values_fraud = explainer_fraud(X_fraud_test)
    if len(shap_values_fraud.shape) == 3:  # some tree classifiers return one slice per class
        shap_values_fraud = shap_values_fraud[:, :, 1]
    print(f"Used TreeExplainer for {fraud_model_name}")
else:
    background = shap.sample(X_fraud_train, 100, random_state=RANDOM_STATE)
    explainer_fraud = shap.Explainer(fraud_model.predict_proba, background)
    shap_values_fraud = explainer_fraud(X_fraud_test.iloc[:100])
    print(f"Used generic (permutation) Explainer for {fraud_model_name} -- 100-row sample for speed")

# GLOBAL EXPLAINABILITY : BEESWARM SUMMARY PLOT (each dot is one test claim)
# Position = how much a feature pushed the claim toward fraud (right) or away (left).
# Color = the feature's own value for that claim (red=high, blue=low).

plt.figure()
shap.summary_plot(shap_values_fraud, X_fraud_test, show=False, max_display=15)
plt.title("SHAP Summary -- Fraud Model (Global Explainability)")
plt.tight_layout()
plt.savefig("plots/14_shap_fraud_summary.png", dpi=120, bbox_inches="tight")
plt.show()

# GLOBAL EXPLAINABILITY : MEAN |SHAP VALUE| BAR CHART

plt.figure()
shap.summary_plot(shap_values_fraud, X_fraud_test, plot_type="bar", show=False, max_display=15)
plt.title("Mean |SHAP Value| -- Fraud Model (Feature Impact Ranking)")
plt.tight_layout()
plt.savefig("plots/15_shap_fraud_bar.png", dpi=120, bbox_inches="tight")
plt.show()

# LOCAL EXPLAINABILITY : EXPLAIN ONE SPECIFIC CLAIM
# A waterfall plot shows, for 1 claim, how each feature pushed the
# prediction up (red, toward fraud) or down (blue, toward legitimate) from
# the model's baseline. This is what makes the system usable by a claims
# officer, as they need "why was this claim flagged".
if hasattr(fraud_model, "predict_proba"):
    fraud_probs_test = fraud_model.predict_proba(X_fraud_test)[:, 1]
else:
    fraud_probs_test = fraud_model.decision_function(X_fraud_test)

most_suspicious_idx = int(np.argmax(fraud_probs_test))
actual_label = "Fraud" if y_fraud_test.iloc[most_suspicious_idx] == 1 else "Legitimate"
print(f"Explaining test claim #{most_suspicious_idx} "
      f"(predicted fraud probability: {fraud_probs_test[most_suspicious_idx]:.3f}, "
      f"actual label: {actual_label})")
if actual_label == "Legitimate":
    print("NOTE: this is a FALSE POSITIVE (model very confident, but wrong) -- still a")
    print("genuinely useful example, since explainability is most valuable exactly when")
    print("a model is wrong: it shows an officer which features misled it.")

plt.figure()
shap.plots.waterfall(shap_values_fraud[most_suspicious_idx], max_display=12, show=False)
plt.title("Why This Claim Was Flagged as High Fraud Risk")
plt.tight_layout()
plt.savefig("plots/16_shap_fraud_waterfall_example.png", dpi=120, bbox_inches="tight")
plt.show()

# SAVE PER-CLAIM TOP CONTRIBUTING FEATURES (FOR EVERY TEST CLAIM)
# a clean table of each claim's top-5 contributing features, ready
# to drop into a prompt, instead of making an LLM parse a raw SHAP plot.

def top_features_for_row(shap_row, feature_names, top_n=5):
    values = shap_row.values if hasattr(shap_row, "values") else shap_row
    order = np.argsort(-np.abs(values))[:top_n]
    return [(feature_names[i], round(float(values[i]), 4)) for i in order]

fraud_explanation_rows = []
n_explain = min(len(X_fraud_test), len(shap_values_fraud))
for i in range(n_explain):
    top_feats = top_features_for_row(shap_values_fraud[i], list(X_fraud_test.columns))
    fraud_explanation_rows.append({
        "test_row_index": i,
        "fraud_probability": round(float(fraud_probs_test[i]), 4),
        "actual_label": "Fraud" if y_fraud_test.iloc[i] == 1 else "Legitimate",
        "top_contributing_features": top_feats,
    })
'''
pd.DataFrame(fraud_explanation_rows).to_csv("results/fraud_shap_explanations.csv", index=False)
print("Saved per-claim fraud explanations to results/fraud_shap_explanations.csv")
'''
# SEVERITY REGRESSOR EXPLAINABILITY ($ AMOUNT)
# (paper [8]'s gap): paper [8] applies SHAP only to
# fraud detection, and notes it "lacks claim severity prediction" entirely.
# Explaining the severity models closes exactly that gap.

explainer_severity = shap.TreeExplainer(severity_regressor)
shap_values_severity = explainer_severity(X_sev_test_reg)

plt.figure()
shap.summary_plot(shap_values_severity, X_sev_test_reg, show=False, max_display=15)
plt.title("SHAP Summary -- Severity Regressor (Global Explainability)")
plt.tight_layout()
plt.savefig("plots/17_shap_severity_summary.png", dpi=120, bbox_inches="tight")
plt.show()

severity_preds_test = severity_regressor.predict(X_sev_test_reg)
most_severe_idx = int(np.argmax(severity_preds_test))
print(f"Explaining test claim #{most_severe_idx} "
      f"(predicted claim amount: ${severity_preds_test[most_severe_idx]:,.0f}, "
      f"actual amount: ${y_amount_test.iloc[most_severe_idx]:,.0f})")

plt.figure()
shap.plots.waterfall(shap_values_severity[most_severe_idx], max_display=12, show=False)
plt.title("Why This Claim Was Predicted as High Severity")
plt.tight_layout()
plt.savefig("plots/18_shap_severity_waterfall_example.png", dpi=120, bbox_inches="tight")
plt.show()

severity_explanation_rows = []
n_explain_sev = min(len(X_sev_test_reg), len(shap_values_severity))
for i in range(n_explain_sev):
    top_feats = top_features_for_row(shap_values_severity[i], list(X_sev_test_reg.columns))
    severity_explanation_rows.append({
        "test_row_index": i,
        "predicted_claim_amount": round(float(severity_preds_test[i]), 2),
        "actual_claim_amount": round(float(y_amount_test.iloc[i]), 2),
        "top_contributing_features": top_feats,
    })
'''
pd.DataFrame(severity_explanation_rows).to_csv("results/severity_shap_explanations.csv", index=False)
print("Saved per-claim severity explanations to results/severity_shap_explanations.csv")
'''
# SEVERITY TIER CLASSIFIER EXPLAINABILITY

explainer_tier = shap.TreeExplainer(severity_tier_classifier)
shap_values_tier = explainer_tier(X_sev_test_tier)

severe_class_idx = list(severity_tier_categories).index("Severe")
print(f"Explaining predictions for the 'Severe' tier (class index {severe_class_idx} of "
      f"{len(severity_tier_categories)})")

plt.figure()
shap.summary_plot(shap_values_tier[:, :, severe_class_idx], X_sev_test_tier,
                  show=False, max_display=15)
plt.title("SHAP Summary -- What Pushes a Claim Toward the 'Severe' Tier")
plt.tight_layout()
plt.savefig("plots/19_shap_tier_severe_summary.png", dpi=120, bbox_inches="tight")
plt.show()

# Local explanation: the claim the model is most confident is "Severe"
tier_probs = severity_tier_classifier.predict_proba(X_sev_test_tier)
most_severe_tier_idx = int(np.argmax(tier_probs[:, severe_class_idx]))
actual_tier = severity_tier_categories[y_tier_test_encoded.iloc[most_severe_tier_idx]]
print(f"Explaining test claim #{most_severe_tier_idx} "
      f"(P(Severe) = {tier_probs[most_severe_tier_idx, severe_class_idx]:.3f}, "
      f"actual tier: {actual_tier})")

plt.figure()
shap.plots.waterfall(shap_values_tier[most_severe_tier_idx, :, severe_class_idx],
                     max_display=12, show=False)
plt.title("Why This Claim Was Predicted 'Severe' Tier")
plt.tight_layout()
plt.savefig("plots/20_shap_tier_waterfall_example.png", dpi=120, bbox_inches="tight")
plt.show()

# SAVE PER-CLAIM TIER EXPLANATIONS

tier_explanation_rows = []
tier_pred_encoded = severity_tier_classifier.predict(X_sev_test_tier)
n_explain_tier = min(len(X_sev_test_tier), shap_values_tier.shape[0])
for i in range(n_explain_tier):
    top_feats = top_features_for_row(shap_values_tier[i, :, severe_class_idx],
                                     list(X_sev_test_tier.columns))
    tier_explanation_rows.append({
        "test_row_index": i,
        "predicted_tier": severity_tier_categories[tier_pred_encoded[i]],
        "actual_tier": severity_tier_categories[y_tier_test_encoded.iloc[i]],
        "p_severe": round(float(tier_probs[i, severe_class_idx]), 4),
        "top_features_for_severe_class": top_feats,
    })
"""
pd.DataFrame(tier_explanation_rows).to_csv("results/severity_tier_shap_explanations.csv", index=False)
print("Saved per-claim tier explanations to results/severity_tier_shap_explanations.csv")
"""

# CROSS-CHECK: SHAP RANKING vs BUILT-IN FEATURE IMPORTANCE (FRAUD MODEL)
# A good sanity check: SHAP's global ranking should broadly agree with the
# model's own built-in feature importances.
# They won't be identical (SHAP accounts for direction and interactions),
# but if the TOP few features wildly disagreed, that would be a red flag.

mean_abs_shap_fraud = pd.Series(
    np.abs(shap_values_fraud.values).mean(axis=0), index=X_fraud_test.columns
).sort_values(ascending=False)
print("Top 10 features by SHAP (Fraud model):")
print(mean_abs_shap_fraud.head(10))

if hasattr(fraud_model, "coef_"):
    coef_importance = pd.Series(np.abs(fraud_model.coef_[0]), index=X_fraud_test.columns
                                ).sort_values(ascending=False)
    print("\nTop 10 features by |coefficient| (for comparison):")
    print(coef_importance.head(10))
elif hasattr(fraud_model, "feature_importances_"):
    builtin_importance = pd.Series(fraud_model.feature_importances_, index=X_fraud_test.columns
                                   ).sort_values(ascending=False)
    print("\nTop 10 features by built-in importance (for comparison):")
    print(builtin_importance.head(10))

print("\nDone. Explainability layer now covers all three models: fraud (XGBoost),")
print("severity amount (Random Forest) and severity tier (XGBoost) -- the last of")
print("which is new in this version, since Step 5 now properly compares models for")
print("that task instead of only ever trying Random Forest.")
