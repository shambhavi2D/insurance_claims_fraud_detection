### MODEL TRAINING: FRAUD CLASSIFICATION
"""
SAME imbalance strategies for EVERY model:
(a) "class_weight" : no resampling, the model penalises mistakes on
                            the fraud class more heavily
(b) "smote" : synthetic fraud rows added INSIDE each training
                              fold only (SMOTENC, which respects 0/1 columns)

SAME tuning effort for EVERY model (a small grid of 6 settings each).

Nothing is fitted on data that a model isn't allowed to see: the scaler and
      SMOTE live inside a Pipeline, so they are re-fitted on each training
      fold only.

NESTED, REPEATED cross-validation: the inner loop tunes settings, the
     outer loop scores the result on rows the tuning never touched. Repeating
      it with different shuffles shows how stable each score is (mean +/- std).

The 20% TEST SET is used exactly once, at the very end, on the single
      winning model. It plays no part in choosing the winner.

Reference papers: [1] Linear SVM + imbalance handling, [2]/[5] Random Forest
 and XGBoost, [8] ensemble + SHAP

"""

# importing libraries

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
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold,
    GridSearchCV, cross_validate,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score, precision_score, recall_score,
)
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

RANDOM_STATE = 42

# Cross-validation size. 5 outer folds x 2 repeats = 10 scored folds per model.
# Raise OUTER_REPEATS to 3 for even steadier numbers (it takes longer).

OUTER_SPLITS = 5
OUTER_REPEATS = 2
INNER_SPLITS = 3

"""We load the UNSCALED matrix on purpose. Scaling is done later, inside the
pipeline, using only each fold's training rows. (If we loaded the already
scaled file, the scaler would have been fitted on every row, test rows
included, which is a small but real leak.)
"""

X = pd.read_csv("X_features_unscaled.csv")
y = pd.read_csv("y_fraud.csv").iloc[:, 0]  # 1 = fraud, 0 = legitimate

print("X shape:", X.shape)
print("Fraud share:", y.mean().round(3))

# HOLD OUT THE TEST SET (STRATIFIED) AND LEAVE IT ALONE
# 80% goes into "train", where ALL model comparison and tuning happens (via
# cross-validation). 20% is locked away until later. stratify=y keeps the
# ~25% fraud share in both parts.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("Train:", X_train.shape, "fraud share", y_train.mean().round(3))
print("Test :", X_test.shape, "fraud share", y_test.mean().round(3), "(locked until 4.10)")

"""#### PREPROCESSING RULES (SCALER + WHICH COLUMNS ARE 0/1)
* Continuous / ordinal columns (age, premium, tenure...) get StandardScaler.
* 0/1 columns (one-hot flags) are passed through untouched.

Scaling a 0/1 flag makes it harder to read, and SMOTENC
(below) needs to know which columns are yes/no flags so that the synthetic
rows it creates only contain valid 0 or 1 values, never something like 0.4.
"""

binary_cols = [c for c in X_train.columns if set(X_train[c].unique()) <= {0, 1}]
continuous_cols = [c for c in X_train.columns if c not in binary_cols]

# ColumnTransformer outputs the scaled columns first, then the pass-through
# ones, so this is the column order every model sees:

feature_order = continuous_cols + binary_cols
categorical_idx = list(range(len(continuous_cols), len(feature_order)))

print(f"{len(continuous_cols)} continuous columns, {len(binary_cols)} binary (0/1) columns")


def make_preprocessor():
    return ColumnTransformer(
        [("scale", StandardScaler(), continuous_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

"""##### CANDIDATE MODELS AND THEIR TUNING GRIDS
Every model gets:  
* the same preprocessing   
* the same two imbalance strategies   
* a grid of exactly 6 settings   
* the same tuning metric (F1).

Equal effort means no model wins simply because it was tuned harder.
"""

n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
scale_pos_weight = n_neg / n_pos  # XGBoost's version of class_weight="balanced"


def cw(strategy):

# class_weight only applies in the 'class_weight' strategy. With SMOTE the
# training data is already balanced, so adding weights on top would double
# count the fraud class."""

    return "balanced" if strategy == "class_weight" else None


CANDIDATES = {
    "Logistic Regression": {
        "make": lambda s: LogisticRegression(
            max_iter=5000, class_weight=cw(s), random_state=RANDOM_STATE),
        "grid": {"clf__C": [0.003, 0.01, 0.03, 0.1, 1, 10]},
    },
    "Linear SVM": {  # the model used in paper [1]
        "make": lambda s: LinearSVC(
            dual="auto", max_iter=20000, class_weight=cw(s), random_state=RANDOM_STATE),
        "grid": {"clf__C": [0.003, 0.01, 0.03, 0.1, 1, 10]},
    },
    "Random Forest": {  # papers [2], [5], [8]
        "make": lambda s: RandomForestClassifier(
            n_estimators=200, class_weight=cw(s), random_state=RANDOM_STATE, n_jobs=1),
        "grid": {"clf__max_depth": [4, 8, None], "clf__min_samples_leaf": [1, 3]},
    },
    "XGBoost": {  # papers [2], [5]
        "make": lambda s: XGBClassifier(
            n_estimators=200, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(scale_pos_weight if s == "class_weight" else 1.0),
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1),
        "grid": {"clf__max_depth": [2, 3, 5], "clf__learning_rate": [0.05, 0.1]},
    },
}

STRATEGIES = ["class_weight", "smote"]


def build_pipeline(model_name, strategy):

# (1)preprocess (2)(optional SMOTENC) (3)classifier.
# Because SMOTENC is a step inside the pipeline, it only ever runs on the
# training part of a fold. The validation/test rows are never resampled."""

    steps = [("prep", make_preprocessor())]
    if strategy == "smote":
        steps.append(("smote", SMOTENC(categorical_features=categorical_idx,
                                       random_state=RANDOM_STATE)))
    steps.append(("clf", CANDIDATES[model_name]["make"](strategy)))
    return Pipeline(steps)


def make_tuned(model_name, strategy, inner_splits=INNER_SPLITS):

# A pipeline wrapped in GridSearchCV: it tries every setting in the grid
# using an inner cross-validation, then refits the best one."""

    return GridSearchCV(
        build_pipeline(model_name, strategy),
        CANDIDATES[model_name]["grid"],
        scoring="f1",
        cv=StratifiedKFold(inner_splits, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit=True,
    )

"""#### NESTED CROSS-VALIDATION : THE CORE IDEA

If we tune a model's settings and score it on the same folds, the score is
optimistic, because the settings were picked to look good on those very rows.
Nested CV avoids that with two loops:

Outer loop (5 folds x 2 repeats): hold out 1/5 of the training rows as a "judging" fold.

Inner loop (3 folds): using only the other 4/5, try all 6 settings, pick
the best, refit it.
   
Then score that tuned model on the judging fold it has never seen.

Every model x strategy goes through the SAME outer folds (same random_state),
so they are judged on identical rows. We also report a threshold-free pair of
metrics next to F1, because F1 depends on the 0.5 cut-off:

ROC-AUC : how well fraud is ranked above legitimate overall

PR-AUC  : the same, but focused on the rare fraud class (more honest when classes are imbalanced)
"""

SCORING = {
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}
outer_cv = RepeatedStratifiedKFold(
    n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=RANDOM_STATE
)

"""#### RUN THE COMPARISON
A "know-nothing" reference: guesses fraud at random in proportion to its
frequency. Any real model must clearly beat this to be worth anything.
"""

rows = []
fold_f1 = {}  # keep per-fold F1 so we can compare models fold by fold

dummy = cross_validate(
    DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
    X_train, y_train, cv=outer_cv, scoring=SCORING,
)
rows.append({"model": "Random guessing (reference)", "strategy": "-",
             **{f"{m}_mean": dummy[f"test_{m}"].mean() for m in SCORING},
             **{f"{m}_std": dummy[f"test_{m}"].std() for m in SCORING}})

for model_name in CANDIDATES:
    for strategy in STRATEGIES:
        t0 = time.time()
        cv_out = cross_validate(
            make_tuned(model_name, strategy), X_train, y_train,
            cv=outer_cv, scoring=SCORING, n_jobs=1,
        )
        fold_f1[(model_name, strategy)] = cv_out["test_f1"]
        rows.append({"model": model_name, "strategy": strategy,
                     **{f"{m}_mean": cv_out[f"test_{m}"].mean() for m in SCORING},
                     **{f"{m}_std": cv_out[f"test_{m}"].std() for m in SCORING}})
        print(f"{model_name:20s} | {strategy:12s} | "
              f"F1 {cv_out['test_f1'].mean():.3f} +/- {cv_out['test_f1'].std():.3f} "
              f"| ROC-AUC {cv_out['test_roc_auc'].mean():.3f} "
              f"| {time.time() - t0:.0f}s")

comparison = pd.DataFrame(rows)

# RESULTS TABLE, TIE CHECK AND CHARTS

display_cols = ["model", "strategy", "f1_mean", "f1_std", "precision_mean",
                "recall_mean", "roc_auc_mean", "pr_auc_mean"]
ranked = comparison[comparison["model"] != "Random guessing (reference)"] \
    .sort_values("f1_mean", ascending=False).reset_index(drop=True)
print(ranked[display_cols].round(3).to_string(index=False))
print("\nReference (random guessing):")
print(comparison[comparison["model"] == "Random guessing (reference)"][display_cols].round(3).to_string(index=False))

best = ranked.iloc[0]
best_model_name, best_strategy = best["model"], best["strategy"]

# TIE CHECK (a rule of thumb, not a formal statistical test): if another
# setup's mean F1 is within one standard deviation of the leader, the data
# cannot really tell them apart, and you can choose between them on other
# grounds (explainability, simplicity, fit with paper [8]).

ranked["within_1_std_of_best"] = ranked["f1_mean"] >= best["f1_mean"] - best["f1_std"]
ties = ranked[ranked["within_1_std_of_best"]][["model", "strategy", "f1_mean"]]
print(f"\nLeader by mean F1: {best_model_name} + {best_strategy} "
      f"(F1 {best['f1_mean']:.3f} +/- {best['f1_std']:.3f})")
print("Statistically hard to separate from the leader (within 1 std):")
print(ties.round(3).to_string(index=False))

# F1, ROC-AUC and PR-AUC with error bars

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
plot_df = ranked.copy()
models_order = list(CANDIDATES.keys())
width = 0.38
for ax, (metric, title) in zip(axes, [("f1", "F1-score (fraud class)"),
                                       ("roc_auc", "ROC-AUC"),
                                       ("pr_auc", "PR-AUC")]):
    for j, (strategy, color) in enumerate(zip(STRATEGIES, ["#4c72b0", "#dd8452"])):
        sub = plot_df[plot_df["strategy"] == strategy].set_index("model").loc[models_order]
        ax.bar(np.arange(len(models_order)) + (j - 0.5) * width, sub[f"{metric}_mean"],
               width, yerr=sub[f"{metric}_std"], capsize=3, color=color, label=strategy)
    ref = comparison[comparison["model"] == "Random guessing (reference)"][f"{metric}_mean"].iloc[0]
    ax.axhline(ref, color="grey", linestyle="--", linewidth=1, label="random guessing")
    ax.set_xticks(np.arange(len(models_order)))
    ax.set_xticklabels(models_order, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylim(0, 1)
axes[0].legend(fontsize=8)
plt.suptitle("Fair comparison: repeated nested cross-validation (mean +/- std)")
plt.tight_layout()
plt.savefig("plots/07_model_comparison.png", dpi=120)
plt.show()

"""#### FOLD-BY-FOLD CHECK OF THE LEADING MODEL
All setups were scored on identical outer folds, so we can ask
in how many of the 10 folds did the leader actually beat each rival?

A leader that wins 9/10 folds is a clearer winner than one that wins 6/10.
"""

leader_scores = fold_f1[(best_model_name, best_strategy)]
wins = []
for (m, s), scores in fold_f1.items():
    if (m, s) == (best_model_name, best_strategy):
        continue
    wins.append({"rival": f"{m} + {s}",
                 "leader_wins_folds": int((leader_scores > scores).sum()),
                 "ties": int((leader_scores == scores).sum()),
                 "total_folds": len(scores)})
print(pd.DataFrame(wins).sort_values("leader_wins_folds").to_string(index=False))

"""#### FINAL MODEL: TUNE ON ALL TRAINING DATA, THEN OPEN THE TEST SET ONCE
Now we fit the chosen model + strategy on the full training
split, tuning with a slightly larger inner CV (5 folds), and score it on the
locked 20% test set. Because the test set had no say in the choice, this is
an honest estimate of performance on new claims. It comes from just around 50 fraud cases, so expect it to wobble around the cross-validated number above; the
CV figure is the steadier one to quote, the test figure is the sanity check.
"""

print(f"Final model: {best_model_name} + {best_strategy}")
final_search = make_tuned(best_model_name, best_strategy, inner_splits=5)
final_search.fit(X_train, y_train)
print("Best settings found:", final_search.best_params_)

final_pipe = final_search.best_estimator_
y_pred = final_pipe.predict(X_test)
if hasattr(final_pipe, "predict_proba"):
    y_score = final_pipe.predict_proba(X_test)[:, 1]
else:  # LinearSVC has no probabilities, only a signed distance from the boundary
    y_score = final_pipe.decision_function(X_test)

test_metrics = {
    "f1": f1_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_score),
    "pr_auc": average_precision_score(y_test, y_score),
}
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
print({k: round(v, 3) for k, v in test_metrics.items()})

# FINAL-MODEL CONFUSION MATRIX, ROC AND PRECISION-RECALL CURVES

# The confusion matrix shows the types of mistakes. A missed fraud (false
# negative) costs the insurer money; a wrongly flagged legitimate claim (false
# positive) costs officer time and customer trust.

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=["Legit", "Fraud"]).plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion matrix (test set)")

fpr, tpr, _ = roc_curve(y_test, y_score)
axes[1].plot(fpr, tpr, label=f"AUC = {test_metrics['roc_auc']:.3f}")
axes[1].plot([0, 1], [0, 1], "k--", label="random")
axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
axes[1].set_title("ROC curve"); axes[1].legend()

prec, rec, _ = precision_recall_curve(y_test, y_score)
axes[2].plot(rec, prec, label=f"PR-AUC = {test_metrics['pr_auc']:.3f}")
axes[2].axhline(y_test.mean(), color="grey", linestyle="--", label="random (= fraud share)")
axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
axes[2].set_title("Precision-recall curve"); axes[2].legend()
plt.suptitle(f"Final model: {best_model_name} + {best_strategy}")
plt.tight_layout()
plt.savefig("plots/08_final_model_evaluation.png", dpi=120)
plt.show()

# BUILT-IN FEATURE IMPORTANCE (PREVIEW BEFORE SHAP)
# Tree models expose feature_importances_, linear models expose coef_.
# Both tell how much the model leans on each feature, but not which direction it pushes a claim.
# SHAP (next step) adds that.

final_clf = final_pipe.named_steps["clf"]
if hasattr(final_clf, "feature_importances_"):
    imp = pd.Series(final_clf.feature_importances_, index=feature_order)
elif hasattr(final_clf, "coef_"):
    imp = pd.Series(np.abs(final_clf.coef_[0]), index=feature_order)
else:
    imp = None

if imp is not None:
    top15 = imp.sort_values(ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=top15.values, y=top15.index, color="#4c72b0")
    plt.title(f"Top 15 built-in importances: {best_model_name}")
    plt.tight_layout()
    plt.savefig("plots/10_feature_importance.png", dpi=120)
    plt.show()
    print(top15.round(4))

'''
# SAVE EVERYTHING

# We save the pieces separately because
# the SHAP step needs the bare classifier, and
# the backend needs the fitted preprocessor to turn a new raw claim
# into exactly the numbers the classifier was trained on.

joblib.dump(final_clf, "models/best_fraud_model.joblib")
joblib.dump(final_pipe.named_steps["prep"], "models/fraud_preprocessor.joblib")
joblib.dump(feature_order, "models/fraud_feature_names.joblib")

metadata = {
    "model": best_model_name,
    "imbalance_strategy": best_strategy,
    "best_params": {k: (None if v is None else (v if isinstance(v, (int, float, str)) else str(v)))
                    for k, v in final_search.best_params_.items()},
    "cv_f1_mean": float(best["f1_mean"]), "cv_f1_std": float(best["f1_std"]),
    "cv_roc_auc_mean": float(best["roc_auc_mean"]),
    "test_metrics": {k: float(v) for k, v in test_metrics.items()},
}
with open("models/fraud_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

comparison.to_csv("results/fraud_model_comparison.csv", index=False)
pd.DataFrame([{"model": best_model_name, "strategy": best_strategy, **test_metrics}]) \
    .to_csv("results/fraud_final_test_metrics.csv", index=False)

print("Saved: models/best_fraud_model.joblib, fraud_preprocessor.joblib,")
print("       fraud_feature_names.joblib, fraud_model_metadata.json")
print("Saved: results/fraud_model_comparison.csv, fraud_final_test_metrics.csv")
print("\nNext: Step 6 (SHAP) loads these files, so re-run 06 after this notebook.")
'''
