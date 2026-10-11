"""
Backend for the Meridian Auto Insurance claim review frontend
(claim_review.html + review_log.html).

Serves POST /api/assess-claim, returning EXACTLY the JSON shape
claim_review.html's getModelResult() placeholder currently fabricates:

    {
      verdict: "flagged" | "clear",
      verdictLabel: "likely fraudulent" | "likely genuine",
      probability: 0-100 (int),
      factors: [{ label: str, weight: float }, ...]   (top 5, signed)
      severityAmount: float,
      severityTier: "Low" | "Medium" | "High" | "Severe",
      priority: "Urgent" | "High" | "Standard" | "Low",
      aiSummary: str
    }

Run with:  uvicorn main:app --reload --port 8000
"""
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List

from feature_engineering import engineer_features, encode_for_model, CATEGORICAL_COLS

warnings.filterwarnings("ignore")

# Compatibility shim: these .joblib files were pickled with an older
# scikit-learn (1.6.x) that briefly used an internal `_RemainderColsList`
# helper class for ColumnTransformer's "remainder" bookkeeping. Newer
# scikit-learn versions removed it, which makes unpickling fail outright
# (not just warn) unless something with that name exists to deserialize into.
# A plain list subclass is a safe stand-in: it's only ever used internally for
# tracking column indices, which behave identically as a plain list.
import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _ct._RemainderColsList = _RemainderColsList

# ---- environment / secrets ---------------------------------------------------
# Loads KEY=value lines from a .env-style file so secrets such as
# ANTHROPIC_API_KEY never have to be typed into the terminal or committed in
# code. Both ".env" and "dotenv.env" are recognised, in backend/ or its
# parent folder. override=False: a real environment variable always wins over
# the file. NEVER commit these files (see .gitignore / .dockerignore).
try:
    from dotenv import load_dotenv
    _here = os.path.dirname(os.path.abspath(__file__))
    for _name in (".env", "dotenv.env"):
        for _dir in (_here, os.path.join(_here, "..")):
            load_dotenv(os.path.join(_dir, _name), override=False)
except ImportError:
    pass  # python-dotenv not installed: only real environment variables are used


def _find_models_dir() -> str:
    """
    Where the trained .joblib files live. Checked in order:
      1. MODELS_DIR environment variable (explicit override)
      2. backend/models            (self-contained deployments, e.g. Docker)
      3. ../ML/data/models         (this repo's layout, straight from training)
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("MODELS_DIR"),
        os.path.join(here, "models"),
        os.path.join(here, "..", "ML", "data", "models"),
    ]
    for c in candidates:
        if c and (os.path.isfile(os.path.join(c, "best_fraud_model.json"))
                  or os.path.isfile(os.path.join(c, "best_fraud_model.joblib"))):
            return os.path.abspath(c)
    raise RuntimeError(
        "Could not find the trained models. Looked in: "
        + ", ".join(str(c) for c in candidates if c)
        + ". Set the MODELS_DIR environment variable to the folder containing best_fraud_model.joblib."
    )


MODELS_DIR = _find_models_dir()

app = FastAPI(title="Meridian Claim Assessment API")

# Wide open for local development against claim_review.html opened as a plain
# file (origin "null") or served from any dev port. TIGHTEN THIS before any
# real deployment -- replace "*" with your actual frontend's origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# LOAD TRAINED ARTIFACTS ONCE AT STARTUP
# =============================================================================
def _load_xgb_classifier(stem: str):
    """
    Loads an XGBoost classifier. Prefers XGBoost's own native format
    (<stem>.json), which is version-stable and not fragile the way pickling
    a booster is; falls back to the pickled <stem>.joblib if no .json exists.
    """
    native = os.path.join(MODELS_DIR, stem + ".json")
    if os.path.isfile(native):
        model = XGBClassifier()
        model.load_model(native)
        return model
    try:
        return joblib.load(os.path.join(MODELS_DIR, stem + ".joblib"))
    except Exception as e:
        if type(e).__name__ == "XGBoostError":
            raise RuntimeError(
                f"XGBoost could not read the pickled model '{stem}.joblib' ({e}). "
                "This can happen when a pickle saved on one operating system is loaded on "
                f"another. Put '{stem}.json' (XGBoost's portable native format) into "
                f"{MODELS_DIR} and restart - it is loaded in preference to the pickle."
            ) from e
        raise


fraud_model = _load_xgb_classifier("best_fraud_model")
fraud_preprocessor = joblib.load(os.path.join(MODELS_DIR, "fraud_preprocessor.joblib"))
fraud_feature_names = joblib.load(os.path.join(MODELS_DIR, "fraud_feature_names.joblib"))

severity_regressor = joblib.load(os.path.join(MODELS_DIR, "severity_regressor.joblib"))
severity_regressor_preprocessor = joblib.load(os.path.join(MODELS_DIR, "severity_regressor_preprocessor.joblib"))
severity_feature_names = joblib.load(os.path.join(MODELS_DIR, "severity_feature_names.joblib"))

severity_tier_classifier = _load_xgb_classifier("severity_tier_classifier")
severity_tier_preprocessor = joblib.load(os.path.join(MODELS_DIR, "severity_tier_preprocessor.joblib"))
tier_categories = joblib.load(os.path.join(MODELS_DIR, "severity_tier_categories.joblib"))  # alphabetical! see below


def apply_scaler(preprocessor, df: pd.DataFrame, all_columns: list) -> pd.DataFrame:
    """
    Scales `df` the same way the saved ColumnTransformer `preprocessor` would,
    WITHOUT calling its .transform() -- these preprocessors were pickled with
    an older scikit-learn (1.6.x), and newer versions' ColumnTransformer
    silently mishandles the "remainder=passthrough" columns on unpickled
    objects (it drops them instead of passing them through -- a version
    incompatibility, not a code bug). Reading out the fitted StandardScaler's
    own mean_/scale_ and applying them by hand sidesteps that entirely: it
    only depends on StandardScaler's two plain numpy arrays, not on
    ColumnTransformer's internal transform logic.
    """
    continuous_cols = preprocessor.transformers[0][2]
    scaler = preprocessor.named_transformers_["scale"]
    out = df.copy()
    out[continuous_cols] = (out[continuous_cols].to_numpy() - scaler.mean_) / scaler.scale_
    return out[all_columns]

with open(os.path.join(MODELS_DIR, "fraud_model_metadata.json")) as f:
    FRAUD_META = json.load(f)
with open(os.path.join(MODELS_DIR, "severity_model_metadata.json")) as f:
    SEVERITY_META = json.load(f)

# All three final models (XGBoost, Random Forest, XGBoost) are tree-based, so
# TreeExplainer needs no background dataset -- it reads the tree structure
# directly. If a future retrain picks a linear model instead, this will need
# shap.LinearExplainer(model, background_df) the way notebook 06 does it.
fraud_explainer = shap.TreeExplainer(fraud_model)
severity_regressor_explainer = shap.TreeExplainer(severity_regressor)
severity_tier_explainer = shap.TreeExplainer(severity_tier_classifier)

# IMPORTANT: tier_categories is ALPHABETICAL ("High","Low","Medium","Severe"),
# not the logical Low<Medium<High<Severe order -- this is how pandas'
# .astype('category').cat.codes assigned the integer labels during training
# (notebook 05). Decoding a predicted class index MUST use this exact list.
SEVERITY_RANK = {"Low": 0, "Medium": 1, "High": 2, "Severe": 3}

# ---- human-readable labels for the SHAP "factors" list ----------------------
BASE_LABELS = {
    "policy_state": "Policy state", "policy_csl": "Policy CSL", "insured_sex": "Insured sex",
    "insured_occupation": "Occupation", "insured_hobbies": "Hobby", "insured_relationship": "Relationship",
    "incident_type": "Incident type", "collision_type": "Collision type",
    "incident_severity": "Incident severity", "authorities_contacted": "Authorities contacted",
    "incident_state": "Incident state", "incident_city": "Incident city", "auto_make": "Vehicle make",
    "incident_time_bucket": "Time of incident", "property_damage": "Property damage reported",
    "police_report_available": "Police report available",
}
NUMERIC_LABELS = {
    "customer_tenure_days": "Customer tenure (days)", "vehicle_age_at_incident": "Vehicle age at incident",
    "net_capital": "Net capital position", "claim_to_premium_ratio": "Claim-to-premium ratio",
    "incident_complexity": "Incident complexity (vehicles + injuries + witnesses)",
    "insured_education_level": "Education level (ordinal)", "months_as_customer": "Months as customer",
    "age": "Policyholder age", "policy_deductable": "Policy deductible",
    "policy_annual_premium": "Annual premium", "umbrella_limit": "Umbrella limit",
    "injury_claim": "Injury claim amount", "property_claim": "Property claim amount",
    "vehicle_claim": "Vehicle claim amount", "injury_claim_ratio": "Injury share of claim",
    "property_claim_ratio": "Property share of claim", "vehicle_claim_ratio": "Vehicle share of claim",
    "incident_hour_of_the_day": "Incident hour", "number_of_vehicles_involved": "Vehicles involved",
    "bodily_injuries": "Bodily injuries", "witnesses": "Witnesses", "auto_year": "Vehicle year",
    "incident_day_of_week": "Incident day of week", "incident_is_weekend": "Incident on a weekend",
}


def _base_field(col_name: str):
    """Returns the original categorical field a one-hot column came from, or None."""
    for prefix in BASE_LABELS:
        if col_name.startswith(prefix + "_"):
            return prefix
    return None


def top_shap_factors(shap_row: np.ndarray, columns: List[str], raw_engineered_row: dict, top_n=5):
    """
    Top contributing factors, one entry per ORIGINAL field.

    One-hot columns from the same field (e.g. incident_severity_Minor Damage,
    incident_severity_Total Loss, ...) are summed into a single contribution
    and labelled with the claim's ACTUAL value for that field. Listing a raw
    one-hot column by itself would be misleading: a claim that is "Major
    Damage" can get a large push from the Minor Damage column simply because
    that column is 0 -- the officer should see "Incident severity: Major
    Damage", not a category the claim doesn't have.
    """
    grouped = {}  # key -> [summed shap, label]
    for col, val in zip(columns, shap_row):
        base = _base_field(col)
        if base is not None:
            key = base
            label = f"{BASE_LABELS[base]}: {raw_engineered_row.get(base)}"
        elif col in NUMERIC_LABELS:
            key = col
            v = raw_engineered_row.get(col)
            v = round(v, 2) if isinstance(v, float) else v
            label = f"{NUMERIC_LABELS[col]}: {v}"
        else:
            key = col
            label = col.replace("_", " ").capitalize()
        if key not in grouped:
            grouped[key] = [0.0, label]
        grouped[key][0] += float(val)

    ranked = sorted(grouped.values(), key=lambda g: -abs(g[0]))[:top_n]
    return [{"label": label, "weight": round(total, 4)} for total, label in ranked]


# =============================================================================
# PRIORITY + AI SUMMARY
# =============================================================================
def fraud_label(prob: float) -> str:
    if prob >= 0.6:
        return "High"
    if prob >= 0.3:
        return "Medium"
    return "Low"


def compute_priority(prob: float, severity_tier: str) -> str:
    """
    Same rule the frontend placeholder uses, now driven by the real model's
    probability:
      Urgent   = High fraud risk AND (High or Severe severity)
      High     = High fraud risk, OR Severe severity
      Standard = Medium fraud risk
      Low      = everything else (Low fraud risk, Low/Medium severity)
    These thresholds (0.3 / 0.6) are a reasonable starting point, not derived
    from the data -- recalibrate against real claim volumes once deployed.
    """
    flabel = fraud_label(prob)
    if flabel == "High" and severity_tier in ("High", "Severe"):
        return "Urgent"
    if flabel == "High" or severity_tier == "Severe":
        return "High"
    if flabel == "Medium":
        return "Standard"
    return "Low"


def generate_ai_summary(prob, flabel, severity_amount, severity_tier, priority,
                        fraud_factors, severity_factors) -> str:
    """
    Plain-English synthesis of the assessment (the "AI-assisted insight" the
    problem statement calls for), built from the SHAP factors above.

    This is a deterministic template by default -- no external dependency, so
    the API always returns something. If ANTHROPIC_API_KEY is set in the
    environment, it instead asks Claude to write a more natural version of
    the exact same facts (never given free rein -- the model is only asked to
    rephrase inputs we already computed, not to invent new claims).
    """
    # Only cite factors that push in the direction the sentence claims:
    # positive SHAP weight = raises fraud risk / raises the severity estimate.
    risk_up = [f["label"] for f in fraud_factors if f["weight"] > 0]
    risk_down = [f["label"] for f in fraud_factors if f["weight"] < 0]
    sev_up = [f["label"] for f in severity_factors if f["weight"] > 0]
    sev_clause = (f"driven mainly by {sev_up[0].lower()}" if sev_up else "based on the overall claim profile")

    if flabel == "High":
        reasons = " and ".join(r.lower() for r in risk_up[:2]) or "its overall profile"
        template = (
            f"This claim is flagged for review primarily due to {reasons}. "
            f"Estimated severity is ${severity_amount:,.0f} ({severity_tier} tier), {sev_clause}. "
            f"Combined with a high fraud-risk score ({prob*100:.0f}%), this claim has been assigned "
            f"{priority} priority. Recommend a closer review before processing."
        )
    elif flabel == "Medium":
        reasons = " and ".join(r.lower() for r in risk_up[:2]) or "a few minor factors"
        template = (
            f"This claim shows moderate fraud indicators (risk score {prob*100:.0f}%), mainly {reasons}. "
            f"Estimated severity is ${severity_amount:,.0f} ({severity_tier} tier), {sev_clause}, "
            f"giving it {priority} priority. A brief check before processing is reasonable."
        )
    else:
        reassurance = (f" The factor most in its favour was {risk_down[0].lower()}." if risk_down else "")
        template = (
            f"No strong fraud indicators were found for this claim (risk score {prob*100:.0f}%).{reassurance} "
            f"Estimated severity is ${severity_amount:,.0f} ({severity_tier} tier), {sev_clause}, "
            f"giving this claim {priority} priority. It appears suitable for standard processing, "
            f"pending your review."
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return template

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5"),
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    "Rewrite the following claims-assessment note in clear, plain English, "
                    "3-4 sentences, for a claims officer. Do not add any facts, numbers, or "
                    "claims that are not already in the note below -- only rephrase for clarity "
                    f"and tone.\n\nNote:\n{template}"
                ),
            }],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip() or template
    except Exception as e:
        # Any API/network issue -> fall back to the deterministic template so
        # the endpoint never fails just because the LLM call didn't work. The
        # reason is logged so a bad key / wrong model name is visible.
        import logging
        logging.getLogger("uvicorn.error").warning(
            "LLM summary failed (%s: %s) - using the built-in template instead.",
            type(e).__name__, str(e)[:200],
        )
        return template


# =============================================================================
# REQUEST / RESPONSE
# =============================================================================
class AssessRequest(BaseModel):
    class Config:
        extra = "allow"  # accept the raw claim dict as-is (keys like "capital-gains" aren't valid Python identifiers)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "fraud_model": FRAUD_META["model"],
        "severity_regressor": SEVERITY_META["regressor"]["model"],
        "severity_tier_classifier": SEVERITY_META["tier_classifier"]["model"],
        "llm_summaries_enabled": bool(os.environ.get("GOOGLE_API_KEY")),
    }


@app.post("/api/assess-claim")
def assess_claim(claim: AssessRequest) -> Dict[str, Any]:
    raw = claim.dict()

    try:
        engineered = engineer_features(raw)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=f"Missing or invalid claim field: {e}")

    # ---- FRAUD ----
    X_fraud_raw = encode_for_model(engineered, fraud_feature_names, drop_leaky=False)
    X_fraud_scaled = apply_scaler(fraud_preprocessor, X_fraud_raw, fraud_feature_names)
    fraud_prob = float(fraud_model.predict_proba(X_fraud_scaled)[0, 1])

    fraud_shap = fraud_explainer(X_fraud_scaled)
    fraud_shap_row = np.array(fraud_shap.values)[0]
    if fraud_shap_row.ndim == 2:  # (n_features, n_classes) -> take the "fraud" class
        fraud_shap_row = fraud_shap_row[:, 1]
    fraud_factors = top_shap_factors(fraud_shap_row, fraud_feature_names, engineered)

    # ---- SEVERITY ($ amount) ----
    X_sev_raw = encode_for_model(engineered, severity_feature_names, drop_leaky=True)
    X_sev_scaled = apply_scaler(severity_regressor_preprocessor, X_sev_raw, severity_feature_names)
    severity_amount = float(severity_regressor.predict(X_sev_scaled)[0])

    sev_shap = severity_regressor_explainer(X_sev_scaled)
    sev_shap_row = np.array(sev_shap.values)[0]
    severity_factors = top_shap_factors(sev_shap_row, severity_feature_names, engineered)

    # ---- SEVERITY TIER ----
    X_tier_scaled = apply_scaler(severity_tier_preprocessor, X_sev_raw, severity_feature_names)
    tier_pred_idx = int(severity_tier_classifier.predict(X_tier_scaled)[0])
    severity_tier = tier_categories[tier_pred_idx]  # decode using the SAVED (alphabetical) order

    # ---- PRIORITY ----
    priority = compute_priority(fraud_prob, severity_tier)
    flabel = fraud_label(fraud_prob)

    # ---- AI SUMMARY ----
    ai_summary = generate_ai_summary(
        fraud_prob, flabel, severity_amount, severity_tier, priority, fraud_factors, severity_factors
    )

    probability_pct = round(fraud_prob * 100)
    verdict = "flagged" if fraud_prob >= 0.5 else "clear"

    return {
        "verdict": verdict,
        "verdictLabel": "likely fraudulent" if verdict == "flagged" else "likely genuine",
        "probability": probability_pct,
        "factors": fraud_factors,
        "severityAmount": round(severity_amount, 2),
        "severityTier": severity_tier,
        "priority": priority,
        "aiSummary": ai_summary,
    }
