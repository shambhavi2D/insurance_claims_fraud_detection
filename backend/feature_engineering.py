"""
Replicates the feature engineering from notebook 03 (03_feature_extraction)
for ONE new, raw claim submitted through the API -- not a batch of training
rows. This is the piece that makes /api/assess-claim work on data the models
have never seen, by turning it into exactly the same shape of numbers the
models were trained on.

WHY A SEPARATE MODULE: this logic has to stay in lockstep with notebook 03.
If you change the feature engineering there, mirror the change here, or the
backend's predictions will silently stop matching what the notebooks report.
"""
import pandas as pd
import numpy as np

# Same ordinal mapping used in notebook 03's fix (Step 3.4): there is a real
# order to education levels, so this is encoded as a single ordinal number
# rather than one-hot columns.
EDUCATION_ORDER = ["High School", "College", "Associate", "JD", "Masters", "MD", "PhD"]

# The 7 columns notebook 05 removes before modeling severity, because they are
# mathematically derived from total_claim_amount (the thing being predicted).
# Kept here so both the regressor and the tier classifier can reuse the same
# "drop these" list the training notebooks used.
LEAKY_SEVERITY_COLS = [
    "injury_claim", "property_claim", "vehicle_claim",
    "injury_claim_ratio", "property_claim_ratio", "vehicle_claim_ratio",
    "claim_to_premium_ratio",
]

# Raw categorical columns that get one-hot encoded (everything else that
# isn't ordinal or already numeric). Matches notebook 03 Step 3.4/3.5.
CATEGORICAL_COLS = [
    "policy_state", "policy_csl", "insured_sex", "insured_occupation",
    "insured_hobbies", "insured_relationship", "incident_type", "collision_type",
    "incident_severity", "authorities_contacted", "incident_state", "incident_city",
    "auto_make", "incident_time_bucket", "property_damage", "police_report_available",
]


def _time_of_day_bucket(hour: int) -> str:
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


def engineer_features(raw: dict) -> dict:
    """
    Takes a raw claim dict with the SAME keys as the frontend's CLAIMS objects
    (e.g. months_as_customer, policy_bind_date, incident_date, "capital-gains",
    injury_claim, ...) and returns a flat dict of engineered features: some
    numeric (ready to use as-is), some still categorical text (to be one-hot
    encoded next, in encode_for_model()).

    This mirrors notebook 03's Step 3.1 (feature engineering) exactly.
    """
    policy_bind_date = pd.to_datetime(raw["policy_bind_date"])
    incident_date = pd.to_datetime(raw["incident_date"])

    injury_claim = float(raw["injury_claim"])
    property_claim = float(raw["property_claim"])
    vehicle_claim = float(raw["vehicle_claim"])
    total_claim_amount = injury_claim + property_claim + vehicle_claim  # not an output column itself

    incident_hour = int(raw["incident_hour_of_the_day"])
    incident_day_of_week = incident_date.dayofweek

    row = {
        # ---- engineered numeric features (Step 3.1) ----
        "customer_tenure_days": (incident_date - policy_bind_date).days,
        "incident_day_of_week": incident_day_of_week,
        "incident_is_weekend": int(incident_day_of_week >= 5),
        "vehicle_age_at_incident": incident_date.year - int(raw["auto_year"]),
        "net_capital": float(raw["capital-gains"]) + float(raw["capital-loss"]),
        "injury_claim_ratio": injury_claim / total_claim_amount if total_claim_amount else 0.0,
        "property_claim_ratio": property_claim / total_claim_amount if total_claim_amount else 0.0,
        "vehicle_claim_ratio": vehicle_claim / total_claim_amount if total_claim_amount else 0.0,
        "claim_to_premium_ratio": total_claim_amount / float(raw["policy_annual_premium"]),
        "incident_complexity": (
            int(raw["number_of_vehicles_involved"]) + int(raw["bodily_injuries"]) + int(raw["witnesses"])
        ),
        # ---- raw numeric passthroughs (unchanged from the dataset) ----
        "months_as_customer": float(raw["months_as_customer"]),
        "age": float(raw["age"]),
        "policy_deductable": float(raw["policy_deductable"]),
        "policy_annual_premium": float(raw["policy_annual_premium"]),
        "umbrella_limit": float(raw["umbrella_limit"]),
        "capital-gains": float(raw["capital-gains"]),
        "capital-loss": float(raw["capital-loss"]),
        "incident_hour_of_the_day": incident_hour,
        "number_of_vehicles_involved": float(raw["number_of_vehicles_involved"]),
        "bodily_injuries": float(raw["bodily_injuries"]),
        "witnesses": float(raw["witnesses"]),
        "auto_year": float(raw["auto_year"]),
        "injury_claim": injury_claim,
        "property_claim": property_claim,
        "vehicle_claim": vehicle_claim,
        # ---- ordinal ----
        "insured_education_level": EDUCATION_ORDER.index(raw["insured_education_level"]),
        # ---- still-categorical text, one-hot encoded in encode_for_model() ----
        "policy_state": raw["policy_state"],
        "policy_csl": raw["policy_csl"],
        "insured_sex": raw["insured_sex"],
        "insured_occupation": raw["insured_occupation"],
        "insured_hobbies": raw["insured_hobbies"],
        "insured_relationship": raw["insured_relationship"],
        "incident_type": raw["incident_type"],
        "collision_type": raw["collision_type"],
        "incident_severity": raw["incident_severity"],
        "authorities_contacted": raw["authorities_contacted"],
        "incident_state": raw["incident_state"],
        "incident_city": raw["incident_city"],
        "auto_make": raw["auto_make"],
        "incident_time_bucket": _time_of_day_bucket(incident_hour),
        "property_damage": raw["property_damage"],
        "police_report_available": raw["police_report_available"],
    }
    return row


def encode_for_model(engineered_row: dict, feature_names: list, drop_leaky: bool) -> pd.DataFrame:
    """
    Turns one engineered row into a single-row DataFrame with EXACTLY the
    columns (name and order) a trained model expects.

    How this replicates training-time one-hot encoding for a single row:
    pd.get_dummies() on a one-row frame creates one "<column>_<value>=1"
    column per categorical field. reindex(columns=feature_names, fill_value=0)
    then aligns it to the full training-time column list: any dummy column
    this row didn't produce (because the category doesn't apply, or -- for
    an unseen category never seen during training -- no column exists at
    all) is filled with 0. That's the correct, standard way to encode a
    single new row against a saved one-hot schema: an unseen category is
    treated like the implicit "all dummies are 0" baseline.
    """
    df = pd.DataFrame([engineered_row])
    if drop_leaky:
        df = df.drop(columns=[c for c in LEAKY_SEVERITY_COLS if c in df.columns])
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns])
    df = df.reindex(columns=feature_names, fill_value=0)
    return df
