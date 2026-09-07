# FEATURE EXTRACTION
"""
This turns the CLEANED dataset into a numeric feature matrix that a model (Random Forest / XGBoost / SVM can actually be trained on.)

Feature extraction here has 3 parts:

   1. Feature ENGINEERING   -create new, more informative columns

   2. Feature SELECTION     -drop columns that hamper

   3. Encoding + Scaling    -convert everything to numbers, on a common scale
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("data/processed_insurance_claims.csv")

print("Starting shape:", df.shape)

print(df[["policy_bind_date", "incident_date"]].dtypes)

"""### FEATURE ENGINEERING

#### 1. Customer tenure at time of incident (in days)

months_as_customer already exists, but it's static / self-reported.
Computing it ourselves from the two dates cross-checks that column and
gives a precise, continuous tenure feature.

A very short tenure followed by a big claim is a classic fraud-risk pattern.
"""

df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])
df["incident_date"] = pd.to_datetime(df["incident_date"])

df["customer_tenure_days"] = (df["incident_date"] - df["policy_bind_date"]).dt.days

"""#### 2. Time-based features from the incident date
Fraud patterns can correlate with day-of-week (e.g. claims reported right after a weekend) and time-of-day is already given to us directly as incident_hour_of_the_day. We derive weekday and a coarse time-of-day bucket.
"""

df["incident_day_of_week"] = df["incident_date"].dt.dayofweek  # 0=Mon ... 6=Sun
df["incident_is_weekend"] = (df["incident_day_of_week"] >= 5).astype(int)

def time_of_day_bucket(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

df["incident_time_bucket"] = df["incident_hour_of_the_day"].apply(time_of_day_bucket)

"""
#### 3. Vehicle age at time of incident

auto_year alone (e.g. "2004") isn't directly meaningful to a model the way "this car was 11 years old when the claim was filed" is. Older/newer vehicles have different theft, damage, and total-loss profiles."""

df["vehicle_age_at_incident"] = df["incident_date"].dt.year - df["auto_year"]

"""#### 3. Net capital position
capital-gains and capital-loss are two separate columns; their net value is a cleaner single indicator of the policyholder's recent financial position (financial pressure is a commonly cited fraud-risk factor).
"""

df["net_capital"] = df["capital-gains"] + df["capital-loss"]  # loss is already negative

"""#### 4. Claim composition ratios (instead of raw duplicate totals)
Ties directly to the multicollinearity finding: rather than feeding the model total_claim_amount AND its three raw components (which are perfectly collinear by construction: total = injury + property + vehicle), we replace the *raw total* with the *proportion* each component contributes. This keeps the useful "shape" of the claim without duplicating information -- e.g. "80% of this claim is vehicle damage" is a distinct, non-redundant signal that a raw total can't give us.


"""

df["injury_claim_ratio"] = df["injury_claim"] / df["total_claim_amount"].replace(0, np.nan)
df["property_claim_ratio"] = df["property_claim"] / df["total_claim_amount"].replace(0, np.nan)
df["vehicle_claim_ratio"] = df["vehicle_claim"] / df["total_claim_amount"].replace(0, np.nan)
df[["injury_claim_ratio", "property_claim_ratio", "vehicle_claim_ratio"]] = (df[["injury_claim_ratio", "property_claim_ratio", "vehicle_claim_ratio"]].fillna(0))

"""#### 5. Claim-to-premium ratio
A claim amount is meaningless in isolation; $50,000 is unremarkable for a customer paying a high premium on a high-value policy, but alarming for someone on a minimal policy.

Normalizing claim size against the customer's own annual premium creates a much stronger, self-relative risk signal than either raw column alone.

 This is the kind of derived "risk ratio" feature paper[2] describesfor its risk-measurement stage.
"""

df["claim_to_premium_ratio"] = df["total_claim_amount"] / df["policy_annual_premium"]

"""
#### 6. Severity label for the "claim severity estimation" objective
The problem statement explicitly asks for claim SEVERITY estimation alongside fraud detection.

"incident_severity" is a categorical description (Trivial/Minor/Major/Total Loss) supplied by the reporting party: useful, but subjective.

We additionally bucket total_claim_amount itself into a data-driven severity tier using quartiles, so we have an objective, amount-based target ready for our future severity-prediction model, separate from the fraud target.
"""

df["claim_severity_tier"] = pd.qcut(
    df["total_claim_amount"], q=4, labels=["Low", "Medium", "High", "Severe"]
)

"""#### 6. Total number of parties/witnesses/injuries involved

A single composite "incident complexity" score is often more stable and less noisy than several small raw counts, especially with only 1000 rows.

"""

df["incident_complexity"] = (
    df["number_of_vehicles_involved"] + df["bodily_injuries"] + df["witnesses"]
)

print("Shape after feature engineering:", df.shape)
df[[
    "customer_tenure_days", "vehicle_age_at_incident", "net_capital",
    "claim_to_premium_ratio", "claim_severity_tier", "incident_complexity"
]].head()

"""### FEATURE SELECTION -- DROP COLUMNS THAT DON'T HELP THE MODEL

PURE IDENTIFIERS - unique (or near-unique) per row, so the model can
only "memorize" them; they carry no pattern that generalizes to a new,
unseen claim.
        policy_number   (1000/1000 unique)
        insured_zip     (995/1000 unique)
        incident_location (1000/1000 unique -- literally the street address)

RAW DATE COLUMNS -- we've already extracted the useful signal from
      them into customer_tenure_days / incident_day_of_week / etc. Keeping
      the raw datetime objects too would be redundant and most ML libraries
      can't consume a datetime column directly.
       ( policy_bind_date, incident_date )

REDUNDANT FINANCIAL COLUMN -- total_claim_amount is dropped
      because it is now fully reconstructable from the three claim ratios
      we engineered + was the perfectly-collinear sum of
      injury/property/vehicle claim.
      NOTE: total_claim_amount is NOT dropped as we intend to use it as the
      regression TARGET for severity prediction. We separate it
      into y_severity before dropping it from X.

VERY HIGH-CARDINALITY, LOW-SIGNAL CATEGORICAL -- auto_model has 39
      unique values across only 1000 rows; one-hot encoding it would add 39
      sparse columns for very little gain.
"""

df['auto_model'].unique()

id_and_leak_cols = [
    "policy_number", "insured_zip", "incident_location",
    "policy_bind_date", "incident_date",
    "auto_model"]

df_features = df.drop(columns=id_and_leak_cols)
print("Shape after dropping identifiers/raw dates/high-cardinality col:", df_features.shape)

"""#### SEPARATE TARGETS FROM FEATURES
 This project has TWO learning objectives:

y_fraud: binary classification target  (fraud_reported: Y/N)

y_severity: claim severity

(we give both a regression version [total_claim_amount] and a classification version [claim_severity_tier] so we can pick whichever modeling approach you go with later)

All three must be removed from X (the feature matrix); leaving any of them in X would let the model "cheat" by training on the answer.
"""

y_fraud = df_features["fraud_reported"].map({"Y": 1, "N": 0})
y_severity_regression = df_features["total_claim_amount"]
y_severity_class = df_features["claim_severity_tier"]

X = df_features.drop(columns=["fraud_reported", "total_claim_amount", "claim_severity_tier"])

print("Fraud target distribution:\n", y_fraud.value_counts())
print("\nFeature matrix shape (before encoding):", X.shape)

print(X["insured_education_level"].unique())
print(X["insured_education_level"].dtype)

# ---- One-hot encoding for nominal categorical columns ----

nominal_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("Nominal columns to one-hot encode:", nominal_cols)

# One-hot encoding
X_encoded = pd.get_dummies(
    X,
    columns=nominal_cols,
    drop_first=True
)

# Convert boolean columns to 0/1
bool_cols = X_encoded.select_dtypes(include="bool").columns
X_encoded[bool_cols] = X_encoded[bool_cols].astype(int)

print("Shape after one-hot encoding:", X_encoded.shape)

"""### FEATURE SCALING
Our numeric features are on very different scales; e.g. age (19-64)
 vs policy_annual_premium (500-2500) vs customer_tenure_days (0-17000).
 Distance or gradient-based models (SVM, logistic regression, neural nets) are
 sensitive to this and let large-scale columns dominate purely because
 of their units, not their actual importance.

  Tree-based models (RandomForest/XGBoost) don't strictly need scaling, but since paper [1] specifically uses a Linear SVM, we scale here so the feature set is ready for ANY of the candidate models from our reference papers.

 We fit the scaler only on numeric columns; the one-hot (0/1) columns are
 left as-is, since scaling binary indicator columns is unnecessary and
 actually makes them harder to interpret.
"""

numeric_feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
# (insured_education_level is now numeric/ordinal too, so it's included here)

scaler = StandardScaler()
X_encoded[numeric_feature_cols] = scaler.fit_transform(X_encoded[numeric_feature_cols])

print("Scaled numeric columns:", numeric_feature_cols)
X_encoded[numeric_feature_cols].describe().T[["mean", "std"]].round(2)

# sanity checks
print("Final feature matrix shape:", X_encoded.shape)
print("Any missing values left: ", X_encoded.isnull().sum().sum())
print("Any non-numeric columns left: ",
      X_encoded.select_dtypes(exclude=[np.number]).columns.tolist())

# SAVE OUTPUTS FOR MODEL TRAINING

X_encoded.to_csv("X_features.csv", index=False)
y_fraud.to_csv("y_fraud.csv", index=False)
y_severity_regression.to_csv("y_severity_regression.csv", index=False)
y_severity_class.to_csv("y_severity_class.csv", index=False)

print("Saved: X_features.csv, y_fraud.csv, y_severity_regression.csv, y_severity_class.csv")

print("\nThese four files are the direct input to our next notebook: model")
print("training for fraud classification (Random Forest / XGBoost / SVM per")
print("papers [1],[2],[5]) and severity estimation, followed by SHAP-based")
print("explainability (papers [3],[6],[8]).")

"""X_features.csv: All the input features after preprocessing, feature extraction, ordinal encoding, and one-hot encoding (Input X for all ML models)

y_target: The fraud target — whether the insurance claim was fraudulent or not(For Fraud classification)

y_severity_regression.csv: The claim severity value as a continuous numerical value(For severity regression)

y_severity_class.csv: The claim severity converted into classes/categories(for severity classification)
"""
