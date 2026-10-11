# Meridian Claim Assessment API

Backend for `claim_review.html` / `review_log.html`. Loads the trained fraud
model, severity regressor, and severity tier classifier, and serves one
endpoint the frontend calls directly.

## Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `claim_review.html` in a browser (as a local file is fine — CORS is
wide open by default for local testing). It calls `http://localhost:8000` by
default; change `API_BASE_URL` near the top of the `<script>` block in
`claim_review.html` if you host the API elsewhere.

## Endpoints

- `GET /health` — confirms the three models loaded and reports which model
  type won each (XGBoost / Random Forest / XGBoost, per your training runs).
- `POST /api/assess-claim` — body is one claim object with the same fields as
  the `CLAIMS` entries in `claim_review.html` (raw dataset fields: policy and
  policyholder details, incident details, and injury/property/vehicle claim
  amounts). Returns:

  ```json
  {
    "verdict": "flagged" | "clear",
    "verdictLabel": "likely fraudulent" | "likely genuine",
    "probability": 0-100,
    "factors": [{ "label": "...", "weight": 0.42 }, ...],
    "severityAmount": 63411.88,
    "severityTier": "Low" | "Medium" | "High" | "Severe",
    "priority": "Urgent" | "High" | "Standard" | "Low",
    "aiSummary": "..."
  }
  ```

## Notes

- **sklearn version**: `models/*_preprocessor.joblib` were pickled with an
  older scikit-learn whose `ColumnTransformer` internals changed slightly in
  newer versions. `main.py` works around this (see the comment block at the
  top and `apply_scaler()`) by reading the fitted `StandardScaler`'s own
  `mean_`/`scale_` directly rather than calling the unpickled
  `ColumnTransformer.transform()`. This is more robust across sklearn
  versions than relying on the pickled object's transform logic, and verified
  end-to-end to produce correct results — but if you retrain and re-save
  fresh preprocessors in a matching sklearn version, you could simplify this
  back to a plain `.transform()` call if you prefer.
- **Priority thresholds** (`fraud_label()` in `main.py`, cut at 30%/60%) are a
  starting point, not derived from the data — recalibrate against real claim
  volumes once deployed.
- **AI summary**: deterministic by default. Set the `ANTHROPIC_API_KEY`
  environment variable to have Claude rephrase the same computed facts into
  more natural language (it's only ever asked to reword inputs already
  computed, never to add new claims — if the API call fails for any reason,
  it falls back to the deterministic template so the endpoint never errors
  out because of the LLM step).
- **CORS** is wide open (`allow_origins=["*"]`) for local development.
  Restrict this in `main.py` before any real deployment.
- Severity tier accuracy is modest (~40%, vs. 25% random baseline for 4
  balanced classes) — see `models/severity_model_metadata.json`. The dollar
  amount from the regressor (R² ≈ 0.69) is the more reliable severity signal.
- An unseen category in a categorical field (e.g. a `insured_hobbies` value
  never seen during training) is handled gracefully: it just doesn't light up
  any one-hot column, which the model treats like the implicit baseline
  category rather than erroring out.
