"""
Run from the backend folder:   python diagnose_models.py
Then paste the ENTIRE output back. It answers, in order:
  1. Which Python / library versions is this machine really using?
  2. Does XGBoost's save/load work at all here? (tiny self-made model)
  3. Are the model files byte-identical to the originals? (size + hash)
  4. Does each way of loading the real models work? (joblib / raw pickle / native .json)
"""
import hashlib
import os
import pickle
import platform
import sys
import tempfile
import traceback
import warnings

warnings.filterwarnings("ignore")


def hr(title):
    print("\n" + "=" * 8, title, "=" * 8)


def attempt(label, fn):
    try:
        out = fn()
        print(f"  OK    {label}" + (f"  -> {out}" if out else ""))
        return True
    except Exception as e:
        last = traceback.format_exc().strip().splitlines()[-1]
        print(f"  FAIL  {label}\n        {type(e).__name__}: {str(e)[:160]}")
        return False


hr("1. ENVIRONMENT")
print("python      :", sys.version.split()[0], "|", platform.platform())
print("executable  :", sys.executable)
import numpy, joblib, sklearn, xgboost  # noqa: E402

print("numpy       :", numpy.__version__)
print("sklearn     :", sklearn.__version__)
print("joblib      :", joblib.__version__)
print("xgboost     :", xgboost.__version__)
print("xgboost file:", xgboost.__file__)

hr("2. DOES XGBOOST SERIALIZATION WORK AT ALL ON THIS MACHINE?")
from xgboost import XGBClassifier  # noqa: E402

rng = numpy.random.default_rng(0)
Xs = rng.normal(size=(200, 5))
ys = (Xs[:, 0] + rng.normal(size=200) > 0).astype(int)
tiny = XGBClassifier(n_estimators=10, max_depth=2).fit(Xs, ys)
attempt("tiny model: pickle round-trip in memory", lambda: pickle.loads(pickle.dumps(tiny)) and None)
tmpdir = tempfile.mkdtemp()
tiny_json = os.path.join(tmpdir, "tiny.json")
tiny.save_model(tiny_json)
attempt("tiny model: native .json save + load", lambda: XGBClassifier().load_model(tiny_json))
tiny_pkl = os.path.join(tmpdir, "tiny.joblib")
joblib.dump(tiny, tiny_pkl)
attempt("tiny model: joblib dump + load via a file", lambda: joblib.load(tiny_pkl) and None)

hr("3. MODEL FILES (expected sizes/hashes from the original zip)")
EXPECT = {
    "best_fraud_model.joblib": (189916, "16b5262a1266a4f0"),
    "severity_tier_classifier.joblib": (717139, "e225bbe863ea4793"),
    "severity_regressor.joblib": (465489, "0d9e99343cb2d3bd"),
    "fraud_preprocessor.joblib": (9499, "89170788db03c1aa"),
}
here = os.path.dirname(os.path.abspath(__file__))
candidates = [os.environ.get("MODELS_DIR"), os.path.join(here, "models"),
              os.path.join(here, "..", "ML", "data", "models")]
models_dir = next((os.path.abspath(c) for c in candidates
                   if c and os.path.isfile(os.path.join(c, "best_fraud_model.joblib"))), None)
print("models dir used:", models_dir)
for c in candidates:
    if c:
        print("  candidate:", os.path.abspath(c), "| exists:", os.path.isdir(c))
if not models_dir:
    sys.exit("No models folder found - stop here and tell me.")

for name, (size, h) in EXPECT.items():
    p = os.path.join(models_dir, name)
    if not os.path.isfile(p):
        print(f"  MISSING {name}")
        continue
    b = open(p, "rb").read()
    ok = len(b) == size and hashlib.sha256(b).hexdigest().startswith(h)
    print(f"  {'MATCH ' if ok else 'DIFFERS'} {name}: {len(b)} bytes (expected {size}), sha256 {hashlib.sha256(b).hexdigest()[:16]} (expected {h})")

hr("4. LOADING THE REAL MODELS")
fraud_pkl = os.path.join(models_dir, "best_fraud_model.joblib")
fraud_json = os.path.join(models_dir, "best_fraud_model.json")
attempt("fraud model via joblib", lambda: type(joblib.load(fraud_pkl)).__name__)


def raw_pickle():
    with open(fraud_pkl, "rb") as f:
        head = f.read(2)
    return f"first bytes: {head!r}"


attempt("fraud model file header", raw_pickle)
if os.path.isfile(fraud_json):
    attempt("fraud model via native .json", lambda: XGBClassifier().load_model(fraud_json))
else:
    print("  (no best_fraud_model.json in the models folder)")
tier_pkl = os.path.join(models_dir, "severity_tier_classifier.joblib")
tier_json = os.path.join(models_dir, "severity_tier_classifier.json")
attempt("tier classifier via joblib", lambda: type(joblib.load(tier_pkl)).__name__)
if os.path.isfile(tier_json):
    attempt("tier classifier via native .json", lambda: XGBClassifier().load_model(tier_json))
attempt("severity regressor via joblib", lambda: type(joblib.load(os.path.join(models_dir, "severity_regressor.joblib"))).__name__)

print("\nDone. Paste everything above.")
