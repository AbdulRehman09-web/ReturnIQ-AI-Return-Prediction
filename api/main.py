import os, sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ══════════════════════════════════════════════════════════════
# FIX: Render-compatible BASE path (DO NOT use D:\ path)
# ══════════════════════════════════════════════════════════════
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UTILS_DIR = os.path.join(BASE, "utils")
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pickle
from scipy.sparse import hstack, csr_matrix

# ══════════════════════════════════════════════════════════════
# FIX: correct module import from utils folder
# ══════════════════════════════════════════════════════════════
try:
    from utils.explain import explain_prediction
    print("[OK]   Loaded explain.py from utils/")
except ImportError as e:
    print(f"[ERROR] Could not import explain.py: {e}")
    explain_prediction = None


# ══════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════
app = FastAPI(
    title="ReturnIQ — E-Commerce Return Prediction API",
    description="AI-Powered Return Risk & Revenue Loss Estimation",
    version="3.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(BASE, "models")

# ══════════════════════════════════════════════════════════════
# THRESHOLDS
# ══════════════════════════════════════════════════════════════
RETURN_THRESHOLD        = 0.35
RULE_OVERRIDE_THRESHOLD = 3

MAX_RULE_SCORE = 10


# ══════════════════════════════════════════════════════════════
# LOAD SAVED ARTEFACTS
# ══════════════════════════════════════════════════════════════
def _load(path: str, label: str):
    if not os.path.exists(path):
        print(f"[WARN] {label} not found → {path}")
        return None
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[OK]   Loaded {label}")
    return obj

clf     = _load(os.path.join(MODEL_DIR, "best_classification_model_XGBoost.pkl"), "Classifier (XGBoost)")
reg     = _load(os.path.join(MODEL_DIR, "best_model_RandomForest.pkl"), "Regressor (RandomForest)")
scaler  = _load(os.path.join(MODEL_DIR, "scaler.pkl"), "Scaler")
tfidf   = _load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "TF-IDF Vectorizer")
le_dict = _load(os.path.join(MODEL_DIR, "label_encoders.pkl"), "Label Encoders")


# ══════════════════════════════════════════════════════════════
# REQUEST SCHEMA
# ══════════════════════════════════════════════════════════════
class OrderInput(BaseModel):
    payment_value:                 float
    price:                         float
    freight_value:                 float
    review_score:                  Optional[int] = 3
    payment_installments:          Optional[int] = 1
    payment_sequential:            Optional[int] = 1
    product_name_lenght:           Optional[float] = 40.0
    product_description_lenght:    Optional[float] = 200.0
    product_photos_qty:            Optional[float] = 2.0
    product_weight_g:              Optional[float] = 500.0
    product_length_cm:             Optional[float] = 20.0
    product_height_cm:             Optional[float] = 10.0
    product_width_cm:              Optional[float] = 15.0
    payment_type:                  Optional[str] = "credit_card"
    customer_state:                Optional[str] = "SP"
    review_comment_message:        Optional[str] = ""
    order_purchase_timestamp:      Optional[str] = ""
    order_delivered_customer_date: Optional[str] = ""


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def build_features(data: OrderInput):
    num = np.array([[
        data.payment_value,
        data.price,
        data.freight_value,
    ]], dtype=float)

    text = data.review_comment_message or ""
    text_feat = tfidf.transform([text]) if tfidf is not None else csr_matrix((1, 300))

    X = hstack([csr_matrix(num), text_feat])

    if scaler is not None:
        X = scaler.transform(X)

    return X


# ══════════════════════════════════════════════════════════════
# RULE-BASED RISK SCORER
# ══════════════════════════════════════════════════════════════
def compute_rule_score(data: OrderInput):
    score = 0
    triggered = []

    safe_price    = max(data.price, 0.01)
    freight_ratio = data.freight_value / safe_price
    payment_gap   = data.price - data.payment_value

    if data.review_score is not None and data.review_score <= 2:
        score += 2
        triggered.append("low_review_score")

    if freight_ratio > 0.30:
        score += 1
        triggered.append("high_freight_ratio")

    if payment_gap > 10:
        score += 1
        triggered.append("payment_gap_underpaid")

    if payment_gap < -10:
        score += 1
        triggered.append("payment_gap_overpaid")

    if data.payment_installments is not None and data.payment_installments > 6:
        score += 1
        triggered.append("long_instalment_plan")

    if data.product_weight_g is not None and data.product_weight_g > 5000:
        score += 1
        triggered.append("heavy_product")

    return score, triggered


# ══════════════════════════════════════════════════════════════
# BLENDED PROBABILITY
# ══════════════════════════════════════════════════════════════
def compute_blended_probability(ml_proba: float | None, rule_score: int) -> float:
    rule_signal = min(rule_score / MAX_RULE_SCORE, 1.0)

    if ml_proba is not None:
        blended = 0.55 * ml_proba + 0.45 * rule_signal
    else:
        blended = rule_signal

    return float(min(max(blended, 0.0), 1.0))


# ══════════════════════════════════════════════════════════════
# PREDICT ENDPOINT
# ══════════════════════════════════════════════════════════════
@app.post("/predict")
def predict(data: OrderInput):

    if clf is None:
        raise HTTPException(503, "Classifier not loaded")
    if reg is None:
        raise HTTPException(503, "Regressor not loaded")
    if explain_prediction is None:
        raise HTTPException(503, "explain.py not loaded")

    X = build_features(data)

    ml_proba = float(clf.predict_proba(X)[0][1]) if hasattr(clf, "predict_proba") else None

    rule_score, triggered_rules = compute_rule_score(data)

    blended_proba = compute_blended_probability(ml_proba, rule_score)

    if rule_score >= RULE_OVERRIDE_THRESHOLD:
        pred = 1
        decision_source = "rule_override"
    elif ml_proba is not None and ml_proba >= RETURN_THRESHOLD:
        pred = 1
        decision_source = "ml_model"
    else:
        pred = 0
        decision_source = "ml_model"

    loss = float(reg.predict(X)[0]) if pred == 1 else 0.0

    explanation = explain_prediction(
        data=data,
        pred=pred,
        loss=loss,
        ml_proba=ml_proba,
        rule_score=rule_score,
        triggered_rules=triggered_rules,
        decision_source=decision_source,
    )

    return {
        "return": bool(pred),
        "return_probability": round(blended_proba * 100, 1),
        "loss": round(loss, 2),
        "explanation": explanation,
        "debug": {
            "ml_probability": ml_proba,
            "rule_score": rule_score,
            "triggered_rules": triggered_rules,
            "decision_source": decision_source,
        }
    }


# ══════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════
@app.get("/")
def home():
    return {"status": "API running"}

@app.get("/health")
def health():
    return {
        "model": clf is not None,
        "regressor": reg is not None,
        "tfidf": tfidf is not None,
        "scaler": scaler is not None,
        "explain": explain_prediction is not None
    }