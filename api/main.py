import os, sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE      = r"D:\Data Science (Atomcamp)\E-Commerce Product Return Prediction & Revenue Loss Estimation"
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

# ── Import explain ONCE at startup ──────────────────────────────
try:
    from explain import explain_prediction
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
RETURN_THRESHOLD        = 0.35   # ML probability threshold
RULE_OVERRIDE_THRESHOLD = 3      # rule score >= 3 → force return = True

# Maximum rule score possible (used for blending normalisation).
# Count your rules:  low_review=2, freight=1, gap_under=1, gap_over=1,
#                    high_value=1, instalment=1, heavy=1, neg_text=2  → 10
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
reg     = _load(os.path.join(MODEL_DIR, "best_model_RandomForest.pkl"),            "Regressor (RandomForest)")
scaler  = _load(os.path.join(MODEL_DIR, "scaler.pkl"),                             "Scaler")
tfidf   = _load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),                   "TF-IDF Vectorizer")
le_dict = _load(os.path.join(MODEL_DIR, "label_encoders.pkl"),                     "Label Encoders")

# ══════════════════════════════════════════════════════════════
# REQUEST SCHEMA
# ══════════════════════════════════════════════════════════════
class OrderInput(BaseModel):
    payment_value:                 float
    price:                         float
    freight_value:                 float
    review_score:                  Optional[int]   = 3
    payment_installments:          Optional[int]   = 1
    payment_sequential:            Optional[int]   = 1
    product_name_lenght:           Optional[float] = 40.0
    product_description_lenght:    Optional[float] = 200.0
    product_photos_qty:            Optional[float] = 2.0
    product_weight_g:              Optional[float] = 500.0
    product_length_cm:             Optional[float] = 20.0
    product_height_cm:             Optional[float] = 10.0
    product_width_cm:              Optional[float] = 15.0
    payment_type:                  Optional[str]   = "credit_card"
    customer_state:                Optional[str]   = "SP"
    review_comment_message:        Optional[str]   = ""
    order_purchase_timestamp:      Optional[str]   = ""
    order_delivered_customer_date: Optional[str]   = ""


# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# Mirrors training Cell 10 exactly:
#   3 numeric features + TF-IDF(300) = 303 total
# ══════════════════════════════════════════════════════════════
def build_features(data: OrderInput):
    num = np.array([[
        data.payment_value,
        data.price,
        data.freight_value,
    ]], dtype=float)

    text = data.review_comment_message or ""
    text_feat = tfidf.transform([text]) if tfidf is not None \
                else csr_matrix(np.zeros((1, 300)))

    X = hstack([csr_matrix(num), text_feat])   # (1, 303)
    if scaler is not None:
        X = scaler.transform(X)
    return X


# ══════════════════════════════════════════════════════════════
# RULE-BASED RISK SCORER
#
# Covers the fields that are NOT in the ML feature matrix:
#   review_score, installments, weight, payment gap.
# Score >= RULE_OVERRIDE_THRESHOLD (3) → force return = True
# MAX possible score = 10 (used for probability blending below)
# ══════════════════════════════════════════════════════════════
def compute_rule_score(data: OrderInput):
    """Returns (score: int, triggered_rules: list[str])"""
    score     = 0
    triggered = []

    safe_price    = max(data.price, 0.01)
    freight_ratio = data.freight_value / safe_price
    payment_gap   = data.price - data.payment_value   # + = underpaid, - = overpaid

    # Rule 1 — Very low review score (worth 2: dominant predictor)
    if data.review_score is not None and data.review_score <= 2:
        score += 2
        triggered.append("low_review_score")

    # Rule 2 — High freight-to-price ratio
    if freight_ratio > 0.30:
        score += 1
        triggered.append("high_freight_ratio")

    # Rule 3a — Customer underpaid (discount / coupon anomaly)
    if payment_gap > 10:
        score += 1
        triggered.append("payment_gap_underpaid")

    # Rule 3b — Customer overpaid (checkout error / fraud)
    if payment_gap < -10:
        score += 1
        triggered.append("payment_gap_overpaid")

    # Rule 4 — High-value transaction
    if data.payment_value > 200:
        score += 1
        triggered.append("high_value_order")

    # Rule 5 — Long instalment plan
    if data.payment_installments is not None and data.payment_installments > 6:
        score += 1
        triggered.append("long_instalment_plan")

    # Rule 6 — Heavy product (transit damage risk)
    if data.product_weight_g is not None and data.product_weight_g > 5000:
        score += 1
        triggered.append("heavy_product")

    # Rule 7 — Negative keywords in review text
    HIGH_RISK_KEYWORDS = [
        "damaged", "broken", "refund", "terrible", "awful", "worst",
        "defective", "return", "fraud", "scam", "late", "wrong",
        "disappointed", "horrible", "never", "cheat", "fake",
        "missing", "poor", "useless", "angry", "complaint",
    ]
    review_lower = (data.review_comment_message or "").lower()
    hits = sum(1 for kw in HIGH_RISK_KEYWORDS if kw in review_lower)
    if hits >= 3:
        score += 2
        triggered.append(f"negative_review_text_{hits}_keywords")
    elif hits >= 1:
        score += 1
        triggered.append(f"negative_review_text_{hits}_keyword")

    return score, triggered


# ══════════════════════════════════════════════════════════════
# BLENDED PROBABILITY
#
# WHY blend instead of showing raw ml_proba only?
# ─────────────────────────────────────────────────────────────
# The ML model was trained on only 3 numeric features + TF-IDF
# text (303 features total).  Fields like review_score,
# payment_installments, product_weight_g, and customer_state
# are collected in the sidebar but were NOT in the training
# feature matrix — so changing them has zero effect on ml_proba,
# making the displayed probability look random/unchanging.
#
# The rule engine covers exactly those missing fields, so we
# blend both signals into one honest probability:
#
#   blended = 0.55 × ml_proba + 0.45 × (rule_score / MAX_RULE_SCORE)
#
# Weights chosen so ML still leads (it saw 100k rows of data)
# but the rule signal is strong enough to be visible when
# review_score, installments, or weight change meaningfully.
#
# The blended value is used ONLY for the probability display
# and the probability bar.  The binary return/no-return decision
# is still made by the original hybrid logic (rule override OR
# ml_proba >= threshold) — blending does not change verdicts.
# ══════════════════════════════════════════════════════════════
def compute_blended_probability(ml_proba: float | None, rule_score: int) -> float:
    """
    Returns a 0–1 blended probability that reflects both the ML
    model output and the rule engine score.
    """
    rule_signal = min(rule_score / MAX_RULE_SCORE, 1.0)

    if ml_proba is not None:
        blended = 0.55 * ml_proba + 0.45 * rule_signal
    else:
        # No ML model — rely entirely on the rule signal
        blended = rule_signal

    # Clamp to [0, 1] as a safety measure
    return float(min(max(blended, 0.0), 1.0))


# ══════════════════════════════════════════════════════════════
# PREDICTION ENDPOINT
# ══════════════════════════════════════════════════════════════
@app.post("/predict")
def predict(data: OrderInput):

    if clf is None:
        raise HTTPException(503, "Classifier not loaded. Run training notebook first.")
    if reg is None:
        raise HTTPException(503, "Regressor not loaded. Run training notebook first.")
    if explain_prediction is None:
        raise HTTPException(503, "explain.py failed to load. Check utils/ folder and sys.path.")

    try:
        # ── 1. Feature matrix ─────────────────────────────────
        X = build_features(data)

        # ── 2. Raw ML probability (from 303-feature XGBoost) ──
        ml_proba = float(clf.predict_proba(X)[0][1]) \
                   if hasattr(clf, "predict_proba") else None

        # ── 3. Rule score (covers fields missing from ML) ─────
        rule_score, triggered_rules = compute_rule_score(data)

        # ── 4. Blended probability (what we display in the UI) ─
        #
        # This is the key fix: the blended value incorporates
        # review_score, installments, weight, etc. via the rule
        # signal, so the probability bar responds to ALL sidebar
        # inputs — not just payment_value, price, freight_value.
        blended_proba = compute_blended_probability(ml_proba, rule_score)

        # ── 5. Hybrid binary decision ──────────────────────────
        # Decision logic is unchanged from v3.2 — we do NOT use
        # blended_proba for the verdict, only for display.
        if rule_score >= RULE_OVERRIDE_THRESHOLD:
            pred            = 1
            decision_source = "rule_override"
        elif ml_proba is not None and ml_proba >= RETURN_THRESHOLD:
            pred            = 1
            decision_source = "ml_model"
        else:
            pred            = 0
            decision_source = "ml_model"

        # ── 6. Revenue loss estimate ───────────────────────────
        loss = 0.0
        if pred == 1:
            raw_loss = float(reg.predict(X)[0])
            loss     = max(raw_loss, 0.0)

        # ── 7. Explanation ─────────────────────────────────────
        explanation = explain_prediction(
            data            = data,
            pred            = pred,
            loss            = loss,
            ml_proba        = ml_proba,
            rule_score      = rule_score,
            triggered_rules = triggered_rules,
            decision_source = decision_source,
        )

        # ── 8. Return full response ────────────────────────────
        # return_probability now uses blended_proba so the UI
        # probability bar and metric tile reflect all 12 inputs.
        return {
            "return":             bool(pred),
            "return_probability": round(blended_proba * 100, 1),
            "loss":               round(loss, 2),
            "explanation":        explanation,
            "debug": {
                "ml_probability":    round(ml_proba * 100, 1) if ml_proba is not None else None,
                "rule_score":        rule_score,
                "triggered_rules":   triggered_rules,
                "decision_source":   decision_source,
                "threshold_used":    RETURN_THRESHOLD,
                "blended_proba":     round(blended_proba * 100, 1),
                "blend_weights":     "55% ML + 45% rule signal",
            },
            "input_echo": {
                "payment_value":        data.payment_value,
                "price":                data.price,
                "freight_value":        data.freight_value,
                "review_score":         data.review_score,
                "payment_type":         data.payment_type,
                "payment_installments": data.payment_installments,
                "product_weight_g":     data.product_weight_g,
                "customer_state":       data.customer_state,
            }
        }

    except HTTPException:
        raise   # don't wrap 503s

    except Exception as e:
        import traceback
        traceback.print_exc()   # full stack in uvicorn terminal for debugging
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ══════════════════════════════════════════════════════════════
# DIAGNOSTICS ENDPOINT
# Visit http://127.0.0.1:8000/debug/imports to verify all
# modules and models loaded correctly.
# ══════════════════════════════════════════════════════════════
@app.get("/debug/imports")
def debug_imports():
    return {
        "explain_loaded":     explain_prediction is not None,
        "classifier_loaded":  clf     is not None,
        "regressor_loaded":   reg     is not None,
        "scaler_loaded":      scaler  is not None,
        "tfidf_loaded":       tfidf   is not None,
        "utils_dir":          UTILS_DIR,
        "utils_on_sys_path":  UTILS_DIR in sys.path,
        "sys_path":           sys.path[:6],
        "blend_config": {
            "ml_weight":       0.55,
            "rule_weight":     0.45,
            "max_rule_score":  MAX_RULE_SCORE,
        },
    }


# ══════════════════════════════════════════════════════════════
# HEALTH & ROOT
# ══════════════════════════════════════════════════════════════
@app.get("/")
def home():
    return {
        "message":                 "ReturnIQ API running 🚀",
        "version":                 "3.3",
        "threshold":               RETURN_THRESHOLD,
        "rule_override_threshold": RULE_OVERRIDE_THRESHOLD,
        "probability_mode":        "blended (55% ML + 45% rule signal)",
    }

@app.get("/health")
def health():
    return {
        "classifier_loaded":  clf               is not None,
        "regressor_loaded":   reg               is not None,
        "scaler_loaded":      scaler            is not None,
        "tfidf_loaded":       tfidf             is not None,
        "label_enc_loaded":   le_dict           is not None,
        "explain_loaded":     explain_prediction is not None,
        "return_threshold":   RETURN_THRESHOLD,
        "rule_override_at":   RULE_OVERRIDE_THRESHOLD,
        "max_rule_score":     MAX_RULE_SCORE,
        "blend_weights":      "55% ML + 45% rule signal",
    }