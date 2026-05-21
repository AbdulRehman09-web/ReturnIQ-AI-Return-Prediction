"""
explain.py — Explanation Engine for ReturnIQ
════════════════════════════════════════════════════════════════
This module is ONLY responsible for turning prediction outputs
into human-readable explanations. It has zero ML logic.

Why a separate file?
─────────────────────
Keeping explanation logic separate from main.py means:
  • You can unit-test explanations without running FastAPI
  • You can swap explanation style without touching API routing
  • main.py stays clean and focused on request/response flow

How it connects to main.py:
─────────────────────────────
main.py calls explain_prediction() AFTER:
  1. The ML model has produced ml_proba
  2. The rule engine has produced rule_score + triggered_rules
  3. The hybrid logic has produced the final pred (0 or 1)

explain_prediction() uses ALL of those to build an honest,
coherent explanation that matches the actual verdict shown
in the UI — fixing the original disconnected-logic bug.
════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported for type hints — avoids circular import at runtime
    from main import OrderInput


# ══════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════

def _risk_level_label(ml_proba: float | None, rule_score: int) -> str:
    """
    Convert probability + rule score into a readable risk tier.

    WHY combine both?
    The ML model probability reflects patterns learned from data.
    The rule score reflects domain knowledge (strong business rules).
    Using both gives a more honest risk label than either alone.

    Tiers:
        CRITICAL  — rule_score ≥ 4  OR  ml_proba ≥ 0.70
        HIGH      — rule_score ≥ 3  OR  ml_proba ≥ 0.50
        MODERATE  — rule_score ≥ 2  OR  ml_proba ≥ 0.35
        LOW       — everything else
    """
    p = ml_proba or 0.0

    if rule_score >= 4 or p >= 0.70:
        return "CRITICAL"
    if rule_score >= 3 or p >= 0.50:
        return "HIGH"
    if rule_score >= 2 or p >= 0.35:
        return "MODERATE"
    return "LOW"


def _probability_band(ml_proba: float | None) -> str:
    """
    Converts raw probability into a readable confidence band
    shown in the explanation summary.
    """
    if ml_proba is None:
        return "unknown confidence"
    p = ml_proba * 100
    if p >= 70:
        return f"very high model confidence ({p:.0f}%)"
    if p >= 50:
        return f"high model confidence ({p:.0f}%)"
    if p >= 35:
        return f"moderate model confidence ({p:.0f}%)"
    return f"low model confidence ({p:.0f}%)"


# ══════════════════════════════════════════════════════════════
# RISK FACTOR BUILDERS
#
# Each function checks ONE risk dimension and returns either
# a (reason_str, recommendation_str) tuple if the risk fires,
# or (None, None) if it does not.
#
# WHY functions instead of one big if-block?
# ─────────────────────────────────────────────────────────────
# • Each risk is independently testable
# • Easy to add/remove/tune individual checks
# • Avoids 100-line if-elif chains that are hard to read
# ══════════════════════════════════════════════════════════════

def _check_review_score(review_score):
    if review_score is None:
        return None, None
    if review_score <= 2:
        severity = "extremely low" if review_score == 1 else "very low"
        return (
            f"⭐ {severity.capitalize()} review score ({review_score}/5) — "
            f"this is the single strongest predictor of return intent in the model.",
            "Investigate product quality issues immediately. "
            "Trigger a post-purchase support email for this customer within 24 hours."
        )
    return None, None


def _check_freight_ratio(price, freight_value):
    if price <= 0:
        return None, None
    ratio = freight_value / price
    if ratio > 0.50:
        return (
            f"🚚 Extreme freight-to-price ratio ({ratio:.0%}) — "
            f"shipping cost is more than half the product price. "
            f"Customers feel overcharged and are highly likely to return.",
            "Review logistics partner pricing for this product category. "
            "Consider absorbing freight cost into product price or offering free shipping above a threshold."
        )
    if ratio > 0.30:
        return (
            f"🚚 High freight-to-price ratio ({ratio:.0%}) — "
            f"shipping cost significantly reduces perceived value.",
            "Introduce free-shipping thresholds or negotiate better carrier rates "
            "to reduce logistics cost on this route."
        )
    return None, None


def _check_payment_gap(price, payment_value):
    gap = price - payment_value
    if gap > 50:
        return (
            f"💰 Large price-payment gap (${gap:.2f}) — "
            f"product listed at ${price:.2f} but only ${payment_value:.2f} was collected. "
            f"This suggests a problematic discount code or checkout error.",
            "Audit active discount codes and checkout flows. "
            "Verify the order was not placed using an unauthorised coupon."
        )
    if gap > 10:
        return (
            f"💰 Minor price-payment discrepancy (${gap:.2f}) — "
            f"possible promotion or rounding issue. Monitor for pattern.",
            "Check if a promotional campaign is causing unintended discounts on this SKU."
        )
    return None, None


def _check_high_value(payment_value):
    if payment_value > 500:
        return (
            f"💳 Very high transaction value (${payment_value:.2f}) — "
            f"premium orders face higher scrutiny and fraud-related returns.",
            "Enable manual order verification for orders above $500. "
            "Send a personalised order confirmation call or email within 2 hours."
        )
    if payment_value > 200:
        return (
            f"💳 High transaction value (${payment_value:.2f}) — "
            f"elevated return scrutiny is typical above $200.",
            "Flag for post-purchase satisfaction check. "
            "Consider adding a care package or personalised thank-you to reduce buyer's remorse."
        )
    return None, None


def _check_installments(payment_installments):
    if payment_installments is None:
        return None, None
    if payment_installments > 12:
        return (
            f"📋 Very long instalment plan ({payment_installments} instalments) — "
            f"extended payment windows strongly correlate with buyer's remorse and cancellation.",
            "Limit instalment plans beyond 12 months to high-trust customers only. "
            "Send a mid-plan satisfaction check to reduce late-stage cancellations."
        )
    if payment_installments > 6:
        return (
            f"📋 Long instalment plan ({payment_installments} instalments) — "
            f"associated with buyer's remorse on non-essential purchases.",
            "Add a cooling-off period reminder at instalment 2 to confirm customer satisfaction."
        )
    return None, None


def _check_product_weight(product_weight_g):
    if product_weight_g is None:
        return None, None
    if product_weight_g > 10000:
        return (
            f"📦 Very heavy product ({product_weight_g/1000:.1f} kg) — "
            f"high damage-in-transit risk and difficult reverse logistics.",
            "Use double-wall corrugated packaging and add fragile stickers. "
            "Include a damage-on-arrival photo return policy in shipment notes."
        )
    if product_weight_g > 5000:
        return (
            f"📦 Heavy product ({product_weight_g/1000:.1f} kg) — "
            f"above-average shipping damage probability.",
            "Review packaging guidelines for products above 5 kg. "
            "Consider adding insurance to the shipment."
        )
    return None, None


# ══════════════════════════════════════════════════════════════
# POSITIVE SIGNAL BUILDERS
#
# Mirror the risk checks but in the favourable direction.
# Positive signals appear in the UI under "Positive Signals".
# ══════════════════════════════════════════════════════════════

def _positive_signals(data) -> list[str]:
    positives = []
    freight_ratio = data.freight_value / max(data.price, 1)

    if data.review_score and data.review_score >= 4:
        positives.append(
            f"✅ Strong review score ({data.review_score}/5) — "
            f"satisfied customers rarely initiate returns."
        )

    if freight_ratio < 0.10:
        positives.append(
            "✅ Very low freight-to-price ratio — "
            "customer perceives good value; shipping cost is unlikely to cause friction."
        )
    elif freight_ratio < 0.20:
        positives.append(
            "✅ Reasonable freight cost — within acceptable customer tolerance."
        )

    if data.payment_installments and data.payment_installments <= 2:
        positives.append(
            "✅ Short instalment plan — minimal buyer's-remorse exposure."
        )

    if data.payment_value and 50 <= data.payment_value <= 150:
        positives.append(
            f"✅ Mid-range transaction value (${data.payment_value:.2f}) — "
            f"lowest return-rate bracket historically."
        )

    if data.product_weight_g and data.product_weight_g < 1000:
        positives.append(
            "✅ Lightweight product — low shipping damage risk."
        )

    return positives


# ══════════════════════════════════════════════════════════════
# DECISION SOURCE CONTEXT
#
# Tells the UI HOW the verdict was reached so the team
# understands whether to trust ML or the rule engine more.
# ══════════════════════════════════════════════════════════════

def _decision_context(decision_source: str, ml_proba, rule_score: int) -> str:
    p = (ml_proba or 0) * 100
    if decision_source == "rule_override":
        return (
            f"⚙️ Decision made by rule engine (score {rule_score}) — "
            f"multiple strong business signals triggered a return flag "
            f"even though ML model probability was only {p:.0f}%. "
            f"This is expected behaviour when key risk features (installments, "
            f"review score, weight) are absent from the ML feature matrix."
        )
    if decision_source == "ml_model" and p >= 35:
        return (
            f"🤖 Decision made by ML model (probability {p:.0f}%) — "
            f"XGBoost detected a return pattern in the numeric and text features."
        )
    return (
        f"🤖 ML model returned {p:.0f}% probability — below the 35% threshold. "
        f"Rule engine score was {rule_score} — below the override threshold of 3. "
        f"Order classified as low risk."
    )


# ══════════════════════════════════════════════════════════════
# SUMMARY BUILDER
# ══════════════════════════════════════════════════════════════

def _build_summary(
    pred: int,
    loss: float,
    ml_proba,
    rule_score: int,
    risk_level: str,
    n_risks: int,
    decision_source: str,
) -> str:
    p_band = _probability_band(ml_proba)

    if pred == 1:
        estimated = f"${loss:.2f}" if loss > 0 else f"~${0:.2f}"
        return (
            f"⚠️ {risk_level} RETURN RISK — "
            f"Estimated revenue at risk: {estimated}. "
            f"{n_risks} risk factor(s) identified with {p_band}. "
            f"Immediate review recommended."
        )
    else:
        if rule_score >= 2:
            # Model says no return but some rules fired — honest moderate warning
            return (
                f"🟡 LOW–MODERATE RISK — ML model predicts no return ({p_band}), "
                f"but {rule_score} minor risk signal(s) were detected. "
                f"No immediate action required; monitor order progress."
            )
        return (
            f"✅ LOW RETURN RISK — {p_band}. "
            f"Order profile is stable. No action required."
        )


# ══════════════════════════════════════════════════════════════
# MAIN PUBLIC FUNCTION
# ══════════════════════════════════════════════════════════════

def explain_prediction(
    data,
    pred: int,
    loss: float,
    ml_proba: float | None      = None,
    rule_score: int             = 0,
    triggered_rules: list       = None,
    decision_source: str        = "ml_model",
) -> dict:
    """
    Build a complete explanation dict for the /predict response.

    Parameters
    ──────────
    data             : OrderInput — the raw user inputs
    pred             : int        — final prediction (0 or 1) from hybrid logic
    loss             : float      — estimated revenue loss from regressor
    ml_proba         : float|None — raw XGBoost probability (0–1)
    rule_score       : int        — number of risk rules triggered
    triggered_rules  : list[str]  — rule IDs that fired (for debug)
    decision_source  : str        — "ml_model" or "rule_override"

    Returns
    ───────
    dict with keys:
        summary, risk_factors, positive_signals,
        recommendations, decision_context, risk_level
    """
    triggered_rules = triggered_rules or []

    # ── 1. Compute derived values ──────────────────────────────
    freight_ratio = data.freight_value / max(data.price, 1)
    risk_level    = _risk_level_label(ml_proba, rule_score)

    # ── 2. Collect all risk factors + recommendations ──────────
    #
    # Each checker returns (reason | None, recommendation | None).
    # We only append non-None results, keeping lists clean.
    # ──────────────────────────────────────────────────────────
    checks = [
        _check_review_score(data.review_score),
        _check_freight_ratio(data.price, data.freight_value),
        _check_payment_gap(data.price, data.payment_value),
        _check_high_value(data.payment_value),
        _check_installments(data.payment_installments),
        _check_product_weight(data.product_weight_g),
    ]

    risk_factors    = []
    recommendations = []

    for reason, rec in checks:
        if reason:
            risk_factors.append(reason)
        if rec:
            recommendations.append(rec)

    # ── 3. Positive signals ────────────────────────────────────
    positives = _positive_signals(data)

    # ── 4. Decision context (explains HOW verdict was reached) ─
    decision_ctx = _decision_context(decision_source, ml_proba, rule_score)

    # ── 5. Summary headline ────────────────────────────────────
    summary = _build_summary(
        pred           = pred,
        loss           = loss,
        ml_proba       = ml_proba,
        rule_score     = rule_score,
        risk_level     = risk_level,
        n_risks        = len(risk_factors),
        decision_source= decision_source,
    )

    # ── 6. De-duplicate recommendations ───────────────────────
    seen  = set()
    dedup = []
    for r in recommendations:
        if r not in seen:
            seen.add(r)
            dedup.append(r)

    return {
        "summary":          summary,
        "risk_level":       risk_level,
        "risk_factors":     risk_factors,
        "positive_signals": positives,
        "recommendations":  dedup,
        "decision_context": decision_ctx,
    }