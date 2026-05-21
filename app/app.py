import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import requests

# Backend API URL
API_URL = "https://returniq-ai-return-prediction-backend.onrender.com/predict"

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ReturnIQ — AI Return Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE       = r"D:\Data Science (Atomcamp)\E-Commerce Product Return Prediction & Revenue Loss Estimation"
RESULT_DIR = os.path.join(BASE, "notebooks", "results")
GRAPH_DIR  = os.path.join(BASE, "notebooks", "graphs")
API_URL    = "http://127.0.0.1:8000"
RETURN_THRESHOLD = 0.35

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500&display=swap');
*, html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; box-sizing: border-box; }
.stApp { background: #080c14; color: #e8e2d6; }
section[data-testid="stSidebar"] { background: #060912 !important; border-right: 1px solid rgba(212,175,55,0.15) !important; }
section[data-testid="stSidebar"] * { color: #c9c3b5 !important; }
section[data-testid="stSidebar"] .stSlider > div > div > div { background: #d4af37 !important; }
.hero { position:relative; padding:52px 44px 44px; border-radius:20px; background:linear-gradient(135deg,#0d1525 0%,#111c30 60%,#0a1020 100%); border:1px solid rgba(212,175,55,0.22); overflow:hidden; margin-bottom:28px; }
.hero::before { content:''; position:absolute; top:-80px; right:-80px; width:340px; height:340px; background:radial-gradient(circle,rgba(212,175,55,0.10) 0%,transparent 70%); pointer-events:none; }
.hero-eyebrow { font-size:11px; font-weight:500; letter-spacing:4px; text-transform:uppercase; color:#d4af37; margin-bottom:12px; }
.hero-title { font-family:'Cormorant Garamond',serif; font-size:58px; font-weight:600; line-height:1.08; color:#f5f0e8; margin-bottom:14px; }
.hero-title span { color:#d4af37; }
.hero-sub { font-size:15px; font-weight:300; color:#8b9ab5; max-width:540px; line-height:1.7; }
.glass { background:rgba(255,255,255,0.035); border:1px solid rgba(212,175,55,0.12); border-radius:18px; padding:28px 30px; margin-bottom:22px; }
.metric-row { display:flex; gap:18px; margin-bottom:22px; flex-wrap:wrap; }
.metric-tile { flex:1; min-width:160px; background:linear-gradient(145deg,#0f1828,#131f34); border:1px solid rgba(212,175,55,0.18); border-radius:16px; padding:26px 22px; text-align:center; }
.metric-label { font-size:10px; font-weight:500; letter-spacing:3px; text-transform:uppercase; color:#5a6a88; margin-bottom:10px; }
.metric-value { font-family:'Cormorant Garamond',serif; font-size:42px; font-weight:600; line-height:1; }
.metric-value.gold  { color:#d4af37; }
.metric-value.red   { color:#e05252; }
.metric-value.green { color:#4caf82; }
.metric-value.blue  { color:#5b9bd5; }
.verdict { border-radius:18px; padding:30px 34px; margin-bottom:22px; }
.verdict.danger { background:linear-gradient(135deg,#1a0a0a,#200d0d); border:1px solid rgba(224,82,82,0.35); }
.verdict.safe   { background:linear-gradient(135deg,#081a10,#0b2014); border:1px solid rgba(76,175,130,0.35); }
.verdict-icon { font-size:40px; margin-bottom:10px; }
.verdict-title { font-family:'Cormorant Garamond',serif; font-size:34px; font-weight:600; margin-bottom:8px; }
.verdict.danger .verdict-title { color:#e87878; }
.verdict.safe   .verdict-title { color:#72c9a0; }
.verdict-sub { font-size:14px; color:#7a8a9a; font-weight:300; }
.factor-card   { background:rgba(255,255,255,0.025); border-left:3px solid #d4af37; border-radius:0 12px 12px 0; padding:14px 18px; margin-bottom:10px; font-size:14px; color:#c8c0b0; line-height:1.6; }
.positive-card { background:rgba(76,175,130,0.06); border-left:3px solid #4caf82; border-radius:0 12px 12px 0; padding:14px 18px; margin-bottom:10px; font-size:14px; color:#a0c8b5; line-height:1.6; }
.rec-card      { background:rgba(91,155,213,0.06); border-left:3px solid #5b9bd5; border-radius:0 12px 12px 0; padding:14px 18px; margin-bottom:10px; font-size:14px; color:#a0b8d0; line-height:1.6; }
.warn-card     { background:rgba(224,82,82,0.06); border-left:3px solid #e05252; border-radius:0 12px 12px 0; padding:14px 18px; margin-bottom:10px; font-size:13px; color:#c08080; line-height:1.6; }
.debug-card    { background:rgba(255,255,255,0.02); border:1px solid rgba(212,175,55,0.10); border-radius:12px; padding:16px 20px; font-size:13px; color:#5a6a88; font-family:monospace; }
.sec-head { font-family:'Cormorant Garamond',serif; font-size:26px; font-weight:500; color:#d4af37; margin:32px 0 16px; padding-bottom:10px; border-bottom:1px solid rgba(212,175,55,0.15); }
hr { border:none; border-top:1px solid rgba(212,175,55,0.10) !important; margin:28px 0 !important; }
.stButton > button { width:100% !important; height:54px !important; border-radius:14px !important; border:1px solid rgba(212,175,55,0.5) !important; background:linear-gradient(90deg,#1a1500 0%,#2a2000 100%) !important; color:#d4af37 !important; font-size:14px !important; font-weight:500 !important; letter-spacing:2px !important; text-transform:uppercase !important; transition:all 0.25s ease !important; }
.stButton > button:hover { background:linear-gradient(90deg,#d4af37 0%,#f0cc60 100%) !important; color:#080c14 !important; box-shadow:0 0 28px rgba(212,175,55,0.35) !important; }
.stNumberInput input, .stTextInput input, .stSelectbox > div > div { background:#0f1828 !important; border:1px solid rgba(212,175,55,0.18) !important; border-radius:10px !important; color:#e8e2d6 !important; }
.stNumberInput label, .stTextInput label, .stSelectbox label, .stSlider label, .stTextArea label { color:#8b9ab5 !important; font-size:12px !important; }
.stTabs [data-baseweb="tab-list"] { background:transparent !important; gap:8px; }
.stTabs [data-baseweb="tab"] { border-radius:10px !important; background:rgba(255,255,255,0.03) !important; color:#7a8a9a !important; border:1px solid rgba(212,175,55,0.10) !important; padding:8px 20px !important; font-size:13px !important; }
.stTabs [aria-selected="true"] { background:rgba(212,175,55,0.12) !important; color:#d4af37 !important; border-color:rgba(212,175,55,0.35) !important; }
.stSpinner > div { border-top-color:#d4af37 !important; }
.prob-bar-wrap { margin:18px 0; }
.prob-bar-label { font-size:11px; letter-spacing:2px; color:#5a6a88; margin-bottom:6px; text-transform:uppercase; }
.prob-bar-bg { height:8px; background:rgba(255,255,255,0.06); border-radius:4px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:4px; }
.footer { text-align:center; padding:40px 0 20px; color:#3a4a60; font-size:12px; letter-spacing:1px; }
.footer strong { color:#5a6a80; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">◈ AI Intelligence Platform</div>
    <div class="hero-title">Return<span>IQ</span></div>
    <div class="hero-sub">
        Enterprise-grade machine learning system for predicting e-commerce return risk
        and estimating revenue loss — powered by XGBoost, Random Forest, LSTM & CNN.
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HELPER — show image or a clear "missing" message
# ══════════════════════════════════════════════════════════════
def show_result_image(filename: str, caption: str):
    """Try RESULT_DIR first, then GRAPH_DIR. Show path if missing."""
    for directory in [RESULT_DIR, GRAPH_DIR]:
        full = os.path.join(directory, filename)
        if os.path.exists(full):
            st.image(full, caption=caption, use_container_width=True)
            return
    # File not found — show a helpful message with the expected path
    st.markdown(
        f'<div class="warn-card">'
        f'📂 <b>{filename}</b> not found.<br>'
        f'Expected at: <code>{os.path.join(RESULT_DIR, filename)}</code><br>'
        f'Run the training notebook (Cell 19–24) to generate it.'
        f'</div>',
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════
# SIDEBAR — INPUT FORM
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ◈ Order Intelligence")
    st.markdown("---")
    st.markdown("**Core Financials**")
    payment_value = st.number_input("Payment Value ($)", min_value=0.0, value=150.0, step=5.0)
    price         = st.number_input("Product Price ($)",  min_value=0.0, value=120.0, step=5.0)
    freight_value = st.number_input("Freight Cost ($)",   min_value=0.0, value=18.0,  step=1.0)

    st.markdown("---")
    st.markdown("**Customer & Product**")
    review_score         = st.selectbox("Review Score", options=[1,2,3,4,5], index=2)
    payment_type         = st.selectbox("Payment Method", ["credit_card","boleto","voucher","debit_card"])
    payment_installments = st.slider("Installments", 1, 24, 1)
    customer_state       = st.selectbox("Customer State",
        ["SP","RJ","MG","RS","BA","PR","SC","GO","PE","CE",
         "MA","PA","MT","MS","PB","RN","PI","AL","SE","TO",
         "RO","AM","ES","AC","AP","RR","DF"])

    st.markdown("---")
    st.markdown("**Product Specs**")
    product_weight = st.number_input("Weight (g)",  min_value=0.0, value=500.0, step=50.0)
    product_length = st.number_input("Length (cm)", min_value=0.0, value=20.0,  step=1.0)
    product_height = st.number_input("Height (cm)", min_value=0.0, value=10.0,  step=1.0)
    product_width  = st.number_input("Width (cm)",  min_value=0.0, value=15.0,  step=1.0)

    st.markdown("---")
    st.markdown("**Review Text**")
    review_text = st.text_area(
        "Customer Review Message", value="",
        placeholder="Paste customer review for NLP analysis…",
        height=90
    )
    st.markdown("---")
    analyze_btn = st.button("◈  ANALYZE ORDER")

# ══════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Model Performance", "🧠 Explainable AI"])

# ──────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION
# ──────────────────────────────────────────────────────────────
with tab1:
    if not analyze_btn:
        st.markdown("""
        <div class="glass" style="text-align:center; padding:60px 30px;">
            <div style="font-size:52px; margin-bottom:20px; opacity:0.3;">◈</div>
            <div style="font-family:'Cormorant Garamond',serif; font-size:28px; color:#3a4a60; font-weight:300;">
                Configure order parameters in the sidebar,<br>
                then press <strong style="color:#5a6a80;">Analyze Order</strong> to run AI inference.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner("Running AI inference pipeline…"):
            try:
                payload = {
                    "payment_value":          payment_value,
                    "price":                  price,
                    "freight_value":          freight_value,
                    "review_score":           review_score,
                    "payment_type":           payment_type,
                    "payment_installments":   payment_installments,
                    "customer_state":         customer_state,
                    "product_weight_g":       product_weight,
                    "product_length_cm":      product_length,
                    "product_height_cm":      product_height,
                    "product_width_cm":       product_width,
                    "review_comment_message": review_text or "",
                }

                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=15
                )

                # ── Always check HTTP status BEFORE parsing JSON ──
                if response.status_code != 200:
                    st.error(f"❌ API returned HTTP {response.status_code}")
                    with st.expander("Show API error detail"):
                        st.code(response.text[:2000])
                    st.stop()

                result = response.json()

                # ── Safe key extraction ───────────────────────────
                is_return = result.get("return", False)
                loss      = result.get("loss", 0.0) or 0.0
                prob      = result.get("return_probability")        # may be None
                expl      = result.get("explanation", {})
                summary   = expl.get("summary", "—")
                risks     = expl.get("risk_factors", [])
                positives = expl.get("positive_signals", [])
                recs      = expl.get("recommendations", [])
                dbg       = result.get("debug", {})

                # ── Verdict banner ───────────────────────────────
                if is_return:
                    st.markdown(f"""
                    <div class="verdict danger">
                        <div class="verdict-icon">⚠️</div>
                        <div class="verdict-title">High Return Risk Detected</div>
                        <div class="verdict-sub">{summary}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict safe">
                        <div class="verdict-icon">✅</div>
                        <div class="verdict-title">Low Return Risk</div>
                        <div class="verdict-sub">{summary}</div>
                    </div>""", unsafe_allow_html=True)

                # ── Metric tiles ─────────────────────────────────
                prob_val     = prob or 0
                prob_display = f"{prob}%" if prob is not None else "N/A"
                loss_display = f"${loss:,.2f}" if is_return else "$0.00"
                risk_count   = len(risks)
                rec_count    = len(recs)
                prob_color   = "red" if prob_val > 60 else ("gold" if prob_val > 35 else "green")
                loss_color   = "red" if loss > 100 else ("gold" if loss > 0 else "green")

                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-tile">
                        <div class="metric-label">Return Probability</div>
                        <div class="metric-value {prob_color}">{prob_display}</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-label">Revenue at Risk</div>
                        <div class="metric-value {loss_color}">{loss_display}</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-label">Risk Factors</div>
                        <div class="metric-value {'red' if risk_count > 2 else 'gold'}">{risk_count}</div>
                    </div>
                    <div class="metric-tile">
                        <div class="metric-label">Actions Suggested</div>
                        <div class="metric-value blue">{rec_count}</div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # ── Probability bar ──────────────────────────────
                if prob is not None:
                    bar_color = "#e05252" if prob_val > 60 else ("#d4af37" if prob_val > 35 else "#4caf82")
                    st.markdown(f"""
                    <div class="prob-bar-wrap">
                        <div class="prob-bar-label">Return Probability — {prob}%</div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{min(prob_val,100)}%; background:{bar_color};"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                # ── Risk factors + Recommendations ───────────────
                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown('<div class="sec-head">Risk Factors Identified</div>', unsafe_allow_html=True)
                    if risks:
                        for r in risks:
                            st.markdown(f'<div class="factor-card">{r}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="positive-card">✅ No significant risk factors detected.</div>', unsafe_allow_html=True)

                    if positives:
                        st.markdown('<div class="sec-head" style="margin-top:24px;">Positive Signals</div>', unsafe_allow_html=True)
                        for p in positives:
                            st.markdown(f'<div class="positive-card">{p}</div>', unsafe_allow_html=True)

                with col_right:
                    st.markdown('<div class="sec-head">Business Recommendations</div>', unsafe_allow_html=True)
                    if recs:
                        for rec in recs:
                            st.markdown(f'<div class="rec-card">💡 {rec}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="rec-card">No additional actions required.</div>', unsafe_allow_html=True)

                # ── Decision Context ──────────────────────────────
                ctx = expl.get("decision_context", "")
                if ctx:
                    st.markdown('<div class="sec-head">Decision Context</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="glass" style="font-size:14px; color:#8b9ab5; line-height:1.8;">{ctx}</div>', unsafe_allow_html=True)

                # ── Debug panel ──────────────────────────────────
                with st.expander("🔧 Debug — Decision Trace"):
                    st.markdown(f"""
                    <div class="debug-card">
                        <b style="color:#d4af37;">Decision Source :</b> {dbg.get('decision_source','—')}<br>
                        <b style="color:#d4af37;">ML Probability  :</b> {dbg.get('ml_probability','N/A')}%
                            &nbsp;(threshold = {dbg.get('threshold_used', RETURN_THRESHOLD)})<br>
                        <b style="color:#d4af37;">Rule Score      :</b> {dbg.get('rule_score', 0)}
                            &nbsp;(override at ≥ 3)<br>
                        <b style="color:#d4af37;">Triggered Rules :</b> {', '.join(dbg.get('triggered_rules',[])) or 'none'}
                    </div>""", unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                st.markdown("""
                <div class="glass" style="border-color:rgba(224,82,82,0.3);">
                    <div style="font-size:24px; color:#e05252; font-family:'Cormorant Garamond',serif; margin-bottom:12px;">
                        ⚠️ Backend API Not Running
                    </div>
                    <div style="color:#8b9ab5; font-size:14px; line-height:1.8;">
                        Start the FastAPI server first, then retry:
                    </div>
                </div>""", unsafe_allow_html=True)
                st.code("uvicorn api.main:app --reload --host 127.0.0.1 --port 8000", language="bash")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out after 15 s. Is the API overloaded?")

            except Exception as exc:
                import traceback
                st.error(f"Unexpected Streamlit error: {exc}")
                with st.expander("Full traceback"):
                    st.code(traceback.format_exc(), language="python")

# ──────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANCE
# Searches RESULT_DIR for each expected image.
# Shows a clear "file not found" card if missing instead of
# silently rendering nothing.
# ──────────────────────────────────────────────────────────────
with tab2:

    # ── quick folder health-check ──
    if not os.path.isdir(RESULT_DIR):
        st.error(f"Results folder not found: `{RESULT_DIR}`\nRun the training notebook first.")
    else:
        found  = os.listdir(RESULT_DIR)
        n_pngs = sum(1 for f in found if f.endswith(".png"))
        # st.caption(f"📂 `{RESULT_DIR}` — {n_pngs} PNG(s) found")

    st.markdown('<div class="sec-head">Classification vs Regression Accuracy</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        show_result_image("classification_model_results.png", "Classification — Accuracy Comparison")
    with col2:
        show_result_image("final_regression_comparison.png",  "Regression — R² Score Comparison")

    st.markdown('<div class="sec-head">ROC Curve — All Models</div>', unsafe_allow_html=True)
    show_result_image("roc_all_models.png", "ROC Curves — All Classification Models")

    st.markdown('<div class="sec-head">DNN Training Curves</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        show_result_image("dnn_loss_curve.png",     "DNN — Loss Curve")
    with col4:
        show_result_image("dnn_accuracy_curve.png", "DNN — Accuracy Curve")

    st.markdown('<div class="sec-head">LSTM Diagnostics</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        show_result_image("lstm_roc_curve.png",              "LSTM — ROC Curve")
    with col6:
        show_result_image("lstm_prediction_distribution.png","LSTM — Prediction Distribution")

    st.markdown('<div class="sec-head">Confusion Matrices</div>', unsafe_allow_html=True)
    cm_models = ["DecisionTree", "RandomForest", "XGBoost", "DNN", "LSTM", "CNN"]
    for row_models in [cm_models[:3], cm_models[3:]]:
        cols = st.columns(3)
        for col, model_name in zip(cols, row_models):
            with col:
                show_result_image(
                    f"confusion_matrix_{model_name}.png",
                    model_name
                )

# ──────────────────────────────────────────────────────────────
# TAB 3 — EXPLAINABLE AI
# ──────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-head">SHAP Explainability — XGBoost</div>', unsafe_allow_html=True)
    col7, col8 = st.columns(2)
    with col7:
        show_result_image("shap_summary_xgb.png", "SHAP Summary — Feature Impact Distribution")
    with col8:
        show_result_image("shap_bar_xgb.png",     "SHAP Bar — Global Feature Importance Ranking")

    st.markdown('<div class="sec-head">How the AI Makes Decisions</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
        <p style="color:#8b9ab5; font-size:14px; line-height:2.0; margin:0;">
            <strong style="color:#d4af37;">Payment Value</strong>
              — High-value transactions carry elevated return scrutiny and fraud risk.<br>
            <strong style="color:#d4af37;">Price vs Payment Gap</strong>
              — Discrepancies between listed price and amount collected signal checkout errors
                or unauthorised discount codes.<br>
            <strong style="color:#d4af37;">Freight-to-Price Ratio</strong>
              — Shipping cost exceeding 30 % of product price sharply reduces perceived value.<br>
            <strong style="color:#d4af37;">Review Score (1–2)</strong>
              — Strongest single predictor of return intent; added with double weight in the rule engine.<br>
            <strong style="color:#d4af37;">Review Text (TF-IDF 300 features)</strong>
              — Linguistic patterns in customer comments capture dissatisfaction invisible to numeric features.<br>
            <strong style="color:#d4af37;">Instalment Count</strong>
              — Extended payment plans (> 6) correlate with buyer's remorse.<br>
            <strong style="color:#d4af37;">Product Weight</strong>
              — Heavy items (> 5 kg) face higher transit-damage probability.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Hybrid Decision Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass">
        <p style="color:#8b9ab5; font-size:14px; line-height:2.0; margin:0;">
            <strong style="color:#d4af37;">Step 1 — ML Model (XGBoost)</strong>
              — Processes 3 numeric features + 300 TF-IDF text features.
                Returns probability 0–1. Threshold = 0.35.<br>
            <strong style="color:#d4af37;">Step 2 — Rule Engine</strong>
              — Evaluates 7 business rules over fields NOT in the ML feature matrix
                (review_score, installments, weight, payment gap).
                Score ≥ 3 → force return = True regardless of ML output.<br>
            <strong style="color:#d4af37;">Step 3 — Revenue Regressor (Random Forest)</strong>
              — Estimates revenue at risk only when return = True.<br>
            <strong style="color:#d4af37;">Step 4 — Explanation Engine (explain.py)</strong>
              — Converts all signals into human-readable risk factors,
                positive signals, and business recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="footer">
    <strong>ReturnIQ</strong> · AI End Semester Project<br>
    Arzoo Sarwari · Talha Ahmed Khan · Abdul Rehman<br>
    <span style="color:#2a3a50;">Powered by XGBoost · Random Forest · LSTM · CNN · SHAP</span>
</div>
""", unsafe_allow_html=True)
