import streamlit as st
import pickle
import re

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Fake News AI",
    page_icon="🧠",
    layout="wide"
)

# --------------------------
# SESSION STATE INIT
# --------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# --------------------------
# SMOOTH THEME TOGGLE
# --------------------------
toggle_col1, toggle_col2 = st.columns([10, 1])

with toggle_col2:
    if st.button("🌙" if st.session_state.dark_mode else "☀️", key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
# --------------------------
# DYNAMIC THEME COLORS
# --------------------------
if st.session_state.dark_mode:
    bg_color = "#0b0d10"
    card_color = "#111418"
    text_color = "#ffffff"
    sub_text = "#cbd5e1"
    input_text = "#ffffff"
    input_placeholder = "#9ca3af"
    border_color = "#2b3138"
    button_gradient = "linear-gradient(90deg,#7c3aed,#2563eb)"
else:
    bg_color = "#f8fafc"
    card_color = "#ffffff"
    text_color = "#111827"
    sub_text = "#374151"
    input_text = "#111827"
    input_placeholder = "#6b7280"
    border_color = "#d1d5db"
    button_gradient = "linear-gradient(90deg,#f59e0b,#f97316)"

# --------------------------
# PREMIUM CSS
# --------------------------
st.markdown(f"""
<style>
#MainMenu {{visibility:hidden;}}
footer {{visibility:hidden;}}
header {{visibility:hidden;}}

.stApp {{
    background: {bg_color};
    color: {text_color};
}}

.block-container {{
    max-width: 1200px;
    padding-top: 1rem;
}}

.navbar {{
    display:flex;
    justify-content:space-between;
    align-items:center;
    background:{card_color};
    padding:18px 28px;
    border-radius:18px;
    margin-bottom:50px;
    border:1px solid {border_color};
}}

.hero h1 {{
    font-size: clamp(48px, 8vw, 92px);
    line-height:0.95;
    font-weight:800;
    letter-spacing:-2px;
    color:{text_color};
}}

.hero p {{
    font-size: clamp(16px, 2.5vw, 24px);
    color:{sub_text};
    max-width:700px;
}}

.stTextArea textarea {{
    background:{card_color} !important;
    color:{input_text} !important;
    border:1px solid {border_color} !important;
    border-radius:18px;
    padding:18px;
    font-size: clamp(16px, 2.5vw, 20px);
}}

.stTextArea textarea::placeholder {{
    color:{input_placeholder} !important;
    opacity:1;
}}

.stButton button {{
    position: relative;
    overflow: hidden;
    background: linear-gradient(
        135deg,
        #0f766e 0%,
        #155e75 45%,
        #1e3a8a 100%
    );
    color: #ffffff;
    font-size: clamp(16px, 2.5vw, 18px);
    font-weight: 700;
    letter-spacing: -0.3px;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 999px;
    padding: 15px 24px;
    transition: all 0.28s ease;
    box-shadow:
        0 6px 16px rgba(30,58,138,0.20),
        inset 0 1px 0 rgba(255,255,255,0.12);
    -webkit-font-smoothing: antialiased;
    text-rendering: geometricPrecision;
}}

.stButton button:hover {{
    transform: translateY(-2px);
    box-shadow:
        0 10px 20px rgba(15,118,110,0.22),
        0 4px 10px rgba(30,58,138,0.18),
        inset 0 1px 0 rgba(255,255,255,0.14);
    filter: brightness(1.15);
}}

.stButton button:active {{
    transform: translateY(0px) scale(0.99);
}}

/* --------------------------
   MOBILE RESPONSIVE (FIXED)
--------------------------- */
@media (max-width: 768px) {{

    .block-container {{
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
        max-width: 100% !important;
    }}

    /* THEME TOGGLE RIGHT */
    .stButton {{
        display: flex;
        justify-content: flex-end;
    }}

    /* MOBILE NAVBAR */
    .navbar {{
        display: block !important;
        padding: 16px 18px !important;
        border-radius: 18px !important;
        margin-bottom: 24px !important;
    }}

    .navbar div:first-child {{
        font-size: 28px !important;
        font-weight: 800 !important;
        margin-bottom: 6px !important;
    }}

    .navbar div:last-child {{
        font-size: 15px !important;
        color: {sub_text} !important;
        line-height: 1.5 !important;
    }}

    /* HERO */
    .hero {{
        text-align: left !important;
    }}

    .hero h1 {{
        font-size: 65px !important;
        line-height: 1.08 !important;
        letter-spacing: -1px !important;
        text-align: left !important;
    }}

    .hero p {{
        font-size: 16px !important;
        line-height: 1.7 !important;
        text-align: left !important;
        margin-top: 10px !important;
    }}

    /* INPUT */
    .stTextArea textarea {{
        min-height: 145px !important;
        border-radius: 18px !important;
        padding: 16px !important;
    }}

    /* BUTTON */
    .stButton button {{
        width: 100% !important;
        border-radius: 18px !important;
        padding: 14px 18px !important;
    }}
}}
</style>
""", unsafe_allow_html=True)

# --------------------------
# LOAD MODEL
# --------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# --------------------------
# NAVBAR
# --------------------------
st.markdown(f"""
<div class="navbar">
    <div style="font-size:28px;font-weight:800;">🧠 FAKE NEWS AI</div>
    <div style="font-size:18px;color:{sub_text};">ML-Powered Fake News Intelligence</div>
</div>
""", unsafe_allow_html=True)

# --------------------------
# HERO
# --------------------------
st.markdown(f"""
<div class="hero">
    <h1>Detect Fake<br>Before It Spreads</h1>
    <p>Advanced machine learning content analysis system for identifying misinformation patterns with confidence-driven insights.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------
# INPUT
# --------------------------
user_input = st.text_area(
    "",
    height=220,
    placeholder="Paste suspicious article or headline here..."
)

analyze = st.button("⚡ Analyze Content", use_container_width=True)

# --------------------------
# RESULT
# --------------------------
if analyze and user_input.strip():
    cleaned = re.sub(r'[^a-zA-Z\s]', '', user_input.lower())
    vector = vectorizer.transform([cleaned])

    score = model.decision_function(vector)[0]

    import math
    real_prob = 1 / (1 + math.exp(-score))
    real_prob *= 100
    fake_prob = 100 - real_prob

    # Result Panel
    st.markdown(f"""
    <div style="
        background:{card_color};
        border:1px solid {border_color};
        color:{text_color};
        border-radius:28px;
        padding:32px;
        margin-top:40px;
    ">
        <div style="font-size:28px;font-weight:800;margin-bottom:6px;">
            📊 Analysis Report
        </div>
        <div style="color:{sub_text};font-size:18px;margin-bottom:25px;">
            Confidence-driven misinformation scoring
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""
        <div style="
            background:{bg_color};
            border-radius:22px;
            padding:24px;
            border:1px solid {border_color};
        ">
            <div style="color:#f87171;font-size:18px;">🔴 Fake Probability</div>
            <div style="font-size:54px;font-weight:800;">{fake_prob:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="
            background:{bg_color};
            border-radius:22px;
            padding:24px;
            border:1px solid {border_color};
        ">
            <div style="color:#4ade80;font-size:18px;">🟢 Authenticity Score</div>
            <div style="font-size:54px;font-weight:800;">{real_prob:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.progress(int(max(fake_prob, real_prob)))

    if fake_prob > real_prob:
        st.error("❌ Verdict: High Risk of Fake News")
    else:
        st.success("✅ Verdict: Content Appears Authentic")

    st.markdown("</div>", unsafe_allow_html=True)
