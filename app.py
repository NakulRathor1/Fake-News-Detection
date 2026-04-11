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
    font-size:92px;
    line-height:0.95;
    font-weight:800;
    letter-spacing:-2px;
    color:{text_color};
}}

.hero p {{
    font-size:24px;
    color:{sub_text};
    max-width:700px;
}}

.stTextArea textarea {{
    background:{card_color} !important;
    color:{input_text} !important;
    border:1px solid {border_color} !important;
    border-radius:18px;
    padding:18px;
    font-size:20px;
}}

.stTextArea textarea::placeholder {{
    color:{input_placeholder} !important;
    opacity:1;
}}

.stButton button {{
    background: {button_gradient};
    color:white;
    font-size:18px;
    font-weight:700;
    border:none;
    border-radius:999px;
    padding:14px 22px;
    transition: all 0.35s ease;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
}}

.stButton button:hover {{
    transform: translateY(-2px) scale(1.02);
}}
</style>
""", unsafe_allow_html=True)

# --------------------------
# LOAD MODEL
# --------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("c:\\vs code\\4TH-SEM\\Macro_4th_Sem\\model.pkl", "rb"))
    vectorizer = pickle.load(open("c:\\vs code\\4TH-SEM\\Macro_4th_Sem\\vectorizer.pkl", "rb"))
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

    probs = model.predict_proba(vector)[0]
    fake_prob = probs[0] * 100
    real_prob = probs[1] * 100

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