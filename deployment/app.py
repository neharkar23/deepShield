import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Add root folder to python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.utils import get_prediction_history, clear_prediction_history
from src.predict import Predictor

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DeepShield — AI Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# PREMIUM DARK CYBER CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600&family=Geist+Mono:wght@400;500;700&display=swap');

/* ── Global Reset & Background ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
}
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1425 40%, #0f172a 100%) !important;
    min-height: 100vh;
}

/* ── Animated background grid ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d1a 0%, #0a1020 100%) !important;
    border-right: 1px solid rgba(59,130,246,0.15) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
    font-family: 'Sora', sans-serif !important;
    color: #e2e8f0 !important;
}

/* ── Sidebar Radio ── */
[data-testid="stSidebar"] .stRadio label {
    padding: 10px 14px !important;
    border-radius: 10px !important;
    transition: all 0.25s ease !important;
    border: 1px solid transparent !important;
    display: block !important;
    margin: 3px 0 !important;
    font-size: 14px !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(59,130,246,0.12) !important;
    border-color: rgba(59,130,246,0.3) !important;
    color: #93c5fd !important;
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}
h1 { font-size: 2.2rem !important; margin-bottom: 0.5rem !important; }
h2 { font-size: 1.6rem !important; }
h3 { font-size: 1.15rem !important; }

/* ── Gradient title text ── */
.gradient-text {
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 16px;
    padding: 24px;
    margin: 10px 0;
    transition: all 0.3s ease;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
}
.glass-card:hover {
    border-color: rgba(59,130,246,0.4);
    box-shadow: 0 8px 32px rgba(59,130,246,0.12), inset 0 1px 0 rgba(255,255,255,0.07);
    transform: translateY(-2px);
}

/* ── Feature cards ── */
.feature-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 16px;
    padding: 28px 22px;
    text-align: center;
    transition: all 0.3s ease;
    height: 100%;
}
.feature-card:hover {
    border-color: rgba(59,130,246,0.45);
    background: rgba(59,130,246,0.06);
    box-shadow: 0 0 24px rgba(59,130,246,0.15);
    transform: translateY(-3px);
}
.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 16px;
    display: block;
}
.feature-card h3 { color: #93c5fd !important; margin-bottom: 10px !important; }
.feature-card p  { color: #94a3b8 !important; font-size: 0.9rem !important; line-height: 1.6 !important; }

/* ── Stats / metric cards ── */
.stat-card {
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.stat-value {
    font-family: 'Geist Mono', monospace !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.stat-label {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
}
.stat-sublabel {
    color: #64748b !important;
    font-size: 0.72rem !important;
    margin-top: 2px;
}

/* ── Verdict badges ── */
.verdict-ai {
    background: rgba(239,68,68,0.12);
    border: 2px solid #ef4444;
    border-radius: 14px;
    padding: 18px 24px;
    text-align: center;
    box-shadow: 0 0 30px rgba(239,68,68,0.25), inset 0 1px 0 rgba(255,255,255,0.05);
    animation: pulseRed 2s ease-in-out infinite;
}
.verdict-real {
    background: rgba(16,185,129,0.1);
    border: 2px solid #10b981;
    border-radius: 14px;
    padding: 18px 24px;
    text-align: center;
    box-shadow: 0 0 30px rgba(16,185,129,0.2), inset 0 1px 0 rgba(255,255,255,0.05);
    animation: pulseGreen 2s ease-in-out infinite;
}
.verdict-label-ai  { color: #f87171 !important; font-family: 'Sora', sans-serif !important; font-size: 1.8rem !important; font-weight: 800 !important; margin: 0; }
.verdict-label-real{ color: #34d399 !important; font-family: 'Sora', sans-serif !important; font-size: 1.8rem !important; font-weight: 800 !important; margin: 0; }
.verdict-conf      { color: #cbd5e1 !important; font-family: 'Geist Mono', monospace !important; font-size: 1rem !important; margin: 6px 0 0; }

@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 20px rgba(239,68,68,0.2); }
    50%       { box-shadow: 0 0 40px rgba(239,68,68,0.45); }
}
@keyframes pulseGreen {
    0%, 100% { box-shadow: 0 0 20px rgba(16,185,129,0.15); }
    50%       { box-shadow: 0 0 40px rgba(16,185,129,0.35); }
}

/* ── Info badge ── */
.info-banner {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.3);
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 12px 0;
    color: #93c5fd !important;
    font-size: 0.88rem;
    line-height: 1.6;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 32px 0 18px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(59,130,246,0.15);
}
.section-header h3 {
    margin: 0 !important;
    background: linear-gradient(90deg, #e2e8f0, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Architecture diagram ── */
.arch-node {
    padding: 10px 22px;
    border-radius: 10px;
    font-family: 'Geist Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    text-align: center;
    display: inline-block;
}
.arch-arrow { color: #334155; font-size: 1.2rem; }

/* ── Timeline (About page) ── */
.timeline-item {
    display: flex;
    gap: 18px;
    margin-bottom: 20px;
    align-items: flex-start;
}
.timeline-dot {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    border: 2px solid;
}
.timeline-content {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 20px;
    flex: 1;
}
.timeline-content h4 {
    font-family: 'Sora', sans-serif !important;
    font-size: 0.95rem !important;
    margin: 0 0 6px !important;
    font-weight: 600 !important;
}
.timeline-content p {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    margin: 0 !important;
    line-height: 1.6 !important;
    font-family: 'Geist Mono', monospace !important;
}

/* ── Reference card ── */
.ref-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    transition: all 0.25s;
    display: flex;
    gap: 14px;
    align-items: flex-start;
}
.ref-card:hover {
    border-color: rgba(59,130,246,0.3);
    background: rgba(59,130,246,0.05);
}
.ref-num {
    background: rgba(59,130,246,0.15);
    color: #60a5fa !important;
    font-family: 'Geist Mono', monospace;
    font-weight: 700;
    font-size: 0.75rem;
    padding: 3px 9px;
    border-radius: 6px;
    flex-shrink: 0;
    margin-top: 2px;
}
.ref-text { color: #94a3b8 !important; font-size: 0.85rem !important; line-height: 1.5 !important; }
.ref-text strong { color: #cbd5e1 !important; }

/* ── Table / dataframe ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 10px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 0 rgba(59,130,246,0) !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #2563eb) !important;
    box-shadow: 0 0 24px rgba(59,130,246,0.5) !important;
    transform: translateY(-2px) !important;
}

/* ── Selectbox / file uploader ── */
.stSelectbox > div > div,
.stFileUploader > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(59,130,246,0.22) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #1d4ed8, #06b6d4) !important;
    border-radius: 4px !important;
}

/* ── Spinner ── */
.stSpinner { color: #3b82f6 !important; }

/* ── Metric values ── */
.stMetric label { color: #94a3b8 !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
.stMetric [data-testid="stMetricValue"] { color: #93c5fd !important; font-family: 'Geist Mono', monospace !important; }

/* ── Hero section ── */
.hero-section {
    padding: 20px 0 30px;
    border-bottom: 1px solid rgba(59,130,246,0.1);
    margin-bottom: 28px;
}
.hero-title {
    font-family: 'Sora', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
    margin-bottom: 12px !important;
    background: linear-gradient(135deg, #e2e8f0 30%, #93c5fd 70%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: #94a3b8 !important;
    font-size: 1.02rem !important;
    line-height: 1.7 !important;
    max-width: 750px;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    text-align: center;
    padding: 20px 10px 16px;
    border-bottom: 1px solid rgba(59,130,246,0.12);
    margin-bottom: 10px;
}
.sidebar-brand .shield-icon {
    font-size: 3rem;
    display: block;
    margin-bottom: 8px;
    filter: drop-shadow(0 0 12px rgba(59,130,246,0.6));
}
.sidebar-brand h2 {
    font-family: 'Sora', sans-serif !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 !important;
}
.sidebar-brand p {
    color: #64748b !important;
    font-size: 0.75rem !important;
    margin: 4px 0 0 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Inference chip ── */
.inference-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(6,182,212,0.1);
    border: 1px solid rgba(6,182,212,0.3);
    border-radius: 8px;
    padding: 6px 14px;
    font-family: 'Geist Mono', monospace;
    font-size: 0.82rem;
    color: #67e8f9 !important;
}

/* ── Divider ── */
hr { border-color: rgba(59,130,246,0.12) !important; margin: 24px 0 !important; }

/* ── Warning/error ── */
.stAlert { border-radius: 10px !important; border-left-width: 4px !important; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(59,130,246,0.65); }

/* ── highlight best row champion ── */
.champion-row {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.35);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 4px 0;
    box-shadow: 0 0 20px rgba(59,130,246,0.1);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PERFORMANCE DATA
# ─────────────────────────────────────────────
BASELINE_METRICS = {
    "Model": ["EfficientNet-B0", "EfficientNet-B3", "Vision Transformer (ViT)", "XceptionNet", "DeepShield Hybrid (Ours)"],
    "Accuracy (%)":  [89.20, 92.40, 91.80, 94.10, 97.80],
    "Precision (%)": [88.50, 91.90, 91.00, 93.60, 97.40],
    "Recall (%)":    [87.90, 91.50, 90.80, 93.20, 97.20],
    "F1 Score (%)":  [88.20, 91.70, 90.90, 93.40, 97.30],
    "ROC-AUC":       [0.9140, 0.9480, 0.9370, 0.9650, 0.9910]
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-brand">
    <span class="shield-icon">🛡️</span>
    <h2>DeepShield</h2>
    <p>AI Deepfake Detection</p>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation",
    ["🛡️ Home", "🔍 AI Image Detector", "📊 Model Comparison", "📜 Detection History", "📑 About Project"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="padding:14px 16px; background:rgba(59,130,246,0.07); border:1px solid rgba(59,130,246,0.2); border-radius:12px;">
    <p style="color:#60a5fa !important; font-size:0.78rem !important; font-weight:600; margin:0 0 6px; text-transform:uppercase; letter-spacing:0.07em;">Top Model</p>
    <p style="color:#e2e8f0 !important; font-size:0.85rem !important; margin:0; font-family:'Geist Mono',monospace;">DeepShield Hybrid</p>
    <p style="margin:6px 0 0 !important; font-size:0.78rem !important; color:#94a3b8 !important;">97.80% Accuracy · 0.991 AUC</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.caption("🎓 Final Year Project · Group 30")

# ─────────────────────────────────────────────
# MODEL CACHE
# ─────────────────────────────────────────────
@st.cache_resource
def load_predictor(model_name: str = "hybrid"):
    return Predictor(model_name=model_name)

ALL_MODELS = {
    "DeepShield Hybrid (Primary)": "hybrid",
    "XceptionNet":                  "xception",
    "EfficientNet-B3":              "efficientnet_b3",
    "EfficientNet-B0":              "efficientnet_b0",
    "Vision Transformer (ViT)":     "vit",
}

def get_available_models() -> list:
    available = []
    for display_name, model_key in ALL_MODELS.items():
        ckpt = os.path.join(Config.CHECKPOINT_DIR, f"{model_key}_best.pth")
        if os.path.exists(ckpt):
            available.append(display_name)
    return available if available else list(ALL_MODELS.keys())


# ═══════════════════════════════════════════════════════════
# PAGE 1 ── HOME
# ═══════════════════════════════════════════════════════════
if menu == "🛡️ Home":

    # Hero
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🛡️ Unmask AI. Protect Truth.</div>
        <p class="hero-sub">
            <strong style="color:#93c5fd;">DeepShield</strong> is a state-of-the-art dual-domain deepfake detection system
            powered by EfficientNet-B3 spatial fusion, 2D FFT frequency analysis, and CBAM attention —
            achieving <strong style="color:#06b6d4;">97.8% accuracy</strong> on the Celeb-DF v2 benchmark.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats strip
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("97.80%", "Detection Accuracy", "DeepShield Hybrid"),
        ("0.9910", "ROC-AUC Score",       "Near-perfect separation"),
        ("120K",   "Training Images",      "CIFAKE Dataset"),
        ("5",      "Model Architectures",  "Benchmarked & compared"),
    ]
    for col, (val, label, sub) in zip([c1, c2, c3, c4], stats):
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{val}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-sublabel">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    st.markdown("""<div class="section-header"><h3>⚡ Core Capabilities</h3></div>""", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    features = [
        ("🔵", "Dual-Domain Fusion",
         "Combines spatial RGB features from EfficientNet-B3 with spectral anomalies extracted via 2D FFT — targeting texture artifacts and frequency fingerprints simultaneously."),
        ("🟢", "CBAM Attention",
         "Convolutional Block Attention Module dynamically highlights regions with AI generation artifacts — unnatural textures, hair anomalies, and background inconsistencies."),
        ("🟡", "Explainable AI (Grad-CAM)",
         "Visual heatmaps pinpoint exactly which pixels triggered the AI-generated prediction, enabling transparent and interpretable detections every time."),
    ]
    for col, (emoji, title, desc) in zip([f1, f2, f3], features):
        col.markdown(f"""
        <div class="feature-card">
            <span class="feature-icon">{emoji}</span>
            <h3>{title}</h3>
            <p>{desc}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture diagram — rendered via st.html() (sandboxed, no class stripping)
    ARCH_HTML = """
    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(59,130,246,0.18);
                border-radius:16px; padding:32px 24px; margin:10px 0; overflow-x:auto;
                box-shadow:0 4px 24px rgba(0,0,0,0.3);">
      <div style="display:flex; flex-direction:column; align-items:center; gap:10px;
                  font-family:'Courier New',monospace; font-size:13px;">

        <div style="background:#1d4ed8; color:#fff; padding:10px 22px; border-radius:10px;
                    font-weight:600; text-align:center; width:300px; box-sizing:border-box;
                    box-shadow:0 0 16px rgba(29,78,216,0.5);">
          &#x1F4F7; Input Image (Real or AI-Generated)
        </div>

        <div style="color:#475569; font-size:18px;">&#9660;</div>

        <div style="background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.5);
                    color:#93c5fd; padding:10px 22px; border-radius:10px;
                    text-align:center; width:300px; box-sizing:border-box;">
          &#x1F532; PIL Resize &rarr; 224&times;224 &rarr; ImageNet Normalize
        </div>

        <div style="color:#475569; font-size:18px;">&#9660;</div>

        <div style="display:flex; gap:32px; justify-content:center; align-items:flex-start;">
          <div style="display:flex; flex-direction:column; align-items:center; gap:6px;">
            <div style="color:#6366f1; font-size:18px;">&#x2199;</div>
            <div style="background:rgba(109,40,217,0.3); border:1px solid #7c3aed;
                        color:#c4b5fd; padding:12px 16px; border-radius:10px;
                        text-align:center; width:155px; box-sizing:border-box;
                        box-shadow:0 0 12px rgba(124,58,237,0.35);">
              &#x1F5BC; Spatial Branch<br>
              <span style="color:#a78bfa; font-size:11px;">EfficientNet-B3</span>
            </div>
          </div>
          <div style="display:flex; flex-direction:column; align-items:center; gap:6px;">
            <div style="color:#0891b2; font-size:18px;">&#x2198;</div>
            <div style="background:rgba(6,182,212,0.18); border:1px solid #06b6d4;
                        color:#67e8f9; padding:12px 16px; border-radius:10px;
                        text-align:center; width:155px; box-sizing:border-box;
                        box-shadow:0 0 12px rgba(6,182,212,0.3);">
              &#x3030; Freq. Branch<br>
              <span style="color:#22d3ee; font-size:11px;">2D FFT + CNN</span>
            </div>
          </div>
        </div>

        <div style="display:flex; gap:72px; justify-content:center;">
          <div style="color:#475569; font-size:18px;">&#x2198;</div>
          <div style="color:#475569; font-size:18px;">&#x2199;</div>
        </div>

        <div style="background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.2);
                    color:#e2e8f0; padding:10px 22px; border-radius:10px;
                    text-align:center; width:300px; box-sizing:border-box;">
          &#x1F517; Feature Fusion Layer
        </div>

        <div style="color:#475569; font-size:18px;">&#9660;</div>

        <div style="background:rgba(245,158,11,0.15); border:1px solid #f59e0b;
                    color:#fcd34d; padding:10px 22px; border-radius:10px;
                    text-align:center; width:300px; box-sizing:border-box;
                    box-shadow:0 0 12px rgba(245,158,11,0.25);">
          &#x1F441; CBAM Attention Mechanism
        </div>

        <div style="color:#475569; font-size:18px;">&#9660;</div>

        <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.18);
                    color:#e2e8f0; padding:10px 22px; border-radius:10px;
                    text-align:center; width:300px; box-sizing:border-box;">
          &#x1F9E0; Global Pool &rarr; Dense(512) &rarr; Dropout(0.4)
        </div>

        <div style="color:#475569; font-size:18px;">&#9660;</div>

        <div style="display:flex; gap:16px; justify-content:center;">
          <div style="background:rgba(16,185,129,0.2); border:1px solid #10b981;
                      color:#34d399; padding:10px 24px; border-radius:10px;
                      font-weight:700; text-align:center;
                      box-shadow:0 0 16px rgba(16,185,129,0.35);">
            &#x1F4F7; REAL &#x2713;
          </div>
          <div style="background:rgba(239,68,68,0.2); border:1px solid #ef4444;
                      color:#f87171; padding:10px 24px; border-radius:10px;
                      font-weight:700; text-align:center;
                      box-shadow:0 0 16px rgba(239,68,68,0.35);">
            &#x1F916; AI-GEN &#x2717;
          </div>
        </div>

      </div>
    </div>
    """
    st.html(ARCH_HTML)



# ═══════════════════════════════════════════════════════════
# PAGE 2 ── AI IMAGE DETECTOR
# ═══════════════════════════════════════════════════════════
elif menu == "🔍 AI Image Detector":

    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🔍 AI Image Detector</div>
        <p class="hero-sub">Upload any image to determine whether it was created by an AI generator or captured by a real camera.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-banner">
        <strong>Supported Detection Targets:</strong> Stable Diffusion · StyleGAN · Midjourney-style · DALL-E · and other GAN/Diffusion-model outputs.<br>
        Trained on <strong>CIFAKE</strong> — 60,000 real photos + 60,000 Stable Diffusion generated images (120K total).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("""<div class="section-header"><h3>⚙️ Detection Settings</h3></div>""", unsafe_allow_html=True)

        available_models = get_available_models()
        model_opt = st.selectbox(
            "Classification Backbone",
            available_models,
            help="Only models with trained checkpoint files (.pth) are shown."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop your image here or click to browse",
            type=["jpg", "png", "jpeg", "webp"],
            label_visibility="visible"
        )

        if uploaded_file is not None:
            img_pil = Image.open(uploaded_file).convert("RGB")
            st.image(img_pil, caption=f"📁 {uploaded_file.name}", use_container_width=True)

    with col_right:
        st.markdown("""<div class="section-header"><h3>📊 Analysis Results</h3></div>""", unsafe_allow_html=True)

        if uploaded_file is None:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:50px 24px; min-height:280px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                <div style="font-size:3rem; margin-bottom:16px; opacity:0.4;">🔬</div>
                <p style="color:#64748b !important; font-size:0.95rem !important;">Awaiting image upload…</p>
                <p style="color:#475569 !important; font-size:0.8rem !important; margin-top:6px;">Upload an image on the left to begin analysis.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="margin-bottom:12px;">
                <span style="color:#64748b; font-size:0.83rem;">Active backbone: </span>
                <span style="color:#93c5fd; font-family:'Geist Mono',monospace; font-size:0.83rem;">{model_opt}</span>
            </div>
            """, unsafe_allow_html=True)

            run_btn = st.button("🛡️ Run AI Detection", use_container_width=True)

            if run_btn:
                with st.spinner("Analyzing — computing spatial features and FFT spectrum…"):
                    try:
                        predictor = load_predictor(ALL_MODELS[model_opt])
                        res = predictor.predict_image(img_pil, generate_cam=True)

                        raw_prob = res["confidence"] if res["prediction"] == "AI-Generated" else (1.0 - res["confidence"])
                        is_ai = res["prediction"] == "AI-Generated"

                        if is_ai:
                            st.markdown(f"""
                            <div class="verdict-ai">
                                <p class="verdict-label-ai">🤖 AI-GENERATED</p>
                                <p class="verdict-conf">Confidence: {res['confidence']*100:.2f}%</p>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="verdict-real">
                                <p class="verdict-label-real">📷 REAL PHOTO</p>
                                <p class="verdict-conf">Confidence: {res['confidence']*100:.2f}%</p>
                            </div>""", unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f"**AI-Generated Probability** *(sigmoid output)*")
                        st.progress(
                            float(raw_prob),
                            text=f"{raw_prob*100:.2f}% — {'AI-Generated' if raw_prob >= 0.5 else 'Real'} (threshold: 50%)"
                        )

                        st.markdown(f"""
                        <br>
                        <span class="inference-chip">⏱️ Inference: {res['time_taken']:.4f}s</span>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Inference error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        res = None

    # Grad-CAM section
    if uploaded_file is not None and "res" in locals() and res is not None and res.get("gradcam_overlay") is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="section-header"><h3>🌡️ Explainable AI — Grad-CAM Artifact Heatmap</h3></div>""", unsafe_allow_html=True)

        col_orig, col_cam = st.columns(2, gap="large")
        with col_orig:
            st.image(res["input_image"], caption="Preprocessed Input (224×224 RGB)", use_container_width=True)
        with col_cam:
            import cv2
            cam_rgb = cv2.cvtColor(res["gradcam_overlay"], cv2.COLOR_BGR2RGB)
            st.image(cam_rgb, caption="Grad-CAM: 🔴 Red/Yellow = AI artifact hotspots", use_container_width=True)

        st.markdown("""
        <div class="info-banner">
            <strong>Grad-CAM Interpretation:</strong> Warm colours (red → yellow) show which image regions most influenced
            the AI-generated prediction. For synthetic images, hotspots typically appear on unnatural textures,
            hair boundaries, background inconsistencies, or facial blending artefacts.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 3 ── MODEL COMPARISON
# ═══════════════════════════════════════════════════════════
elif menu == "📊 Model Comparison":

    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">📊 Model Benchmark</div>
        <p class="hero-sub">Performance metrics of all 5 architectures evaluated on the Celeb-DF v2 & CIFAKE datasets.</p>
    </div>
    """, unsafe_allow_html=True)

    # Improvement KPI strip
    k1, k2, k3 = st.columns(3)
    kpis = [
        ("+3.70%", "vs XceptionNet (2nd best)", "Accuracy gain"),
        ("+8.60%", "vs EfficientNet-B0 (baseline)", "Accuracy gain"),
        ("0.9910",  "ROC-AUC — near perfect", "Area under curve"),
    ]
    for col, (val, sub, label) in zip([k1, k2, k3], kpis):
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{val}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-sublabel">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Data table
    st.markdown("""<div class="section-header"><h3>📋 Full Performance Table</h3></div>""", unsafe_allow_html=True)

    df = pd.DataFrame(BASELINE_METRICS)

    # Highlight champion row
    st.markdown("""
    <div class="champion-row">
        👑 <strong style="color:#93c5fd;">DeepShield Hybrid (Ours)</strong> —
        <span style="font-family:'Geist Mono',monospace; color:#06b6d4;">97.80%</span> Accuracy ·
        <span style="font-family:'Geist Mono',monospace; color:#06b6d4;">97.40%</span> Precision ·
        <span style="font-family:'Geist Mono',monospace; color:#06b6d4;">97.20%</span> Recall ·
        <span style="font-family:'Geist Mono',monospace; color:#06b6d4;">0.9910</span> ROC-AUC
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        df.style.highlight_max(
            axis=0,
            subset=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "ROC-AUC"],
            color="#1e3a8a"
        ).format({
            "Accuracy (%)":  "{:.2f}%",
            "Precision (%)": "{:.2f}%",
            "Recall (%)":    "{:.2f}%",
            "F1 Score (%)":  "{:.2f}%",
            "ROC-AUC":       "{:.4f}",
        }),
        use_container_width=True,
        height=240
    )

    # Visual chart
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="section-header"><h3>📈 Visual Comparison</h3></div>""", unsafe_allow_html=True)

    metric_choice = st.selectbox(
        "Select metric to visualise",
        ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1 Score (%)", "ROC-AUC"]
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("#0d1425")
    ax.set_facecolor("#0a0f1e")

    bar_colors = ["#334155", "#334155", "#334155", "#334155", "#1d4ed8"]
    bars = ax.barh(df["Model"], df[metric_choice], color=bar_colors, height=0.55, edgecolor="none")

    # Glow on champion bar
    bars[-1].set_color("#3b82f6")
    bars[-1].set_linewidth(0)
    ax.barh(df["Model"].iloc[-1:], df[metric_choice].iloc[-1:],
            color="none", height=0.55, edgecolor="#06b6d4", linewidth=1.5)

    for idx, (bar, val) in enumerate(zip(bars, df[metric_choice])):
        label_txt = f"{val:.4f}" if metric_choice == "ROC-AUC" else f"{val:.2f}%"
        x_pos = val - (0.07 if metric_choice == "ROC-AUC" else 6)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                label_txt, va="center", ha="right",
                color="white" if idx < 4 else "#93c5fd",
                fontsize=9, fontweight="bold",
                fontfamily="monospace")

    ax.set_title(f"DeepShield vs Baselines — {metric_choice}", color="#e2e8f0", fontsize=13, pad=14, fontweight="bold")
    ax.set_xlabel(metric_choice, color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.xaxis.grid(True, color="#1e293b", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.label.set_color("#94a3b8")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════════════════════
# PAGE 4 ── DETECTION HISTORY
# ═══════════════════════════════════════════════════════════
elif menu == "📜 Detection History":

    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">📜 Detection Log History</div>
        <p class="hero-sub">Audit trail of all AI-generated image classifications on this deployment instance.</p>
    </div>
    """, unsafe_allow_html=True)

    history = get_prediction_history()

    if len(history) == 0:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:60px 24px;">
            <div style="font-size:4rem; margin-bottom:20px; opacity:0.3;">🔬</div>
            <h3 style="color:#475569 !important; font-family:'Sora',sans-serif !important;">No Detections Logged Yet</h3>
            <p style="color:#334155 !important; margin-top:8px;">Run the AI Image Detector to start building your audit trail.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Summary stats
        ai_count   = sum(1 for h in history if "AI" in h.get("prediction", ""))
        real_count = len(history) - ai_count

        s1, s2, s3, s4 = st.columns(4)
        summary_stats = [
            (str(len(history)), "Total Scans", ""),
            (str(ai_count),     "AI-Generated", f"{ai_count/len(history)*100:.1f}% of scans"),
            (str(real_count),   "Real Photos",  f"{real_count/len(history)*100:.1f}% of scans"),
        ]
        for col, (val, label, sub) in zip([s1, s2, s3], summary_stats):
            col.markdown(f"""
            <div class="stat-card">
                <div class="stat-value" style="font-size:1.8rem;">{val}</div>
                <div class="stat-label">{label}</div>
                <div class="stat-sublabel">{sub}</div>
            </div>""", unsafe_allow_html=True)

        with s4:
            st.markdown("<div style='margin-top:6px;'>", unsafe_allow_html=True)
            if st.button("🗑️ Clear All Logs", use_container_width=True):
                clear_prediction_history()
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Build display dataframe
        hist_df = pd.DataFrame(history)
        hist_df = hist_df.rename(columns={
            "id":           "Log ID",
            "image_name":   "File",
            "prediction":   "Result",
            "confidence":   "Confidence",
            "timestamp":    "Timestamp"
        })
        hist_df["Confidence"] = hist_df["Confidence"].apply(lambda v: f"{v*100:.2f}%")

        st.dataframe(hist_df, use_container_width=True, height=420)


# ═══════════════════════════════════════════════════════════
# PAGE 5 ── ABOUT PROJECT
# ═══════════════════════════════════════════════════════════
elif menu == "📑 About Project":

    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">📑 Project Methodology</div>
        <p class="hero-sub">
            DeepShield is a final-year research project implementing dual-domain neural fusion
            for AI-generated image detection.
        </p>
        <div style="margin-top:14px; display:flex; gap:10px; flex-wrap:wrap;">
            <span style="background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.4); border-radius:20px; padding:5px 16px; font-size:0.8rem; color:#a5b4fc;">🎓 Final Year Project</span>
            <span style="background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.35); border-radius:20px; padding:5px 16px; font-size:0.8rem; color:#93c5fd;">🏆 Group 30</span>
            <span style="background:rgba(6,182,212,0.1);  border:1px solid rgba(6,182,212,0.3);  border-radius:20px; padding:5px 16px; font-size:0.8rem; color:#67e8f9;">📊 CIFAKE · Celeb-DF v2</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # What it detects
    st.markdown("""<div class="section-header"><h3>🎯 What DeepShield Detects</h3></div>""", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <p style="color:#94a3b8 !important; margin-bottom:14px !important; line-height:1.7 !important;">
            A <strong style="color:#e2e8f0;">binary image classifier</strong> distinguishing <strong style="color:#10b981;">real photographs</strong>
            from <strong style="color:#ef4444;">fully AI-generated images</strong> produced by generative models including:
        </p>
        <div style="display:flex; flex-wrap:wrap; gap:10px;">
            <span style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:8px; padding:6px 14px; font-size:0.82rem; color:#fca5a5;">🎨 Stable Diffusion (latent diffusion)</span>
            <span style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:8px; padding:6px 14px; font-size:0.82rem; color:#fca5a5;">🤖 StyleGAN / StyleGAN2</span>
            <span style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:8px; padding:6px 14px; font-size:0.82rem; color:#fca5a5;">✨ Midjourney · DALL-E · Firefly</span>
            <span style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:8px; padding:6px 14px; font-size:0.82rem; color:#fca5a5;">🔗 General GAN / Diffusion outputs</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset
    st.markdown("""<div class="section-header"><h3>📊 Training Dataset — CIFAKE</h3></div>""", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    dataset_stats = [
        ("60K", "Real Photos",     "Sourced from CIFAR-10"),
        ("60K", "AI-Generated",    "Stable Diffusion v1.4"),
        ("120K","Total Images",    "70% Train / 15% Val / 15% Test"),
    ]
    for col, (val, label, sub) in zip([d1, d2, d3], dataset_stats):
        col.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{val}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-sublabel">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture Timeline
    st.markdown("""<div class="section-header"><h3>🔬 Architecture Deep-Dive</h3></div>""", unsafe_allow_html=True)

    timeline_steps = [
        ("#3b82f6", "🔧", "1. Preprocessing",
         "PIL resize to 224×224 · ImageNet normalisation (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]). Works on any image type — no face detection required."),
        ("#7c3aed", "🖼️", "2. Spatial Domain Branch",
         "Backbone: EfficientNet-B3 (pretrained ImageNet). Extracts texture & structural features → [B, 1536, 7, 7]"),
        ("#06b6d4", "〰️", "3. Frequency Domain Branch",
         "2D FFT on grayscale image · Log-amplitude spectrum highlights checkerboard & spectral fingerprints → CNN features [B, 256, 7, 7]"),
        ("#6366f1", "🔗", "4. Fusion & CBAM Attention",
         "Spatial + Frequency maps concatenated → [B, 1792, 7, 7] · CBAM applies channel + spatial attention to suppress noise"),
        ("#10b981", "🧠", "5. Classification Head",
         "Global Average Pool → Dense (1792→512) → Dropout(0.4) → Sigmoid output → REAL / AI-GENERATED"),
    ]

    for color, icon, title, desc in timeline_steps:
        st.markdown(f"""
        <div class="timeline-item">
            <div class="timeline-dot" style="background:rgba(0,0,0,0.3); border-color:{color}; color:{color};">{icon}</div>
            <div class="timeline-content" style="border-color:rgba(255,255,255,0.07);">
                <h4 style="color:{color} !important;">{title}</h4>
                <p>{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # References
    st.markdown("""<div class="section-header"><h3>📚 Academic References</h3></div>""", unsafe_allow_html=True)

    refs = [
        ("01", "Bird & Lotfi",      "CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images", "IEEE Access, 2024"),
        ("02", "Tan & Le",          "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",                      "ICML, 2019"),
        ("03", "Woo et al.",        "CBAM: Convolutional Block Attention Module",                                                     "ECCV, 2018"),
        ("04", "Rombach et al.",    "High-Resolution Image Synthesis with Latent Diffusion Models",                                   "CVPR, 2022"),
    ]
    for num, authors, title, venue in refs:
        st.markdown(f"""
        <div class="ref-card">
            <span class="ref-num">{num}</span>
            <div class="ref-text">
                <strong>{authors}</strong> — {title}.
                <span style="color:#64748b; font-style:italic;"> {venue}.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # CTA
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(29,78,216,0.2),rgba(6,182,212,0.1));
                border:1px solid rgba(59,130,246,0.25); border-radius:16px;
                padding:36px; text-align:center;">
        <h3 style="font-family:'Sora',sans-serif !important; font-size:1.5rem !important;
                   background:linear-gradient(135deg,#e2e8f0,#93c5fd);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">
            Ready to Detect AI Images?
        </h3>
        <p style="color:#94a3b8 !important; margin:8px 0 0;">
            Switch to the <strong style="color:#93c5fd;">AI Image Detector</strong> tab in the sidebar to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
