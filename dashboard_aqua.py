"""
AQUA-AI Dashboard — Streamlit v2.0
Interface de supervision ONEA — Burkina Faso
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import json

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION PAGE
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AQUA-AI | ONEA Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# CSS GLOBAL — Thème industriel bleu nuit + cyan électrique
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* ── Variables ─────────────────────────────────────────────────────── */
:root {
    --bg-deep:      #050d1a;
    --bg-card:      #0a1628;
    --bg-card2:     #0d1e35;
    --bg-input:     #0f2040;
    --border:       #1a3a5c;
    --border-light: #1e4a72;
    --cyan:         #00d4ff;
    --cyan-dim:     #0099bb;
    --cyan-glow:    rgba(0, 212, 255, 0.15);
    --green:        #00e676;
    --green-dim:    #00a854;
    --orange:       #ff9100;
    --red:          #ff3d57;
    --yellow:       #ffe033;
    --text-prim:    #e8f4fd;
    --text-sec:     #7ba8c9;
    --text-dim:     #3d6680;
    --font-mono:    'Space Mono', monospace;
    --font-main:    'Outfit', sans-serif;
}

/* ── Base ───────────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    background-color: var(--bg-deep) !important;
    color: var(--text-prim) !important;
    font-family: var(--font-main) !important;
}
.block-container { padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1600px !important; }
h1,h2,h3,h4 { font-family: var(--font-main) !important; letter-spacing: -0.02em; }

/* ── Sidebar ────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060e1c 0%, #091525 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-prim) !important; }
section[data-testid="stSidebar"] .stSlider > label,
section[data-testid="stSidebar"] .stSelectbox > label,
section[data-testid="stSidebar"] .stNumberInput > label {
    color: var(--text-sec) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] { background: var(--border) !important; }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--cyan) !important;
    border-color: var(--cyan) !important;
    box-shadow: 0 0 8px var(--cyan) !important;
}

/* ── Boutons ────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #003d5c, #005580) !important;
    color: var(--cyan) !important;
    border: 1px solid var(--cyan-dim) !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #005580, #007aaa) !important;
    box-shadow: 0 0 16px var(--cyan-glow) !important;
    border-color: var(--cyan) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--cyan-dim), var(--cyan)) !important;
    color: var(--bg-deep) !important;
    border: none !important;
    font-weight: 800 !important;
}

/* ── Inputs ─────────────────────────────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 6px !important;
    color: var(--text-prim) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}
.stSelectbox [data-baseweb="select"] > div:focus-within,
.stNumberInput input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px rgba(0,212,255,0.2) !important;
}

/* ── Divider ────────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; opacity: 0.5 !important; }

/* ── Checkbox & toggle ──────────────────────────────────────────────── */
.stCheckbox label { color: var(--text-sec) !important; font-size: 0.82rem !important; }

/* ── Metric cards natifs ────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="metric-container"] label { color: var(--text-sec) !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--cyan) !important; font-family: var(--font-mono) !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Dataframe ──────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* ── Expander ───────────────────────────────────────────────────────── */
details summary { background: var(--bg-card2) !important; color: var(--text-sec) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; padding: 0.6rem 1rem !important; }

/* ── Progress bar ───────────────────────────────────────────────────── */
.stProgress > div > div { background: var(--bg-card) !important; border-radius: 4px !important; }
.stProgress > div > div > div { background: linear-gradient(90deg, var(--cyan-dim), var(--cyan)) !important; }

/* ── Scrollbar ──────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan-dim); }

/* ── Tabs ───────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card) !important; border-bottom: 1px solid var(--border) !important; gap: 4px !important; padding: 0 8px !important; border-radius: 8px 8px 0 0 !important; }
.stTabs [data-baseweb="tab"] { color: var(--text-dim) !important; font-family: var(--font-mono) !important; font-size: 0.78rem !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; padding: 0.7rem 1.2rem !important; border: none !important; background: transparent !important; }
.stTabs [aria-selected="true"] { color: var(--cyan) !important; border-bottom: 2px solid var(--cyan) !important; }
.stTabs [data-baseweb="tab-panel"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-top: none !important; border-radius: 0 0 8px 8px !important; padding: 1.5rem !important; }

/* ── Alertes custom ─────────────────────────────────────────────────── */
.stAlert { border-radius: 8px !important; border-left-width: 3px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# COMPOSANTS HTML CUSTOM
# ══════════════════════════════════════════════════════════════════════

def header_bar():
    now = datetime.now()
    st.markdown(f"""
    <div style="
        display:flex; align-items:center; justify-content:space-between;
        background: linear-gradient(135deg, #060e1c 0%, #0a1628 60%, #0d1e35 100%);
        border: 1px solid #1a3a5c;
        border-radius: 12px;
        padding: 1rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(0,212,255,0.08);
    ">
        <div style="display:flex; align-items:center; gap:1rem;">
            <div style="
                width:44px; height:44px;
                background: linear-gradient(135deg, #003d5c, #005580);
                border:1px solid #00d4ff;
                border-radius:10px;
                display:flex; align-items:center; justify-content:center;
                font-size:1.4rem;
                box-shadow: 0 0 12px rgba(0,212,255,0.3);
            ">💧</div>
            <div>
                <div style="font-family:'Outfit',sans-serif; font-size:1.4rem; font-weight:800; color:#e8f4fd; letter-spacing:-0.03em; line-height:1;">
                    AQUA<span style="color:#00d4ff;">·AI</span>
                </div>
                <div style="font-size:0.68rem; color:#3d6680; font-family:'Space Mono',monospace; letter-spacing:0.1em; text-transform:uppercase; margin-top:1px;">
                    ONEA — Burkina Faso · Supervision Énergétique
                </div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:2rem;">
            <div style="text-align:right;">
                <div style="font-family:'Space Mono',monospace; font-size:1rem; font-weight:700; color:#00d4ff;">{now.strftime('%H:%M:%S')}</div>
                <div style="font-size:0.68rem; color:#3d6680; font-family:'Space Mono',monospace;">{now.strftime('%a %d %b %Y')}</div>
            </div>
            <div style="
                display:flex; align-items:center; gap:0.4rem;
                background:#0a2818; border:1px solid #00e676;
                border-radius:20px; padding:0.3rem 0.8rem;
            ">
                <div style="width:7px; height:7px; border-radius:50%; background:#00e676; box-shadow:0 0 8px #00e676; animation: pulse 2s infinite;"></div>
                <span style="font-family:'Space Mono',monospace; font-size:0.68rem; color:#00e676; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">Système En Ligne</span>
            </div>
        </div>
    </div>
    <style>
    @keyframes pulse {{ 0%,100%{{opacity:1;box-shadow:0 0 8px #00e676;}} 50%{{opacity:0.5;box-shadow:0 0 3px #00e676;}} }}
    </style>
    """, unsafe_allow_html=True)


def kpi_card(titre, valeur, unite, icone, couleur="#00d4ff", tendance=None, tendance_label=""):
    tendance_html = ""
    if tendance is not None:
        t_color = "#00e676" if tendance >= 0 else "#ff3d57"
        t_arrow = "▲" if tendance >= 0 else "▼"
        tendance_html = f'<div style="font-size:0.7rem; color:{t_color}; font-family:Space Mono,monospace; margin-top:4px;">{t_arrow} {abs(tendance):.1f} {tendance_label}</div>'

    return f"""
    <div style="
        background: linear-gradient(135deg, #0a1628 0%, #0d1e35 100%);
        border: 1px solid #1a3a5c;
        border-top: 2px solid {couleur};
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        height: 100%;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        transition: border-color 0.2s;
    ">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <div style="font-size:0.65rem; color:#3d6680; text-transform:uppercase; letter-spacing:0.1em; font-weight:600; font-family:'Outfit',sans-serif;">{titre}</div>
                <div style="font-family:'Space Mono',monospace; font-size:1.6rem; font-weight:700; color:{couleur}; line-height:1.1; margin-top:0.3rem;">{valeur}<span style="font-size:0.7rem; color:#7ba8c9; margin-left:4px; font-weight:400;">{unite}</span></div>
                {tendance_html}
            </div>
            <div style="font-size:1.6rem; opacity:0.6;">{icone}</div>
        </div>
    </div>
    """


def jauge_niveau(nom, valeur, seuil_crit=20, seuil_att=35, unite="%"):
    if valeur <= seuil_crit:
        color = "#ff3d57"
        status = "CRITIQUE"
        glow = "rgba(255,61,87,0.4)"
    elif valeur <= seuil_att:
        color = "#ff9100"
        status = "ATTENTION"
        glow = "rgba(255,145,0,0.3)"
    elif valeur >= 92:
        color = "#ffe033"
        status = "PLEIN"
        glow = "rgba(255,224,51,0.3)"
    else:
        color = "#00e676"
        status = "NORMAL"
        glow = "rgba(0,230,118,0.25)"

    pct = min(max(valeur, 0), 100)
    return f"""
    <div style="
        background: linear-gradient(135deg, #0a1628, #0d1e35);
        border: 1px solid #1a3a5c;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 0.6rem;
    ">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.7rem;">
            <span style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#7ba8c9; font-weight:600;">{nom}</span>
            <span style="font-family:'Space Mono',monospace; font-size:0.7rem; color:{color}; font-weight:700; 
                background:rgba(0,0,0,0.3); padding:2px 8px; border-radius:3px; border:1px solid {color}40;">
                {status}
            </span>
        </div>
        <div style="display:flex; align-items:center; gap:0.8rem;">
            <div style="flex:1; height:10px; background:#0f2040; border-radius:5px; overflow:hidden; border:1px solid #1a3a5c;">
                <div style="
                    width:{pct}%;
                    height:100%;
                    background: linear-gradient(90deg, {color}88, {color});
                    border-radius:5px;
                    box-shadow: 0 0 8px {glow};
                    transition: width 0.5s ease;
                "></div>
            </div>
            <span style="font-family:'Space Mono',monospace; font-size:1rem; font-weight:700; color:{color}; min-width:48px; text-align:right;">{valeur:.0f}{unite}</span>
        </div>
    </div>
    """


def pompe_status(num, on, alerte, score, vibration, temp, efficacite):
    if alerte == "CRITIQUE":
        bg, border, dot, status_color = "#1a0a0a", "#ff3d57", "#ff3d57", "#ff3d57"
    elif alerte == "ATTENTION":
        bg, border, dot, status_color = "#1a1000", "#ff9100", "#ff9100", "#ff9100"
    else:
        bg, border, dot, status_color = "#0a1628", "#1a3a5c", "#00e676" if on else "#3d6680", "#00e676" if on else "#7ba8c9"

    etat_txt = "EN MARCHE" if on else "ARRÊT"
    etat_color = "#00e676" if on else "#7ba8c9"

    score_bars = ""
    for i in range(10):
        filled = i < int(score * 10)
        bar_color = "#ff3d57" if filled and score > 0.7 else ("#ff9100" if filled and score > 0.4 else ("#00e676" if filled else "#1a3a5c"))
        score_bars += f'<div style="flex:1; height:16px; background:{bar_color}; border-radius:2px; margin:0 1px;"></div>'

    return f"""
    <div style="
        background: linear-gradient(135deg, {bg}, #0d1e35);
        border: 1px solid {border};
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 0.6rem;
        box-shadow: {'0 0 12px ' + border + '30' if alerte != 'NORMAL' else 'none'};
    ">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.9rem;">
            <div style="display:flex; align-items:center; gap:0.6rem;">
                <div style="width:10px; height:10px; border-radius:50%; background:{dot}; box-shadow: 0 0 6px {dot};"></div>
                <span style="font-family:'Space Mono',monospace; font-size:0.8rem; font-weight:700; color:#e8f4fd;">POMPE {num}</span>
            </div>
            <div style="display:flex; gap:0.5rem; align-items:center;">
                <span style="font-size:0.65rem; color:{etat_color}; font-weight:700; text-transform:uppercase; letter-spacing:0.06em;">{etat_txt}</span>
                <span style="font-size:0.65rem; color:{status_color}; background:{status_color}15; border:1px solid {status_color}40; padding:1px 7px; border-radius:3px; font-weight:700;">
                    {alerte}
                </span>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.5rem; margin-bottom:0.8rem;">
            <div style="text-align:center; background:#05101f; border-radius:6px; padding:0.4rem;">
                <div style="font-size:0.58rem; color:#3d6680; text-transform:uppercase; letter-spacing:0.06em;">Efficacité</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.85rem; font-weight:700; color:#00d4ff;">{efficacite*100:.0f}%</div>
            </div>
            <div style="text-align:center; background:#05101f; border-radius:6px; padding:0.4rem;">
                <div style="font-size:0.58rem; color:#3d6680; text-transform:uppercase; letter-spacing:0.06em;">Vibration</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.85rem; font-weight:700; color:#{'ff9100' if vibration>3 else '00d4ff'};">{vibration:.1f} mm/s</div>
            </div>
            <div style="text-align:center; background:#05101f; border-radius:6px; padding:0.4rem;">
                <div style="font-size:0.58rem; color:#3d6680; text-transform:uppercase; letter-spacing:0.06em;">Temp. Moteur</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.85rem; font-weight:700; color:#{'ff3d57' if temp>80 else 'ff9100' if temp>70 else '00d4ff'};">{temp:.0f} °C</div>
            </div>
        </div>
        <div style="margin-top:0.3rem;">
            <div style="font-size:0.58rem; color:#3d6680; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:4px;">Score Anomalie : {score:.2f}</div>
            <div style="display:flex; height:16px;">{score_bars}</div>
        </div>
    </div>
    """


def decision_badge(action_nom, source, puissance, cout, decideur, override, confiance):
    src_icons = {'sonabel': '⚡', 'solaire': '☀️', 'diesel': '⛽', 'none': '⛔'}
    src_colors = {'sonabel': '#00d4ff', 'solaire': '#ffe033', 'diesel': '#ff9100', 'none': '#ff3d57'}
    icon = src_icons.get(source, '⚡')
    color = src_colors.get(source, '#00d4ff')
    conf_pct = int(confiance * 100)
    override_html = f'<div style="margin-top:0.8rem; padding:0.5rem 0.8rem; background:#1a0a0a; border:1px solid #ff3d57; border-radius:6px; font-size:0.7rem; color:#ff3d57; font-weight:600;">⚠️ OVERRIDE SÉCURITÉ ACTIF</div>' if override else ""

    return f"""
    <div style="
        background: linear-gradient(135deg, #0a1628, #0d1e35);
        border: 1px solid {color}60;
        border-left: 3px solid {color};
        border-radius: 10px;
        padding: 1.3rem;
        box-shadow: 0 0 20px {color}15;
    ">
        <div style="font-size:0.65rem; color:#3d6680; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.5rem; font-weight:600;">Décision IA — Cycle Actuel</div>
        <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.8rem;">
            <span style="font-size:1.6rem;">{icon}</span>
            <div>
                <div style="font-family:'Space Mono',monospace; font-size:1rem; font-weight:700; color:{color};">{action_nom}</div>
                <div style="font-size:0.68rem; color:#7ba8c9; margin-top:1px;">{decideur}</div>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin-bottom:0.8rem;">
            <div style="background:#05101f; border-radius:6px; padding:0.5rem 0.7rem;">
                <div style="font-size:0.58rem; color:#3d6680; text-transform:uppercase; letter-spacing:0.06em;">Puissance</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.95rem; font-weight:700; color:#e8f4fd;">{puissance:.0f} kW</div>
            </div>
            <div style="background:#05101f; border-radius:6px; padding:0.5rem 0.7rem;">
                <div style="font-size:0.58rem; color:#3d6680; text-transform:uppercase; letter-spacing:0.06em;">Coût estimé/h</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.95rem; font-weight:700; color:#e8f4fd;">{cout:,.0f} <span style="font-size:0.65rem; color:#7ba8c9;">FCFA</span></div>
            </div>
        </div>
        <div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="font-size:0.65rem; color:#3d6680; text-transform:uppercase; letter-spacing:0.06em;">Confiance IA</span>
                <span style="font-family:'Space Mono',monospace; font-size:0.68rem; color:{'#00e676' if conf_pct>=80 else '#ff9100' if conf_pct>=60 else '#ff3d57'}; font-weight:700;">{conf_pct}%</span>
            </div>
            <div style="height:5px; background:#0f2040; border-radius:3px; overflow:hidden;">
                <div style="width:{conf_pct}%; height:100%; background:linear-gradient(90deg, {'#00e676' if conf_pct>=80 else '#ff9100' if conf_pct>=60 else '#ff3d57'}88, {'#00e676' if conf_pct>=80 else '#ff9100' if conf_pct>=60 else '#ff3d57'}); border-radius:3px;"></div>
            </div>
        </div>
        {override_html}
    </div>
    """


def alerte_item(msg, niveau="info"):
    configs = {
        "critique": ("#ff3d57", "#1a0a0a", "🔴"),
        "attention": ("#ff9100", "#1a0f00", "🟠"),
        "info":      ("#00d4ff", "#001a24", "🔵"),
        "normal":    ("#00e676", "#001a0e", "🟢"),
    }
    color, bg, icon = configs.get(niveau, configs["info"])
    return f"""
    <div style="
        background:{bg}; border:1px solid {color}40; border-left:3px solid {color};
        border-radius:6px; padding:0.55rem 0.8rem; margin-bottom:0.35rem;
        display:flex; align-items:center; gap:0.6rem;
    ">
        <span style="font-size:0.75rem;">{icon}</span>
        <span style="font-size:0.75rem; color:#c5dde8; font-weight:500;">{msg}</span>
    </div>
    """


def section_title(titre, sous_titre=""):
    s = f'<div style="font-size:0.7rem; color:#3d6680; margin-top:2px; font-family:\'Space Mono\',monospace;">{sous_titre}</div>' if sous_titre else ""
    return f"""
    <div style="margin-bottom:1rem; padding-bottom:0.6rem; border-bottom:1px solid #1a3a5c;">
        <div style="font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#7ba8c9;">{titre}</div>
        {s}
    </div>
    """


# ══════════════════════════════════════════════════════════════════════
# SIMULATION DÉCISION (sans appel à l'orchestrateur réel)
# ══════════════════════════════════════════════════════════════════════

def simuler_decision(s):
    """Logique de décision simplifiée pour la démo."""
    alertes = []
    override = False
    raison_override = ""

    # Sécurité
    if s['niveau_chateau'] < 20:
        override = True
        if s['coupure_sonabel']:
            action_code, action_nom, source = 6, "POMPE1-DIESEL", "diesel"
        else:
            action_code, action_nom, source = 3, "POMPES12-SONABEL", "sonabel"
        raison_override = f"URGENCE : Château critique {s['niveau_chateau']:.0f}%"
        alertes.append(("critique", f"🚨 Château CRITIQUE : {s['niveau_chateau']:.0f}% — Pompage forcé"))
    elif s['niveau_bache'] < 15:
        override = True
        action_code, action_nom, source = 0, "STOP", "none"
        raison_override = f"URGENCE : Bâche critique {s['niveau_bache']:.0f}%"
        alertes.append(("critique", f"🚨 Bâche CRITIQUE : {s['niveau_bache']:.0f}% — ARRÊT pompes (cavitation)"))
    elif s['coupure_sonabel'] and s['stock_diesel'] < 10:
        override = True
        action_code, action_nom, source = 0, "STOP", "none"
        raison_override = f"Coupure + diesel faible ({s['stock_diesel']:.0f}%)"
        alertes.append(("critique", f"Coupure SONABEL + diesel faible ({s['stock_diesel']:.0f}%) → ARRÊT"))
    else:
        # Décision tarifaire
        if s['coupure_sonabel']:
            if s['niveau_chateau'] < 40:
                action_code, action_nom, source = 6, "POMPE1-DIESEL", "diesel"
            else:
                action_code, action_nom, source = 0, "STOP", "none"
        elif s['solaire_kw'] > 200:
            action_code, action_nom, source = 4, "POMPE1-SOLAIRE", "solaire"
        elif s['plage'] == 'HC':
            if s['niveau_bache'] > 50:
                action_code, action_nom, source = 3, "POMPES12-SONABEL", "sonabel"
            else:
                action_code, action_nom, source = 1, "POMPE1-SONABEL", "sonabel"
        else:  # HP
            if s['niveau_chateau'] > 55:
                action_code, action_nom, source = 0, "STOP", "none"
            else:
                action_code, action_nom, source = 1, "POMPE1-SONABEL", "sonabel"
                alertes.append(("attention", f"Pompage en HP ({s['tarif']:.0f} FCFA/kWh) — château bas"))

    # Château plein
    if s['niveau_chateau'] > 92 and not override:
        action_code, action_nom, source = 0, "STOP", "none"
        alertes.append(("normal", "Château plein (>92%) — pompage suspendu"))

    # Alertes niveaux
    if 20 <= s['niveau_chateau'] < 35:
        alertes.append(("attention", f"Château bas : {s['niveau_chateau']:.0f}%"))
    if 15 <= s['niveau_bache'] < 25:
        alertes.append(("attention", f"Bâche basse : {s['niveau_bache']:.0f}%"))

    # Pompes
    alerte_p1 = "CRITIQUE" if (s['vib_p1'] > 5 or s['temp_p1'] > 85 or s['eff_p1'] < 0.70) else \
                "ATTENTION" if (s['vib_p1'] > 3.5 or s['temp_p1'] > 75 or s['eff_p1'] < 0.78) else "NORMAL"
    alerte_p2 = "CRITIQUE" if (s['vib_p2'] > 5 or s['temp_p2'] > 85 or s['eff_p2'] < 0.70) else \
                "ATTENTION" if (s['vib_p2'] > 3.5 or s['temp_p2'] > 75 or s['eff_p2'] < 0.78) else "NORMAL"

    score_p1 = min(1.0, max(0, (max(0, s['vib_p1']-1.5)/4 + max(0, s['temp_p1']-55)/40 + max(0, 0.87-s['eff_p1'])/0.2) / 3))
    score_p2 = min(1.0, max(0, (max(0, s['vib_p2']-1.5)/4 + max(0, s['temp_p2']-55)/40 + max(0, 0.85-s['eff_p2'])/0.2) / 3))

    if alerte_p1 != "NORMAL":
        alertes.append(("critique" if alerte_p1 == "CRITIQUE" else "attention", f"🔧 Pompe 1 : {alerte_p1} (vib={s['vib_p1']:.1f} mm/s, T={s['temp_p1']:.0f}°C)"))
    if alerte_p2 != "NORMAL":
        alertes.append(("critique" if alerte_p2 == "CRITIQUE" else "attention", f"🔧 Pompe 2 : {alerte_p2} (vib={s['vib_p2']:.1f} mm/s, T={s['temp_p2']:.0f}°C)"))

    PUISS = {0:0, 1:185, 2:200, 3:385, 4:185, 5:200, 6:185, 7:385}
    POMPE1_ON = {0:False,1:True,2:False,3:True,4:True,5:False,6:True,7:True}
    POMPE2_ON = {0:False,1:False,2:True,3:True,4:False,5:True,6:False,7:True}
    puissance = PUISS[action_code]
    cout = puissance * 300 if source == 'diesel' else 0 if source in ['solaire','none'] else puissance * s['tarif']

    confiance = 1.0
    if override: confiance -= 0.3
    if alerte_p1 == "CRITIQUE" or alerte_p2 == "CRITIQUE": confiance -= 0.2
    elif alerte_p1 == "ATTENTION" or alerte_p2 == "ATTENTION": confiance -= 0.1
    confiance = max(0.3, confiance)

    h = s['heure']
    profil = 0.6 + 0.4*math.exp(-((h-7)**2)/8) + 0.3*math.exp(-((h-19)**2)/6)
    pred_debit = round(750 * profil, 1)
    pred_puiss = round(560 * profil, 1)
    icp = s['icp']

    return {
        'action_code': action_code, 'action_nom': action_nom, 'source': source,
        'pompe1_on': POMPE1_ON[action_code], 'pompe2_on': POMPE2_ON[action_code],
        'puissance': puissance, 'cout': cout,
        'alerte_p1': alerte_p1, 'alerte_p2': alerte_p2,
        'score_p1': round(score_p1, 3), 'score_p2': round(score_p2, 3),
        'override': override, 'raison_override': raison_override,
        'confiance': confiance,
        'alertes': alertes,
        'pred_debit': pred_debit, 'pred_puiss': pred_puiss,
        'icp': icp,
    }


def generer_historique(heure_actuelle, n=24):
    """Génère un historique fictif des 24 dernières heures."""
    rows = []
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    for i in range(n, 0, -1):
        ts = now - timedelta(hours=i)
        h = ts.hour
        profil = 0.6 + 0.4*math.exp(-((h-7)**2)/8) + 0.3*math.exp(-((h-19)**2)/6)
        plage = 'HP' if h >= 17 else 'HC'
        tarif = 165 if plage == 'HP' else 84
        puiss = 185 * profil * (1 + np.random.normal(0, 0.05))
        debit = 750 * profil * (1 + np.random.normal(0, 0.04))
        cout = puiss * tarif
        rows.append({
            'heure': ts.strftime('%H:%M'),
            'débit_m3h': round(debit, 1),
            'puissance_kw': round(puiss, 1),
            'coût_fcfa': round(cout, 0),
            'plage': plage,
            'tarif': tarif,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# GRAPHIQUES (Plotly)
# ══════════════════════════════════════════════════════════════════════

def chart_historique(df):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        colors_plage = ['#ff3d57' if p == 'HP' else '#00d4ff' for p in df['plage']]

        fig.add_trace(go.Bar(
            x=df['heure'], y=df['coût_fcfa'],
            name='Coût FCFA', marker_color=colors_plage, opacity=0.75,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df['heure'], y=df['puissance_kw'],
            name='Puissance kW', line=dict(color='#00d4ff', width=2),
            mode='lines+markers', marker=dict(size=4),
        ), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=df['heure'], y=df['débit_m3h'],
            name='Débit m³/h', line=dict(color='#00e676', width=1.5, dash='dot'),
        ), secondary_y=True)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Mono', color='#7ba8c9', size=10),
            legend=dict(orientation='h', x=0, y=1.1, font=dict(size=9)),
            margin=dict(l=10, r=10, t=30, b=10),
            height=220,
            xaxis=dict(showgrid=False, tickfont=dict(size=9), color='#3d6680'),
            yaxis=dict(showgrid=True, gridcolor='#1a3a5c', tickfont=dict(size=9), color='#3d6680', title_text='Coût FCFA', title_font=dict(size=8)),
            yaxis2=dict(showgrid=False, tickfont=dict(size=9), color='#3d6680', title_text='kW / m³/h', title_font=dict(size=8)),
            bargap=0.15,
        )
        return fig
    except ImportError:
        return None


def chart_predictions(heure_actuelle):
    try:
        import plotly.graph_objects as go
        heures = [(heure_actuelle + i) % 24 for i in range(1, 13)]
        labels = [f"{h:02d}:00" for h in heures]
        debits = [round(750 * (0.6 + 0.4*math.exp(-((h-7)**2)/8) + 0.3*math.exp(-((h-19)**2)/6)), 1) for h in heures]
        puiss  = [round(560 * (0.6 + 0.4*math.exp(-((h-7)**2)/8) + 0.3*math.exp(-((h-19)**2)/6)), 1) for h in heures]
        couts  = [p * (165 if h >= 17 else 84) for p, h in zip(puiss, heures)]
        colors = ['#ff3d57' if h >= 17 else 'rgba(0,212,255,0.25)' for h in heures]

        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=labels, y=couts, name='Coût FCFA prévu', marker_color=colors, opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(x=labels, y=debits, name='Débit prévu', line=dict(color='#00e676', width=2), fill='tozeroy', fillcolor='rgba(0,230,118,0.06)'), secondary_y=True)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Space Mono', color='#7ba8c9', size=10),
            legend=dict(orientation='h', x=0, y=1.1, font=dict(size=9)),
            margin=dict(l=10, r=10, t=30, b=10), height=200,
            xaxis=dict(showgrid=False, tickfont=dict(size=9), color='#3d6680'),
            yaxis=dict(showgrid=True, gridcolor='#1a3a5c', tickfont=dict(size=9), color='#3d6680'),
            yaxis2=dict(showgrid=False, tickfont=dict(size=9), color='#3d6680'),
        )
        return fig
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — Paramètres station
# ══════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:0.8rem 0; border-bottom:1px solid #1a3a5c; margin-bottom:1.2rem;">
            <div style="font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em; color:#3d6680; font-weight:700;">Paramètres Station</div>
            <div style="font-size:0.75rem; color:#7ba8c9; margin-top:2px; font-family:'Space Mono',monospace;">STATION_OUAGA_01</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**💧 Réservoirs**")
        niveau_chateau = st.slider("Château d'eau (%)", 0, 100, 65, 1)
        niveau_bache   = st.slider("Bâche d'aspiration (%)", 0, 100, 72, 1)

        st.markdown("**⚡ Énergie**")
        plage = st.selectbox("Plage tarifaire", ["HC", "HP"], index=0)
        tarif = 84 if plage == "HC" else 165
        st.markdown(f'<div style="font-family:Space Mono,monospace; font-size:0.75rem; color:#00d4ff; margin:-8px 0 8px 0;">Tarif actif : {tarif} FCFA/kWh</div>', unsafe_allow_html=True)

        solaire_kw    = st.slider("Puissance solaire (kW)", 0, 600, 0, 10)
        coupure       = st.checkbox("🔴 Coupure SONABEL", value=False)
        stock_diesel  = st.slider("Stock diesel (%)", 0, 100, 80, 1)

        st.markdown("**🕐 Contexte**")
        heure         = st.slider("Heure actuelle", 0, 23, datetime.now().hour, 1)
        temperature   = st.slider("Température (°C)", 20, 48, 32, 1)

        st.markdown("**🔧 Pompe 1**")
        eff_p1  = st.slider("Efficacité P1", 0.60, 1.00, 0.87, 0.01)
        vib_p1  = st.slider("Vibration P1 (mm/s)", 0.5, 8.0, 1.4, 0.1)
        temp_p1 = st.slider("Temp. moteur P1 (°C)", 30, 100, 58, 1)

        st.markdown("**🔧 Pompe 2**")
        eff_p2  = st.slider("Efficacité P2", 0.60, 1.00, 0.85, 0.01)
        vib_p2  = st.slider("Vibration P2 (mm/s)", 0.5, 8.0, 1.6, 0.1)
        temp_p2 = st.slider("Temp. moteur P2 (°C)", 30, 100, 60, 1)

        st.markdown("**📊 Hydraulique**")
        p_entree = st.slider("Pression entrée (bar)", 0.5, 4.0, 2.1, 0.1)
        p_sortie = st.slider("Pression sortie (bar)", 1.0, 8.0, 4.2, 0.1)
        icp      = st.slider("ICP (kWh/m³)", 0.3, 1.2, 0.57, 0.01)

        st.markdown("---")
        if st.button("🔄 Lancer le cycle IA", use_container_width=True, type="primary"):
            st.session_state['run_cycle'] = True

    return {
        'niveau_chateau': niveau_chateau,
        'niveau_bache':   niveau_bache,
        'plage': plage, 'tarif': tarif,
        'solaire_kw': solaire_kw,
        'coupure_sonabel': int(coupure),
        'stock_diesel': stock_diesel,
        'heure': heure, 'temperature': temperature,
        'eff_p1': eff_p1, 'vib_p1': vib_p1, 'temp_p1': temp_p1,
        'eff_p2': eff_p2, 'vib_p2': vib_p2, 'temp_p2': temp_p2,
        'p_entree': p_entree, 'p_sortie': p_sortie, 'icp': icp,
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════

def main():
    # Init session
    if 'run_cycle' not in st.session_state:
        st.session_state['run_cycle'] = False
    if 'decision_hist' not in st.session_state:
        st.session_state['decision_hist'] = []

    # Sidebar
    params = render_sidebar()

    # Calcul décision
    dec = simuler_decision(params)

    # Header
    header_bar()

    # ── TABS ──────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  Supervision", "🤖  Décision IA", "🔧  Maintenance", "📈  Analytique"
    ])

    # ══════════════════════════════════════════════════════════════════
    with tab1:
        # KPIs rapides
        c1 = "#ff3d57" if params['niveau_chateau']<20 else "#ff9100" if params['niveau_chateau']<35 else "#00e676"
        c2 = "#ff3d57" if params['niveau_bache']<15 else "#ff9100" if params['niveau_bache']<25 else "#00d4ff"
        c3 = "#ff9100" if params['plage']=='HP' else "#00e676"
        c4 = "#ffe033" if params['solaire_kw']>100 else "#3d6680"
        c5 = "#ff3d57" if params['stock_diesel']<10 else "#ff9100" if params['stock_diesel']<25 else "#00d4ff"
        st.markdown(
            '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0.75rem;margin-bottom:0.8rem;">'
            + kpi_card("Château d'eau", f"{params['niveau_chateau']:.0f}", "%", "🏛️", c1)
            + kpi_card("Bâche Aspiration", f"{params['niveau_bache']:.0f}", "%", "🪣", c2)
            + kpi_card("Tarif Actif", f"{params['tarif']}", "FCFA/kWh", "⚡", c3)
            + kpi_card("Solaire", f"{params['solaire_kw']:.0f}", "kW", "☀️", c4)
            + kpi_card("Stock Diesel", f"{params['stock_diesel']:.0f}", "%", "⛽", c5)
            + '</div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1.6, 1])

        with col_left:
            st.markdown(section_title("Niveaux Réservoirs", "Mise à jour en temps réel"), unsafe_allow_html=True)
            st.markdown(jauge_niveau("Château d'eau — Tour principale", params['niveau_chateau']), unsafe_allow_html=True)
            st.markdown(jauge_niveau("Bâche d'aspiration — Pompage", params['niveau_bache'], seuil_crit=15, seuil_att=25), unsafe_allow_html=True)
            st.markdown(jauge_niveau("Stock Diesel — Groupe électrogène", params['stock_diesel'], seuil_crit=10, seuil_att=20), unsafe_allow_html=True)

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            st.markdown(section_title("Hydraulique", "Pression & débit réseau"), unsafe_allow_html=True)
            ch1 = "#ff3d57" if params['p_entree']<1.5 else "#00d4ff"
            ch2 = "#ff3d57" if params['p_sortie']>6 else "#ff9100" if params['p_sortie']>5 else "#00d4ff"
            ch3 = "#ff9100" if params['icp']>0.7 else "#00e676"
            st.markdown(
                '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem;">'
                + kpi_card("P. Entrée", f"{params['p_entree']:.1f}", "bar", "🔽", ch1)
                + kpi_card("P. Sortie", f"{params['p_sortie']:.1f}", "bar", "🔼", ch2)
                + kpi_card("ICP", f"{params['icp']:.2f}", "kWh/m³", "📏", ch3)
                + '</div>',
                unsafe_allow_html=True
            )

        with col_right:
            st.markdown(section_title("Alertes Système", f"{len(dec['alertes'])} notification(s)"), unsafe_allow_html=True)
            if not dec['alertes']:
                st.markdown(alerte_item("Aucune alerte — Tous les paramètres nominaux", "normal"), unsafe_allow_html=True)
            else:
                for niveau, msg in dec['alertes']:
                    st.markdown(alerte_item(msg, niveau), unsafe_allow_html=True)

            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            st.markdown(section_title("Sources Énergie", "Disponibilité"), unsafe_allow_html=True)

            sources = [
                ("SONABEL", "⚡", not params['coupure_sonabel'], f"Tarif {params['tarif']} FCFA/kWh"),
                ("Solaire", "☀️", params['solaire_kw'] > 50, f"{params['solaire_kw']:.0f} kW disponibles"),
                ("Diesel", "⛽", params['stock_diesel'] > 10, f"Stock {params['stock_diesel']:.0f}%"),
            ]
            for nom, icon, dispo, detail in sources:
                color = "#00e676" if dispo else "#ff3d57"
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                    background:#0a1628; border:1px solid #1a3a5c; border-radius:8px;
                    padding:0.6rem 0.9rem; margin-bottom:0.4rem;">
                    <div style="display:flex; align-items:center; gap:0.6rem;">
                        <span style="font-size:1rem;">{icon}</span>
                        <div>
                            <div style="font-size:0.75rem; font-weight:600; color:#e8f4fd;">{nom}</div>
                            <div style="font-size:0.62rem; color:#3d6680;">{detail}</div>
                        </div>
                    </div>
                    <div style="display:flex; align-items:center; gap:0.4rem;">
                        <div style="width:8px; height:8px; border-radius:50%; background:{color}; box-shadow:0 0 5px {color};"></div>
                        <span style="font-size:0.65rem; color:{color}; font-weight:700;">{'ON' if dispo else 'OFF'}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    with tab2:
        d_col1, d_col2 = st.columns([1, 1.2])

        with d_col1:
            st.markdown(section_title("Décision Cycle Actuel", f"Heure {params['heure']:02d}:00"), unsafe_allow_html=True)
            st.markdown(decision_badge(
                dec['action_nom'], dec['source'], dec['puissance'],
                dec['cout'], "Agent Tarif (DQN)" if not dec['override'] else "Agent Sécurité",
                dec['override'], dec['confiance']
            ), unsafe_allow_html=True)

            if dec['override']:
                st.markdown(f"""
                <div style="background:#1a0505; border:1px solid #ff3d5780; border-left:3px solid #ff3d57;
                    border-radius:8px; padding:0.9rem 1.1rem; margin-top:0.8rem;">
                    <div style="font-size:0.65rem; color:#ff3d57; text-transform:uppercase; font-weight:700; letter-spacing:0.08em; margin-bottom:0.3rem;">
                        ⚠️ Override Sécurité Actif
                    </div>
                    <div style="font-size:0.75rem; color:#e8f4fd;">{dec['raison_override']}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            st.markdown(section_title("Prédictions LSTM — 12h"), unsafe_allow_html=True)

            fig_pred = chart_predictions(params['heure'])
            if fig_pred:
                st.plotly_chart(fig_pred, use_container_width=True, config={'displayModeBar': False})
            else:
                st.markdown(f"""
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem;">
                    <div style="background:#0a1628; border:1px solid #1a3a5c; border-radius:8px; padding:0.8rem; text-align:center;">
                        <div style="font-size:0.6rem; color:#3d6680; text-transform:uppercase;">Débit H+1</div>
                        <div style="font-family:Space Mono,monospace; font-size:1.1rem; color:#00d4ff; font-weight:700;">{dec['pred_debit']:.0f} m³/h</div>
                    </div>
                    <div style="background:#0a1628; border:1px solid #1a3a5c; border-radius:8px; padding:0.8rem; text-align:center;">
                        <div style="font-size:0.6rem; color:#3d6680; text-transform:uppercase;">Puissance H+1</div>
                        <div style="font-family:Space Mono,monospace; font-size:1.1rem; color:#00d4ff; font-weight:700;">{dec['pred_puiss']:.0f} kW</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with d_col2:
            st.markdown(section_title("Pipeline Décisionnel — 5 Agents"), unsafe_allow_html=True)

            agents = [
                ("01", "Agent Sécurité", "Seuils critiques — Override absolu",
                 "⚠️ OVERRIDE" if dec['override'] else "✅ RAS", "critique" if dec['override'] else "normal"),
                ("02", "Agent Maintenance", "Autoencoder — État pompes",
                 f"P1:{dec['alerte_p1']} P2:{dec['alerte_p2']}", "critique" if "CRITIQUE" in [dec['alerte_p1'],dec['alerte_p2']] else "attention" if "ATTENTION" in [dec['alerte_p1'],dec['alerte_p2']] else "normal"),
                ("03", "Agent Pompage", "LSTM 48h→24h — Prédictions",
                 f"Débit {dec['pred_debit']:.0f} m³/h | {dec['pred_puiss']:.0f} kW", "info"),
                ("04", "Agent Tarif", "DQN — Optimisation coût/énergie",
                 f"→ {dec['action_nom']}", "info"),
                ("05", "Agent Réseau", "Validation hydraulique",
                 "Contraintes vérifiées", "normal"),
            ]

            for num, nom, desc, result, niveau in agents:
                colors = {"normale":"#00e676","normal":"#00e676","critique":"#ff3d57","attention":"#ff9100","info":"#00d4ff"}
                c = colors.get(niveau, "#00d4ff")
                st.markdown(f"""
                <div style="display:flex; gap:0.8rem; margin-bottom:0.5rem; align-items:flex-start;">
                    <div style="
                        min-width:32px; height:32px; border-radius:50%;
                        background:{c}20; border:1px solid {c}60;
                        display:flex; align-items:center; justify-content:center;
                        font-family:'Space Mono',monospace; font-size:0.65rem; color:{c}; font-weight:700;
                    ">{num}</div>
                    <div style="
                        flex:1; background:#0a1628; border:1px solid #1a3a5c;
                        border-left:2px solid {c}; border-radius:0 8px 8px 0;
                        padding:0.5rem 0.8rem;
                    ">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:0.75rem; font-weight:600; color:#e8f4fd;">{nom}</span>
                            <span style="font-size:0.62rem; color:{c}; font-weight:700; background:{c}15; padding:1px 7px; border-radius:3px;">{result}</span>
                        </div>
                        <div style="font-size:0.63rem; color:#3d6680; margin-top:2px;">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            st.markdown(section_title("Résumé Coûts — Scénario Actuel"), unsafe_allow_html=True)

            tarif_hc, tarif_hp = 84, 165
            cout_hc_h = dec['puissance'] * tarif_hc
            cout_hp_h = dec['puissance'] * tarif_hp
            eco_h = max(0, cout_hp_h - dec['cout'])

            st.markdown(f"""
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.5rem; margin-bottom:0.8rem;">
                <div style="background:#0a1628; border:1px solid #1a3a5c; border-radius:8px; padding:0.7rem; text-align:center;">
                    <div style="font-size:0.58rem; color:#3d6680; text-transform:uppercase; letter-spacing:0.06em;">Coût/h actuel</div>
                    <div style="font-family:Space Mono,monospace; font-size:0.9rem; font-weight:700; color:#00d4ff;">{dec['cout']:,.0f}</div>
                    <div style="font-size:0.58rem; color:#3d6680;">FCFA</div>
                </div>
                <div style="background:#0a1628; border:1px solid #1a3a5c; border-radius:8px; padding:0.7rem; text-align:center;">
                    <div style="font-size:0.58rem; color:#3d6680; text-transform:uppercase; letter-spacing:0.06em;">Si HP</div>
                    <div style="font-family:Space Mono,monospace; font-size:0.9rem; font-weight:700; color:#ff9100;">{cout_hp_h:,.0f}</div>
                    <div style="font-size:0.58rem; color:#3d6680;">FCFA</div>
                </div>
                <div style="background:#0a1a0e; border:1px solid #00e67640; border-radius:8px; padding:0.7rem; text-align:center;">
                    <div style="font-size:0.58rem; color:#3d6680; text-transform:uppercase; letter-spacing:0.06em;">Économie/h</div>
                    <div style="font-family:Space Mono,monospace; font-size:0.9rem; font-weight:700; color:#00e676;">+{eco_h:,.0f}</div>
                    <div style="font-size:0.58rem; color:#3d6680;">FCFA</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    with tab3:
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(section_title("Pompe 1 — Diagnostic Autoencoder"), unsafe_allow_html=True)
            st.markdown(pompe_status(
                1, dec['pompe1_on'], dec['alerte_p1'], dec['score_p1'],
                params['vib_p1'], params['temp_p1'], params['eff_p1']
            ), unsafe_allow_html=True)
        with m2:
            st.markdown(section_title("Pompe 2 — Diagnostic Autoencoder"), unsafe_allow_html=True)
            st.markdown(pompe_status(
                2, dec['pompe2_on'], dec['alerte_p2'], dec['score_p2'],
                params['vib_p2'], params['temp_p2'], params['eff_p2']
            ), unsafe_allow_html=True)

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown(section_title("Indicateurs Combinés"), unsafe_allow_html=True)

        diff_p = params['p_sortie'] - params['p_entree']
        eff_moy = (params['eff_p1']+params['eff_p2'])/2
        temp_max = max(params['temp_p1'], params['temp_p2'])
        cm1 = "#ff3d57" if diff_p < 1.5 else "#00d4ff"
        cm2 = "#ff9100" if params['icp']>0.7 else "#00e676"
        cm3 = "#ff3d57" if eff_moy < 0.75 else "#00d4ff"
        cm4 = "#ff3d57" if temp_max>85 else "#ff9100" if temp_max>70 else "#00d4ff"
        st.markdown(
            '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.75rem;">'
            + kpi_card("Diff. Pression", f"{diff_p:.1f}", "bar", "\u21d5", cm1)
            + kpi_card("ICP Global", f"{params['icp']:.2f}", "kWh/m³", "\U0001f4ca", cm2)
            + kpi_card("Efficacité moy.", f"{eff_moy*100:.0f}", "%", "\u2699\ufe0f", cm3)
            + kpi_card("Temp. moy. moteur", f"{(params['temp_p1']+params['temp_p2'])/2:.0f}", "°C", "\U0001f321\ufe0f", cm4)
            + '</div>',
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        with st.expander("📋 Recommandations Maintenance"):
            recs = []
            if params['vib_p1'] > 4: recs.append(("critique", "Pompe 1 : vibrations élevées → inspection paliers/roulements urgente"))
            elif params['vib_p1'] > 3: recs.append(("attention", "Pompe 1 : vibrations légèrement élevées → surveillance renforcée"))
            if params['temp_p1'] > 80: recs.append(("critique", "Pompe 1 : surchauffe moteur → vérifier ventilation et lubrification"))
            if params['eff_p1'] < 0.75: recs.append(("attention", "Pompe 1 : efficacité dégradée → nettoyage roue ou remplacement garnitures"))
            if params['vib_p2'] > 4: recs.append(("critique", "Pompe 2 : vibrations élevées → inspection paliers/roulements urgente"))
            if params['temp_p2'] > 80: recs.append(("critique", "Pompe 2 : surchauffe moteur → vérifier ventilation"))
            if params['icp'] > 0.75: recs.append(("attention", "ICP élevé → audit consommation, vérifier pertes réseau"))
            if not recs: recs.append(("normal", "Toutes les pompes sont dans les paramètres nominaux."))
            for niv, msg in recs:
                st.markdown(alerte_item(msg, niv), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    with tab4:
        hist_df = generer_historique(params['heure'])

        an1, an2 = st.columns([1.6, 1])

        with an1:
            st.markdown(section_title("Historique 24h — Coût / Puissance / Débit"), unsafe_allow_html=True)
            fig_hist = chart_historique(hist_df)
            if fig_hist:
                st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
            else:
                st.dataframe(hist_df.set_index('heure'), use_container_width=True)

        with an2:
            st.markdown(section_title("Résumé 24h"), unsafe_allow_html=True)
            cout_total = hist_df['coût_fcfa'].sum()
            cout_hc    = hist_df[hist_df['plage']=='HC']['coût_fcfa'].sum()
            cout_hp    = hist_df[hist_df['plage']=='HP']['coût_fcfa'].sum()
            puiss_moy  = hist_df['puissance_kw'].mean()
            debit_moy  = hist_df['débit_m3h'].mean()

            st.markdown(
                '<div style="display:grid;grid-template-columns:1fr;gap:0.6rem;">'
                + kpi_card("Coût total 24h", f"{cout_total/1e6:.2f}", "M FCFA", "💰", "#00d4ff")
                + kpi_card("Coût HC", f"{cout_hc/1e6:.2f}", "M FCFA", "🟢", "#00e676")
                + kpi_card("Coût HP", f"{cout_hp/1e6:.2f}", "M FCFA", "🔴", "#ff3d57")
                + kpi_card("Puissance moy.", f"{puiss_moy:.0f}", "kW", "⚡", "#00d4ff")
                + '</div>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown(section_title("Données Brutes — Dernières 24h"), unsafe_allow_html=True)
        st.dataframe(
            hist_df.style.applymap(
                lambda v: 'color: #ff3d57' if isinstance(v, str) and v == 'HP' else 'color: #00e676' if isinstance(v, str) and v == 'HC' else ''
            ),
            use_container_width=True, hide_index=True
        )

    # ── Footer ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        margin-top:2rem; padding:1rem 1.5rem;
        border-top:1px solid #1a3a5c;
        display:flex; justify-content:space-between; align-items:center;
    ">
        <div style="font-size:0.62rem; color:#1e4a72; font-family:'Space Mono',monospace;">
            AQUA-AI v2.0 · ONEA Burkina Faso · Système de supervision énergétique
        </div>
        <div style="font-size:0.62rem; color:#1e4a72; font-family:'Space Mono',monospace;">
            Agents actifs : Sécurité · Maintenance · Pompage · Tarif · Réseau
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
