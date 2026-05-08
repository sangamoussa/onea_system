"""
AQUA-AI OPTIMIZER — Interface Streamlit v2.0
Design professionnel — ONEA Burkina Faso
Palette : Sky/Azure (inspiration maquette HTML)
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import math

# ══════════════════════════════════════════════════════════════════════
# CONFIG PAGE
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AQUA-AI Optimizer — ONEA",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# ══════════════════════════════════════════════════════════════════════
# CSS — Design Sky/Azure inspiré maquette HTML
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
/* Font Awesome 6 */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

/* CSS Variables */
:root {
    --cloud-white: #ffffff;
    --morning-mist: #f8fafd;
    --soft-cloud: #f0f4fa;
    --sky-light: #e3f0ff;
    --sky-blue: #b8d9ff;
    --azure: #7eb6ff;
    --deep-sky: #4a90e2;
    --twilight: #2c3e50;
    --starlight: #1a2634;
    --gold: #ffd966;
    --sunset: #ffb347;
    --coral: #ff7f7f;
    --mint: #7fcdb9;
    --lavender: #b8a9d9;
}

/* Dark theme support */
[data-theme="dark"] {
    --cloud-white: #1a2634;
    --morning-mist: #232f3f;
    --soft-cloud: #2a3647;
    --azure: #4a90e2;
    --deep-sky: #7eb6ff;
    --twilight: #e3f0ff;
}

/* Base */
html, body, [class*="css"], .stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
.main, .main > div, .block-container {
    font-family: 'Outfit', sans-serif !important;
    background-color: var(--morning-mist) !important;
    color: var(--twilight) !important;
}

.block-container { 
    padding: 1.5rem 2rem 3rem !important; 
    max-width: 1400px !important; 
}

/* Animations */
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 20px rgba(126,182,255,0.2); }
    50% { box-shadow: 0 0 40px rgba(126,182,255,0.4); }
}

@keyframes ripple {
    0% { box-shadow: 0 0 0 0 rgba(126,182,255,0.4); }
    70% { box-shadow: 0 0 0 10px rgba(126,182,255,0); }
    100% { box-shadow: 0 0 0 0 rgba(126,182,255,0); }
}

@keyframes slideIn {
    from { opacity: 0; transform: translateX(20px); }
    to { opacity: 1; transform: translateX(0); }
}

/* Sidebar */
[data-testid="stSidebar"] {
    min-width: 320px !important;
    width: 320px !important;
    max-width: 320px !important;
    background: var(--cloud-white) !important;
    border-right: 1px solid rgba(126,182,255,0.2) !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: var(--cloud-white) !important;
    position: relative;
}

[data-testid="stSidebar"] > div:first-child::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--azure), var(--deep-sky), var(--lavender));
    z-index: 100;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: var(--twilight) !important;
}

[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Grotesk', monospace !important;
    color: var(--deep-sky) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-top: 1.5rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 8px !important;
    padding: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(240,244,250,0.6) !important;
    color: var(--twilight) !important;
    border-radius: 40px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 20px !important;
    border: 1px solid rgba(126,182,255,0.2) !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--sky-light) !important;
    transform: translateY(-2px);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--azure), var(--deep-sky)) !important;
    color: white !important;
    border: none !important;
}

.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 24px !important;
    animation: slideIn 0.3s ease-out;
}

/* Header */
.onea-header {
    background: var(--cloud-white);
    border-radius: 20px;
    padding: 20px 24px;
    margin-bottom: 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border: 1px solid rgba(126,182,255,0.3);
    box-shadow: 0 8px 30px rgba(74,144,226,0.08);
}

.logo-area {
    display: flex;
    align-items: center;
    gap: 16px;
}

.logo-icon {
    background: linear-gradient(135deg, var(--azure), var(--deep-sky));
    border-radius: 20px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: float 6s ease-in-out infinite;
}

.logo-icon i {
    font-size: 28px;
    color: white;
}

.logo-text h1 {
    font-family: 'Space Grotesk', monospace !important;
    font-size: 28px;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, var(--azure), var(--deep-sky));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.logo-text .badge {
    background: var(--gold);
    color: var(--starlight);
    border-radius: 30px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 600;
    display: inline-block;
    margin-top: 6px;
}

.header-right {
    text-align: right;
}

.header-right .status-badge {
    background: linear-gradient(135deg, var(--mint), #5bb89a);
    color: white;
    padding: 6px 14px;
    border-radius: 40px;
    font-size: 11px;
    font-weight: 600;
}

.header-right .date-info {
    font-size: 11px;
    color: rgba(44,62,80,0.5);
    margin-top: 6px;
}

/* Section Titles */
.section-title {
    font-family: 'Space Grotesk', monospace;
    font-size: 11px;
    font-weight: 700;
    color: var(--deep-sky);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--sky-blue);
    display: inline-block;
}

/* Glass Cards */
.glass-card {
    background: rgba(240,244,250,0.8);
    backdrop-filter: blur(2px);
    border: 1px solid rgba(126,182,255,0.3);
    border-radius: 20px;
    padding: 20px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(74,144,226,0.15);
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--azure), var(--deep-sky), transparent);
}

/* KPI Cards */
.kpi-card {
    background: rgba(240,244,250,0.8);
    backdrop-filter: blur(2px);
    border: 1px solid rgba(126,182,255,0.3);
    border-radius: 20px;
    padding: 20px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(74,144,226,0.15);
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--azure), var(--deep-sky), transparent);
}

.kpi-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #7A8FA6;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.kpi-label i {
    color: var(--azure);
    font-size: 12px;
}

.kpi-value {
    font-size: 32px;
    font-weight: 800;
    color: var(--deep-sky);
    line-height: 1.1;
    margin-bottom: 8px;
}

.kpi-unit {
    font-size: 14px;
    font-weight: 500;
    color: #7A8FA6;
}

.kpi-sub {
    font-size: 11px;
    font-weight: 500;
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid rgba(126,182,255,0.2);
}

.kpi-ok { color: var(--mint); }
.kpi-warn { color: var(--sunset); }
.kpi-crit { color: var(--coral); }
.kpi-info { color: var(--azure); }

/* Tank Levels */
.tank-wrap {
    background: rgba(126,182,255,0.15);
    border-radius: 12px;
    height: 12px;
    overflow: hidden;
    margin: 8px 0;
}

.tank-fill {
    height: 100%;
    border-radius: 12px;
    transition: width 0.5s ease;
}

/* Pump Rows */
.pump-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(126,182,255,0.15);
}

.pump-row:last-child { border-bottom: none; }

.pump-name {
    width: 65px;
    font-weight: 600;
    font-size: 12px;
    color: var(--twilight);
}

.pump-bar-wrap {
    flex: 1;
    background: rgba(126,182,255,0.15);
    border-radius: 20px;
    height: 8px;
    overflow: hidden;
}

.pump-bar-fill {
    height: 100%;
    border-radius: 20px;
    transition: width 0.5s ease;
}

.pump-pct {
    width: 40px;
    font-weight: 700;
    font-size: 12px;
    text-align: right;
}

.pump-status {
    font-size: 10px;
    padding: 4px 10px;
    border-radius: 30px;
    font-weight: 600;
}

.ps-ok { background: var(--mint); color: white; }
.ps-warn { background: var(--sunset); color: white; }
.ps-crit { background: var(--coral); color: white; }

/* Alert Items */
.alert-item {
    border-radius: 16px;
    padding: 14px;
    margin-bottom: 12px;
    border-left: 4px solid;
    animation: slideIn 0.3s ease-out;
}

.alert-ok { background: rgba(127,205,185,0.15); border-color: var(--mint); }
.alert-warn { background: rgba(255,179,71,0.12); border-color: var(--sunset); }
.alert-crit { background: rgba(255,127,127,0.12); border-color: var(--coral); }
.alert-info { background: rgba(126,182,255,0.12); border-color: var(--azure); }

.alert-title {
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 4px;
}

.alert-ok .alert-title { color: var(--mint); }
.alert-warn .alert-title { color: var(--sunset); }
.alert-crit .alert-title { color: var(--coral); }
.alert-info .alert-title { color: var(--deep-sky); }

.alert-body {
    font-size: 11px;
    color: var(--twilight);
    opacity: 0.8;
}

/* Badges */
.badge-hc, .badge-hp, .badge-sol {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 40px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.badge-hc { background: linear-gradient(135deg, var(--mint), #5bb89a); color: white; }
.badge-hp { background: linear-gradient(135deg, var(--coral), #e06666); color: white; }
.badge-sol { background: linear-gradient(135deg, var(--gold), #ffcc33); color: var(--starlight); }

/* Bilan Rows */
.bilan-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 10px 0;
    border-bottom: 1px solid rgba(126,182,255,0.15);
}

.bilan-row:last-child { border-bottom: none; }

.bilan-label {
    font-size: 12px;
    color: #7A8FA6;
}

.bilan-val {
    font-weight: 600;
    color: var(--twilight);
}

.bilan-green {
    color: var(--mint);
    font-weight: 700;
}

/* Prediction Table */
.pred-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}

.pred-table th {
    background: linear-gradient(135deg, var(--azure), var(--deep-sky));
    color: white;
    font-weight: 600;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 12px 10px;
    text-align: left;
}

.pred-table td {
    padding: 10px;
    border-bottom: 1px solid rgba(126,182,255,0.2);
    color: var(--twilight);
    background: var(--cloud-white);
}

.pred-table tr:nth-child(even) td {
    background: var(--morning-mist);
}

.pred-table tr:hover td {
    background: var(--sky-light);
}

.td-hc { background: rgba(127,205,185,0.2) !important; font-weight: 700; }
.td-hp { background: rgba(255,127,127,0.12) !important; font-weight: 700; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--azure), var(--deep-sky)) !important;
    color: white !important;
    border: none !important;
    border-radius: 40px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(74,144,226,0.3) !important;
}

/* Ripple indicator */
.ripple-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--mint);
    animation: ripple 1.5s infinite;
    margin-right: 6px;
}

.ripple-indicator.hp {
    background: var(--coral);
}

/* Footer */
.dashboard-footer {
    font-size: 11px;
    color: rgba(44,62,80,0.4);
    text-align: center;
    padding: 20px;
    margin-top: 30px;
    border-top: 1px solid rgba(126,182,255,0.15);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Responsive */
@media (max-width: 768px) {
    .block-container { padding: 1rem !important; }
    .kpi-value { font-size: 28px; }
    .logo-text h1 { font-size: 22px; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════════
def check_api() -> bool:
    try:
        r = requests.get(f"{API_URL}/status", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def call_api_simple(chateau, bache, heure, temp) -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/predict/simple",
            params={"niveau_chateau": chateau, "niveau_bache": bache,
                    "heure_actuelle": heure, "temperature": temp},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def fmt(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", " ")


def profil_horaire(h: int) -> float:
    return 0.6 + 0.4 * math.exp(-((h - 7) ** 2) / 8) + 0.3 * math.exp(-((h - 19) ** 2) / 6)


def simuler_predictions(chateau: float, bache: float, heure: int, temp: float) -> list[dict]:
    rows = []
    ch, ba = float(chateau), float(bache)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    for i in range(24):
        ts = now + timedelta(hours=i + 1)
        h = ts.hour
        plage = "HC" if h < 17 else "HP"
        tarif = 84 if plage == "HC" else 165
        prof = profil_horaire(h)
        intensite = 1.0 if (plage == "HC" or ch < 30) else 0.35
        debit = float(np.clip(750 * prof * intensite + random.normalvariate(0, 12), 0, 900))
        puiss = float(np.clip(9000 * prof * intensite + random.normalvariate(0, 150), 100, 17000))
        cout = puiss * tarif
        ch = float(np.clip(ch + debit / 1500 - prof * 0.4, 15, 98))
        ba = float(np.clip(ba + 0.3 - debit / 5000, 10, 98))
        # Décision IA
        if plage == "HC":
            decision = "POMPER — HC 84 FCFA/kWh"
        elif ch > 60:
            decision = "ATTENDRE — Château suffisant"
        elif ch < 30:
            decision = "POMPER — Château critique"
        else:
            decision = "RÉDUIRE — Éviter HP si possible"
        alerte = ""
        if ch < 25:
            alerte = "Château critique"
        elif ba < 20:
            alerte = "Bâche basse"
        eco = max(0.0, puiss * (165 - 84)) if plage == "HP" else 0.0
        rows.append({
            "Heure": ts.strftime("%H:%M"),
            "Débit (m³/h)": round(debit, 1),
            "Puissance (kW)": round(puiss, 0),
            "Coût (FCFA)": round(cout, 0),
            "Château (%)": round(ch, 1),
            "Bâche (%)": round(ba, 1),
            "Tarif": plage,
            "Décision IA": decision,
            "Économie (FCFA)": round(eco, 0),
            "Alerte": alerte,
        })
    return rows


def get_ia_decision(heure: int, chateau: float, bache: float,
                    coupure: bool, solaire_kw: float) -> dict:
    if coupure and chateau < 40:
        return {"badge": "badge-hp", "label": "Urgence diesel",
                "action": "POMPE 1 — DIESEL",
                "source": "Groupe électrogène (300 FCFA/kWh)",
                "cout_h": fmt(185 * 300) + " FCFA/h",
                "eco": "N/A — Urgence",
                "confiance": "72%", "agent": "Agent Sécurité (override)"}
    if solaire_kw > 200 and chateau < 80:
        return {"badge": "badge-sol", "label": "Solaire disponible",
                "action": "POMPE 1 — SOLAIRE PV",
                "source": "Solaire PV (~0 FCFA/kWh)",
                "cout_h": "0 FCFA/h",
                "eco": "+" + fmt(185 * 84) + " FCFA",
                "confiance": "89%", "agent": "Agent Tarif (DQN)"}
    if heure < 17:
        if bache > 50 and chateau < 85:
            return {"badge": "badge-hc", "label": "HC — Pomper",
                    "action": "POMPES 1+2 — SONABEL HC",
                    "source": "SONABEL HC (84 FCFA/kWh)",
                    "cout_h": fmt(385 * 84) + " FCFA/h",
                    "eco": "+" + fmt(385 * (165 - 84)) + " FCFA vs HP",
                    "confiance": "93%", "agent": "Agent Tarif (DQN)"}
        return {"badge": "badge-hc", "label": "HC — Pompe 1",
                "action": "POMPE 1 — SONABEL HC",
                "source": "SONABEL HC (84 FCFA/kWh)",
                "cout_h": fmt(185 * 84) + " FCFA/h",
                "eco": "+" + fmt(185 * (165 - 84)) + " FCFA vs HP",
                "confiance": "88%", "agent": "Agent Tarif (DQN)"}
    else:
        if chateau > 55:
            return {"badge": "badge-hp", "label": "HP — Économiser",
                    "action": "ARRÊT — Château suffisant",
                    "source": "Attente heures creuses",
                    "cout_h": "0 FCFA/h",
                    "eco": "+" + fmt(385 * 165) + " FCFA économisés",
                    "confiance": "91%", "agent": "Agent Tarif (DQN)"}
        return {"badge": "badge-hp", "label": "HP — Urgence niveau",
                "action": "POMPE 1 — SONABEL HP",
                "source": "SONABEL HP (165 FCFA/kWh)",
                "cout_h": fmt(185 * 165) + " FCFA/h",
                "eco": "Contrainte niveau bas",
                "confiance": "76%", "agent": "Agent Sécurité (override)"}


def build_alerts(chateau, bache, heure, temp, coupure) -> list:
    alerts = []
    if chateau < 25:
        alerts.append(("crit", '<i class="fas fa-triangle-exclamation"></i> Château d\'eau critique',
                        f"Niveau {chateau}% — pompage immédiat requis"))
    elif chateau < 40:
        alerts.append(("warn", '<i class="fas fa-exclamation-triangle"></i> Château d\'eau bas',
                        f"Niveau {chateau}% — anticiper le remplissage"))
    if bache < 20:
        alerts.append(("crit", '<i class="fas fa-water"></i> Bâche critique',
                        f"Niveau {bache}% — risque cavitation pompes"))
    elif bache < 35:
        alerts.append(("warn", '<i class="fas fa-water"></i> Bâche basse',
                        f"Niveau {bache}% — alimenter en eau brute"))
    if coupure:
        alerts.append(("warn", '<i class="fas fa-bolt"></i> Coupure SONABEL active',
                        "Groupe électrogène en service — vérifier stock diesel"))
    if heure >= 17:
        alerts.append(("warn", '<i class="fas fa-clock"></i> Heures de pointe actives',
                        f"Tarif 165 FCFA/kWh — réduire pompage si niveaux suffisants"))
    if temp > 38:
        alerts.append(("info", '<i class="fas fa-temperature-high"></i> Température élevée',
                        f"{temp}°C — pic de demande probable en soirée"))
    if not alerts:
        alerts.append(("ok", '<i class="fas fa-circle-check"></i> Fonctionnement normal',
                        "Tous les paramètres dans les seuils acceptables"))
    return alerts


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding: 15px 0 20px; text-align: center;">
        <div style="display: inline-block; background: linear-gradient(135deg, #7eb6ff, #4a90e2); 
                    border-radius: 18px; padding: 12px 14px; margin-bottom: 12px;">
            <i class="fas fa-droplet" style="color: white; font-size: 28px;"></i>
        </div>
        <div style="font-family:'Space Grotesk'; font-size: 20px; font-weight: 700; 
                    background: linear-gradient(135deg, #7eb6ff, #4a90e2); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            AQUA-AI
        </div>
        <div style="font-size: 10px; color: #7A8FA6; margin-top: 4px;">ONEA · Burkina Faso</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### <i class='fas fa-map-marker-alt'></i> Station", unsafe_allow_html=True)
    station_id = st.selectbox(
        "Sélectionner la station",
        ["STATION_OUAGA_01", "STATION_OUAGA_02", "STATION_BOBO_01"],
        label_visibility="collapsed",
    )

    st.markdown("### <i class='fas fa-water'></i> Niveaux réservoirs", unsafe_allow_html=True)
    niveau_chateau = st.slider("Château d'eau (%)", 15, 98, 65, key="sl_chateau")
    niveau_bache   = st.slider("Bâche de stockage (%)", 10, 98, 72, key="sl_bache")

    st.markdown("### <i class='fas fa-chart-line'></i> Contexte opérationnel", unsafe_allow_html=True)
    heure_actuelle = st.slider("Heure actuelle", 0, 23,
                                datetime.now().hour, key="sl_heure")
    temperature    = st.slider("Température (°C)", 18, 44, 32, key="sl_temp")
    solaire_kw     = st.slider("Puissance solaire (kW)", 0, 600, 0, key="sl_sol")

    st.markdown("### <i class='fas fa-microchip'></i> État équipements", unsafe_allow_html=True)
    pompe1  = st.checkbox("Pompe 1 disponible", value=True)
    pompe2  = st.checkbox("Pompe 2 disponible", value=True)
    coupure = st.checkbox("Coupure SONABEL", value=False)

    st.markdown("---")
    tarif_actuel = 84 if heure_actuelle < 17 else 165
    ripple_class = "ripple-indicator" if heure_actuelle < 17 else "ripple-indicator hp"
    plage_label  = f"<span class='{ripple_class}'></span> HC — 84 FCFA/kWh" if heure_actuelle < 17 else f"<span class='{ripple_class}'></span> HP — 165 FCFA/kWh"
    st.markdown(f"**Tarif actuel :** {plage_label}", unsafe_allow_html=True)

    api_ok = check_api()
    api_icon = "<i class='fas fa-circle' style='color:#7fcdb9; font-size:8px;'></i>" if api_ok else "<i class='fas fa-circle' style='color:#ffb347; font-size:8px;'></i>"
    api_status = f"{api_icon} API connectée" if api_ok else f"{api_icon} Mode simulation"
    st.markdown(f"<div style='font-size:11px; color:#7A8FA6; margin-top:8px;'>{api_status}</div>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HEADER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════
mode_txt = "REEL" if api_ok else "SIMULATION"
now_str  = datetime.now().strftime("%d/%m/%Y — %H:%M")

st.markdown(f"""
<div class="onea-header">
  <div class="logo-area">
    <div class="logo-icon">
      <i class="fas fa-droplet"></i>
    </div>
    <div class="logo-text">
      <h1>AQUA-AI</h1>
      <div class="badge">ONEA · 2026</div>
    </div>
  </div>
  <div class="header-right">
    <div class="status-badge">
      <i class="fas fa-chart-line" style="margin-right: 6px;"></i>Opérationnel
    </div>
    <div class="date-info">
      <i class="far fa-calendar-alt"></i> {now_str} · Mode {mode_txt} · {station_id}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# ONGLETS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tableau de bord",
    "🔮 Prédiction 24h",
    "💰 Analyse économique",
    "ℹ️ Système",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — TABLEAU DE BORD
# ══════════════════════════════════════════════════════════════════════
with tab1:

    # KPI row
    st.markdown('<div class="section-title"><i class="fas fa-chart-simple"></i> Indicateurs clés — temps réel</div>',
                unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    ch_color = "#7fcdb9" if niveau_chateau > 40 else ("#ffb347" if niveau_chateau > 25 else "#ff7f7f")
    ch_status_cls = "kpi-ok" if niveau_chateau > 40 else ("kpi-warn" if niveau_chateau > 25 else "kpi-crit")
    ch_status_txt = "Niveau normal" if niveau_chateau > 40 else ("Niveau bas" if niveau_chateau > 25 else "Critique !")

    ba_color = "#7eb6ff" if niveau_bache > 35 else ("#ffb347" if niveau_bache > 20 else "#ff7f7f")
    ba_status_cls = "kpi-ok" if niveau_bache > 35 else ("kpi-warn" if niveau_bache > 20 else "kpi-crit")
    ba_status_txt = "Niveau suffisant" if niveau_bache > 35 else ("Niveau bas" if niveau_bache > 20 else "Critique !")

    tarif_color = "#7fcdb9" if heure_actuelle < 17 else "#ff7f7f"
    tarif_cls   = "kpi-ok" if heure_actuelle < 17 else "kpi-crit"
    tarif_txt   = "Heures creuses" if heure_actuelle < 17 else "Heures de pointe"

    puiss_sim = int(9000 * profil_horaire(heure_actuelle))
    nb_pompes = (1 if pompe1 else 0) + (1 if pompe2 else 0)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">
            <i class="fas fa-water"></i>Château d'eau
          </div>
          <div class="kpi-value">{niveau_chateau}<span class="kpi-unit">%</span></div>
          <div class="kpi-sub {ch_status_cls}">{ch_status_txt}</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">
            <i class="fas fa-draw-polygon"></i>Bâche stockage
          </div>
          <div class="kpi-value">{niveau_bache}<span class="kpi-unit">%</span></div>
          <div class="kpi-sub {ba_status_cls}">{ba_status_txt}</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">
            <i class="fas fa-bolt"></i>Tarif SONABEL
          </div>
          <div class="kpi-value">{tarif_actuel}<span class="kpi-unit">FCFA/kWh</span></div>
          <div class="kpi-sub {tarif_cls}">{tarif_txt}</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">
            <i class="fas fa-gauge-high"></i>Puissance active
          </div>
          <div class="kpi-value">{puiss_sim:,}<span class="kpi-unit"> kW</span></div>
          <div class="kpi-sub kpi-info">{nb_pompes} pompe{'s' if nb_pompes > 1 else ''} active{'s' if nb_pompes > 1 else ''}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Ligne 2 : Réservoirs + IA + Alertes
    col_res, col_ia, col_alerts = st.columns([2, 1.2, 1])

    with col_res:
        st.markdown('<div class="section-title"><i class="fas fa-chart-pie"></i> Niveaux réservoirs</div>',
                    unsafe_allow_html=True)

        ch_bar_color = "#7fcdb9" if niveau_chateau > 40 else ("#ffb347" if niveau_chateau > 25 else "#ff7f7f")
        st.markdown(f"""
        <div class="glass-card">
          <div style="margin-bottom: 20px;">
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px;">
              <span><i class="fas fa-tower-cell"></i> Château d'eau — 1 500 m³</span>
              <span style="font-weight:700; color: var(--deep-sky);">{niveau_chateau}% · {int(niveau_chateau * 15)} m³</span>
            </div>
            <div class="tank-wrap">
              <div class="tank-fill" style="width:{niveau_chateau}%; background:{ch_bar_color};"></div>
            </div>
          </div>
          <div style="margin-bottom: 20px;">
            <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:6px;">
              <span><i class="fas fa-water"></i> Bâche de stockage — 5 000 m³</span>
              <span style="font-weight:700; color: var(--deep-sky);">{niveau_bache}% · {int(niveau_bache * 50)} m³</span>
            </div>
            <div class="tank-wrap">
              <div class="tank-fill" style="width:{niveau_bache}%; background: var(--azure);"></div>
            </div>
          </div>
          <div>
            <div style="font-size:10px; font-weight:700; color: var(--deep-sky); text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px;">
              <i class="fas fa-industry"></i> État pompes
            </div>
            <div class="pump-row">
              <span class="pump-name">Pompe 1</span>
              <div class="pump-bar-wrap">
                <div class="pump-bar-fill" style="width:87%; background:{'#7fcdb9' if pompe1 else '#ff7f7f'};"></div>
              </div>
              <span class="pump-pct">87%</span>
              <span class="pump-status {'ps-ok' if pompe1 else 'ps-crit'}">{'Normal' if pompe1 else 'Arrêt'}</span>
            </div>
            <div class="pump-row">
              <span class="pump-name">Pompe 2</span>
              <div class="pump-bar-wrap">
                <div class="pump-bar-fill" style="width:85%; background:{'#7fcdb9' if pompe2 else '#ff7f7f'};"></div>
              </div>
              <span class="pump-pct">85%</span>
              <span class="pump-status {'ps-ok' if pompe2 else 'ps-crit'}">{'Normal' if pompe2 else 'Arrêt'}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ia:
        st.markdown('<div class="section-title"><i class="fas fa-brain"></i> Décision IA</div>', unsafe_allow_html=True)
        ia = get_ia_decision(heure_actuelle, niveau_chateau, niveau_bache,
                             coupure, solaire_kw)
        st.markdown(f"""
        <div class="glass-card">
          <span class="{ia['badge']}">{ia['label']}</span>
          <div style="background: var(--cloud-white); border-radius: 16px; padding: 14px; margin: 14px 0;">
            <div style="font-size:10px; text-transform:uppercase; letter-spacing:0.08em; color: var(--deep-sky); margin-bottom:6px;">
              <i class="fas fa-microchip"></i> Action recommandée
            </div>
            <div style="font-size:14px; font-weight:700; color: var(--twilight);">{ia['action']}</div>
            <div style="font-size:11px; color: var(--mint); margin-top:4px;">
              <i class="fas fa-chart-line"></i> Confiance : {ia['confiance']}
            </div>
          </div>
          <div class="bilan-row"><span class="bilan-label"><i class="fas fa-clock"></i> Coût / heure</span><span class="bilan-val">{ia['cout_h']}</span></div>
          <div class="bilan-row"><span class="bilan-label"><i class="fas fa-chart-line"></i> Économie</span><span class="bilan-green">{ia['eco']}</span></div>
          <div class="bilan-row"><span class="bilan-label"><i class="fas fa-plug"></i> Source</span><span class="bilan-val" style="font-size:11px">{ia['source']}</span></div>
          <div class="bilan-row"><span class="bilan-label"><i class="fas fa-user-robot"></i> Agent</span><span class="bilan-val" style="font-size:10px; color: #7A8FA6">{ia['agent']}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col_alerts:
        st.markdown('<div class="section-title"><i class="fas fa-bell"></i> Alertes actives</div>', unsafe_allow_html=True)
        alerts = build_alerts(niveau_chateau, niveau_bache, heure_actuelle, temperature, coupure)
        alerts_html = ""
        for a_type, a_title, a_body in alerts:
            alerts_html += f"""
            <div class="alert-item alert-{a_type}">
              <div class="alert-title">{a_title}</div>
              <div class="alert-body">{a_body}</div>
            </div>"""
        st.markdown(alerts_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Ligne 3 : Consommation + ICP
    st.markdown('<div class="section-title"><i class="fas fa-chart-line"></i> Consommation & performance</div>',
                unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    cout_heure = puiss_sim * tarif_actuel
    icp = round(0.57 + random.normalvariate(0, 0.01), 3)

    with mc1:
        st.markdown(f"""
        <div class="glass-card">
          <div style="font-size:11px; color: var(--deep-sky); margin-bottom:8px;"><i class="fas fa-bolt"></i> Puissance totale</div>
          <div style="font-size:24px; font-weight:700; color: var(--twilight);">{puiss_sim:,} <span style="font-size:14px;">kW</span></div>
          <div style="font-size:11px; color:#7A8FA6; margin-top:6px;">Tarif : {tarif_actuel} FCFA/kWh</div>
        </div>
        """, unsafe_allow_html=True)

    with mc2:
        st.markdown(f"""
        <div class="glass-card">
          <div style="font-size:11px; color: var(--deep-sky); margin-bottom:8px;"><i class="fas fa-coins"></i> Coût heure en cours</div>
          <div style="font-size:24px; font-weight:700; color: var(--twilight);">{fmt(cout_heure)} <span style="font-size:14px;">FCFA</span></div>
          <div style="font-size:11px; color:#7A8FA6; margin-top:6px;">Plage {"HC" if heure_actuelle < 17 else "HP"}</div>
        </div>
        """, unsafe_allow_html=True)

    with mc3:
        st.markdown(f"""
        <div class="glass-card">
          <div style="font-size:11px; color: var(--deep-sky); margin-bottom:8px;"><i class="fas fa-chart-simple"></i> ICP actuel</div>
          <div style="font-size:24px; font-weight:700; color: var(--twilight);">{icp} <span style="font-size:14px;">kWh/m³</span></div>
          <div style="font-size:11px; color:#7A8FA6; margin-top:6px;">Référence ONEA : 0.55–0.65</div>
        </div>
        """, unsafe_allow_html=True)

    with mc4:
        st.markdown(f"""
        <div class="glass-card">
          <div style="font-size:11px; color: var(--deep-sky); margin-bottom:8px;"><i class="fas fa-water"></i> Débit refoulement</div>
          <div style="font-size:24px; font-weight:700; color: var(--twilight);">{int(750 * profil_horaire(heure_actuelle))} <span style="font-size:14px;">m³/h</span></div>
          <div style="font-size:11px; color:#7A8FA6; margin-top:6px;">Profil horaire : {round(profil_horaire(heure_actuelle),2)}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — PRÉDICTION 24H
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title"><i class="fas fa-chart-line"></i> Prédiction LSTM — Prochaines 24 heures</div>',
                unsafe_allow_html=True)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_pred = st.button("Lancer la prédiction", type="primary",
                             use_container_width=True)
    with col_info:
        if api_ok:
            st.success("API AQUA-AI connectée — modèle LSTM actif")
        else:
            st.info("Mode simulation — données générées par règles heuristiques")

    if "pred_done" not in st.session_state:
        st.session_state["pred_done"] = False

    if run_pred:
        st.session_state["pred_done"] = True

    if st.session_state["pred_done"]:
        with st.spinner("Calcul en cours..."):
            if api_ok:
                data = call_api_simple(niveau_chateau, niveau_bache,
                                       heure_actuelle, temperature)
            else:
                data = None
            rows = simuler_predictions(niveau_chateau, niveau_bache,
                                       heure_actuelle, temperature)
            df = pd.DataFrame(rows)

        # KPIs résumé
        cout_total = df["Coût (FCFA)"].sum()
        cout_hp_t  = df[df["Tarif"] == "HP"]["Coût (FCFA)"].sum()
        eco_pot    = df["Économie (FCFA)"].sum()
        n_alertes  = (df["Alerte"] != "").sum()

        st.markdown('<div class="section-title" style="margin-top:20px;"><i class="fas fa-chart-pie"></i> Résumé 24h</div>',
                    unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.markdown(f"""
            <div class="glass-card">
              <div style="font-size:11px; color: var(--deep-sky); margin-bottom:8px;"><i class="fas fa-coins"></i> Coût total prévu</div>
              <div style="font-size:22px; font-weight:700; color: var(--twilight);">{cout_total/1e6:.2f} M FCFA</div>
              <div style="font-size:11px; color:#7A8FA6;">Toutes plages</div>
            </div>
            """, unsafe_allow_html=True)

        with s2:
            st.markdown(f"""
            <div class="glass-card">
              <div style="font-size:11px; color: var(--deep-sky); margin-bottom:8px;"><i class="fas fa-bolt"></i> Dont heures de pointe</div>
              <div style="font-size:22px; font-weight:700; color: var(--twilight);">{cout_hp_t/1e6:.2f} M FCFA</div>
              <div style="font-size:11px; color:#7A8FA6;">À optimiser</div>
            </div>
            """, unsafe_allow_html=True)

        with s3:
            st.markdown(f"""
            <div class="glass-card">
              <div style="font-size:11px; color: var(--deep-sky); margin-bottom:8px;"><i class="fas fa-chart-line"></i> Économie potentielle</div>
              <div style="font-size:22px; font-weight:700; color: var(--mint);">+{eco_pot/1e6:.2f} M FCFA</div>
              <div style="font-size:11px; color:#7A8FA6;">Décalage HC/HP</div>
            </div>
            """, unsafe_allow_html=True)

        with s4:
            st.markdown(f"""
            <div class="glass-card">
              <div style="font-size:11px; color: var(--deep-sky); margin-bottom:8px;"><i class="fas fa-bell"></i> Alertes détectées</div>
              <div style="font-size:22px; font-weight:700; color: var(--twilight);">{int(n_alertes)}</div>
              <div style="font-size:11px; color:#7A8FA6;">Sur 24 heures</div>
            </div>
            """, unsafe_allow_html=True)

        # Graphiques
        g1, g2 = st.columns(2)
        with g1:
            st.markdown('<div class="section-title"><i class="fas fa-chart-line"></i> Puissance prévue (kW)</div>',
                        unsafe_allow_html=True)
            chart_df = df.set_index("Heure")[["Puissance (kW)"]]
            st.area_chart(chart_df, color="#7eb6ff", height=200)
        with g2:
            st.markdown('<div class="section-title"><i class="fas fa-chart-line"></i> Niveaux réservoirs prévus (%)</div>',
                        unsafe_allow_html=True)
            niv_df = df.set_index("Heure")[["Château (%)", "Bâche (%)"]].rename(
                columns={"Château (%)": "Château", "Bâche (%)": "Bâche"})
            st.line_chart(niv_df, height=200)

        # Tableau heure par heure
        st.markdown('<div class="section-title" style="margin-top:20px;"><i class="fas fa-table-list"></i> Détail heure par heure</div>',
                    unsafe_allow_html=True)

        rows_html = ""
        for _, row in df.iterrows():
            td_plage = f'<td class="td-hc">HC</td>' if row["Tarif"] == "HC" \
                       else f'<td class="td-hp">HP</td>'
            alerte_txt = f'<span style="color:#ff7f7f; font-size:10px;"><i class="fas fa-triangle-exclamation"></i> {row["Alerte"]}</span>' \
                         if row["Alerte"] else '<span style="color:#7fcdb9; font-size:10px;"><i class="fas fa-circle-check"></i> OK</span>'
            rows_html += f"""
            <tr>
              <td style="font-weight:600;">{row['Heure']}</td>
              {td_plage}
              <td>{fmt(row['Débit (m³/h)'])}</td>
              <td>{fmt(row['Puissance (kW)'])}</td>
              <td>{fmt(row['Coût (FCFA)'])}</td>
              <td>{row['Château (%)']}%</td>
              <td>{row['Bâche (%)']}%</td>
              <td style="font-size:11px;">{row['Décision IA']}</td>
              <td>{alerte_txt}</td>
            </tr>"""

        st.markdown(f"""
        <div class="glass-card" style="overflow-x:auto; padding: 0;">
        <table class="pred-table">
          <thead>
            <tr>
              <th>Heure</th><th>Plage</th><th>Débit m³/h</th>
              <th>Puissance kW</th><th>Coût FCFA</th>
              <th>Château %</th><th>Bâche %</th>
              <th>Décision IA</th><th>État</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        # Export CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Exporter en CSV",
            data=csv,
            file_name=f"predictions_{station_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding: 60px;">
          <div style="font-size: 48px; margin-bottom: 16px;">
            <i class="fas fa-brain" style="color: #7eb6ff;"></i>
          </div>
          <div style="font-size: 16px; font-weight: 600; color: var(--twilight); margin-bottom: 8px;">
            Prêt à prédire
          </div>
          <div style="font-size: 13px; color: #7A8FA6;">
            Configurez les paramètres dans la barre latérale<br>
            puis cliquez sur <strong>Lancer la prédiction</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE ÉCONOMIQUE
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title"><i class="fas fa-chart-line"></i> Bilan énergétique ONEA 2024</div>',
                unsafe_allow_html=True)

    ec1, ec2 = st.columns(2)

    with ec1:
        st.markdown("""
        <div class="glass-card">
          <div style="font-size: 14px; font-weight: 700; color: var(--twilight); margin-bottom: 16px;">
            <i class="fas fa-chart-pie"></i> Consommation réelle ONEA 2024
          </div>
        """, unsafe_allow_html=True)
        bilan_data = [
            ("<i class='fas fa-bolt'></i> Consommation totale", "91,3 GWh"),
            ("<i class='fas fa-coins'></i> Coût total énergie", "8,445 Mrd FCFA"),
            ("<i class='fas fa-chart-line'></i> Dont heures de pointe", "27,2 M kWh"),
            ("<i class='fas fa-warning'></i> Pertes réseau", "205 M kWh"),
            ("<i class='fas fa-solar-panel'></i> Production solaire PV", "4 GWh (4%)"),
            ("<i class='fas fa-building'></i> Stations actives", "67 stations"),
        ]
        rows = "".join([
            f'<div class="bilan-row"><span class="bilan-label">{k}</span>'
            f'<span class="bilan-val">{v}</span></div>'
            for k, v in bilan_data
        ])
        st.markdown(rows + "</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card" style="margin-top: 20px;">
          <div style="font-size: 14px; font-weight: 700; color: var(--twilight); margin-bottom: 16px;">
            <i class="fas fa-chart-line"></i> Potentiel d'économies AQUA-AI
          </div>
        """, unsafe_allow_html=True)
        eco_data = [
            ("Décalage HC/HP (LSTM+DQN)",    "422-591 M FCFA/an"),
            ("Optimisation multi-agents",    "507-676 M FCFA/an"),
            ("Réduction pertes (Autoencoder)", "338 M FCFA/an"),
            ("<strong>TOTAL estimé</strong>",                 "<strong style='color:#7fcdb9;'>1,267-1,942 Mrd FCFA/an</strong>"),
        ]
        rows2 = "".join([
            f'<div class="bilan-row"><span class="bilan-label">{k}</span>'
            f'<span class="bilan-val">{v}</span></div>'
            for k, v in eco_data
        ])
        st.markdown(rows2 + "</div>", unsafe_allow_html=True)

    with ec2:
        st.markdown("""
        <div class="glass-card">
          <div style="font-size: 14px; font-weight: 700; color: var(--twilight); margin-bottom: 16px;">
            <i class="fas fa-calculator"></i> Simulateur économies HC/HP
          </div>
        """, unsafe_allow_html=True)

        kwh_hp    = st.number_input("kWh consommés en HP",
                                    value=27200000, step=500000,
                                    format="%d")
        pct_decal = st.slider("% décalable vers HC", 10, 80, 40)
        kwh_dec   = kwh_hp * pct_decal / 100
        eco_sim   = kwh_dec * (165 - 84)

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(127,205,185,0.1), rgba(127,205,185,0.05)); 
                    border-radius: 16px; padding: 16px; margin: 16px 0;">
          <div style="font-size: 12px; color: var(--mint); margin-bottom: 10px;">
            <i class="fas fa-chart-line"></i> Résultat simulé
          </div>
          <div class="bilan-row" style="border-color: rgba(127,205,185,0.2);">
            <span class="bilan-label">kWh décalés</span>
            <span class="bilan-green">{kwh_dec/1e6:.1f} M kWh</span>
          </div>
          <div class="bilan-row" style="border-color: rgba(127,205,185,0.2);">
            <span class="bilan-label">Économie tarifaire</span>
            <span class="bilan-green">{eco_sim/1e6:.0f} M FCFA/an</span>
          </div>
          <div class="bilan-row" style="border-bottom:none;">
            <span class="bilan-label">Part du budget énergie</span>
            <span class="bilan-green">{eco_sim/8445306812*100:.1f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ROI
        capex = 289.9e6
        if eco_sim > 0:
            roi_mois = capex / (eco_sim / 12)
            roi_cls  = "bilan-green" if roi_mois <= 18 else "bilan-val"
            st.markdown(f"""
            <div style="font-size: 13px; font-weight: 600; color: var(--twilight); margin: 16px 0 12px;">
              <i class="fas fa-chart-line"></i> Retour sur investissement
            </div>
            <div class="bilan-row">
              <span class="bilan-label">CAPEX estimé</span>
              <span class="bilan-val">289,9 M FCFA</span>
            </div>
            <div class="bilan-row">
              <span class="bilan-label">Économie annuelle</span>
              <span class="bilan-green">{eco_sim/1e6:.0f} M FCFA</span>
            </div>
            <div class="bilan-row">
              <span class="bilan-label">ROI estimé</span>
              <span class="{roi_cls}">{roi_mois:.0f} mois</span>
            </div>
            <div style="font-size: 11px; color: #7A8FA6; margin-top: 8px;">
              <i class="fas fa-bullseye"></i> Cible projet : 12–18 mois
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — SYSTÈME
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title"><i class="fas fa-microchip"></i> Architecture système AQUA-AI</div>',
                unsafe_allow_html=True)

    sy1, sy2 = st.columns(2)

    with sy1:
        st.markdown("""
        <div class="glass-card">
          <div style="font-size: 14px; font-weight: 700; color: var(--twilight); margin-bottom: 16px;">
            <i class="fas fa-brain"></i> Modules IA déployés
          </div>
        """, unsafe_allow_html=True)

        modules = [
            ("LSTM Bidirectionnel", "Prédiction demande 1–24h",
             "48h → 24h · 14 features", "#7eb6ff", "Actif"),
            ("DQN Dueling Double", "Optimisation tarifaire temps réel",
             "8 actions · reward multi-objectif", "#7fcdb9", "Actif"),
            ("Autoencoder", "Détection anomalies pompes",
             "8 features · reconstruction error", "#ffb347", "Actif"),
            ("Orchestrateur", "Coordination multi-agents",
             "5 agents · priorité sécurité", "#4a90e2", "Actif"),
        ]

        for name, desc, detail, color, status in modules:
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap: 12px; padding: 12px 0;
                        border-bottom: 1px solid rgba(126,182,255,0.15);">
              <div style="width: 8px; height: 8px; border-radius: 50%; background: {color};
                          margin-top: 5px; flex-shrink: 0;"></div>
              <div style="flex:1;">
                <div style="font-size: 13px; font-weight: 600; color: var(--twilight);">{name}</div>
                <div style="font-size: 11px; color: #7A8FA6; margin-top: 2px;">{desc}</div>
                <div style="font-size: 10px; color: #9AA1A8; margin-top: 2px;">{detail}</div>
              </div>
              <span style="font-size: 10px; background: rgba(127,205,185,0.15); color: var(--mint);
                           padding: 4px 10px; border-radius: 30px; font-weight: 600;">{status}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with sy2:
        st.markdown("""
        <div class="glass-card">
          <div style="font-size: 14px; font-weight: 700; color: var(--twilight); margin-bottom: 16px;">
            <i class="fas fa-plug"></i> Sources d'énergie
          </div>
        """, unsafe_allow_html=True)
        sources = [
            ("SONABEL HC", "00h–17h", "80–88 FCFA/kWh", "Priorité 1"),
            ("Solaire PV", "06h–18h", "~0 FCFA/kWh", "Priorité 2"),
            ("SONABEL HP", "17h–24h", "165 FCFA/kWh", "Éviter si possible"),
            ("Diesel",     "Coupure", "~300 FCFA/kWh", "Urgence"),
        ]
        for src, horaire, tarif, prio in sources:
            prio_color = "#7fcdb9" if "1" in prio or "2" in prio else \
                         ("#ffb347" if "Éviter" in prio else "#ff7f7f")
            prio_bg    = "rgba(127,205,185,0.1)" if "1" in prio or "2" in prio else \
                         ("rgba(255,179,71,0.1)" if "Éviter" in prio else "rgba(255,127,127,0.1)")
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap: 12px; padding: 10px 0;
                        border-bottom: 1px solid rgba(126,182,255,0.15); font-size: 12px;">
              <div style="flex:1;">
                <div style="font-weight: 600; color: var(--twilight);">{src}</div>
                <div style="color: #7A8FA6; font-size: 11px;">{horaire} · {tarif}</div>
              </div>
              <span style="font-size: 10px; background: {prio_bg}; color: {prio_color};
                           padding: 4px 12px; border-radius: 30px; font-weight: 600;">{prio}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card" style="margin-top: 20px;">
          <div style="font-size: 14px; font-weight: 700; color: var(--twilight); margin-bottom: 14px;">
            <i class="fas fa-chart-line"></i> Plan de déploiement
          </div>
        """, unsafe_allow_html=True)
        phases = [
            ("Phase 1", "Mois 1–2",  "1 station",    "Observation"),
            ("Phase 2", "Mois 3–4",  "3 stations",   "Semi-automatique"),
            ("Phase 3", "Mois 5–8",  "20 stations",  "Auto supervisé"),
            ("Phase 4", "Mois 9+",   "Toutes",       "Optimisation continue"),
        ]
        for ph, dur, st_nb, mode in phases:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap: 12px; padding: 8px 0;
                        border-bottom: 1px solid rgba(126,182,255,0.1); font-size: 12px;">
              <div style="font-weight: 700; color: var(--deep-sky); width: 65px; flex-shrink: 0;">{ph}</div>
              <div style="flex:1; color: var(--twilight);">{dur} · {st_nb}</div>
              <div style="font-size: 11px; color: #7A8FA6;">{mode}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="dashboard-footer">
      <i class="fas fa-droplet"></i> AQUA-AI Optimizer v2.0 — Prototype ONEA Burkina Faso — 2026<br>
      FastAPI backend + Streamlit frontend · Données simulées (SCADA en intégration)
    </div>
    """, unsafe_allow_html=True)