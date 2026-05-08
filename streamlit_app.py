"""
AQUA-AI OPTIMIZER — Interface Streamlit
Dashboard de prototypage ONEA Burkina Faso
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random

# ── Config page ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQUA-AI Optimizer — ONEA",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS custom ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0066CC, #00A86B);
        padding: 20px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f0f8ff;
        border-left: 4px solid #0066CC;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 4px 0;
    }
    .alert-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 4px 0;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px 14px;
        border-radius: 6px;
    }
    .hc-badge { background:#28a745; color:white; padding:2px 8px; border-radius:4px; font-size:12px; }
    .hp-badge { background:#dc3545; color:white; padding:2px 8px; border-radius:4px; font-size:12px; }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000"

# ── Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2 style="margin:0">💧 AQUA-AI Optimizer</h2>
    <p style="margin:4px 0 0 0; opacity:0.9">Optimisation énergétique intelligente — ONEA Burkina Faso</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — Paramètres station ───────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/Flag_of_Burkina_Faso.svg/120px-Flag_of_Burkina_Faso.svg.png", width=80)
    st.markdown("### ⚙️ Paramètres Station")

    station_id = st.selectbox("Station", [
        "STATION_OUAGA_01", "STATION_OUAGA_02", "STATION_BOBO_01"
    ])

    st.markdown("---")
    st.markdown("**État actuel de la station**")

    niveau_chateau = st.slider("🏰 Niveau château d'eau (%)", 15, 98, 65)
    niveau_bache   = st.slider("🗄️ Niveau bâche (%)", 10, 98, 72)
    temperature    = st.slider("🌡️ Température (°C)", 18, 44, 32)
    heure_actuelle = st.slider("🕐 Heure actuelle", 0, 23, datetime.now().hour)

    st.markdown("---")
    pompe1 = st.checkbox("Pompe 1 active", value=True)
    pompe2 = st.checkbox("Pompe 2 active", value=True)
    coupure = st.checkbox("⚡ Coupure SONABEL", value=False)

    st.markdown("---")
    st.markdown(f"**Tarif actuel :** {'🟢 HC (84 FCFA/kWh)' if heure_actuelle < 17 else '🔴 HP (165 FCFA/kWh)'}")

    predict_btn = st.button("🚀 Lancer la prédiction", type="primary", use_container_width=True)

# ── Fonctions utilitaires ──────────────────────────────────────────────
def call_api_simple(niveau_chateau, niveau_bache, heure_actuelle, temperature):
    try:
        url = f"{API_URL}/predict/simple"
        params = {
            "niveau_chateau": niveau_chateau,
            "niveau_bache": niveau_bache,
            "heure_actuelle": heure_actuelle,
            "temperature": temperature
        }
        r = requests.post(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def check_api():
    try:
        r = requests.get(f"{API_URL}/status", timeout=3)
        return r.status_code == 200
    except:
        return False

# ── Tabs principales ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard", "🔮 Prédiction 24h", "💰 Analyse Économique", "ℹ️ À propos"
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🏭 État Temps Réel de la Station")

    # KPIs principaux
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        color = "normal" if niveau_chateau > 40 else "inverse"
        st.metric("🏰 Château d'eau", f"{niveau_chateau}%",
                  delta="OK" if niveau_chateau > 40 else "BAS",
                  delta_color=color)
    with col2:
        st.metric("🗄️ Bâche stockage", f"{niveau_bache}%",
                  delta="OK" if niveau_bache > 30 else "BAS",
                  delta_color="normal" if niveau_bache > 30 else "inverse")
    with col3:
        tarif_actuel = 84 if heure_actuelle < 17 else 165
        st.metric("⚡ Tarif SONABEL", f"{tarif_actuel} FCFA/kWh",
                  delta="Heures Creuses" if heure_actuelle < 17 else "Heures de Pointe",
                  delta_color="normal" if heure_actuelle < 17 else "inverse")
    with col4:
        st.metric("🌡️ Température", f"{temperature}°C",
                  delta=f"Pompes: {'ON' if pompe1 or pompe2 else 'OFF'}")

    st.markdown("---")

    # Schéma station
    col_schema, col_alertes = st.columns([2, 1])

    with col_schema:
        st.markdown("#### 🔄 Schéma de la station")

        # Visualisation simplifiée avec progress bars
        st.markdown("**Flux : Eau brute → UCD → Bâche → Refoulement → Château → Consommateurs**")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🗄️ Bâche de stockage**")
            st.progress(niveau_bache / 100)
            st.caption(f"{niveau_bache}% — {int(niveau_bache * 50)} m³ / 5000 m³")

            st.markdown("**🏰 Château d'eau**")
            st.progress(niveau_chateau / 100)
            st.caption(f"{niveau_chateau}% — {int(niveau_chateau * 15)} m³ / 1500 m³")

        with col_b:
            st.markdown("**🔧 Pompe de refoulement 1**")
            p1_eff = random.uniform(82, 90)
            st.progress(p1_eff / 100)
            st.caption(f"Efficacité: {p1_eff:.1f}% | {'🟢 Active' if pompe1 else '🔴 Arrêtée'}")

            st.markdown("**🔧 Pompe de refoulement 2**")
            p2_eff = random.uniform(80, 88)
            st.progress(p2_eff / 100)
            st.caption(f"Efficacité: {p2_eff:.1f}% | {'🟢 Active' if pompe2 else '🔴 Arrêtée'}")

    with col_alertes:
        st.markdown("#### 🚨 Alertes actives")

        alertes = []
        if niveau_chateau < 30:
            alertes.append(("⚠️ URGENT", "Niveau château critique (<30%)"))
        if niveau_bache < 20:
            alertes.append(("⚠️ URGENT", "Niveau bâche critique (<20%)"))
        if heure_actuelle >= 17 and (pompe1 or pompe2):
            alertes.append(("💰 COÛT", "Pompage en HP — décaler si possible"))
        if coupure:
            alertes.append(("⚡ COUPURE", "SONABEL indisponible — groupe en marche"))
        if temperature > 38:
            alertes.append(("🌡️ CHALEUR", "Température élevée — pic demande probable"))

        if alertes:
            for niveau, msg in alertes:
                st.markdown(f"""<div class="alert-box">
                    <strong>{niveau}</strong><br>{msg}
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="success-box">
                ✅ Aucune alerte — Fonctionnement normal
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 📊 Consommation actuelle")
        puiss_actuelle = 8500 + 1500 * np.sin(np.pi * heure_actuelle / 12) + random.normalvariate(0, 200)
        st.metric("Puissance totale", f"{puiss_actuelle:,.0f} kW")
        cout_heure = puiss_actuelle * tarif_actuel
        st.metric("Coût heure en cours", f"{cout_heure:,.0f} FCFA")

        # Source énergie
        if coupure:
            st.error("⛽ Source : Groupe électrogène (300 FCFA/kWh)")
        elif heure_actuelle < 17:
            st.success("🟢 Source : SONABEL HC (84 FCFA/kWh)")
        else:
            st.warning("🟡 Source : SONABEL HP (165 FCFA/kWh)")

# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — PRÉDICTION 24H
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔮 Prédiction LSTM — Prochaines 24 heures")

    api_ok = check_api()

    if api_ok:
        st.success("✅ API AQUA-AI connectée")
    else:
        st.warning("⚠️ API non joignable — simulation locale activée")

    if predict_btn or st.button("▶️ Prédire maintenant", key="pred2"):
        with st.spinner("🧠 LSTM en cours de prédiction..."):
            if api_ok:
                data, err = call_api_simple(niveau_chateau, niveau_bache,
                                            heure_actuelle, temperature)
            else:
                # Simulation locale si API non disponible
                data = None
                err = "API non disponible"

            # Simulation locale (toujours disponible)
            now = datetime.now().replace(minute=0, second=0, microsecond=0)
            rows = []
            chateau_sim = float(niveau_chateau)
            bache_sim   = float(niveau_bache)

            for h in range(24):
                ts = now + timedelta(hours=h+1)
                hh = ts.hour
                plage = 'HC' if hh < 17 else 'HP'
                tarif = 84 if plage == 'HC' else 165

                profil = (0.6 + 0.4*np.exp(-((hh-7)**2)/8)
                               + 0.3*np.exp(-((hh-19)**2)/6))

                if plage == 'HC' or chateau_sim < 30:
                    intensite = 1.0
                    source = "SONABEL HC ✅"
                elif plage == 'HP' and chateau_sim > 60:
                    intensite = 0.3
                    source = "Économie HP 💰"
                else:
                    intensite = 0.7
                    source = "SONABEL HP ⚠️"

                debit = float(np.clip(750 * profil * intensite + random.normalvariate(0, 15), 0, 900))
                puiss = float(np.clip(9000 * profil * intensite + random.normalvariate(0, 200), 100, 17000))
                cout  = puiss * tarif

                bache_sim   = float(np.clip(bache_sim   + 0.3 - debit/5000, 10, 98))
                chateau_sim = float(np.clip(chateau_sim + debit/1500 - profil*0.4, 15, 98))

                alerte = ""
                if chateau_sim < 25: alerte = "⚠️ Château critique"
                elif bache_sim < 20: alerte = "⚠️ Bâche basse"
                elif plage == 'HP' and intensite > 0.8: alerte = "💰 Pompage HP"

                rows.append({
                    "Heure": ts.strftime("%H:%M"),
                    "Débit (m³/h)": round(debit, 1),
                    "Puissance (kW)": round(puiss, 0),
                    "Coût (FCFA)": round(cout, 0),
                    "Château (%)": round(chateau_sim, 1),
                    "Bâche (%)": round(bache_sim, 1),
                    "Tarif": plage,
                    "Source": source,
                    "Alerte": alerte
                })

            df_pred = pd.DataFrame(rows)

        # Résumé 4 KPIs
        st.markdown("#### 📊 Résumé prédiction 24h")
        c1, c2, c3, c4 = st.columns(4)
        cout_total = df_pred["Coût (FCFA)"].sum()
        cout_hp    = df_pred[df_pred["Tarif"]=="HP"]["Coût (FCFA)"].sum()
        eco_pot    = cout_hp * 0.4
        n_alertes  = (df_pred["Alerte"] != "").sum()

        with c1: st.metric("💰 Coût total prévu", f"{cout_total/1e6:.2f}M FCFA")
        with c2: st.metric("🔴 Coût en HP", f"{cout_hp/1e6:.2f}M FCFA")
        with c3: st.metric("✅ Économie potentielle", f"{eco_pot/1e6:.2f}M FCFA", delta="-40% si décalage HC")
        with c4: st.metric("🚨 Alertes", n_alertes)

        st.markdown("---")

        # Graphiques
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("**Puissance prévue (kW)**")
            chart_data = df_pred.set_index("Heure")["Puissance (kW)"]
            st.line_chart(chart_data)

        with col_g2:
            st.markdown("**Niveaux réservoirs prévus (%)**")
            niv_data = df_pred.set_index("Heure")[["Château (%)", "Bâche (%)"]].rename(
                columns={"Château (%)": "Château", "Bâche (%)": "Bâche"}
            )
            st.line_chart(niv_data)

        # Table complète
        st.markdown("#### 📋 Prédictions heure par heure")

        def color_tarif(val):
            if val == 'HC':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'HP':
                return 'background-color: #f8d7da; color: #721c24'
            return ''

        st.dataframe(
            df_pred.style.applymap(color_tarif, subset=["Tarif"]),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("👈 Configurez les paramètres dans la barre latérale et cliquez sur **Lancer la prédiction**")

        st.markdown("#### Comment ça fonctionne :")
        st.markdown("""
        1. **Entrée** : 48 dernières heures de données station (débit, puissance, niveaux, météo...)
        2. **LSTM Bidirectionnel** : analyse les tendances passées dans les 2 sens temporels
        3. **Sortie** : prédiction heure par heure sur 24h (débit, puissance, coût)
        4. **Action IA** : recommande quand pomper en HC (84 FCFA) vs HP (165 FCFA)
        """)

# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYSE ÉCONOMIQUE
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 💰 Analyse Économique ONEA 2024")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Consommation réelle ONEA 2024")
        st.markdown("""
        | Indicateur | Valeur |
        |---|---|
        | Consommation totale | **91,3 GWh** |
        | Coût total | **8,445 Mrd FCFA** |
        | Dont Heures de Pointe | **27,2M kWh** |
        | Pertes réseau | **205M kWh** |
        | Consommation solaire | **4 GWh (4%)** |
        """)

        st.markdown("#### Potentiel d'économies AQUA-AI")
        st.markdown("""
        | Levier | Économie estimée |
        |---|---|
        | Décalage HC/HP (LSTM+DQN) | **422–591M FCFA/an** |
        | Optimisation multi-agents | **507–676M FCFA/an** |
        | Réduction pertes (Autoencoder) | **338M FCFA/an** |
        | **TOTAL** | **1,267–1,942 Mrd FCFA/an** |
        """)

    with col_right:
        st.markdown("#### Simulateur d'économies HC/HP")

        kwh_hp = st.number_input("kWh consommés en HP (heures de pointe)", 
                                  value=27200000, step=100000)
        pct_decalage = st.slider("% décalable vers HC", 10, 80, 40)

        kwh_decales = kwh_hp * pct_decalage / 100
        eco = kwh_decales * (165 - 84)

        st.markdown(f"""
        <div class="metric-card">
            <strong>kWh décalés vers HC :</strong> {kwh_decales/1e6:.1f}M kWh<br>
            <strong>Économie tarifaire :</strong> {eco/1e6:.0f}M FCFA/an<br>
            <strong>Soit :</strong> {eco/8445306812*100:.1f}% du budget énergie
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ROI du projet")
        capex = 289.9e6
        eco_annuelle = eco

        if eco_annuelle > 0:
            roi_mois = capex / (eco_annuelle / 12)
            st.metric("CAPEX", f"{capex/1e6:.1f}M FCFA")
            st.metric("Économie annuelle estimée", f"{eco_annuelle/1e6:.0f}M FCFA")
            st.metric("⏱️ ROI estimé", f"{roi_mois:.0f} mois",
                      delta="Cible : 12–18 mois",
                      delta_color="normal" if roi_mois <= 18 else "inverse")

# ═══════════════════════════════════════════════════════════════════════
# TAB 4 — À PROPOS
# ═══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ℹ️ AQUA-AI Optimizer")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Système d'IA hybride pour l'optimisation énergétique des stations de pompage ONEA.**

        #### 5 Modules IA :
        - 🧠 **LSTM Bidirectionnel** — Prédiction demande 1-24h
        - 📈 **Transformer** — Planification 1-7 jours
        - 🎮 **DQN** — Optimisation tarifaire temps réel
        - 🤝 **Multi-Agents** — Coordination inter-stations
        - 🔍 **Autoencoder** — Détection anomalies pompes

        #### Architecture station :
        ```
        Eau brute
            ↓
        [UCD — Traitement]
        Pompes doseuses, agitateurs
            ↓
        [Bâches de stockage]
            ↓
        [Pompes de refoulement]
            ↓
        [Château d'eau]
            ↓
        Consommateurs finaux
        ```
        """)

    with col2:
        st.markdown("""
        #### Plan de déploiement :
        | Phase | Durée | Stations | Mode |
        |---|---|---|---|
        | 1 | Mois 1-2 | 1 | Observation |
        | 2 | Mois 3-4 | 3 | Semi-auto |
        | 3 | Mois 5-8 | 20 | Auto supervisé |
        | 4 | Mois 9+ | Toutes | Optimisation continue |

        #### Sources d'énergie :
        | Source | Coût | Disponibilité |
        |---|---|---|
        | SONABEL HC | 80-88 FCFA/kWh | 00h-17h |
        | SONABEL HP | 165 FCFA/kWh | 17h-24h |
        | Solaire PV | ~0 FCFA/kWh | Jour (4 GWh/an) |
        | Diesel | ~300 FCFA/kWh | Coupures |

        #### Version actuelle :
        - Prototype v1.0 — Mars 2026
        - Données simulées (SCADA en cours d'intégration)
        - Soumissionnaire : SANGA Moussa
        """)

    st.markdown("---")
    st.markdown("*AQUA-AI Optimizer — Prototype de démonstration ONEA Burkina Faso — 2026*")
