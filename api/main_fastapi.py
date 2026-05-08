"""
AQUA-AI OPTIMIZER — FastAPI
Endpoint principal : POST /predict
Simule la réponse du LSTM (en production : charger lstm_aqua_best.keras)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

app = FastAPI(
    title="AQUA-AI Optimizer API",
    description="Prédiction 24h débit, puissance et coût — ONEA Burkina Faso",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schémas Pydantic ───────────────────────────────────────────────────

class HoraireInput(BaseModel):
    timestamp: str
    temperature_C: float
    humidite_pct: float
    puissance_totale_kw: float
    puissance_sonabel_kw: float
    puissance_solaire_kw: float
    puissance_diesel_kw: float
    tarif_sonabel_fcfa_kwh: float
    plage_tarifaire: str          # 'HC' ou 'HP'
    niveau_bache_pct: float
    niveau_chateau_pct: float
    pression_entree_bar: float
    pression_sortie_bar: float
    debit_refoulement_m3h: float
    puissance_refoulement_kw: float
    pompe1_on: int
    pompe2_on: int
    coupure_sonabel: int

class PredictRequest(BaseModel):
    station_id: str = "STATION_OUAGA_01"
    last_48h: List[HoraireInput]

class HeurePrediction(BaseModel):
    heure: str
    debit_refoulement_m3h: float
    puissance_kw: float
    cout_fcfa: float
    niveau_chateau_pct: float
    niveau_bache_pct: float
    plage_tarifaire: str
    source_recommandee: str
    alerte: Optional[str] = None

class PredictResponse(BaseModel):
    station_id: str
    timestamp_prediction: str
    horizon_heures: int
    predictions: List[HeurePrediction]
    resume: dict

# ── Logique de prédiction (simule le LSTM) ────────────────────────────

def simulate_lstm_prediction(last_48h: List[HoraireInput]) -> List[HeurePrediction]:
    """
    En production : charger lstm_aqua_best.keras et appeler predict_next_24h().
    Ici : simulation réaliste basée sur les tendances des 48 dernières heures.
    """
    if len(last_48h) < 2:
        raise ValueError("Minimum 2 heures nécessaires")

    # Extraire les dernières valeurs connues
    last = last_48h[-1]
    last_ts = datetime.fromisoformat(last.timestamp)

    # Moyennes glissantes sur 48h
    debits    = [h.debit_refoulement_m3h for h in last_48h]
    puissances= [h.puissance_totale_kw   for h in last_48h]
    niv_bache = [h.niveau_bache_pct      for h in last_48h]
    niv_chat  = [h.niveau_chateau_pct    for h in last_48h]

    moy_debit    = np.mean(debits[-12:])
    moy_puissance= np.mean(puissances[-12:])
    dernier_bache= niv_bache[-1]
    dernier_chat = niv_chat[-1]

    predictions = []
    bache_sim   = dernier_bache
    chateau_sim = dernier_chat

    for h in range(24):
        ts_pred = last_ts + timedelta(hours=h+1)
        heure   = ts_pred.hour

        # Plage tarifaire
        plage  = 'HC' if heure < 17 else 'HP'
        tarif  = 84 if plage == 'HC' else 165

        # Profil demande (2 pics)
        profil = (0.6
                  + 0.4 * np.exp(-((heure - 7)**2) / 8)
                  + 0.3 * np.exp(-((heure - 19)**2) / 6))

        # Stratégie IA : pomper fort en HC, économiser en HP
        if plage == 'HC' or chateau_sim < 30:
            intensite = 1.0
            source    = "SONABEL_HC ✅"
        elif plage == 'HP' and chateau_sim > 60:
            intensite = 0.3
            source    = "SONABEL_HP ⚠️ (réserve suffisante)"
        else:
            intensite = 0.7
            source    = "SONABEL_HP (château bas)"

        # Prédictions avec bruit réaliste
        debit_pred = float(np.clip(
            moy_debit * profil * intensite + np.random.normal(0, 15),
            0, 900
        ))
        puiss_pred = float(np.clip(
            moy_puissance * profil * intensite + np.random.normal(0, 20),
            100, 5000
        ))
        cout_pred  = float(puiss_pred * tarif)

        # Mise à jour niveaux simulés
        bache_sim   = float(np.clip(bache_sim   + 0.3 - debit_pred/5000, 10, 98))
        chateau_sim = float(np.clip(chateau_sim + debit_pred/1500 - profil*0.4, 15, 98))

        # Alertes
        alerte = None
        if chateau_sim < 25:
            alerte = "⚠️ Niveau château critique (<25%)"
        elif bache_sim < 20:
            alerte = "⚠️ Niveau bâche bas (<20%)"
        elif plage == 'HP' and intensite > 0.8:
            alerte = "💰 Pompage en HP — économie possible"

        predictions.append(HeurePrediction(
            heure=ts_pred.strftime("%Y-%m-%d %H:%M"),
            debit_refoulement_m3h=round(debit_pred, 1),
            puissance_kw=round(puiss_pred, 1),
            cout_fcfa=round(cout_pred, 0),
            niveau_chateau_pct=round(chateau_sim, 1),
            niveau_bache_pct=round(bache_sim, 1),
            plage_tarifaire=plage,
            source_recommandee=source,
            alerte=alerte
        ))

    return predictions


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "AQUA-AI Optimizer",
        "version": "1.0.0",
        "client": "ONEA — Burkina Faso",
        "endpoints": ["/predict", "/status", "/docs"]
    }

@app.get("/status")
def status():
    return {
        "status": "online",
        "model": "LSTM Bidirectionnel (simulé)",
        "horizon": "24h",
        "features": 14,
        "targets": ["debit_refoulement_m3h", "puissance_kw", "cout_fcfa"]
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if len(request.last_48h) < 2:
        raise HTTPException(status_code=400, detail="Fournir au moins 2 heures de données")
    if len(request.last_48h) > 48:
        raise HTTPException(status_code=400, detail="Maximum 48 heures en entrée")

    preds = simulate_lstm_prediction(request.last_48h)

    # Résumé économique
    cout_hp  = sum(p.cout_fcfa for p in preds if p.plage_tarifaire == 'HP')
    cout_hc  = sum(p.cout_fcfa for p in preds if p.plage_tarifaire == 'HC')
    economie = cout_hp * 0.4  # économie potentielle si décalage HC

    alertes = [p.alerte for p in preds if p.alerte]

    return PredictResponse(
        station_id=request.station_id,
        timestamp_prediction=datetime.now().isoformat(),
        horizon_heures=24,
        predictions=preds,
        resume={
            "cout_total_prevu_fcfa": round(sum(p.cout_fcfa for p in preds), 0),
            "cout_en_hc_fcfa": round(cout_hc, 0),
            "cout_en_hp_fcfa": round(cout_hp, 0),
            "economie_potentielle_fcfa": round(economie, 0),
            "heures_hc": sum(1 for p in preds if p.plage_tarifaire == 'HC'),
            "heures_hp": sum(1 for p in preds if p.plage_tarifaire == 'HP'),
            "nb_alertes": len(alertes),
            "alertes": alertes[:5]
        }
    )

@app.post("/predict/simple")
def predict_simple(
    niveau_chateau: float = 65.0,
    niveau_bache: float = 72.0,
    heure_actuelle: int = 14,
    temperature: float = 32.0
):
    """Endpoint simplifié pour tests rapides sans historique complet."""
    # Générer 48h fictives cohérentes
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    fake_48h = []
    for i in range(48):
        ts = now - timedelta(hours=47-i)
        h  = ts.hour
        plage = 'HC' if h < 17 else 'HP'
        fake_48h.append(HoraireInput(
            timestamp=ts.isoformat(),
            temperature_C=temperature + np.random.normal(0, 1),
            humidite_pct=45.0,
            puissance_totale_kw=9000 + 2000*np.sin(np.pi*h/12),
            puissance_sonabel_kw=8500 + 2000*np.sin(np.pi*h/12),
            puissance_solaire_kw=max(0, 400*np.sin(np.pi*(h-6)/13)) if 6<=h<=18 else 0,
            puissance_diesel_kw=0.0,
            tarif_sonabel_fcfa_kwh=84 if plage=='HC' else 165,
            plage_tarifaire=plage,
            niveau_bache_pct=niveau_bache,
            niveau_chateau_pct=niveau_chateau,
            pression_entree_bar=2.1,
            pression_sortie_bar=4.2,
            debit_refoulement_m3h=750.0,
            puissance_refoulement_kw=430.0,
            pompe1_on=1,
            pompe2_on=1,
            coupure_sonabel=0
        ))

    preds = simulate_lstm_prediction(fake_48h)
    return {
        "predictions_24h": [p.dict() for p in preds],
        "resume": {
            "cout_total_fcfa": sum(p.cout_fcfa for p in preds),
            "economie_potentielle_fcfa": sum(p.cout_fcfa for p in preds if p.plage_tarifaire=='HP') * 0.4
        }
    }
