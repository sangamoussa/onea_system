"""
AQUA-AI OPTIMIZER — FastAPI v2.0
Branche le vrai modèle LSTM (lstm_aqua_best.keras + scalers)
Si les fichiers modèle sont absents → bascule automatiquement en mode simulation

Structure attendue dans models/ :
    models/
    ├── lstm_aqua_best.keras   ← modèle entraîné
    ├── scaler_X.pkl           ← StandardScaler features
    └── scalers_y.pkl          ← dict de StandardScaler par target
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aqua-ai")

# ── Chemins modèle ─────────────────────────────────────────────────────
MODEL_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
MODEL_PATH   = os.path.join(MODEL_DIR, 'lstm_aqua_best.keras')
SCALERX_PATH = os.path.join(MODEL_DIR, 'scaler_X.pkl')
SCALERY_PATH = os.path.join(MODEL_DIR, 'scalers_y.pkl')

model     = None
scaler_X  = None
scalers_y = None
MODE_REEL = False
orche       = None
EtatStation = None

def _load_orchestrateur():
    global orche, EtatStation
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from orchestrateur import Orchestrateur, EtatStation as _ES
        EtatStation = _ES
        orche = Orchestrateur(api_url="http://localhost:8000")
        logger.info("✅ Orchestrateur initialise")
    except Exception as e:
        logger.error(f"Erreur Orchestrateur : {e}")


def load_model():
    global model, scaler_X, scalers_y, MODE_REEL
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Modele absent : {MODEL_PATH} -> mode SIMULATION")
        return
    try:
        import tensorflow as tf, joblib
        logger.info("Chargement modele LSTM...")
        model     = tf.keras.models.load_model(MODEL_PATH)
        scaler_X  = joblib.load(SCALERX_PATH)
        scalers_y = joblib.load(SCALERY_PATH)
        MODE_REEL = True
        logger.info("Modele LSTM charge OK")
    except Exception as e:
        logger.error(f"Erreur chargement : {e} -> SIMULATION")

# ── Config LSTM (identique au notebook) ───────────────────────────────
WINDOW_IN   = 48
HORIZON_OUT = 24

FEATURES = [
    'debit_refoulement_m3h', 'puissance_totale_kw',
    'temperature_C', 'humidite_pct',
    'heure_sin', 'heure_cos', 'mois_sin', 'mois_cos',
    'is_weekend', 'is_ferie',
    'tarif_sonabel_fcfa_kwh', 'part_solaire_pct',
    'part_diesel_pct', 'puissance_solaire_kw',
]
TARGETS = ['debit_refoulement_m3h', 'puissance_totale_kw', 'cout_total_fcfa']

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AQUA-AI Optimizer API",
    description="Optimisation energetique ONEA — Burkina Faso | LSTM 48h→24h",
    version="2.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup():
    load_model()
    _load_orchestrateur()

# ── Schemas ────────────────────────────────────────────────────────────
class HoraireInput(BaseModel):
    timestamp: str
    # 14 features LSTM
    debit_refoulement_m3h: float
    puissance_totale_kw: float
    temperature_C: float
    humidite_pct: float
    heure_sin: float
    heure_cos: float
    mois_sin: float
    mois_cos: float
    is_weekend: int
    is_ferie: int
    tarif_sonabel_fcfa_kwh: float
    part_solaire_pct: float
    part_diesel_pct: float
    puissance_solaire_kw: float
    # Colonnes physiques station (optionnelles, pour alertes)
    niveau_bache_pct: Optional[float] = 70.0
    niveau_chateau_pct: Optional[float] = 65.0
    pression_entree_bar: Optional[float] = 2.1
    pression_sortie_bar: Optional[float] = 4.2
    debit_refoulement_m3h: Optional[float] = 750.0
    puissance_refoulement_kw: Optional[float] = 430.0
    pompe1_on: Optional[int] = 1
    pompe2_on: Optional[int] = 1
    coupure_sonabel: Optional[int] = 0
    plage_tarifaire: Optional[str] = "HC"

class PredictRequest(BaseModel):
    station_id: str = "STATION_OUAGA_01"
    last_48h: List[HoraireInput]

class HeurePrediction(BaseModel):
    heure: str
    debit_m3h: float
    puissance_kw: float
    cout_fcfa: float
    plage_tarifaire: str
    tarif_fcfa_kwh: int
    decision_ia: str
    economie_vs_hp_fcfa: float
    niveau_chateau_pct: Optional[float] = None
    niveau_bache_pct: Optional[float] = None
    alerte: Optional[str] = None

class PredictResponse(BaseModel):
    station_id: str
    mode: str
    timestamp_prediction: str
    horizon_heures: int
    predictions: List[HeurePrediction]
    resume: dict

# ── Inférence ─────────────────────────────────────────────────────────
def predict_lstm_reel(padded: List[HoraireInput]) -> np.ndarray:
    records = [{f: getattr(h, f) for f in FEATURES} for h in padded]
    df_in   = pd.DataFrame(records)
    X_norm  = scaler_X.transform(df_in[FEATURES].values)
    X_norm  = X_norm.reshape(1, WINDOW_IN, len(FEATURES)).astype(np.float32)
    y_norm  = model.predict(X_norm, verbose=0)          # (1, 24, 3)
    out     = np.zeros((HORIZON_OUT, len(TARGETS)))
    for i, col in enumerate(TARGETS):
        vals = y_norm[0, :, i].reshape(-1, 1)
        out[:, i] = scalers_y[col].inverse_transform(vals).ravel()
    return out

def predict_simulation(padded: List[HoraireInput]) -> np.ndarray:
    moy_debit = float(np.mean([h.debit_refoulement_m3h       for h in padded[-12:]]))
    moy_puiss = float(np.mean([h.puissance_totale_kw for h in padded[-12:]]))
    last_ts   = datetime.fromisoformat(padded[-1].timestamp)
    out       = np.zeros((HORIZON_OUT, 3))
    for h in range(HORIZON_OUT):
        heure  = (last_ts + timedelta(hours=h+1)).hour
        profil = 0.6 + 0.4*np.exp(-((heure-7)**2)/8) + 0.3*np.exp(-((heure-19)**2)/6)
        tarif  = 84 if heure < 17 else 165
        debit  = float(np.clip(moy_debit * profil + np.random.normal(0, 5000), 0, 240000))
        puiss  = float(np.clip(moy_puiss * profil + np.random.normal(0, 300),  0, 18000))
        out[h] = [debit, puiss, puiss * tarif]
    return out

def build_predictions(raw: np.ndarray, padded: List[HoraireInput]) -> List[HeurePrediction]:
    last_ts     = datetime.fromisoformat(padded[-1].timestamp)
    chateau_sim = padded[-1].niveau_chateau_pct or 65.0
    bache_sim   = padded[-1].niveau_bache_pct   or 70.0
    preds       = []

    for h in range(HORIZON_OUT):
        ts_pred = last_ts + timedelta(hours=h+1)
        heure   = ts_pred.hour
        plage   = 'HC' if heure < 17 else 'HP'
        tarif   = 84   if heure < 17 else 165

        debit = float(max(0, raw[h, 0]))
        puiss = float(max(0, raw[h, 1]))
        cout  = float(max(0, raw[h, 2]))
        eco   = max(0.0, puiss * (165 - 84)) if plage == 'HP' else 0.0

        # Décision IA
        if plage == 'HC':
            decision = "POMPER — HC 84 FCFA/kWh"
        elif chateau_sim > 70:
            decision = "REDUIRE — Chateau plein, eviter HP"
        elif chateau_sim < 30:
            decision = "POMPER — Chateau critique malgre HP"
        else:
            decision = "ATTENDRE — Remplir avant 17h si possible"

        # Mise à jour niveaux
        profil      = 0.6 + 0.4*np.exp(-((heure-7)**2)/8) + 0.3*np.exp(-((heure-19)**2)/6)
        intensite   = 1.0 if plage == 'HC' or chateau_sim < 30 else 0.3
        chateau_sim = float(np.clip(chateau_sim + debit/150000*intensite - profil*0.4, 15, 98))
        bache_sim   = float(np.clip(bache_sim   + 0.3 - debit/500000, 10, 98))

        alerte = None
        if chateau_sim < 25:  alerte = "Chateau critique (<25%)"
        elif bache_sim < 20:  alerte = "Bache basse (<20%)"
        elif plage=='HP' and cout > 1_500_000: alerte = f"Cout HP eleve : {cout:,.0f} FCFA"

        preds.append(HeurePrediction(
            heure=ts_pred.strftime("%Y-%m-%d %H:%M"),
            debit_m3h=round(debit, 1),
            puissance_kw=round(puiss, 1),
            cout_fcfa=round(cout, 0),
            plage_tarifaire=plage,
            tarif_fcfa_kwh=tarif,
            decision_ia=decision,
            economie_vs_hp_fcfa=round(eco, 0),
            niveau_chateau_pct=round(chateau_sim, 1),
            niveau_bache_pct=round(bache_sim, 1),
            alerte=alerte
        ))
    return preds

# ── Endpoints ──────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "service": "AQUA-AI Optimizer v2.0",
        "mode"   : "REEL" if MODE_REEL else "SIMULATION",
        "client" : "ONEA — Burkina Faso",
        "docs"   : "/docs"
    }

@app.get("/status", tags=["Info"])
def status():
    return {
        "status"       : "online",
        "mode"         : "REEL" if MODE_REEL else "SIMULATION",
        "modele_charge": MODE_REEL,
        "window_in"    : WINDOW_IN,
        "horizon_out"  : HORIZON_OUT,
        "features"     : FEATURES,
        "targets"      : TARGETS,
        "instructions" : "OK" if MODE_REEL else f"Deposez les 3 fichiers dans {MODEL_DIR}"
    }

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    n = len(request.last_48h)
    if n < 2:
        raise HTTPException(400, "Minimum 2 heures requises")
    if n > WINDOW_IN:
        raise HTTPException(400, f"Maximum {WINDOW_IN} heures (recu {n})")

    # Padding si < 48h
    if n < WINDOW_IN:
        padded = [request.last_48h[0]] * (WINDOW_IN - n) + list(request.last_48h)
    else:
        padded = list(request.last_48h)

    try:
        if MODE_REEL:
            raw, mode_used = predict_lstm_reel(padded), "REEL"
        else:
            raw, mode_used = predict_simulation(padded), "SIMULATION"
    except Exception as e:
        logger.error(f"Erreur prediction : {e}")
        raw, mode_used = predict_simulation(padded), "SIMULATION (fallback)"

    preds = build_predictions(raw, padded)

    cout_total = sum(p.cout_fcfa for p in preds)
    cout_hp    = sum(p.cout_fcfa for p in preds if p.plage_tarifaire == 'HP')
    cout_hc    = sum(p.cout_fcfa for p in preds if p.plage_tarifaire == 'HC')
    eco_totale = sum(p.economie_vs_hp_fcfa for p in preds)
    alertes    = [p.alerte for p in preds if p.alerte]

    return PredictResponse(
        station_id=request.station_id,
        mode=mode_used,
        timestamp_prediction=datetime.now().isoformat(),
        horizon_heures=HORIZON_OUT,
        predictions=preds,
        resume={
            "cout_total_prevu_fcfa"     : round(cout_total, 0),
            "cout_en_hc_fcfa"           : round(cout_hc, 0),
            "cout_en_hp_fcfa"           : round(cout_hp, 0),
            "economie_potentielle_fcfa" : round(eco_totale, 0),
            "pct_economie"              : round(eco_totale / max(cout_total, 1) * 100, 1),
            "heures_hc": sum(1 for p in preds if p.plage_tarifaire == 'HC'),
            "heures_hp": sum(1 for p in preds if p.plage_tarifaire == 'HP'),
            "nb_alertes": len(alertes),
            "alertes"   : alertes[:5]
        }
    )

@app.post("/predict/simple", tags=["Prediction"])
def predict_simple(
    niveau_chateau : float = 65.0,
    niveau_bache   : float = 72.0,
    heure_actuelle : int   = 14,
    temperature    : float = 32.0,
    station_id     : str   = "STATION_OUAGA_01"
):
    """Test rapide sans historique — génère 48h fictives automatiquement."""
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    fake = []
    for i in range(WINDOW_IN):
        ts  = now - timedelta(hours=WINDOW_IN - 1 - i)
        h, m = ts.hour, ts.month
        plage = 'HC' if h < 17 else 'HP'
        profil = 0.6 + 0.4*np.exp(-((h-7)**2)/8) + 0.3*np.exp(-((h-19)**2)/6)
        fake.append(HoraireInput(
            timestamp             = ts.isoformat(),
            debit_refoulement_m3h = float(np.clip(130000*profil + np.random.normal(0,3000), 65000, 240000)),
            puissance_totale_kw   = float(np.clip(9000*profil   + np.random.normal(0,300),  100, 17000)),
            temperature_C         = float(temperature + np.random.normal(0,1)),
            humidite_pct          = 45.0,
            heure_sin             = float(np.sin(2*np.pi*h/24)),
            heure_cos             = float(np.cos(2*np.pi*h/24)),
            mois_sin              = float(np.sin(2*np.pi*m/12)),
            mois_cos              = float(np.cos(2*np.pi*m/12)),
            is_weekend            = int(ts.weekday() >= 5),
            is_ferie              = 0,
            tarif_sonabel_fcfa_kwh= 84.0 if plage=='HC' else 165.0,
            part_solaire_pct      = max(0.0, float(4*np.sin(np.pi*(h-6)/13))) if 6<=h<=18 else 0.0,
            part_diesel_pct       = 0.0,
            puissance_solaire_kw  = max(0.0, float(400*np.sin(np.pi*(h-6)/13)+np.random.normal(0,20))) if 6<=h<=18 else 0.0,
            niveau_bache_pct      = float(niveau_bache),
            niveau_chateau_pct    = float(niveau_chateau),
            plage_tarifaire       = plage
        ))
    return predict(PredictRequest(station_id=station_id, last_48h=fake))
