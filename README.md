# 💧 AQUA-AI Optimizer — Prototype ONEA

Système d'optimisation énergétique par IA pour les stations de pompage ONEA (Burkina Faso).

---

## 🗂️ Structure du projet

```
AQUA-AI/
├── data/
│   └── A_energie_horaire_station_2021_2024.csv   # Données simulées (52 colonnes)
├── api/
│   └── main.py          # API FastAPI — endpoint /predict
├── models/              # (à placer ici) lstm_aqua_best.keras + scalers
├── streamlit_app.py     # Interface de démonstration
├── start.sh             # Script de lancement
└── README.md
```

---

## 🚀 Installation et lancement

```bash
# 1. Installer les dépendances
pip install fastapi uvicorn streamlit scikit-learn numpy pandas joblib requests

# 2. Lancer l'API FastAPI
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 3. (autre terminal) Lancer Streamlit
streamlit run streamlit_app.py

# OU tout en une fois :
bash start.sh
```

---

## 📡 API FastAPI — Endpoints

| Endpoint | Méthode | Description |
|---|---|---|
| `/` | GET | Infos service |
| `/status` | GET | État du modèle |
| `/predict` | POST | Prédiction complète (48h → 24h) |
| `/predict/simple` | POST | Prédiction rapide (paramètres seuls) |
| `/docs` | GET | Documentation Swagger interactive |

### Exemple d'appel `/predict/simple` :
```bash
curl -X POST "http://localhost:8000/predict/simple?niveau_chateau=65&niveau_bache=72&heure_actuelle=14&temperature=32"
```

### Exemple de réponse :
```json
{
  "predictions_24h": [
    {
      "heure": "2026-03-15 15:00",
      "debit_refoulement_m3h": 762.3,
      "puissance_kw": 9845.0,
      "cout_fcfa": 826380.0,
      "niveau_chateau_pct": 67.2,
      "niveau_bache_pct": 71.8,
      "plage_tarifaire": "HC",
      "source_recommandee": "SONABEL_HC ✅",
      "alerte": null
    },
    ...
  ],
  "resume": {
    "cout_total_fcfa": 18500000,
    "economie_potentielle_fcfa": 2400000
  }
}
```

---

## 🧠 Architecture LSTM

```
Entrée : 48h × 14 features
    ↓
BiLSTM(128) → BiLSTM(64) → LSTM(32)
    ↓
Dense(64) → Dropout → Dense(32)
    ↓
3 sorties × 24h :
  - debit_refoulement_m3h   → Agent Pompage
  - puissance_totale_kw     → Agent Tarif
  - cout_total_fcfa         → Reporting
```

---

## 📊 Nouvelles colonnes physiques (vs données initiales)

| Colonne | Description | Unité |
|---|---|---|
| `niveau_bache_pct` | Niveau bâche de stockage | % |
| `niveau_chateau_pct` | Niveau château d'eau | % |
| `pression_entree_bar` | Pression entrée pompe | bar |
| `pression_sortie_bar` | Pression sortie pompe | bar |
| `debit_refoulement_m3h` | Débit pompes refoulement | m³/h |
| `puissance_refoulement_kw` | Puissance pompes refoulement | kW |
| `puissance_ucd_kw` | Puissance unité traitement | kW |
| `pompe1_on` / `pompe2_on` | État pompes | 0/1 |
| `vibration_pompe1_mms` | Vibration pompe 1 | mm/s |
| `temp_moteur_pompe1_C` | Température moteur pompe 1 | °C |
| `icp_kwh_m3` | Indice consommation pompage | kWh/m³ |
| `debit_eau_brute_m3h` | Débit eau brute UCD | m³/h |
| `debit_consommateur_m3h` | Débit consommateur final | m³/h |

---

## 💰 Enjeux financiers

- Consommation 2024 : **91,3 GWh — 8,445 Mrd FCFA**
- Potentiel décalage HC/HP : **422–591M FCFA/an**
- ROI cible : **12–18 mois**
- CAPEX : **289,9M FCFA**

---

*Prototype AQUA-AI v1.0 — SANGA Moussa — Mars 2026*
