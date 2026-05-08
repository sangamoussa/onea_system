"""
AQUA-AI — Orchestrateur Multi-Agents v1.0
Chef d'orchestre : coordonne LSTM + DQN + Autoencoder

Priorité absolue : Sécurité > Continuité (99%) > Qualité eau > Coût

5 Agents :
  - Agent Sécurité     : overrides critiques, priorité absolue
  - Agent Maintenance  : Autoencoder → état pompes
  - Agent Pompage      : LSTM → prédiction demande 24h
  - Agent Tarif        : DQN → décision optimale
  - Agent Réseau       : validation contraintes hydrauliques

Usage :
  from orchestrateur import Orchestrateur
  orche = Orchestrateur()
  decision = orche.cycle(etat_station)
"""

import os, json, logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(name)s | %(levelname)s | %(message)s')

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# ── Structures de données ──────────────────────────────────────────────

@dataclass
class EtatStation:
    """État complet d'une station à un instant t."""
    # Réservoirs
    niveau_chateau_pct   : float = 65.0
    niveau_bache_pct     : float = 72.0
    # Énergie
    tarif_fcfa_kwh       : float = 84.0
    plage_tarifaire      : str   = 'HC'
    puissance_solaire_kw : float = 0.0
    coupure_sonabel      : int   = 0
    stock_diesel_pct     : float = 80.0
    # Pompes
    efficacite_pompe1    : float = 0.87
    efficacite_pompe2    : float = 0.85
    vibration_pompe1     : float = 1.4
    vibration_pompe2     : float = 1.6
    temp_moteur_pompe1   : float = 58.0
    temp_moteur_pompe2   : float = 60.0
    icp_kwh_m3           : float = 0.57
    pression_entree_bar  : float = 2.1
    pression_sortie_bar  : float = 4.2
    courant_pompe1_A     : float = 265.0
    courant_pompe2_A     : float = 280.0
    cycles_demarrage_24h : int   = 3
    # Contexte
    temperature_C        : float = 32.0
    heure                : int   = 14
    timestamp            : str   = ""

@dataclass
class DecisionFinale:
    """Décision complète retournée par l'Orchestrateur."""
    action_code          : int   = 1
    action_nom           : str   = "POMPE1-SONABEL"
    source_energie       : str   = "sonabel"
    pompe1_on            : bool  = True
    pompe2_on            : bool  = False
    puissance_kw         : float = 185.0
    cout_estime_fcfa     : float = 0.0
    # Prédictions LSTM
    pred_debit_h1        : float = 0.0
    pred_puissance_h1    : float = 0.0
    pred_debit_h6        : float = 0.0
    # État pompes
    alerte_pompe1        : str   = "NORMAL"
    alerte_pompe2        : str   = "NORMAL"
    score_anomalie_p1    : float = 0.0
    score_anomalie_p2    : float = 0.0
    # Décision
    agent_decideur       : str   = "Agent Tarif (DQN)"
    override_securite    : bool  = False
    raison_override      : str   = ""
    niveau_confiance     : float = 1.0
    alertes              : list  = field(default_factory=list)
    timestamp            : str   = ""

# ── Constantes actions DQN ─────────────────────────────────────────────
ACTION_NAMES = [
    'STOP', 'POMPE1-SONABEL', 'POMPE2-SONABEL', 'POMPES12-SONABEL',
    'POMPE1-SOLAIRE', 'POMPE2-SOLAIRE', 'POMPE1-DIESEL', 'POMPES12-DIESEL'
]
PUISSANCE_ACTION = {0:0, 1:185, 2:200, 3:385, 4:185, 5:200, 6:185, 7:385}
SOURCE_ACTION    = {0:'none', 1:'sonabel', 2:'sonabel', 3:'sonabel',
                    4:'solaire', 5:'solaire', 6:'diesel', 7:'diesel'}
POMPE1_ON = {0:False,1:True,2:False,3:True,4:True,5:False,6:True,7:True}
POMPE2_ON = {0:False,1:False,2:True,3:True,4:False,5:True,6:False,7:True}


# ══════════════════════════════════════════════════════════════════════
# AGENT SÉCURITÉ — Priorité absolue
# ══════════════════════════════════════════════════════════════════════
class AgentSecurite:
    """
    Vérifie les seuils critiques AVANT toute décision.
    Peut forcer une action d'urgence indépendamment du DQN.
    """
    SEUIL_CHATEAU_CRITIQUE = 20.0
    SEUIL_BACHE_CRITIQUE   = 15.0
    SEUIL_CHATEAU_BAS      = 30.0
    PRESSION_MIN           = 2.0
    PRESSION_MAX           = 6.0

    def __init__(self):
        self.logger = logging.getLogger("AgentSecurite")

    def evaluer(self, etat: EtatStation) -> tuple:
        """
        Retourne (override: bool, action_forcee: int, raison: str)
        """
        # ── Château critique → pomper OBLIGATOIREMENT ─────────────────
        if etat.niveau_chateau_pct < self.SEUIL_CHATEAU_CRITIQUE:
            if etat.coupure_sonabel:
                action = 6  # Diesel si coupure
                raison = f"URGENCE : Château {etat.niveau_chateau_pct:.0f}% + coupure SONABEL → diesel"
            else:
                action = 3  # 2 pompes SONABEL
                raison = f"URGENCE : Château critique {etat.niveau_chateau_pct:.0f}% < {self.SEUIL_CHATEAU_CRITIQUE}%"
            self.logger.warning(raison)
            return True, action, raison

        # ── Bâche critique → arrêt pompes (risque cavitation) ─────────
        if etat.niveau_bache_pct < self.SEUIL_BACHE_CRITIQUE:
            raison = f"URGENCE : Bâche critique {etat.niveau_bache_pct:.0f}% < {self.SEUIL_BACHE_CRITIQUE}% → STOP"
            self.logger.warning(raison)
            return True, 0, raison

        # ── Pression hors limites ──────────────────────────────────────
        if etat.pression_sortie_bar > self.PRESSION_MAX:
            raison = f"SÉCURITÉ : Pression {etat.pression_sortie_bar:.1f} bar > {self.PRESSION_MAX} → réduction"
            self.logger.warning(raison)
            return True, 1, raison  # 1 seule pompe

        # ── Coupure SONABEL sans diesel suffisant ──────────────────────
        if etat.coupure_sonabel and etat.stock_diesel_pct < 10:
            raison = f"SÉCURITÉ : Coupure + diesel faible ({etat.stock_diesel_pct:.0f}%) → STOP"
            self.logger.warning(raison)
            return True, 0, raison

        return False, -1, ""


# ══════════════════════════════════════════════════════════════════════
# AGENT MAINTENANCE — Autoencoder
# ══════════════════════════════════════════════════════════════════════
class AgentMaintenance:
    """Évalue l'état des pompes via l'Autoencoder."""

    AE_FEATURES = [
        'efficacite_pct', 'vibration_mm_s', 'temperature_moteur_C',
        'icp_kwh_m3', 'pression_entree_bar', 'pression_sortie_bar',
        'courant_A', 'cycles_demarrage_24h'
    ]

    def __init__(self):
        self.logger  = logging.getLogger("AgentMaintenance")
        self.modele  = None
        self.scaler  = None
        self.seuils  = None
        self._charger()

    def _charger(self):
        ae_path     = os.path.join(MODEL_DIR, 'autoencoder_aqua.keras')
        scaler_path = os.path.join(MODEL_DIR, 'autoencoder_scaler.pkl')
        seuils_path = os.path.join(MODEL_DIR, 'autoencoder_seuils.json')

        if not os.path.exists(ae_path):
            self.logger.warning("Autoencoder absent → mode dégradé")
            return
        try:
            import tensorflow as tf
            import joblib
            self.modele = tf.keras.models.load_model(ae_path)
            self.scaler = joblib.load(scaler_path)
            with open(seuils_path) as f:
                self.seuils = json.load(f)
            self.logger.info("✅ Autoencoder chargé")
        except Exception as e:
            self.logger.error(f"Erreur chargement Autoencoder : {e}")

    def evaluer_pompe(self, efficacite, vibration, temp_moteur,
                      icp, p_entree, p_sortie, courant, cycles) -> dict:
        if self.modele is None:
            # Mode dégradé : règles simples
            score = 0.0
            if vibration > 4.5:    score += 0.4
            if temp_moteur > 80:   score += 0.3
            if efficacite < 72:    score += 0.3
            alerte = 'CRITIQUE' if score > 0.5 else ('ATTENTION' if score > 0.25 else 'NORMAL')
            return {'alerte': alerte, 'score': round(score, 3),
                    'feature_critique': 'vibration_mm_s', 'mode': 'regles'}

        x = np.array([[efficacite, vibration, temp_moteur, icp,
                       p_entree, p_sortie, courant, cycles]], dtype=np.float32)
        x_norm = self.scaler.transform(x)
        x_pred = self.modele.predict(x_norm, verbose=0)
        erreur = float(np.mean(np.power(x_norm - x_pred, 2)))

        seuil_att = self.seuils['seuil_attention']
        seuil_cr  = self.seuils['seuil_critique']
        mean_err  = self.seuils['mean_erreur_normale']

        score  = float(np.clip((erreur - mean_err) / (seuil_cr - mean_err + 1e-8), 0, 1))
        alerte = 'CRITIQUE' if erreur >= seuil_cr else ('ATTENTION' if erreur >= seuil_att else 'NORMAL')

        err_feat = np.power(x_norm - x_pred, 2)[0]
        feat_cr  = self.AE_FEATURES[int(np.argmax(err_feat))]

        return {'alerte': alerte, 'score': round(score, 3),
                'feature_critique': feat_cr, 'erreur': round(erreur, 6), 'mode': 'autoencoder'}

    def evaluer(self, etat: EtatStation) -> dict:
        p1 = self.evaluer_pompe(
            etat.efficacite_pompe1 * 100, etat.vibration_pompe1,
            etat.temp_moteur_pompe1, etat.icp_kwh_m3,
            etat.pression_entree_bar, etat.pression_sortie_bar,
            etat.courant_pompe1_A, etat.cycles_demarrage_24h
        )
        p2 = self.evaluer_pompe(
            etat.efficacite_pompe2 * 100, etat.vibration_pompe2,
            etat.temp_moteur_pompe2, etat.icp_kwh_m3,
            etat.pression_entree_bar, etat.pression_sortie_bar,
            etat.courant_pompe2_A, etat.cycles_demarrage_24h
        )
        # Pompes disponibles (non critiques)
        p1_dispo = p1['alerte'] != 'CRITIQUE'
        p2_dispo = p2['alerte'] != 'CRITIQUE'

        if p1['alerte'] != 'NORMAL':
            self.logger.warning(f"Pompe1 : {p1['alerte']} | score={p1['score']}")
        if p2['alerte'] != 'NORMAL':
            self.logger.warning(f"Pompe2 : {p2['alerte']} | score={p2['score']}")

        return {'pompe1': p1, 'pompe2': p2,
                'p1_disponible': p1_dispo, 'p2_disponible': p2_dispo}


# ══════════════════════════════════════════════════════════════════════
# AGENT POMPAGE — LSTM
# ══════════════════════════════════════════════════════════════════════
class AgentPompage:
    """Appelle le LSTM via FastAPI pour obtenir les prédictions 24h."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.logger  = logging.getLogger("AgentPompage")
        self.api_url = api_url

    def predire(self, etat: EtatStation) -> dict:
        try:
            import requests
            heure = etat.heure
            import math
            params = {
                "niveau_chateau": etat.niveau_chateau_pct,
                "niveau_bache"  : etat.niveau_bache_pct,
                "heure_actuelle": heure,
                "temperature"   : etat.temperature_C,
            }
            r = requests.post(
                f"{self.api_url}/predict/simple",
                params=params, timeout=5
            )
            data = r.json()
            preds = data.get('predictions', [])
            if not preds:
                raise ValueError("Pas de prédictions")

            return {
                'pred_debit_h1'    : preds[0]['debit_m3h']   if len(preds) > 0 else 750.0,
                'pred_puissance_h1': preds[0]['puissance_kw'] if len(preds) > 0 else 560.0,
                'pred_debit_h6'    : preds[5]['debit_m3h']   if len(preds) > 5 else 750.0,
                'source'           : 'lstm_api'
            }
        except Exception as e:
            self.logger.warning(f"LSTM API indisponible ({e}) → simulation")
            return self._simulation(etat)

    def _simulation(self, etat: EtatStation) -> dict:
        """Fallback si API indisponible."""
        h = etat.heure
        profil = 0.6 + 0.4 * np.exp(-((h-7)**2)/8) + 0.3 * np.exp(-((h-19)**2)/6)
        return {
            'pred_debit_h1'    : round(750 * profil, 1),
            'pred_puissance_h1': round(560 * profil, 1),
            'pred_debit_h6'    : round(750 * profil, 1),
            'source'           : 'simulation'
        }


# ══════════════════════════════════════════════════════════════════════
# AGENT TARIF — DQN
# ══════════════════════════════════════════════════════════════════════
class AgentTarif:
    """Décide l'action optimale via le DQN."""

    def __init__(self):
        self.logger  = logging.getLogger("AgentTarif")
        self.modele  = None
        self.stats   = None
        self._charger()

    def _charger(self):
        dqn_path  = os.path.join(MODEL_DIR, 'dqn_aqua_best.keras')
        norm_path = os.path.join(MODEL_DIR, 'dqn_state_stats.json')
        if not os.path.exists(dqn_path):
            self.logger.warning("DQN absent → mode règles")
            return
        try:
            import tensorflow as tf
            self.modele = tf.keras.models.load_model(dqn_path)
            with open(norm_path) as f:
                self.stats = json.load(f)
            self.logger.info("✅ DQN chargé")
        except Exception as e:
            self.logger.error(f"Erreur chargement DQN : {e}")

    def _normaliser(self, etat: EtatStation, pred: dict) -> np.ndarray:
        import math
        vals = {
            'niveau_chateau_pct' : etat.niveau_chateau_pct,
            'niveau_bache_pct'   : etat.niveau_bache_pct,
            'tarif_fcfa_kwh'     : etat.tarif_fcfa_kwh,
            'heure_sin'          : math.sin(2*math.pi*etat.heure/24),
            'heure_cos'          : math.cos(2*math.pi*etat.heure/24),
            'puissance_solaire_kw': etat.puissance_solaire_kw,
            'coupure_sonabel'    : float(etat.coupure_sonabel),
            'efficacite_pompe1'  : etat.efficacite_pompe1,
            'efficacite_pompe2'  : etat.efficacite_pompe2,
            'temperature_C'      : etat.temperature_C,
            'stock_diesel_pct'   : etat.stock_diesel_pct,
            'icp_kwh_m3'         : etat.icp_kwh_m3,
            'pred_debit_h1'      : pred['pred_debit_h1'],
            'pred_puissance_h1'  : pred['pred_puissance_h1'],
            'pred_debit_h6'      : pred['pred_debit_h6'],
        }
        s = self.stats['stats']
        state = []
        for col in self.stats['state_cols']:
            mn  = s[col]['min']
            mx  = s[col]['max']
            val = vals.get(col, 0.0)
            state.append((val - mn) / (mx - mn + 1e-8))
        return np.array(state, dtype=np.float32)

    def decider(self, etat: EtatStation, pred: dict,
                p1_dispo: bool = True, p2_dispo: bool = True) -> tuple:
        """Retourne (action: int, q_values: list)"""
        if self.modele is None:
            return self._regles(etat, p1_dispo, p2_dispo), []

        state  = self._normaliser(etat, pred)
        q_vals = self.modele(state.reshape(1,-1), training=False).numpy()[0]

        # Masquer actions avec pompes indisponibles
        q_masked = q_vals.copy()
        if not p1_dispo:
            for a in [1, 3, 4, 6, 7]: q_masked[a] = -999
        if not p2_dispo:
            for a in [2, 3, 5, 7]:    q_masked[a] = -999
        if etat.coupure_sonabel:
            for a in [1, 2, 3]:        q_masked[a] = -999
        if etat.puissance_solaire_kw < 100:
            for a in [4, 5]:           q_masked[a] = -999

        action = int(np.argmax(q_masked))
        return action, q_vals.tolist()

    def _regles(self, etat: EtatStation, p1_dispo: bool, p2_dispo: bool) -> int:
        """Règles de décision si DQN absent."""
        if etat.coupure_sonabel:
            return 6 if etat.niveau_chateau_pct < 40 and p1_dispo else 0
        if etat.puissance_solaire_kw > 200 and p1_dispo:
            return 4
        if etat.plage_tarifaire == 'HC':
            if p1_dispo and p2_dispo and etat.niveau_bache_pct > 50:
                return 3
            return 1 if p1_dispo else 2
        else:  # HP
            return 0 if etat.niveau_chateau_pct > 55 else (1 if p1_dispo else 2)


# ══════════════════════════════════════════════════════════════════════
# AGENT RÉSEAU — Validation hydraulique
# ══════════════════════════════════════════════════════════════════════
class AgentReseau:
    """Valide que l'action respecte les contraintes hydrauliques."""

    def __init__(self):
        self.logger = logging.getLogger("AgentReseau")

    def valider(self, action: int, etat: EtatStation) -> tuple:
        """
        Retourne (action_validee: int, alertes: list)
        Peut modifier l'action si contrainte hydraulique non respectée.
        """
        alertes = []
        src     = SOURCE_ACTION[action]
        puiss   = PUISSANCE_ACTION[action]

        # Vérification pression réseau
        if etat.pression_sortie_bar < 2.0 and puiss > 0:
            alertes.append(f"⚠️ Pression sortie basse ({etat.pression_sortie_bar:.1f} bar)")

        # Vérification niveau château
        if etat.niveau_chateau_pct > 92 and puiss > 0:
            self.logger.info("Château plein → STOP forcé")
            alertes.append("Château plein (>92%) → arrêt pompage")
            return 0, alertes

        # Vérification niveau bâche suffisant pour pomper
        if etat.niveau_bache_pct < 20 and puiss > 200:
            self.logger.warning("Bâche basse → réduction à 1 pompe")
            alertes.append(f"⚠️ Bâche basse ({etat.niveau_bache_pct:.0f}%) → 1 pompe max")
            # Réduire de 2 pompes à 1 pompe
            if action == 3: action = 1
            if action == 7: action = 6

        return action, alertes


# ══════════════════════════════════════════════════════════════════════
# ORCHESTRATEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════
class Orchestrateur:
    """
    Chef d'orchestre AQUA-AI.
    Coordonne les 5 agents dans le bon ordre de priorité.
    """

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.logger = logging.getLogger("Orchestrateur")
        self.logger.info("Initialisation AQUA-AI Orchestrateur...")

        self.agent_securite   = AgentSecurite()
        self.agent_maintenance= AgentMaintenance()
        self.agent_pompage    = AgentPompage(api_url)
        self.agent_tarif      = AgentTarif()
        self.agent_reseau     = AgentReseau()

        self.logger.info("✅ Tous les agents initialisés")

    def cycle(self, etat: EtatStation) -> DecisionFinale:
        """
        Cycle décisionnel complet — appelé à chaque heure.
        Retourne la DecisionFinale à exécuter sur la station.
        """
        ts = datetime.now().isoformat()
        self.logger.info(f"─── Cycle décisionnel {ts} ───")
        alertes = []

        # ── ÉTAPE 1 : Sécurité (priorité absolue) ─────────────────────
        override, action_urgence, raison_urgence = \
            self.agent_securite.evaluer(etat)

        # ── ÉTAPE 2 : Maintenance (état pompes) ───────────────────────
        etat_pompes = self.agent_maintenance.evaluer(etat)
        p1_dispo    = etat_pompes['p1_disponible']
        p2_dispo    = etat_pompes['p2_disponible']

        for pid, info in [('P1', etat_pompes['pompe1']),
                          ('P2', etat_pompes['pompe2'])]:
            if info['alerte'] != 'NORMAL':
                alertes.append(f"🔧 Pompe {pid} : {info['alerte']} "
                                f"({info.get('feature_critique','')})")

        # ── ÉTAPE 3 : Pompage (prédictions LSTM) ──────────────────────
        pred = self.agent_pompage.predire(etat)
        self.logger.info(f"LSTM → débit h+1={pred['pred_debit_h1']:.0f} m³/h | "
                         f"h+6={pred['pred_debit_h6']:.0f} m³/h")

        # ── ÉTAPE 4 : Tarif (décision DQN) ────────────────────────────
        if override:
            action_dqn = action_urgence
            q_vals     = []
            decideur   = "Agent Sécurité (override)"
        else:
            action_dqn, q_vals = self.agent_tarif.decider(
                etat, pred, p1_dispo, p2_dispo)
            decideur = "Agent Tarif (DQN)"

        # ── ÉTAPE 5 : Réseau (validation hydraulique) ─────────────────
        action_final, alertes_reseau = \
            self.agent_reseau.valider(action_dqn, etat)
        alertes.extend(alertes_reseau)

        if action_final != action_dqn:
            self.logger.info(f"Agent Réseau : action modifiée "
                             f"{ACTION_NAMES[action_dqn]} → {ACTION_NAMES[action_final]}")

        # ── Construction décision finale ───────────────────────────────
        src   = SOURCE_ACTION[action_final]
        puiss = PUISSANCE_ACTION[action_final]
        tarif = etat.tarif_fcfa_kwh
        cout  = (puiss * 300 if src == 'diesel'
                 else 0 if src in ['solaire','none']
                 else puiss * tarif)

        # Niveau de confiance (réduit si override ou pompe dégradée)
        confiance = 1.0
        if override: confiance -= 0.3
        if not p1_dispo or not p2_dispo: confiance -= 0.1
        if pred['source'] == 'simulation': confiance -= 0.1

        # Alerte château bas
        if etat.niveau_chateau_pct < 35:
            alertes.append(f"⚠️ Château bas : {etat.niveau_chateau_pct:.0f}%")
        if etat.niveau_bache_pct < 25:
            alertes.append(f"⚠️ Bâche basse : {etat.niveau_bache_pct:.0f}%")

        decision = DecisionFinale(
            action_code       = action_final,
            action_nom        = ACTION_NAMES[action_final],
            source_energie    = src,
            pompe1_on         = POMPE1_ON[action_final],
            pompe2_on         = POMPE2_ON[action_final],
            puissance_kw      = float(puiss),
            cout_estime_fcfa  = float(cout),
            pred_debit_h1     = pred['pred_debit_h1'],
            pred_puissance_h1 = pred['pred_puissance_h1'],
            pred_debit_h6     = pred['pred_debit_h6'],
            alerte_pompe1     = etat_pompes['pompe1']['alerte'],
            alerte_pompe2     = etat_pompes['pompe2']['alerte'],
            score_anomalie_p1 = etat_pompes['pompe1']['score'],
            score_anomalie_p2 = etat_pompes['pompe2']['score'],
            agent_decideur    = decideur,
            override_securite = override,
            raison_override   = raison_urgence,
            niveau_confiance  = round(max(0.0, confiance), 2),
            alertes           = alertes,
            timestamp         = ts,
        )

        self.logger.info(
            f"✅ Décision : {decision.action_nom} | "
            f"Source : {src} | Puissance : {puiss} kW | "
            f"Coût : {cout:,.0f} FCFA | Confiance : {confiance:.0%}"
        )
        return decision

    def cycle_dict(self, etat_dict: dict) -> dict:
        """Version dict pour intégration FastAPI / Streamlit."""
        etat = EtatStation(**{k: v for k, v in etat_dict.items()
                              if k in EtatStation.__dataclass_fields__})
        d = self.cycle(etat)
        return d.__dict__


# ══════════════════════════════════════════════════════════════════════
# TEST AUTONOME
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n" + "=" * 58)
    print("  TEST ORCHESTRATEUR — 3 scénarios")
    print("=" * 58)

    orche = Orchestrateur()

    scenarios = [
        ("Scénario 1 — Normal HC 15h", EtatStation(
            niveau_chateau_pct=58, niveau_bache_pct=72,
            tarif_fcfa_kwh=84, plage_tarifaire='HC',
            puissance_solaire_kw=320, heure=15,
            efficacite_pompe1=0.87, efficacite_pompe2=0.85,
            vibration_pompe1=1.4, vibration_pompe2=1.6,
            temp_moteur_pompe1=58, temp_moteur_pompe2=60,
        )),
        ("Scénario 2 — Château critique HP 20h", EtatStation(
            niveau_chateau_pct=18, niveau_bache_pct=65,
            tarif_fcfa_kwh=165, plage_tarifaire='HP',
            puissance_solaire_kw=0, heure=20,
            efficacite_pompe1=0.87, efficacite_pompe2=0.85,
            vibration_pompe1=1.4, vibration_pompe2=1.6,
            temp_moteur_pompe1=58, temp_moteur_pompe2=60,
        )),
        ("Scénario 3 — Coupure SONABEL + P1 dégradée", EtatStation(
            niveau_chateau_pct=45, niveau_bache_pct=68,
            tarif_fcfa_kwh=84, plage_tarifaire='HC',
            coupure_sonabel=1, puissance_solaire_kw=0, heure=10,
            efficacite_pompe1=0.68, vibration_pompe1=5.8,
            temp_moteur_pompe1=85, efficacite_pompe2=0.86,
            vibration_pompe2=1.5, temp_moteur_pompe2=60,
        )),
    ]

    for titre, etat in scenarios:
        print(f"\n{'─'*55}")
        print(f"  {titre}")
        print(f"{'─'*55}")
        d = orche.cycle(etat)
        print(f"  Action    : {d.action_nom}")
        print(f"  Source    : {d.source_energie}")
        print(f"  Puissance : {d.puissance_kw:.0f} kW")
        print(f"  Coût/h    : {d.cout_estime_fcfa:,.0f} FCFA")
        print(f"  Pompe1    : {'ON' if d.pompe1_on else 'OFF'} ({d.alerte_pompe1})")
        print(f"  Pompe2    : {'ON' if d.pompe2_on else 'OFF'} ({d.alerte_pompe2})")
        print(f"  Décideur  : {d.agent_decideur}")
        print(f"  Confiance : {d.niveau_confiance:.0%}")
        if d.override_securite:
            print(f"  ⚠️  OVERRIDE : {d.raison_override}")
        if d.alertes:
            for a in d.alertes:
                print(f"  🔔 {a}")
