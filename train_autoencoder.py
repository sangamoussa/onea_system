"""
AQUA-AI — Autoencoder Détection d'Anomalies v1.0
Surveillance temps réel des pompes de refoulement ONEA

Architecture : Encodeur [8→4→2] → Espace latent → Décodeur [2→4→8]
Principe     : Entraîné sur données NORMALES uniquement
               → grande erreur de reconstruction = anomalie

Génère dans models/ :
  - autoencoder_aqua.keras
  - autoencoder_scaler.pkl
  - autoencoder_seuils.json   ← seuils NORMAL/ATTENTION/CRITIQUE
"""

import os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ── Chemins ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'B_pompes_efficacite_2021_2024.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

AE_PATH     = os.path.join(MODEL_DIR, 'autoencoder_aqua.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'autoencoder_scaler.pkl')
SEUILS_PATH = os.path.join(MODEL_DIR, 'autoencoder_seuils.json')

# ── Features Autoencoder (8 dimensions) ───────────────────────────────
AE_FEATURES = [
    'efficacite_pct',          # santé globale pompe
    'vibration_mm_s',          # défauts mécaniques
    'temperature_moteur_C',    # surcharge / bobinage
    'icp_kwh_m3',              # surconsommation
    'pression_entree_bar',     # risque cavitation
    'pression_sortie_bar',     # capacité refoulement
    'courant_A',               # défauts électriques
    'cycles_demarrage_24h',    # usure démarrages
]

# Hyperparamètres
LATENT_DIM = 2      # espace latent 2D (visualisable)
EPOCHS     = 50
BATCH      = 256
LR         = 0.001

print("=" * 58)
print("  AQUA-AI — Autoencoder Détection Anomalies")
print("=" * 58)

# ── 1. Chargement données ──────────────────────────────────────────────
print(f"\n📂 Chargement : {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"   Shape total : {df.shape}")
print(f"   Pompes : {df['pompe_id'].unique()}")
print(f"\n   Distribution alertes :")
print(df['niveau_alerte'].value_counts().to_string())

# ── 2. Split NORMAL / ANOMALIE ─────────────────────────────────────────
# Entraînement sur données NORMALES uniquement (principe Autoencoder)
df_normal   = df[df['niveau_alerte'] == 'NORMAL'].copy()
df_anomalie = df[df['niveau_alerte'] != 'NORMAL'].copy()

print(f"\n   Données normales (train) : {len(df_normal):,}")
print(f"   Données anomalies (test) : {len(df_anomalie):,}")

# Split train/val sur normales
from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(df_normal, test_size=0.15, random_state=42)
print(f"   Train : {len(df_train):,} | Val : {len(df_val):,}")

# ── 3. Normalisation ───────────────────────────────────────────────────
print("\n📐 Normalisation...")
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[AE_FEATURES].values).astype(np.float32)
X_val   = scaler.transform(df_val[AE_FEATURES].values).astype(np.float32)
X_anomalie = scaler.transform(df_anomalie[AE_FEATURES].values).astype(np.float32)

# Dataset complet pour évaluation finale
X_all   = scaler.transform(df[AE_FEATURES].values).astype(np.float32)

joblib.dump(scaler, SCALER_PATH)
print(f"   ✅ Scaler sauvegardé")
for i, col in enumerate(AE_FEATURES):
    print(f"   {col}: mean={scaler.mean_[i]:.2f} | std={scaler.scale_[i]:.2f}")

# ── 4. Architecture Autoencoder ────────────────────────────────────────
print("\n🏗️  Construction Autoencoder...")
import tensorflow as tf
from tensorflow import keras

def build_autoencoder(input_dim, latent_dim):
    """
    Autoencoder symétrique :
    Encodeur : input_dim → 4 → latent_dim
    Décodeur : latent_dim → 4 → input_dim
    """
    # ── Encodeur ───────────────────────────────────────────────────────
    inp = keras.layers.Input(shape=(input_dim,), name='input')
    x   = keras.layers.Dense(4, activation='relu', name='enc_1')(inp)
    x   = keras.layers.BatchNormalization()(x)
    lat = keras.layers.Dense(latent_dim, activation='relu',
                             name='latent')(x)

    # ── Décodeur ───────────────────────────────────────────────────────
    x   = keras.layers.Dense(4, activation='relu', name='dec_1')(lat)
    x   = keras.layers.BatchNormalization()(x)
    out = keras.layers.Dense(input_dim, activation='linear',
                             name='reconstruction')(x)

    autoencoder = keras.Model(inputs=inp, outputs=out,
                              name='autoencoder_aqua')
    encodeur    = keras.Model(inputs=inp, outputs=lat,
                              name='encodeur')

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss='mse',
        metrics=['mae']
    )
    return autoencoder, encodeur

autoencoder, encodeur = build_autoencoder(len(AE_FEATURES), LATENT_DIM)
autoencoder.summary()

# ── 5. Entraînement ────────────────────────────────────────────────────
print(f"\n🚀 Entraînement ({EPOCHS} epochs, données NORMALES uniquement)...")

callbacks = [
    keras.callbacks.EarlyStopping(
        patience=8, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        patience=4, factor=0.5, min_lr=1e-5, verbose=1),
    keras.callbacks.ModelCheckpoint(
        AE_PATH, save_best_only=True, verbose=0),
]

history = autoencoder.fit(
    X_train, X_train,       # ← entrée = sortie (reconstruction)
    validation_data=(X_val, X_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=1
)

# ── 6. Calcul des seuils ───────────────────────────────────────────────
print("\n📏 Calcul des seuils d'anomalie...")

# Erreur de reconstruction sur données NORMALES
X_train_pred = autoencoder.predict(X_train, verbose=0)
erreurs_norm = np.mean(np.power(X_train - X_train_pred, 2), axis=1)

mean_err = float(np.mean(erreurs_norm))
std_err  = float(np.std(erreurs_norm))
p95      = float(np.percentile(erreurs_norm, 95))
p99      = float(np.percentile(erreurs_norm, 99))

# Seuils calibrés
seuil_attention = mean_err + 2 * std_err   # ~95e percentile
seuil_critique  = mean_err + 4 * std_err   # ~99.9e percentile

print(f"   Erreur normale : mean={mean_err:.5f} | std={std_err:.5f}")
print(f"   P95={p95:.5f} | P99={p99:.5f}")
print(f"   Seuil ATTENTION  : {seuil_attention:.5f}")
print(f"   Seuil CRITIQUE   : {seuil_critique:.5f}")

seuils = {
    'mean_erreur_normale': mean_err,
    'std_erreur_normale' : std_err,
    'seuil_attention'    : seuil_attention,
    'seuil_critique'     : seuil_critique,
    'percentile_95'      : p95,
    'percentile_99'      : p99,
    'ae_features'        : AE_FEATURES,
    'latent_dim'         : LATENT_DIM,
}
with open(SEUILS_PATH, 'w') as f:
    json.dump(seuils, f, indent=2)
print(f"   ✅ Seuils sauvegardés")

# ── 7. Évaluation ──────────────────────────────────────────────────────
print("\n📊 Évaluation sur toutes les pompes...")

X_all_pred = autoencoder.predict(X_all, verbose=0)
erreurs_all = np.mean(np.power(X_all - X_all_pred, 2), axis=1)

df['erreur_reconstruction'] = erreurs_all
df['score_ae'] = np.clip(
    (erreurs_all - mean_err) / (seuil_critique - mean_err + 1e-8),
    0, 1
)
df['pred_alerte'] = 'NORMAL'
df.loc[erreurs_all >= seuil_attention, 'pred_alerte'] = 'ATTENTION'
df.loc[erreurs_all >= seuil_critique,  'pred_alerte'] = 'CRITIQUE'

# Résultats par pompe
print(f"\n   Erreur reconstruction par pompe :")
for pid in ['P1', 'P2', 'P3']:
    sub  = df[df['pompe_id'] == pid]
    err  = sub['erreur_reconstruction'].mean()
    n_cr = (sub['pred_alerte'] == 'CRITIQUE').sum()
    n_at = (sub['pred_alerte'] == 'ATTENTION').sum()
    n_ok = (sub['pred_alerte'] == 'NORMAL').sum()
    print(f"   {pid} : erreur={err:.5f} | "
          f"NORMAL={n_ok:,} ATTENTION={n_at:,} CRITIQUE={n_cr:,}")

# Métriques classification binaire
y_true = (df['niveau_alerte'] != 'NORMAL').astype(int)
y_pred = (df['pred_alerte']   != 'NORMAL').astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)

print(f"\n   Métriques détection anomalie (binaire) :")
print(f"   Accuracy  = {acc:.3f}")
print(f"   Precision = {prec:.3f}")
print(f"   Recall    = {rec:.3f}  ← le plus important (ne pas rater de pannes)")
print(f"   F1-Score  = {f1:.3f}")

# ── 8. Analyse par feature ─────────────────────────────────────────────
print(f"\n🔍 Contribution des features aux anomalies...")
erreurs_par_feature = np.power(X_all - X_all_pred, 2)
df_err_feat = pd.DataFrame(erreurs_par_feature, columns=AE_FEATURES)
df_err_feat['pompe_id'] = df['pompe_id'].values

print("   Erreur moyenne par feature (P1 vs P3) :")
for feat in AE_FEATURES:
    err_p1 = df_err_feat[df_err_feat['pompe_id']=='P1'][feat].mean()
    err_p3 = df_err_feat[df_err_feat['pompe_id']=='P3'][feat].mean()
    ratio  = err_p1 / (err_p3 + 1e-8)
    bar    = '█' * min(int(ratio), 20)
    print(f"   {feat:28s}: P1={err_p1:.4f} | P3={err_p3:.4f} | ratio={ratio:.1f}x  {bar}")

# ── 9. Fonction d'inférence ────────────────────────────────────────────
print("\n🔌 Test fonction d'inférence...")

def detect_anomalie(capteurs: dict) -> dict:
    """
    Entrée  : dict avec les 8 features
    Sortie  : score, alerte, feature la plus anormale

    Exemple :
    detect_anomalie({
        'efficacite_pct': 71.0,
        'vibration_mm_s': 5.2,
        'temperature_moteur_C': 83.0,
        'icp_kwh_m3': 0.74,
        'pression_entree_bar': 1.9,
        'pression_sortie_bar': 3.8,
        'courant_A': 285.0,
        'cycles_demarrage_24h': 9
    })
    """
    x      = np.array([[capteurs[f] for f in AE_FEATURES]], dtype=np.float32)
    x_norm = scaler.transform(x)
    x_pred = autoencoder.predict(x_norm, verbose=0)
    erreur = float(np.mean(np.power(x_norm - x_pred, 2)))

    # Feature la plus anormale
    err_par_feat = np.power(x_norm - x_pred, 2)[0]
    feat_anomale = AE_FEATURES[int(np.argmax(err_par_feat))]

    # Niveau alerte
    if erreur >= seuil_critique:   alerte = 'CRITIQUE'
    elif erreur >= seuil_attention: alerte = 'ATTENTION'
    else:                           alerte = 'NORMAL'

    score = float(np.clip(
        (erreur - mean_err) / (seuil_critique - mean_err + 1e-8), 0, 1))

    return {
        'erreur_reconstruction': round(erreur, 6),
        'score_anomalie'       : round(score, 4),
        'niveau_alerte'        : alerte,
        'feature_critique'     : feat_anomale,
        'seuil_attention'      : round(seuil_attention, 6),
        'seuil_critique'       : round(seuil_critique, 6),
    }

# Test pompe normale (P3)
test_normal = {
    'efficacite_pct'      : 87.5,
    'vibration_mm_s'      : 1.4,
    'temperature_moteur_C': 58.0,
    'icp_kwh_m3'          : 0.52,
    'pression_entree_bar' : 2.1,
    'pression_sortie_bar' : 4.3,
    'courant_A'           : 265.0,
    'cycles_demarrage_24h': 3
}

# Test pompe dégradée (P1)
test_anomalie = {
    'efficacite_pct'      : 69.0,
    'vibration_mm_s'      : 5.8,
    'temperature_moteur_C': 84.0,
    'icp_kwh_m3'          : 0.76,
    'pression_entree_bar' : 1.7,
    'pression_sortie_bar' : 3.5,
    'courant_A'           : 310.0,
    'cycles_demarrage_24h': 11
}

res_normal   = detect_anomalie(test_normal)
res_anomalie = detect_anomalie(test_anomalie)

print(f"\n   Test pompe NORMALE (P3 type) :")
print(f"     Alerte  : {res_normal['niveau_alerte']}")
print(f"     Score   : {res_normal['score_anomalie']}")
print(f"     Erreur  : {res_normal['erreur_reconstruction']}")

print(f"\n   Test pompe DÉGRADÉE (P1 type) :")
print(f"     Alerte  : {res_anomalie['niveau_alerte']}")
print(f"     Score   : {res_anomalie['score_anomalie']}")
print(f"     Erreur  : {res_anomalie['erreur_reconstruction']}")
print(f"     Feature critique : {res_anomalie['feature_critique']}")

# ── 10. Résumé final ───────────────────────────────────────────────────
print("\n" + "=" * 58)
print("  FICHIERS GÉNÉRÉS")
print("=" * 58)
print(f"  ✅ {AE_PATH}")
print(f"  ✅ {SCALER_PATH}")
print(f"  ✅ {SEUILS_PATH}")

summary = {
    'date'            : pd.Timestamp.now().isoformat(),
    'ae_features'     : AE_FEATURES,
    'latent_dim'      : LATENT_DIM,
    'n_train_normal'  : int(len(X_train)),
    'metrics'         : {
        'accuracy' : float(acc),
        'precision': float(prec),
        'recall'   : float(rec),
        'f1'       : float(f1),
    },
    'seuils'          : {
        'attention': float(seuil_attention),
        'critique' : float(seuil_critique),
    },
    'test_normal'     : res_normal,
    'test_anomalie'   : res_anomalie,
}
with open(os.path.join(MODEL_DIR, 'autoencoder_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✅ autoencoder_summary.json")
print("=" * 58)
