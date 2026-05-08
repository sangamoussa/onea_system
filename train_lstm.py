"""
AQUA-AI — Script d'entraînement LSTM v2
Corrections :
  - ✅ Double normalisation corrigée (features et targets séparés proprement)
  - ✅ Erreur JSON float32 corrigée
  - ✅ Colonnes adaptées au nouveau CSV enrichi

Usage :
    python train_lstm_v2.py
    python train_lstm_v2.py --quick   (10 epochs pour tester)
"""

import os, sys, argparse, json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ── Arguments ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--csv',    default='data/A_energie_horaire_station_2021_2024.csv')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--quick',  action='store_true')
args = parser.parse_args()

if args.quick:
    args.epochs = 10
    print("⚡ Mode QUICK — 10 epochs")

# ── Chemins ────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, args.csv)
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH   = os.path.join(MODEL_DIR, 'lstm_aqua_best.keras')
SCALERX_PATH = os.path.join(MODEL_DIR, 'scaler_X.pkl')
SCALERY_PATH = os.path.join(MODEL_DIR, 'scalers_y.pkl')

# ── Config ─────────────────────────────────────────────────────────────
WINDOW_IN   = 48
HORIZON_OUT = 24
BATCH       = 64
PATIENCE    = 15
LR          = 0.001

# Features d'entrée — NE contiennent PAS les targets pour éviter la double normalisation
FEATURES = [
    'puissance_totale_kw',       # sera normalisé par scaler_X
    'temperature_C',
    'humidite_pct',
    'heure_sin', 'heure_cos',
    'mois_sin',  'mois_cos',
    'is_weekend',
    'is_ferie',
    'tarif_sonabel_fcfa_kwh',
    'part_solaire_pct',
    'part_diesel_pct',
    'puissance_solaire_kw',
    'niveau_chateau_pct',        # ✅ nouvelle feature physique
]

# Targets — colonnes à prédire (scalers séparés, jamais dans FEATURES)
TARGETS = [
    'debit_refoulement_m3h',     # → Agent Pompage
    'puissance_totale_kw',       # → Agent Tarif
    'cout_total_fcfa',           # → Reporting
]

# ⚠️ Note : puissance_totale_kw est à la fois feature ET target.
# On la normalise avec scaler_X pour les features, et scalers_y pour les targets.
# Cela fonctionne car on travaille sur des copies séparées du dataframe.

print("=" * 55)
print("  AQUA-AI — Entraînement LSTM v2")
print("=" * 55)

# ── 1. Chargement ──────────────────────────────────────────────────────
print(f"\n📂 Chargement : {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(f"❌ Fichier introuvable : {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"   Shape : {df.shape}")

manquantes = [c for c in FEATURES + TARGETS if c not in df.columns]
if manquantes:
    print(f"❌ Colonnes manquantes : {manquantes}")
    sys.exit(1)

# Split temporel strict
train_df = df[df['year'] <= 2023].copy().reset_index(drop=True)
test_df  = df[df['year'] == 2024].copy().reset_index(drop=True)
print(f"   Train 2021–2023 : {len(train_df):,} heures")
print(f"   Test  2024      : {len(test_df):,} heures")

# ── 2. Normalisation — CLEF : séparer features et targets ─────────────
print("\n📐 Normalisation...")

# Scaler FEATURES sur les colonnes features uniquement
scaler_X = StandardScaler()
train_X = train_df[FEATURES].values.astype(np.float32)
test_X  = test_df[FEATURES].values.astype(np.float32)
train_X_norm = scaler_X.fit_transform(train_X)
test_X_norm  = scaler_X.transform(test_X)

# Scaler TARGETS sur les colonnes targets — sur les données BRUTES (avant normalisation features)
scalers_y = {}
train_Y = {}
test_Y  = {}
for col in TARGETS:
    scalers_y[col] = StandardScaler()
    train_Y[col] = scalers_y[col].fit_transform(
        train_df[[col]].values.astype(np.float32)
    )
    test_Y[col] = scalers_y[col].transform(
        test_df[[col]].values.astype(np.float32)
    )
    print(f"   {col}: mean={scalers_y[col].mean_[0]:.1f} | scale={scalers_y[col].scale_[0]:.1f}")

# Sauvegarder scalers
joblib.dump(scaler_X,  SCALERX_PATH)
joblib.dump(scalers_y, SCALERY_PATH)
print(f"   ✅ scaler_X.pkl  sauvegardé")
print(f"   ✅ scalers_y.pkl sauvegardé")

# ── 3. Séquences glissantes ────────────────────────────────────────────
print("\n🔢 Création des séquences...")

def make_sequences(X_norm, Y_dict, targets, win_in, hor_out):
    n = len(X_norm)
    # Assembler Y en array (N, n_targets)
    Y_all = np.concatenate([Y_dict[col] for col in targets], axis=1)
    X_list, Y_list = [], []
    for i in range(win_in, n - hor_out + 1):
        X_list.append(X_norm[i - win_in : i])
        Y_list.append(Y_all[i : i + hor_out])
    return np.array(X_list), np.array(Y_list)

X_train, y_train = make_sequences(train_X_norm, train_Y, TARGETS, WINDOW_IN, HORIZON_OUT)
X_test,  y_test  = make_sequences(test_X_norm,  test_Y,  TARGETS, WINDOW_IN, HORIZON_OUT)

print(f"   X_train : {X_train.shape}  (échantillons, {WINDOW_IN}h, {len(FEATURES)} features)")
print(f"   y_train : {y_train.shape}  (échantillons, {HORIZON_OUT}h, {len(TARGETS)} targets)")
print(f"   X_test  : {X_test.shape}")

# ── 4. Modèle ──────────────────────────────────────────────────────────
print("\n🏗️  Construction du modèle...")
import tensorflow as tf
from tensorflow import keras

def build_model(win_in, n_features, hor_out, n_targets):
    inp = keras.layers.Input(shape=(win_in, n_features))
    x = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True, dropout=0.2))(inp)
    x = keras.layers.Bidirectional(
            keras.layers.LSTM(64,  return_sequences=True, dropout=0.2))(x)
    x = keras.layers.LSTM(32, return_sequences=False)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation='relu')(x)

    outputs = []
    names = ['debit', 'puissance', 'cout']
    for i in range(n_targets):
        out = keras.layers.Dense(hor_out, activation='linear', name=f'out_{i}_{names[i]}')(x)
        out = keras.layers.Reshape((hor_out, 1))(out)
        outputs.append(out)

    merged = keras.layers.Concatenate(axis=-1)(outputs)
    model  = keras.Model(inputs=inp, outputs=merged)
    model.compile(optimizer=keras.optimizers.Adam(LR), loss='huber', metrics=['mae'])
    return model

model = build_model(WINDOW_IN, len(FEATURES), HORIZON_OUT, len(TARGETS))
model.summary()

# ── 5. Entraînement ────────────────────────────────────────────────────
print(f"\n🚀 Entraînement ({args.epochs} epochs max, patience={PATIENCE})...")

callbacks = [
    keras.callbacks.EarlyStopping(
        patience=PATIENCE, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        patience=7, factor=0.5, min_lr=1e-5, verbose=1),
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH, save_best_only=True, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=args.epochs,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=1
)

# ── 6. Évaluation ──────────────────────────────────────────────────────
print("\n📊 Évaluation sur données 2024...")

def denormalize(y_norm_slice, scaler):
    return scaler.inverse_transform(
        y_norm_slice.ravel().reshape(-1, 1)
    ).ravel()

y_pred   = model.predict(X_test, verbose=0)
resultats = {}

for i, col in enumerate(TARGETS):
    true_r = denormalize(y_test[:, :, i], scalers_y[col])
    pred_r = denormalize(y_pred[:, :, i], scalers_y[col])
    mae    = float(mean_absolute_error(true_r, pred_r))
    r2     = float(r2_score(true_r, pred_r))
    smape  = float(100 * np.mean(
        2*np.abs(pred_r - true_r) / (np.abs(true_r) + np.abs(pred_r) + 1e-8)
    ))
    resultats[col] = {'mae': mae, 'r2': r2, 'smape': smape}
    status = "✅" if r2 > 0.85 else "⚠️ "
    print(f"   {status} {col}")
    print(f"      MAE   = {mae:,.1f}")
    print(f"      R²    = {r2:.3f}  (cible > 0.85)")
    print(f"      sMAPE = {smape:.1f}%")

# ── 7. Résumé ──────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  FICHIERS GÉNÉRÉS")
print("=" * 55)
print(f"  ✅ {MODEL_PATH}")
print(f"  ✅ {SCALERX_PATH}")
print(f"  ✅ {SCALERY_PATH}")
print("\n  Relancez uvicorn — mode REEL s'active automatiquement.")
print("=" * 55)

# ── 8. Résumé JSON — float32 converti en float natif ──────────────────
summary = {
    "date_entrainement" : pd.Timestamp.now().isoformat(),
    "csv_source"        : args.csv,
    "n_train"           : int(len(X_train)),
    "n_test"            : int(len(X_test)),
    "features"          : FEATURES,
    "targets"           : TARGETS,
    "window_in"         : WINDOW_IN,
    "horizon_out"       : HORIZON_OUT,
    "resultats"         : {
        col: {k: float(v) for k, v in vals.items()}   # ✅ float32 → float
        for col, vals in resultats.items()
    }
}
summary_path = os.path.join(MODEL_DIR, 'training_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\n  📄 Résumé → {summary_path}")
