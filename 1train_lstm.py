"""
AQUA-AI — Script d'entraînement LSTM
Génère les 3 fichiers dans models/ :
  - lstm_aqua_best.keras
  - scaler_X.pkl
  - scalers_y.pkl

Usage :
    python train_lstm.py
    python train_lstm.py --csv data/mon_fichier.csv
    python train_lstm.py --epochs 50 --quick
"""

import os, sys, argparse, json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ── Arguments ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Entraînement LSTM AQUA-AI")
parser.add_argument('--csv',    default='data/A_energie_horaire_station_2021_2024.csv')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--quick',  action='store_true',
                    help='Mode rapide : 10 epochs, pour vérifier que tout fonctionne')
args = parser.parse_args()

if args.quick:
    args.epochs = 10
    print("⚡ Mode QUICK activé — 10 epochs seulement (pour test)")

# ── Chemins ────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, args.csv)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(MODEL_DIR, 'lstm_aqua_best.keras')
SCALERX_PATH = os.path.join(MODEL_DIR, 'scaler_X.pkl')
SCALERY_PATH = os.path.join(MODEL_DIR, 'scalers_y.pkl')

# ── Hyperparamètres (identiques au notebook) ───────────────────────────
WINDOW_IN   = 48
HORIZON_OUT = 24
BATCH       = 64
PATIENCE    = 15
LR          = 0.001

FEATURES = [
    'debit_refoulement_m3h',
    'puissance_totale_kw',
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
]

TARGETS = [
    'debit_refoulement_m3h',
    'puissance_totale_kw',
    'cout_total_fcfa',
]

print("=" * 55)
print("  AQUA-AI — Entraînement LSTM")
print("=" * 55)

# ── 1. Chargement données ──────────────────────────────────────────────
print(f"\n📂 Chargement : {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(f"❌ Fichier introuvable : {DATA_PATH}")
    print("   Vérifiez le chemin ou utilisez --csv <chemin>")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"   Shape      : {df.shape}")

# Vérifier colonnes manquantes
manquantes = [c for c in FEATURES + TARGETS if c not in df.columns]
if manquantes:
    print(f"❌ Colonnes manquantes dans le CSV : {manquantes}")
    print(f"   Colonnes disponibles : {list(df.columns)}")
    sys.exit(1)

# Split temporel : 2021-2023 train / 2024 test
train_df = df[df['year'] <= 2023].reset_index(drop=True)
test_df  = df[df['year'] == 2024].reset_index(drop=True)
print(f"   Train 2021–2023 : {len(train_df):,} heures")
print(f"   Test  2024      : {len(test_df):,} heures")

# Vérification valeurs négatives
for col in ['puissance_totale_kw', 'debit_refoulement_m3h']:
    neg = (df[col] < 0).sum()
    if neg > 0:
        print(f"⚠️  {col} : {neg} valeurs négatives (remplacées par 0)")
        df[col] = df[col].clip(lower=0)

# ── 2. Normalisation ───────────────────────────────────────────────────
print("\n📐 Normalisation...")
scaler_X  = StandardScaler()
scalers_y = {col: StandardScaler() for col in TARGETS}

train_df[FEATURES] = scaler_X.fit_transform(train_df[FEATURES])
test_df[FEATURES]  = scaler_X.transform(test_df[FEATURES])

for col in TARGETS:
    train_df[[col]] = scalers_y[col].fit_transform(train_df[[col]])
    test_df[[col]]  = scalers_y[col].transform(test_df[[col]])
    print(f"   {col}: mean={scalers_y[col].mean_[0]:.1f} | scale={scalers_y[col].scale_[0]:.1f}")

# Sauvegarder scalers MAINTENANT (avant entraînement, déjà fittés)
joblib.dump(scaler_X,  SCALERX_PATH)
joblib.dump(scalers_y, SCALERY_PATH)
print(f"   ✅ scaler_X.pkl  → {SCALERX_PATH}")
print(f"   ✅ scalers_y.pkl → {SCALERY_PATH}")

# ── 3. Création séquences ──────────────────────────────────────────────
print("\n🔢 Création des séquences glissantes...")

def make_sequences(df, features, targets, win_in, hor_out):
    feat = df[features].values.astype(np.float32)
    tgt  = df[targets].values.astype(np.float32)
    X, y = [], []
    for i in range(win_in, len(df) - hor_out + 1):
        X.append(feat[i - win_in : i])
        y.append(tgt[i : i + hor_out])
    return np.array(X), np.array(y)

X_train, y_train = make_sequences(train_df, FEATURES, TARGETS, WINDOW_IN, HORIZON_OUT)
X_test,  y_test  = make_sequences(test_df,  FEATURES, TARGETS, WINDOW_IN, HORIZON_OUT)

print(f"   X_train : {X_train.shape}  (échantillons, {WINDOW_IN}h, {len(FEATURES)} features)")
print(f"   y_train : {y_train.shape}  (échantillons, {HORIZON_OUT}h, {len(TARGETS)} targets)")
print(f"   X_test  : {X_test.shape}")

# ── 4. Construction modèle ─────────────────────────────────────────────
print("\n🏗️  Construction du modèle...")
import tensorflow as tf
from tensorflow import keras

def build_model(win_in, n_features, hor_out, n_targets):
    inp = keras.layers.Input(shape=(win_in, n_features))

    # Encodeur BiLSTM
    x = keras.layers.Bidirectional(
            keras.layers.LSTM(128, return_sequences=True, dropout=0.2))(inp)
    x = keras.layers.Bidirectional(
            keras.layers.LSTM(64,  return_sequences=True, dropout=0.2))(x)
    x = keras.layers.LSTM(32, return_sequences=False)(x)

    # Tête dense partagée
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation='relu')(x)

    # 3 sorties indépendantes (une par target) — activation LINEAR
    outputs = []
    target_names = ['debit_ea', 'puissanc', 'cout_tot']
    for i in range(n_targets):
        out = keras.layers.Dense(hor_out, activation='linear',
                                 name=f'out_{i}_{target_names[i]}')(x)
        out = keras.layers.Reshape((hor_out, 1))(out)
        outputs.append(out)

    merged = keras.layers.Concatenate(axis=-1)(outputs)
    model  = keras.Model(inputs=inp, outputs=merged)

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss='huber',
        metrics=['mae']
    )
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

def denormalize(y_norm, scaler):
    return scaler.inverse_transform(y_norm.ravel().reshape(-1,1)).ravel()

y_pred = model.predict(X_test, verbose=0)
results = {}

for i, col in enumerate(TARGETS):
    true_r = denormalize(y_test[:,:,i], scalers_y[col])
    pred_r = denormalize(y_pred[:,:,i], scalers_y[col])
    mae    = mean_absolute_error(true_r, pred_r)
    r2     = r2_score(true_r, pred_r)
    smape  = 100 * np.mean(2*np.abs(pred_r-true_r)/(np.abs(true_r)+np.abs(pred_r)+1e-8))
    results[col] = {'mae': mae, 'r2': r2, 'smape': smape}
    status = "✅" if r2 > 0.85 else "⚠️ "
    print(f"   {status} {col}")
    print(f"      MAE   = {mae:,.1f}")
    print(f"      R²    = {r2:.3f}  (cible > 0.85)")
    print(f"      sMAPE = {smape:.1f}%")

# ── 7. Résumé final ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  FICHIERS GÉNÉRÉS")
print("=" * 55)
print(f"  ✅ {MODEL_PATH}")
print(f"  ✅ {SCALERX_PATH}")
print(f"  ✅ {SCALERY_PATH}")
print("\n  Copiez ces 3 fichiers dans le dossier models/ de l'API")
print("  puis relancez uvicorn — le mode REEL s'activera automatiquement.")
print("=" * 55)

# Sauvegarder le résumé JSON
summary = {
    "date_entrainement": pd.Timestamp.now().isoformat(),
    "csv_source": args.csv,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "features": FEATURES,
    "targets": TARGETS,
    "window_in": WINDOW_IN,
    "horizon_out": HORIZON_OUT,
    "resultats": results
}
with open(os.path.join(MODEL_DIR, 'training_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n  📄 Résumé → {os.path.join(MODEL_DIR, 'training_summary.json')}")
