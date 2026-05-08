"""
AQUA-AI — DQN v2 (corrigé)
Corrections :
  - Reward normalisé [-1, +1]  → loss stable
  - Gradient clipping          → plus d'explosion
  - 10 épisodes au lieu de 50  → rapide sur CPU (~10 min)
  - Sous-échantillonnage        → 8 760 steps/épisode (1 an)
"""

import os, json, random
import numpy as np
import pandas as pd
from collections import deque

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'D_dqn_arbitrage_energetique.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
DQN_PATH  = os.path.join(MODEL_DIR, 'dqn_aqua_best.keras')
DQN_NORM  = os.path.join(MODEL_DIR, 'dqn_state_stats.json')

# ── Hyperparamètres corrigés ───────────────────────────────────────────
STATE_DIM    = 15
N_ACTIONS    = 8
GAMMA        = 0.95
LR           = 0.0005       # ✅ réduit (était 0.001)
BATCH_SIZE   = 32           # ✅ réduit (était 64)
BUFFER_SIZE  = 5000
TARGET_UPDATE= 200
EPSILON_START= 1.0
EPSILON_END  = 0.05
EPSILON_DECAY= 0.990        # ✅ décroissance plus rapide
N_EPISODES   = 10           # ✅ 10 au lieu de 50
MIN_BUFFER   = 200
STEPS_PAR_EP = 8760         # ✅ 1 an de données par épisode

STATE_COLS = [
    'niveau_chateau_pct', 'niveau_bache_pct',
    'tarif_fcfa_kwh', 'heure_sin', 'heure_cos',
    'puissance_solaire_kw', 'coupure_sonabel',
    'efficacite_pompe1', 'efficacite_pompe2',
    'temperature_C', 'stock_diesel_pct', 'icp_kwh_m3',
    'pred_debit_h1', 'pred_puissance_h1', 'pred_debit_h6',
]

ACTION_NAMES = [
    'STOP', 'POMPE1-SONABEL', 'POMPE2-SONABEL', 'POMPES12-SONABEL',
    'POMPE1-SOLAIRE', 'POMPE2-SOLAIRE', 'POMPE1-DIESEL', 'POMPES12-DIESEL'
]
PUISSANCE = {0:0, 1:185, 2:200, 3:385, 4:185, 5:200, 6:185, 7:385}
SOURCE    = {0:'none', 1:'sonabel', 2:'sonabel', 3:'sonabel',
             4:'solaire', 5:'solaire', 6:'diesel', 7:'diesel'}

print("=" * 58)
print("  AQUA-AI — DQN v2 (optimisé CPU)")
print("=" * 58)

# ── 1. Données ─────────────────────────────────────────────────────────
print(f"\n📂 Chargement...")
df = pd.read_csv(DATA_PATH)
print(f"   Shape : {df.shape}")

# Normalisation état min-max
state_stats = {}
for col in STATE_COLS:
    mn = float(df[col].min())
    mx = float(df[col].max())
    state_stats[col] = {'min': mn, 'max': mx}

def normalize_state(row):
    s = []
    for col in STATE_COLS:
        mn  = state_stats[col]['min']
        mx  = state_stats[col]['max']
        val = float(row[col])
        s.append((val - mn) / (mx - mn + 1e-8))
    return np.array(s, dtype=np.float32)

with open(DQN_NORM, 'w') as f:
    json.dump({'state_cols': STATE_COLS, 'stats': state_stats,
               'n_actions': N_ACTIONS, 'action_names': ACTION_NAMES}, f, indent=2)

# ── 2. Modèle ──────────────────────────────────────────────────────────
print("\n🏗️  Construction réseau Q...")
import tensorflow as tf
from tensorflow import keras

# ✅ Gradient clipping dans l'optimizer
optimizer = keras.optimizers.Adam(LR, clipnorm=1.0)

def build_q_network(state_dim, n_actions, name="q"):
    inp = keras.layers.Input(shape=(state_dim,))
    x   = keras.layers.Dense(64, activation='relu')(inp)   # ✅ 64 au lieu de 128
    x   = keras.layers.Dense(64, activation='relu')(x)
    out = keras.layers.Dense(n_actions, activation='linear')(x)
    m   = keras.Model(inputs=inp, outputs=out, name=name)
    m.compile(optimizer=optimizer, loss='huber')
    return m

q_net    = build_q_network(STATE_DIM, N_ACTIONS, "q_main")
q_target = build_q_network(STATE_DIM, N_ACTIONS, "q_target")
q_target.set_weights(q_net.get_weights())
q_net.summary()

# ── 3. Replay Buffer ───────────────────────────────────────────────────
buffer  = deque(maxlen=BUFFER_SIZE)
epsilon = EPSILON_START

def push(s, a, r, ns, d):
    buffer.append((s, a, r, ns, d))

def sample_batch():
    batch = random.sample(buffer, BATCH_SIZE)
    s,a,r,ns,d = zip(*batch)
    return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
            np.array(ns), np.array(d, dtype=np.float32))

# ── 4. Reward normalisé [-1, +1] ───────────────────────────────────────
def compute_reward(action, row, next_chateau, next_bache):
    tarif   = float(row['tarif_fcfa_kwh'])
    chateau = float(row['niveau_chateau_pct'])
    solaire = float(row['puissance_solaire_kw'])
    coupure = int(row['coupure_sonabel'])
    src     = SOURCE[action]
    puiss   = PUISSANCE[action]

    # Coût normalisé sur [-0.5, 0]
    cout_max = 385 * 300   # max diesel 2 pompes
    if src == 'diesel':    cout = puiss * 300
    elif src == 'solaire': cout = 0.0
    elif src == 'none':    cout = 0.0
    else:                  cout = puiss * tarif
    r = -0.5 * cout / cout_max

    # Niveaux [-0.5, +0.5]
    if next_chateau < 20:    r -= 0.50
    elif next_chateau < 35:  r -= 0.20
    elif next_chateau > 75 and tarif == 84: r += 0.20
    if next_bache < 15:      r -= 0.40
    elif next_bache < 25:    r -= 0.10

    # Bonus stratégiques [-0.3, +0.3]
    if src == 'solaire' and puiss > 0 and solaire > 150: r += 0.30
    if src == 'diesel' and not coupure:                  r -= 0.30
    if src == 'diesel' and coupure and chateau < 40:     r += 0.15
    if tarif == 165 and src == 'sonabel' and chateau > 65: r -= 0.10

    return float(np.clip(r, -1.0, 1.0))   # ✅ clippé [-1, +1]

# ── 5. Entraînement ────────────────────────────────────────────────────
print(f"\n🚀 Entraînement ({N_EPISODES} épisodes × {STEPS_PAR_EP} steps)...")
print(f"   Durée estimée : ~5–15 min sur CPU")

history_reward, history_loss = [], []
step_count     = 0
best_avg_reward= -999

for episode in range(N_EPISODES):
    ep_rewards, ep_losses = [], []

    # Sous-échantillon aléatoire de STEPS_PAR_EP lignes
    indices = np.random.choice(range(1, len(df)-1), STEPS_PAR_EP, replace=False)

    for idx in indices:
        row   = df.iloc[idx]
        state = normalize_state(row)

        # ε-greedy
        if random.random() < epsilon:
            action = random.randint(0, N_ACTIONS-1)
        else:
            q_vals = q_net(state.reshape(1,-1), training=False).numpy()[0]
            action = int(np.argmax(q_vals))

        next_row     = df.iloc[min(idx+1, len(df)-1)]
        next_state   = normalize_state(next_row)
        next_chateau = float(next_row['next_niveau_chateau'])
        next_bache   = float(next_row['next_niveau_bache'])
        done         = bool(next_row['done'])

        reward = compute_reward(action, row, next_chateau, next_bache)
        ep_rewards.append(reward)

        push(state, action, reward, next_state, float(done))
        step_count += 1

        # Train
        if len(buffer) >= MIN_BUFFER:
            s_b, a_b, r_b, ns_b, d_b = sample_batch()
            q_next   = q_target(ns_b, training=False).numpy()
            q_target_val = r_b + GAMMA * np.max(q_next, axis=1) * (1 - d_b)
            # ✅ Clipper les cibles Q
            q_target_val = np.clip(q_target_val, -10, 10)
            q_curr = q_net(s_b, training=False).numpy()
            for i in range(BATCH_SIZE):
                q_curr[i, a_b[i]] = q_target_val[i]
            loss = q_net.train_on_batch(s_b, q_curr)
            ep_losses.append(float(loss))

        if step_count % TARGET_UPDATE == 0:
            q_target.set_weights(q_net.get_weights())

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    avg_r = float(np.mean(ep_rewards))
    avg_l = float(np.mean(ep_losses)) if ep_losses else 0.0
    history_reward.append(avg_r)
    history_loss.append(avg_l)

    if avg_r > best_avg_reward:
        best_avg_reward = avg_r
        q_net.save(DQN_PATH)

    print(f"   Ep {episode+1:2d}/{N_EPISODES} | "
          f"Reward={avg_r:.4f} | Loss={avg_l:.4f} | "
          f"ε={epsilon:.3f} | Buffer={len(buffer)}")

# ── 6. Évaluation ──────────────────────────────────────────────────────
print("\n📊 Évaluation politique apprise...")
best_model = keras.models.load_model(DQN_PATH)

eval_actions, eval_rewards = [], []
for idx in range(1, min(8760, len(df)-1)):
    row   = df.iloc[idx]
    state = normalize_state(row)
    q_v   = best_model(state.reshape(1,-1), training=False).numpy()[0]
    action= int(np.argmax(q_v))
    nxt   = df.iloc[idx+1]
    r     = compute_reward(action, row,
                           float(nxt['next_niveau_chateau']),
                           float(nxt['next_niveau_bache']))
    eval_actions.append(action)
    eval_rewards.append(r)

from collections import Counter
dist = Counter(eval_actions)
print(f"\n   Distribution actions (1 an) :")
for a, cnt in sorted(dist.items()):
    print(f"     {ACTION_NAMES[a]:22s}: {cnt:5d} ({cnt/len(eval_actions)*100:.1f}%)")
print(f"\n   Reward moyen final : {np.mean(eval_rewards):.4f}")
print(f"   Meilleur reward ep : {best_avg_reward:.4f}")

# ── 7. Résumé ──────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("  FICHIERS GÉNÉRÉS")
print("=" * 58)
print(f"  ✅ {DQN_PATH}")
print(f"  ✅ {DQN_NORM}")

summary = {
    'date': pd.Timestamp.now().isoformat(),
    'episodes': N_EPISODES,
    'steps_par_episode': STEPS_PAR_EP,
    'best_avg_reward': float(best_avg_reward),
    'eval_reward_moyen': float(np.mean(eval_rewards)),
    'epsilon_final': float(epsilon),
    'action_names': ACTION_NAMES,
    'history_reward': [float(r) for r in history_reward],
    'history_loss':   [float(l) for l in history_loss],
}
with open(os.path.join(MODEL_DIR, 'dqn_training_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✅ dqn_training_summary.json")
print("=" * 58)
