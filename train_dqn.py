"""
AQUA-AI — DQN (Deep Q-Network) v1.0
Optimisation tarifaire temps réel pour stations ONEA

Architecture :
  - Replay Buffer (experience replay)
  - Target Network (stabilité entraînement)
  - Epsilon-Greedy (exploration → exploitation)
  - 8 actions : pompes ON/OFF × source énergie

État (15 dimensions) :
  niveau_chateau, niveau_bache, tarif, heure_sin, heure_cos,
  puissance_solaire, coupure_sonabel, efficacite_p1, efficacite_p2,
  temperature, stock_diesel, icp, pred_debit_h1, pred_puissance_h1, pred_debit_h6

Actions (8) :
  0: STOP           — tout éteint
  1: POMPE1-SONABEL — pompe 1, réseau SONABEL
  2: POMPE2-SONABEL — pompe 2, réseau SONABEL
  3: POMPES12-SON   — 2 pompes, SONABEL (max débit)
  4: POMPE1-SOLAIRE — pompe 1, énergie solaire
  5: POMPE2-SOLAIRE — pompe 2, énergie solaire
  6: POMPE1-DIESEL  — pompe 1, groupe électrogène
  7: POMPES12-DIES  — 2 pompes, diesel (urgence)
"""

import os, json, random
import numpy as np
import pandas as pd
from collections import deque

# ── Chemins ────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'D_dqn_arbitrage_energetique.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
DQN_PATH  = os.path.join(MODEL_DIR, 'dqn_aqua_best.keras')
DQN_NORM  = os.path.join(MODEL_DIR, 'dqn_state_stats.json')

# ── Hyperparamètres ────────────────────────────────────────────────────
STATE_DIM    = 15
N_ACTIONS    = 8
GAMMA        = 0.95      # facteur d'actualisation
LR           = 0.001
BATCH_SIZE   = 64
BUFFER_SIZE  = 10000
TARGET_UPDATE= 100       # sync target network toutes les N steps
EPSILON_START= 1.0
EPSILON_END  = 0.05
EPSILON_DECAY= 0.995
N_EPISODES   = 50        # passages sur les données
MIN_BUFFER   = 500       # commencer entraînement après N expériences

# Colonnes état
STATE_COLS = [
    'niveau_chateau_pct', 'niveau_bache_pct',
    'tarif_fcfa_kwh',
    'heure_sin', 'heure_cos',
    'puissance_solaire_kw', 'coupure_sonabel',
    'efficacite_pompe1', 'efficacite_pompe2',
    'temperature_C', 'stock_diesel_pct', 'icp_kwh_m3',
    'pred_debit_h1', 'pred_puissance_h1', 'pred_debit_h6',
]

ACTION_NAMES = [
    'STOP',
    'POMPE1-SONABEL', 'POMPE2-SONABEL', 'POMPES12-SONABEL',
    'POMPE1-SOLAIRE', 'POMPE2-SOLAIRE',
    'POMPE1-DIESEL',  'POMPES12-DIESEL'
]
PUISSANCE = {0:0, 1:185, 2:200, 3:385, 4:185, 5:200, 6:185, 7:385}
SOURCE    = {0:'none',1:'sonabel',2:'sonabel',3:'sonabel',
             4:'solaire',5:'solaire',6:'diesel',7:'diesel'}

print("=" * 58)
print("  AQUA-AI — DQN Entraînement")
print("=" * 58)

# ── 1. Chargement données ──────────────────────────────────────────────
print(f"\n📂 Chargement : {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"   Shape : {df.shape}")

# Normalisation état (min-max)
state_stats = {}
for col in STATE_COLS:
    mn, mx = float(df[col].min()), float(df[col].max())
    state_stats[col] = {'min': mn, 'max': mx}

def normalize_state(row):
    s = []
    for col in STATE_COLS:
        mn = state_stats[col]['min']
        mx = state_stats[col]['max']
        val = float(row[col])
        s.append((val - mn) / (mx - mn + 1e-8))
    return np.array(s, dtype=np.float32)

# Sauvegarder stats normalisation
with open(DQN_NORM, 'w') as f:
    json.dump({'state_cols': STATE_COLS, 'stats': state_stats,
               'n_actions': N_ACTIONS, 'action_names': ACTION_NAMES}, f, indent=2)
print(f"   ✅ Stats normalisation sauvegardées")

# ── 2. Construction modèle Q ───────────────────────────────────────────
print("\n🏗️  Construction réseau Q...")
import tensorflow as tf
from tensorflow import keras

def build_q_network(state_dim, n_actions, name="q_network"):
    """
    Réseau fully-connected : State → Q-valeurs pour chaque action
    Architecture : 15 → 128 → 128 → 64 → 8
    """
    inp = keras.layers.Input(shape=(state_dim,), name='state_input')
    x   = keras.layers.Dense(128, activation='relu')(inp)
    x   = keras.layers.BatchNormalization()(x)
    x   = keras.layers.Dense(128, activation='relu')(x)
    x   = keras.layers.Dropout(0.1)(x)
    x   = keras.layers.Dense(64,  activation='relu')(x)
    out = keras.layers.Dense(n_actions, activation='linear', name='q_values')(x)
    model = keras.Model(inputs=inp, outputs=out, name=name)
    model.compile(optimizer=keras.optimizers.Adam(LR), loss='huber')
    return model

q_network     = build_q_network(STATE_DIM, N_ACTIONS, "q_main")
target_network= build_q_network(STATE_DIM, N_ACTIONS, "q_target")
target_network.set_weights(q_network.get_weights())
q_network.summary()

# ── 3. Replay Buffer ───────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s),  np.array(a),
                np.array(r),  np.array(ns), np.array(d))

    def __len__(self):
        return len(self.buffer)

buffer  = ReplayBuffer(BUFFER_SIZE)
epsilon = EPSILON_START

# ── 4. Fonction reward ─────────────────────────────────────────────────
def compute_reward(action, row, next_chateau, next_bache):
    tarif   = float(row['tarif_fcfa_kwh'])
    chateau = float(row['niveau_chateau_pct'])
    bache   = float(row['niveau_bache_pct'])
    solaire = float(row['puissance_solaire_kw'])
    coupure = int(row['coupure_sonabel'])
    src     = SOURCE[action]
    puiss   = PUISSANCE[action]

    # Coût de l'action
    if src == 'diesel':   cout = puiss * 300
    elif src == 'solaire': cout = 0.0
    elif src == 'none':    cout = 0.0
    else:                  cout = puiss * tarif

    r = -cout / 80000   # pénalité coût normalisée

    # Bonus/malus niveaux
    if next_chateau < 20:   r -= 8.0   # critique — coupure service imminente
    elif next_chateau < 35: r -= 2.0   # bas
    elif next_chateau > 75 and tarif == 84: r += 1.5  # bien rempli en HC
    if next_bache < 15:     r -= 5.0   # bâche critique
    elif next_bache < 25:   r -= 1.0

    # Bonus stratégique
    if src == 'solaire' and puiss > 0 and solaire > 150: r += 2.0
    if src == 'diesel' and not coupure: r -= 3.0  # diesel sans coupure = mauvaise décision
    if src == 'diesel' and coupure and chateau < 40: r += 1.0  # diesel justifié
    if tarif == 165 and src == 'sonabel' and chateau > 65: r -= 1.0  # pompage HP inutile

    # Bonus anticipation : remplir avant 17h
    heure = float(row['heure_sin'])  # proxy heure
    if tarif == 84 and puiss > 0 and next_chateau > chateau: r += 0.5

    return float(r)

# ── 5. Entraînement ────────────────────────────────────────────────────
print(f"\n🚀 Entraînement ({N_EPISODES} épisodes)...")
print(f"   Buffer min avant train : {MIN_BUFFER}")
print(f"   Epsilon : {EPSILON_START} → {EPSILON_END} (decay={EPSILON_DECAY})")

history_reward = []
history_loss   = []
step_count     = 0
best_avg_reward= -999

for episode in range(N_EPISODES):
    ep_rewards = []
    ep_losses  = []

    # Parcourir les données chronologiquement
    indices = list(range(1, len(df)-1))

    for idx in indices:
        row  = df.iloc[idx]
        state= normalize_state(row)

        # ε-greedy : exploration vs exploitation
        if random.random() < epsilon:
            action = random.randint(0, N_ACTIONS-1)
        else:
            q_vals = q_network(state.reshape(1,-1), training=False).numpy()[0]
            action = int(np.argmax(q_vals))

        # État suivant (depuis données réelles simulées)
        next_row    = df.iloc[idx+1]
        next_state  = normalize_state(next_row)
        next_chateau= float(next_row['next_niveau_chateau'])
        next_bache  = float(next_row['next_niveau_bache'])
        done        = bool(next_row['done'])

        # Reward
        reward = compute_reward(action, row, next_chateau, next_bache)
        ep_rewards.append(reward)

        # Stocker expérience
        buffer.push(state, action, reward, next_state, float(done))
        step_count += 1

        # Entraîner si buffer suffisant
        if len(buffer) >= MIN_BUFFER:
            s_b, a_b, r_b, ns_b, d_b = buffer.sample(BATCH_SIZE)

            # Q-values cibles (Bellman)
            q_next   = target_network(ns_b, training=False).numpy()
            q_target = r_b + GAMMA * np.max(q_next, axis=1) * (1 - d_b)

            # Q-values actuelles
            q_curr   = q_network(s_b, training=False).numpy()
            for i in range(BATCH_SIZE):
                q_curr[i, a_b[i]] = q_target[i]

            loss = q_network.train_on_batch(s_b, q_curr)
            ep_losses.append(float(loss))

        # Sync target network
        if step_count % TARGET_UPDATE == 0:
            target_network.set_weights(q_network.get_weights())

    # Décroissance epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    avg_r = np.mean(ep_rewards)
    avg_l = np.mean(ep_losses) if ep_losses else 0
    history_reward.append(avg_r)
    history_loss.append(avg_l)

    # Sauvegarder meilleur modèle
    if avg_r > best_avg_reward:
        best_avg_reward = avg_r
        q_network.save(DQN_PATH)

    if (episode+1) % 5 == 0:
        print(f"   Ep {episode+1:3d}/{N_EPISODES} | "
              f"Reward={avg_r:.4f} | Loss={avg_l:.4f} | "
              f"ε={epsilon:.3f} | Buffer={len(buffer)}")

# ── 6. Évaluation finale ───────────────────────────────────────────────
print("\n📊 Évaluation politique apprise...")

# Charger meilleur modèle
best_model = keras.models.load_model(DQN_PATH)

eval_actions, eval_rewards, eval_couts = [], [], []
for idx in range(1, min(8760, len(df)-1)):  # 1 an
    row   = df.iloc[idx]
    state = normalize_state(row)
    q_vals= best_model(state.reshape(1,-1), training=False).numpy()[0]
    action= int(np.argmax(q_vals))
    nxt   = df.iloc[idx+1]
    reward= compute_reward(action, row, float(nxt['next_niveau_chateau']),
                           float(nxt['next_niveau_bache']))
    src   = SOURCE[action]
    puiss = PUISSANCE[action]
    tarif = float(row['tarif_fcfa_kwh'])
    cout  = puiss * (300 if src=='diesel' else 0 if src in ['solaire','none'] else tarif)
    eval_actions.append(action)
    eval_rewards.append(reward)
    eval_couts.append(cout)

from collections import Counter
dist = Counter(eval_actions)
cout_total  = sum(eval_couts)
cout_hp     = sum(c for i,c in enumerate(eval_couts) if df.iloc[i+1]['tarif_fcfa_kwh']==165)
eco_annuelle= (sum(PUISSANCE[eval_actions[i]] * 165
                   for i in range(len(eval_actions))) - cout_total)

print(f"\n   Distribution actions (1 an simulé) :")
for a, cnt in sorted(dist.items()):
    pct = cnt/len(eval_actions)*100
    print(f"     {ACTION_NAMES[a]:20s} : {cnt:5d} ({pct:.1f}%)")

print(f"\n   Reward moyen       : {np.mean(eval_rewards):.4f}")
print(f"   Coût total annuel  : {cout_total/1e6:.1f}M FCFA")
print(f"   Économie vs tout-HP: {eco_annuelle/1e6:.0f}M FCFA")

# ── 7. Résumé ──────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("  FICHIERS GÉNÉRÉS")
print("=" * 58)
print(f"  ✅ {DQN_PATH}")
print(f"  ✅ {DQN_NORM}")

summary = {
    "date": pd.Timestamp.now().isoformat(),
    "episodes": N_EPISODES,
    "state_dim": STATE_DIM,
    "n_actions": N_ACTIONS,
    "action_names": ACTION_NAMES,
    "best_avg_reward": float(best_avg_reward),
    "eval_reward_moyen": float(np.mean(eval_rewards)),
    "cout_annuel_M_fcfa": float(cout_total/1e6),
    "epsilon_final": float(epsilon),
    "history_reward": [float(r) for r in history_reward],
}
with open(os.path.join(MODEL_DIR, 'dqn_training_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  ✅ {os.path.join(MODEL_DIR, 'dqn_training_summary.json')}")
print("=" * 58)
