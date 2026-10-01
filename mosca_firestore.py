"""
mosca_firestore.py (v2 - mas precision, mas honestidad, mas realismo)
---------------------------------------------------------------------------
Mejoras respecto a la version anterior:

1. DATOS: la entrada ahora tiene 2 variables (retorno de precio + cambio
   de volumen), no solo precio.
2. VALIDACION: en vez de un solo split 80/20 (que puede ser puro azar),
   se hace validacion "walk-forward" con varios cortes cronologicos y se
   promedia - un numero mucho mas honesto de que tan bien funciona esto
   de verdad.
3. MODELO: se prueban 3 radios espectrales distintos para el reservorio
   (0.7, 0.9, 1.1) usando walk-forward, y se usa el mejor. El alpha de
   Ridge se elige automaticamente con validacion cruzada (RidgeCV) en
   vez de estar fijo a mano.
4. DECISION: la direccion (sube/baja/dividido) se sigue decidiendo por
   mayoria de votos como antes (simple, interpretable), pero ahora el
   TAMANO de la posicion se escala por que tan seguras estan las moscas
   (confianza = que tan lejos del 50% esta el promedio de sus
   predicciones), en vez de apostar SIEMPRE el 100% del capital.
5. REALISMO: se descuenta una comision simulada (0.3%, tipico de un swap
   en un DEX como Jupiter) en cada compra/venta, para que el portfolio
   que veas ya no sea el numero "ideal" sino uno mas parecido a lo que
   pasaria con plata real.

Sigue siendo 100% paper trading - no toca ninguna wallet ni clave real.

Uso:
  python mosca_firestore.py train   (entrena y reemplaza el modelo/estado)
  python mosca_firestore.py cycle   (un ciclo de trading - lo corre GitHub
                                      Actions cada 15 min)
"""

import json
import os
import sys
from datetime import datetime, timezone

import firebase_admin
import numpy as np
import requests
from firebase_admin import credentials, firestore
from sklearn.linear_model import RidgeCV

# ========================= CONFIG =========================

COIN_ID = "solana"
VS_CURRENCY = "usd"
HISTORICAL_DAYS = 90
WASHOUT = 50

SPECTRAL_RADIUS_CANDIDATES = [0.7, 0.9, 1.1]
RIDGE_ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0]
N_WALKFORWARD_FOLDS = 5  # cortes cronologicos para validar de forma honesta

N_FLIES = 5
FLY_SEED_BASE = 42
MIN_AGREEMENT = 0.6           # % minimo de votos en el mismo sentido para actuar
FEE_RATE = 0.003              # 0.3% simulado por operacion (tipico de un swap en Jupiter)
MAX_POSITION_FRACTION = 1.0   # tope de cuanto del cash se arriesga aunque la confianza sea maxima

STARTING_BALANCE_USD = 1000.0
ADJACENCY_FILE = "fly_adjacency.npy"


def init_firestore():
    cred_json = os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"]
    cred = credentials.Certificate(json.loads(cred_json))
    firebase_admin.initialize_app(cred)
    return firestore.client()


# ========================= DATOS =========================

def fetch_historical():
    print(f"Descargando {HISTORICAL_DAYS} dias de precio y volumen de {COIN_ID}...")
    resp = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{COIN_ID}/market_chart",
        params={"vs_currency": VS_CURRENCY, "days": str(HISTORICAL_DAYS)},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    prices = np.array([p[1] for p in data["prices"]])
    volumes = np.array([v[1] for v in data["total_volumes"]])
    print(f"Puntos: {len(prices)}")
    return prices, volumes


def fetch_live_price_and_volume():
    for attempt in range(3):
        try:
            resp = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": COIN_ID, "vs_currencies": VS_CURRENCY, "include_24hr_vol": "true"},
                timeout=30,
            )
            resp.raise_for_status()
            d = resp.json()[COIN_ID]
            return float(d["usd"]), float(d["usd_24h_vol"])
        except Exception as e:
            print(f"  [aviso] fallo consultando precio/volumen (intento {attempt + 1}/3): {e}")
    return None, None


# ========================= RESERVORIO =========================

def load_connectome_raw():
    return np.load(ADJACENCY_FILE)


def normalize_spectral_radius(W_raw, target_radius):
    sr = np.max(np.abs(np.linalg.eigvals(W_raw)))
    return W_raw * (target_radius / sr)


def build_fly_win(seed, n_neurons, n_inputs=2):
    rng = np.random.RandomState(seed)
    return rng.uniform(-1, 1, size=(n_neurons, n_inputs)) * 0.5


def step_reservoir(W, W_in, x, input_vec):
    return np.tanh(W @ x + W_in @ input_vec)


def run_reservoir_sequence(W, W_in, n_neurons, inputs, x0=None):
    states = np.zeros((len(inputs), n_neurons))
    x = np.zeros(n_neurons) if x0 is None else x0.copy()
    for i in range(len(inputs)):
        x = step_reservoir(W, W_in, x, inputs[i])
        states[i] = x
    return states, x


# ========================= VALIDACION WALK-FORWARD =========================

def walkforward_folds(n_points, n_folds):
    """Genera cortes cronologicos: cada fold usa todo lo anterior para
    entrenar y un tramo nuevo para probar. Nunca mira el futuro."""
    fold_size = n_points // (n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        test_end = fold_size * (k + 1)
        if test_end > n_points:
            break
        folds.append((train_end, test_end))
    return folds


def evaluate_spectral_radius(W_raw, radius, n_neurons, inputs, targets, seed):
    """Corre walk-forward para UN radio espectral candidato, con una sola
    mosca 'sonda' (mas barato que probarlo con las 5)."""
    W = W_raw * (radius / np.max(np.abs(np.linalg.eigvals(W_raw))))
    W_in = build_fly_win(seed, n_neurons)
    states, _ = run_reservoir_sequence(W, W_in, n_neurons, inputs)

    accs = []
    for train_end, test_end in walkforward_folds(len(inputs), N_WALKFORWARD_FOLDS):
        X_train = states[WASHOUT:train_end - 1]
        y_train = targets[WASHOUT + 1:train_end]
        X_test = states[train_end:test_end - 1]
        y_test = targets[train_end + 1:test_end]
        if len(X_train) < 10 or len(X_test) < 5:
            continue
        readout = RidgeCV(alphas=RIDGE_ALPHAS)
        readout.fit(X_train, y_train)
        pred = (readout.predict(X_test) > 0.5).astype(int)
        accs.append((pred == y_test).mean())
    return float(np.mean(accs)) if accs else 0.0


def train_one_fly_walkforward(seed, W, n_neurons, inputs, targets):
    """Entrena una mosca y devuelve tanto el modelo final (entrenado con
    TODOS los datos, para produccion) como su precision walk-forward
    honesta (promediada sobre varios cortes, no un solo split)."""
    W_in = build_fly_win(seed, n_neurons)
    states, warmed_x = run_reservoir_sequence(W, W_in, n_neurons, inputs)

    fold_accs = []
    fold_preds_for_hive = []  # para calcular precision de LA COLMENA mas adelante
    for train_end, test_end in walkforward_folds(len(inputs), N_WALKFORWARD_FOLDS):
        X_train = states[WASHOUT:train_end - 1]
        y_train = targets[WASHOUT + 1:train_end]
        X_test = states[train_end:test_end - 1]
        y_test = targets[train_end + 1:test_end]
        if len(X_train) < 10 or len(X_test) < 5:
            continue
        readout = RidgeCV(alphas=RIDGE_ALPHAS)
        readout.fit(X_train, y_train)
        pred_prob = readout.predict(X_test)
        pred = (pred_prob > 0.5).astype(int)
        fold_accs.append((pred == y_test).mean())
        fold_preds_for_hive.append((pred, y_test))

    # Modelo final para produccion: entrenado con TODA la historia
    # disponible (mas datos = mejor generalizacion en el uso real).
    X_full = states[WASHOUT:-1]
    y_full = targets[WASHOUT + 1:]
    final_readout = RidgeCV(alphas=RIDGE_ALPHAS)
    final_readout.fit(X_full, y_full)

    return {
        "seed": seed,
        "coef_json": json.dumps(final_readout.coef_.tolist()),
        "intercept": float(final_readout.intercept_),
        "alpha_chosen": float(final_readout.alpha_),
        "walkforward_acc": float(np.mean(fold_accs)) if fold_accs else None,
        "warmed_x_json": json.dumps(warmed_x.tolist()),
    }, fold_preds_for_hive


# ========================= ENTRENAR LA COLMENA =========================

def train():
    db = init_firestore()

    prices, volumes = fetch_historical()
    returns = np.diff(prices) / prices[:-1]
    ret_mean, ret_std = float(returns.mean()), float(returns.std())
    returns_norm = (returns - ret_mean) / ret_std

    vol_changes = np.diff(volumes) / volumes[:-1]
    vol_mean, vol_std = float(vol_changes.mean()), float(vol_changes.std())
    vol_changes_norm = (vol_changes - vol_mean) / vol_std

    inputs = np.stack([returns_norm, vol_changes_norm], axis=1)
    targets = (returns_norm > 0).astype(int)  # prediciendo el signo del PROXIMO retorno se maneja con el offset +1 en los folds

    W_raw = load_connectome_raw()
    n_neurons = W_raw.shape[0]

    print("\nBuscando el mejor radio espectral (walk-forward, mosca sonda)...")
    best_radius, best_acc = SPECTRAL_RADIUS_CANDIDATES[0], -1
    for radius in SPECTRAL_RADIUS_CANDIDATES:
        acc = evaluate_spectral_radius(W_raw, radius, n_neurons, inputs, targets, FLY_SEED_BASE)
        print(f"  radio={radius}: precision walk-forward = {acc:.1%}")
        if acc > best_acc:
            best_radius, best_acc = radius, acc
    print(f"Mejor radio espectral: {best_radius}\n")

    W = normalize_spectral_radius(W_raw, best_radius)

    print(f"Entrenando {N_FLIES} moscas (walk-forward + RidgeCV)...")
    flies_doc = {}
    all_fold_preds = []  # lista de listas [(pred, y_test), ...] por mosca
    for i in range(N_FLIES):
        seed = FLY_SEED_BASE + i
        fly, fold_preds = train_one_fly_walkforward(seed, W, n_neurons, inputs, targets)
        wf = fly["walkforward_acc"]
        print(f"  Mosca #{i} (seed={seed}): walk-forward = {wf:.1%}  alpha elegido = {fly['alpha_chosen']:.2f}")
        flies_doc[f"fly_{i}"] = fly
        all_fold_preds.append(fold_preds)

    # Precision de LA COLMENA (voto mayoritario) sobre los mismos folds,
    # para comparar honestamente contra el promedio individual.
    hive_accs = []
    n_folds_common = min(len(fp) for fp in all_fold_preds)
    for f in range(n_folds_common):
        preds_stack = np.stack([all_fold_preds[m][f][0] for m in range(N_FLIES)], axis=0)
        y_test = all_fold_preds[0][f][1]
        hive_pred = (preds_stack.mean(axis=0) >= 0.5).astype(int)
        hive_accs.append((hive_pred == y_test).mean())
    hive_acc = float(np.mean(hive_accs)) if hive_accs else None
    avg_individual = float(np.mean([flies_doc[f"fly_{i}"]["walkforward_acc"] for i in range(N_FLIES)]))

    print(f"\nPromedio individual (walk-forward): {avg_individual:.1%}")
    print(f"COLMENA (walk-forward, voto mayoritario): {hive_acc:.1%}" if hive_acc else "")
    print("(Esta es la metrica honesta - promedio de varios cortes cronologicos, no un solo split.)\n")

    db.collection("model").document("hive").set({
        "ret_mean": ret_mean, "ret_std": ret_std,
        "vol_mean": vol_mean, "vol_std": vol_std,
        "last_known_price": float(prices[-1]),
        "last_known_volume": float(volumes[-1]),
        "spectral_radius": best_radius,
        "n_neurons": n_neurons,
        "flies": flies_doc,
        "hive_walkforward_acc": hive_acc,
        "avg_individual_walkforward_acc": avg_individual,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    })

    db.collection("state").document("current").set({
        "cash": STARTING_BALANCE_USD,
        "position_sol": 0.0,
        "last_price": float(prices[-1]),
        "last_volume": float(volumes[-1]),
        "buy_hold_sol": STARTING_BALANCE_USD / float(prices[-1]),
        "x_list_json": json.dumps([
            json.loads(flies_doc[f"fly_{i}"]["warmed_x_json"]) for i in range(N_FLIES)
        ]),
        "total_fees_paid": 0.0,
    })
    print("Listo. Modelo (con validacion walk-forward) y estado inicial guardados en Firestore.")


# ========================= CICLO DE TRADING =========================

def cycle():
    db = init_firestore()

    model_snap = db.collection("model").document("hive").get()
    state_snap = db.collection("state").document("current").get()
    if not model_snap.exists or not state_snap.exists:
        print(json.dumps({"error": "no hay modelo/estado - corre primero: python mosca_firestore.py train"}))
        sys.exit(1)
    model = model_snap.to_dict()
    state = state_snap.to_dict()

    W = normalize_spectral_radius(load_connectome_raw(), model["spectral_radius"])
    n_neurons = model["n_neurons"]

    price, volume = fetch_live_price_and_volume()
    if price is None:
        print(json.dumps({"error": "no se pudo obtener precio/volumen en vivo"}))
        sys.exit(1)

    raw_return = (price - state["last_price"]) / state["last_price"]
    norm_return = (raw_return - model["ret_mean"]) / model["ret_std"]
    raw_vol_change = (volume - state["last_volume"]) / state["last_volume"] if state["last_volume"] else 0.0
    norm_vol_change = (raw_vol_change - model["vol_mean"]) / model["vol_std"]
    input_vec = np.array([norm_return, norm_vol_change])

    x_list = json.loads(state["x_list_json"])

    probs = []   # prediccion CONTINUA de cada mosca (no solo 0/1)
    new_x_list = []
    for i in range(N_FLIES):
        fly = model["flies"][f"fly_{i}"]
        W_in = build_fly_win(fly["seed"], n_neurons)
        x = np.array(x_list[i])
        x_new = step_reservoir(W, W_in, x, input_vec)
        new_x_list.append(x_new.tolist())
        coef = json.loads(fly["coef_json"])
        prob = float(np.dot(x_new, coef) + fly["intercept"])
        probs.append(prob)

    votes = [1 if p > 0.5 else 0 for p in probs]
    votes_up = sum(votes)
    agreement_up = votes_up / N_FLIES

    if agreement_up >= MIN_AGREEMENT:
        hive_signal = "sube"
    elif (1 - agreement_up) >= MIN_AGREEMENT:
        hive_signal = "baja"
    else:
        hive_signal = "dividido"

    # Confianza = que tan lejos del 50% esta el promedio de las
    # predicciones continuas. 0 = las moscas estan como tirando una
    # moneda, 1 = todas convencidas al maximo. Esto escala CUANTO se
    # apuesta, no si se apuesta (esa decision la sigue tomando el voto).
    avg_prob = float(np.mean(probs))
    confidence = min(abs(avg_prob - 0.5) * 2, 1.0)
    position_fraction = min(confidence, MAX_POSITION_FRACTION)

    cash = state["cash"]
    position_sol = state["position_sol"]
    action = "hold"
    fee_paid = 0.0

    if hive_signal == "sube" and position_sol == 0.0 and cash > 0:
        invest_usd = cash * position_fraction
        fee_paid = invest_usd * FEE_RATE
        position_sol = (invest_usd - fee_paid) / price
        cash = cash - invest_usd
        action = "buy"
    elif hive_signal == "baja" and position_sol > 0:
        proceeds = position_sol * price
        fee_paid = proceeds * FEE_RATE
        cash = cash + proceeds - fee_paid
        position_sol = 0.0
        action = "sell"

    total_fees_paid = state.get("total_fees_paid", 0.0) + fee_paid
    portfolio_value = cash + position_sol * price
    buy_hold_value = state["buy_hold_sol"] * price
    ts = datetime.now(timezone.utc).isoformat()

    db.collection("state").document("current").set({
        "cash": cash, "position_sol": position_sol,
        "last_price": price, "last_volume": volume,
        "buy_hold_sol": state["buy_hold_sol"],
        "x_list_json": json.dumps(new_x_list),
        "total_fees_paid": total_fees_paid,
    })

    db.collection("trades").add({
        "timestamp_utc": ts, "price_usd": price,
        "votes_up": votes_up, "votes_total": N_FLIES,
        "hive_signal": hive_signal, "action": action,
        "confidence": confidence, "position_fraction": position_fraction,
        "fee_paid": fee_paid, "total_fees_paid": total_fees_paid,
        "cash_usd": cash, "position_sol": position_sol,
        "portfolio_value_usd": portfolio_value,
        "buy_hold_value_usd": buy_hold_value,
    })

    db.collection("neuron_activity").add({
        "timestamp_utc": ts, "price_usd": price,
        "flies_json": json.dumps([
            {"fly_id": i, "activity": [round(v, 4) for v in new_x_list[i]]}
            for i in range(N_FLIES)
        ]),
    })

    print(json.dumps({
        "timestamp": ts, "price": price, "hive_signal": hive_signal,
        "action": action, "confidence": round(confidence, 3),
        "position_fraction": round(position_fraction, 3),
        "fee_paid": round(fee_paid, 4),
        "portfolio_value": portfolio_value,
    }))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cycle"
    if mode == "train":
        train()
    elif mode == "cycle":
        cycle()
    else:
        print("uso: python mosca_firestore.py [train|cycle]")
        sys.exit(1)
