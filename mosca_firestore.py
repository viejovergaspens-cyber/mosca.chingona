"""
mosca_firestore.py
---------------------------------------------------------------------------
Version del ciclo de trading pensada para correr como tarea programada de
GitHub Actions (gratis, SIN tarjeta de credito), escribiendo los
resultados directo en Firestore para que tu amigo los vea en tiempo real.

Por que asi y no con Cloud Functions de Firebase: Firestore (la base de
datos) es gratis en el plan Spark, sin tarjeta. Lo unico que pedia
tarjeta era que el CODIGO que llama a CoinGecko corriera adentro de una
Cloud Function (el plan Spark no deja que las Functions salgan a
internet). Corriendo ese mismo codigo desde GitHub Actions en cambio,
evitamos esa restriccion: GitHub llama a CoinGecko sin problema, y
despues escribe el resultado en Firestore usando una cuenta de servicio
(que se genera gratis, sin tarjeta).

Uso:
  python mosca_firestore.py train   (una sola vez - entrena la colmena
                                      y crea el estado inicial en Firestore)
  python mosca_firestore.py cycle   (un ciclo de trading - esto es lo
                                      que GitHub Actions corre cada 15 min)

Necesita:
  - fly_adjacency.npy en la misma carpeta
  - la variable de entorno FIREBASE_SERVICE_ACCOUNT_JSON con el
    contenido COMPLETO del JSON de la cuenta de servicio de Firebase
    (Firebase Console > icono de tuerca > Project settings > Service
    accounts > Generate new private key - esto no pide tarjeta).
"""

import json
import os
import sys
from datetime import datetime, timezone

import firebase_admin
import numpy as np
import requests
from firebase_admin import credentials, firestore
from sklearn.linear_model import Ridge

# ========================= CONFIG =========================

COIN_ID = "solana"
VS_CURRENCY = "usd"
HISTORICAL_DAYS = 90
WASHOUT = 50
RIDGE_ALPHA = 1.0
SPECTRAL_RADIUS_TARGET = 0.9

N_FLIES = 5
FLY_SEED_BASE = 42
MIN_AGREEMENT = 0.6
STARTING_BALANCE_USD = 1000.0

ADJACENCY_FILE = "fly_adjacency.npy"


def init_firestore():
    cred_json = os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"]
    cred_dict = json.loads(cred_json)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    return firestore.client()


# ========================= RESERVORIO =========================

def load_shared_connectome():
    W = np.load(ADJACENCY_FILE)
    n_neurons = W.shape[0]
    spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (SPECTRAL_RADIUS_TARGET / spectral_radius)
    return W, n_neurons


def step_reservoir(W, W_in, x, norm_return):
    return np.tanh(W @ x + W_in * norm_return)


def run_reservoir_sequence(W, W_in, n_neurons, signal, x0=None):
    states = np.zeros((len(signal), n_neurons))
    x = np.zeros(n_neurons) if x0 is None else x0.copy()
    for i in range(len(signal)):
        x = step_reservoir(W, W_in, x, signal[i])
        states[i] = x
    return states, x


def build_fly_win(seed, n_neurons):
    rng = np.random.RandomState(seed)
    return rng.uniform(-1, 1, size=n_neurons) * 0.5


# ========================= ENTRENAR =========================

def train():
    db = init_firestore()

    print("Descargando historia de SOL...")
    resp = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{COIN_ID}/market_chart",
        params={"vs_currency": VS_CURRENCY, "days": str(HISTORICAL_DAYS)},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    prices = np.array([p[1] for p in data["prices"]])
    returns = np.diff(prices) / prices[:-1]
    ret_mean, ret_std = float(returns.mean()), float(returns.std())
    returns_norm = (returns - ret_mean) / ret_std

    W, n_neurons = load_shared_connectome()
    split = int(len(returns_norm) * 0.8)
    train_signal = returns_norm[:split]
    test_signal = returns_norm[split:]

    print(f"Entrenando {N_FLIES} moscas...")
    flies_doc = {}
    for i in range(N_FLIES):
        seed = FLY_SEED_BASE + i
        W_in = build_fly_win(seed, n_neurons)

        train_states, _ = run_reservoir_sequence(W, W_in, n_neurons, train_signal)
        X_train = train_states[WASHOUT:-1]
        y_train = (train_signal[WASHOUT + 1:] > 0).astype(int)
        readout = Ridge(alpha=RIDGE_ALPHA)
        readout.fit(X_train, y_train)

        test_states, _ = run_reservoir_sequence(W, W_in, n_neurons, test_signal)
        X_test = test_states[WASHOUT:-1]
        y_test = (test_signal[WASHOUT + 1:] > 0).astype(int)
        acc = float(((readout.predict(X_test) > 0.5).astype(int) == y_test).mean())
        print(f"  Mosca #{i} (seed={seed}): precision test = {acc:.1%}")

        _, warmed_x = run_reservoir_sequence(W, W_in, n_neurons, returns_norm)

        flies_doc[f"fly_{i}"] = {
            "seed": seed,
            # Guardados como texto JSON (no como array de Firestore): un
            # array de Firestore indexa cada numero por separado, y con
            # 2000 numeros x 5 moscas eso se pasa del limite de indices
            # que permite un documento. Como texto, cuenta como UN solo
            # campo indexado, sin importar cuantos numeros tenga adentro.
            "coef_json": json.dumps(readout.coef_.tolist()),
            "intercept": float(readout.intercept_),
            "test_acc": acc,
            "warmed_x_json": json.dumps(warmed_x.tolist()),
        }

    db.collection("model").document("hive").set({
        "ret_mean": ret_mean,
        "ret_std": ret_std,
        "last_known_price": float(prices[-1]),
        "n_neurons": n_neurons,
        "flies": flies_doc,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    })

    db.collection("state").document("current").set({
        "cash": STARTING_BALANCE_USD,
        "position_sol": 0.0,
        "last_price": float(prices[-1]),
        "buy_hold_sol": STARTING_BALANCE_USD / float(prices[-1]),
        "x_list_json": json.dumps([
            json.loads(flies_doc[f"fly_{i}"]["warmed_x_json"]) for i in range(N_FLIES)
        ]),
    })
    print("Listo. Modelo y estado inicial guardados en Firestore.")


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

    W, n_neurons = load_shared_connectome()

    price_resp = requests.get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={"ids": COIN_ID, "vs_currencies": VS_CURRENCY}, timeout=30,
    )
    price_resp.raise_for_status()
    price = float(price_resp.json()[COIN_ID][VS_CURRENCY])

    raw_return = (price - state["last_price"]) / state["last_price"]
    norm_return = (raw_return - model["ret_mean"]) / model["ret_std"]

    x_list = json.loads(state["x_list_json"])

    votes = []
    new_x_list = []
    for i in range(N_FLIES):
        fly = model["flies"][f"fly_{i}"]
        W_in = build_fly_win(fly["seed"], n_neurons)
        x = np.array(x_list[i])
        x_new = step_reservoir(W, W_in, x, norm_return)
        new_x_list.append(x_new.tolist())
        coef = json.loads(fly["coef_json"])
        pred = float(np.dot(x_new, coef) + fly["intercept"])
        votes.append(1 if pred > 0.5 else 0)

    votes_up = sum(votes)
    agreement_up = votes_up / N_FLIES
    if agreement_up >= MIN_AGREEMENT:
        hive_signal = "sube"
    elif (1 - agreement_up) >= MIN_AGREEMENT:
        hive_signal = "baja"
    else:
        hive_signal = "dividido"

    cash = state["cash"]
    position_sol = state["position_sol"]
    action = "hold"
    if hive_signal == "sube" and position_sol == 0.0 and cash > 0:
        position_sol = cash / price
        cash = 0.0
        action = "buy"
    elif hive_signal == "baja" and position_sol > 0:
        cash = position_sol * price
        position_sol = 0.0
        action = "sell"

    portfolio_value = cash + position_sol * price
    buy_hold_value = state["buy_hold_sol"] * price
    ts = datetime.now(timezone.utc).isoformat()

    db.collection("state").document("current").set({
        "cash": cash,
        "position_sol": position_sol,
        "last_price": price,
        "buy_hold_sol": state["buy_hold_sol"],
        "x_list_json": json.dumps(new_x_list),
    })

    db.collection("trades").add({
        "timestamp_utc": ts, "price_usd": price,
        "votes_up": votes_up, "votes_total": N_FLIES,
        "hive_signal": hive_signal, "action": action,
        "cash_usd": cash, "position_sol": position_sol,
        "portfolio_value_usd": portfolio_value,
        "buy_hold_value_usd": buy_hold_value,
    })

    # Toda la actividad neuronal de las 5 moscas va en UN solo campo de
    # texto (flies_json), no como arrays anidados - el amigo que lea
    # esto del lado del traductor/3D solo tiene que hacer JSON.parse()
    # sobre ese campo para recuperar la lista de moscas y sus 2000
    # numeros de actividad cada una.
    db.collection("neuron_activity").add({
        "timestamp_utc": ts,
        "price_usd": price,
        "flies_json": json.dumps([
            {"fly_id": i, "activity": [round(v, 4) for v in new_x_list[i]]}
            for i in range(N_FLIES)
        ]),
    })

    result = {
        "timestamp": ts, "price": price, "hive_signal": hive_signal,
        "action": action, "portfolio_value": portfolio_value,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cycle"
    if mode == "train":
        train()
    elif mode == "cycle":
        cycle()
    else:
        print("uso: python mosca_firestore.py [train|cycle]")
        sys.exit(1)
