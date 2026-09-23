"""
paper_trading_mosca.py
---------------------------------------------------------------------------
Simulacion de trading EN VIVO (sin dinero real, sin wallet, sin claves)
usando una COLMENA de reservorios de la mosca que VOTAN por mayoria en
cada decision, en vez de un solo modelo decidiendo solo.

Que es la "colmena":
  Todas las moscas comparten el mismo conectoma real (fly_adjacency.npy,
  la estructura de conexiones entre neuronas). Lo que cambia entre
  moscas es la "sensibilidad" de entrada (W_in): cada mosca tiene su
  propia proyeccion aleatoria de como el precio le llega a las neuronas,
  como si cada una prestara atencion a un subconjunto ligeramente
  distinto de la informacion. Cada mosca entrena su propio lector
  (Ridge) por separado. En vivo, cada mosca vota "sube" o "baja" segun
  su propio estado interno, y la decision final de la colmena es la
  que gana por mayoria de votos. La idea: si una mosca se confunde
  (ruido, un patron raro que nunca vio), las demas la pueden corregir.

Que hace el script:
  1. Entrena N_FLIES lectores independientes con 90 dias de historia
     real de SOL (cada uno con su propia W_in al azar).
  2. "Calienta" el estado de CADA mosca con los datos historicos, para
     no arrancar ninguna en cero.
  3. Cada POLL_INTERVAL_MINUTES minutos, consulta el precio actual
     real, le da de comer a cada mosca, junta los votos, y decide por
     mayoria si "compra", "vende" o "mantiene".
  4. Simula esa decision sobre un portfolio FICTICIO que arranca en
     STARTING_BALANCE_USD. No ejecuta ninguna operacion real.
  5. Anota cada paso (incluyendo el detalle de como voto cada mosca)
     en paper_trading_log.csv, y guarda el estado en
     paper_trading_state.json para poder cortar (Ctrl+C) y reanudar
     sin perder progreso.
  6. Trackea "comprar y mantener" (buy & hold) como punto de
     comparacion honesto.

Como correrlo:
  python paper_trading_mosca.py

Para dejarlo corriendo varios dias en background:
  nohup python paper_trading_mosca.py > paper_trading.log 2>&1 &

Requisitos: numpy, pandas, requests, scikit-learn + fly_adjacency.npy
en la misma carpeta.

IMPORTANTE: esto es una simulacion educativa, no consejo financiero.
La colmena puede votar mal en conjunto igual que una sola mosca - votar
por mayoria reduce el ruido de UNA mosca despistada, pero no garantiza
que el conjunto le gane al mercado. Por eso este script mide todo en
vivo, honestamente, ANTES de pensar en plata real.
"""

import csv
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import requests
from sklearn.linear_model import Ridge

# ========================= CONFIG =========================

COIN_ID = "solana"
VS_CURRENCY = "usd"

POLL_INTERVAL_MINUTES = 15
STARTING_BALANCE_USD = 1000.0

LOG_FILE = "paper_trading_log.csv"
STATE_FILE = "paper_trading_state.json"
ADJACENCY_FILE = "fly_adjacency.npy"

HISTORICAL_DAYS = 90
WASHOUT = 50
RIDGE_ALPHA = 1.0
SPECTRAL_RADIUS_TARGET = 0.9

# --- Colmena ---
N_FLIES = 5              # cantidad de moscas votando (numero impar = nunca hay empate)
FLY_SEED_BASE = 42       # cada mosca usa FLY_SEED_BASE + su indice como semilla
MIN_AGREEMENT = 0.6      # % minimo de votos en el mismo sentido para actuar (si no, "hold")

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 20


# ========================= DATOS =========================

def fetch_historical_prices(days=HISTORICAL_DAYS):
    print(f"Descargando {days} dias de historia de {COIN_ID} (CoinGecko)...")
    resp = requests.get(
        f"https://api.coingecko.com/api/v3/coins/{COIN_ID}/market_chart",
        params={"vs_currency": VS_CURRENCY, "days": str(days)},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    prices = np.array([p[1] for p in data["prices"]])
    print(f"Puntos historicos: {len(prices)}")
    return prices


def fetch_live_price():
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": COIN_ID, "vs_currencies": VS_CURRENCY}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return float(resp.json()[COIN_ID][VS_CURRENCY])
        except Exception as e:
            print(f"  [aviso] fallo consultando precio (intento {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)
    return None


# ========================= RESERVORIO (UNA MOSCA) =========================

def load_shared_connectome():
    """El conectoma (W) es el mismo para todas las moscas - es la
    estructura fisica real del cerebro de mosca. Lo que varia entre
    moscas es W_in (ver train_one_fly)."""
    print("Cargando el conectoma compartido de la mosca...")
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


def train_one_fly(fly_id, seed, W, n_neurons, train_signal, test_signal):
    """Entrena una mosca individual con su propia W_in al azar."""
    rng = np.random.RandomState(seed)
    W_in = rng.uniform(-1, 1, size=n_neurons) * 0.5

    train_states, _ = run_reservoir_sequence(W, W_in, n_neurons, train_signal)
    X_train = train_states[WASHOUT:-1]
    y_train = (train_signal[WASHOUT + 1:] > 0).astype(int)

    readout = Ridge(alpha=RIDGE_ALPHA)
    readout.fit(X_train, y_train)

    test_states, _ = run_reservoir_sequence(W, W_in, n_neurons, test_signal)
    X_test = test_states[WASHOUT:-1]
    y_test = (test_signal[WASHOUT + 1:] > 0).astype(int)
    test_pred = (readout.predict(X_test) > 0.5).astype(int)
    acc = (test_pred == y_test).mean()

    print(f"  Mosca #{fly_id} (seed={seed}): precision individual en test = {acc:.1%}")
    return {"id": fly_id, "W_in": W_in, "readout": readout, "test_acc": acc}, test_pred, y_test


# ========================= ENTRENAR LA COLMENA =========================

def train_hive():
    prices = fetch_historical_prices()
    returns = np.diff(prices) / prices[:-1]
    ret_mean, ret_std = returns.mean(), returns.std()
    returns_norm = (returns - ret_mean) / ret_std

    W, n_neurons = load_shared_connectome()

    split = int(len(returns_norm) * 0.8)
    train_signal = returns_norm[:split]
    test_signal = returns_norm[split:]

    print(f"\nEntrenando la colmena ({N_FLIES} moscas)...")
    flies = []
    all_test_preds = []
    y_test_shared = None
    for i in range(N_FLIES):
        fly, test_pred, y_test = train_one_fly(
            i, FLY_SEED_BASE + i, W, n_neurons, train_signal, test_signal
        )
        flies.append(fly)
        all_test_preds.append(test_pred)
        y_test_shared = y_test  # es el mismo para todas, el target no cambia

    # Backtest de la COLMENA: voto mayoritario de las N moscas, comparado
    # contra el resultado real, sobre los mismos datos de test.
    votes_matrix = np.stack(all_test_preds, axis=0)  # (N_FLIES, n_test_points)
    hive_pred = (votes_matrix.mean(axis=0) >= 0.5).astype(int)
    hive_acc = (hive_pred == y_test_shared).mean()
    baseline_acc = max(y_test_shared.mean(), 1 - y_test_shared.mean())

    avg_individual_acc = np.mean([f["test_acc"] for f in flies])
    print(f"\nPromedio individual de las moscas: {avg_individual_acc:.1%}")
    print(f"COLMENA (voto mayoritario):         {hive_acc:.1%}")
    print(f"Linea base:                          {baseline_acc:.1%}")
    if hive_acc > avg_individual_acc:
        print("-> La colmena le gana al promedio de moscas individuales (el voto ayuda).")
    elif hive_acc == avg_individual_acc:
        print("-> La colmena empata con el promedio individual esta vez.")
    else:
        print("-> Esta vez el voto no mejoro sobre el promedio individual (puede pasar con pocas moscas).")
    print("(Este backtest es solo informativo. El paper trading en vivo mide el desempeno real.)\n")

    # Calentar el estado de CADA mosca con toda la serie historica
    for fly in flies:
        _, warmed_x = run_reservoir_sequence(W, fly["W_in"], n_neurons, returns_norm)
        fly["warmed_x"] = warmed_x

    return {
        "W": W,
        "n_neurons": n_neurons,
        "flies": flies,
        "ret_mean": ret_mean,
        "ret_std": ret_std,
        "last_known_price": float(prices[-1]),
    }


# ========================= ESTADO / PORTFOLIO =========================

def load_or_init_state(model, live_price_now):
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                saved = json.load(f)
            if len(saved["x_list"]) != N_FLIES:
                raise ValueError("el numero de moscas cambio desde la ultima corrida")
            print(f"Retomando estado guardado de una corrida anterior "
                  f"(valor de portfolio: ${saved['cash'] + saved['position_sol'] * live_price_now:.2f})")
            return {
                "cash": saved["cash"],
                "position_sol": saved["position_sol"],
                "last_price": saved["last_price"],
                "x_list": [np.array(x) for x in saved["x_list"]],
                "buy_hold_sol": saved["buy_hold_sol"],
            }
        except Exception as e:
            print(f"  [aviso] no se pudo retomar el estado guardado ({e}). Arranco de cero.")

    print(f"No hay estado previo (o no es compatible). Arrancando con ${STARTING_BALANCE_USD:.2f} ficticios.")
    return {
        "cash": STARTING_BALANCE_USD,
        "position_sol": 0.0,
        "last_price": model["last_known_price"],
        "x_list": [fly["warmed_x"] for fly in model["flies"]],
        "buy_hold_sol": STARTING_BALANCE_USD / model["last_known_price"],
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump({
            "cash": state["cash"],
            "position_sol": state["position_sol"],
            "last_price": state["last_price"],
            "x_list": [x.tolist() for x in state["x_list"]],
            "buy_hold_sol": state["buy_hold_sol"],
        }, f)


def ensure_log_header():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            header = [
                "timestamp_utc", "price_usd", "votes_up", "votes_total",
                "hive_signal", "action", "cash_usd", "position_sol",
                "portfolio_value_usd", "buy_hold_value_usd",
            ]
            csv.writer(f).writerow(header)


def log_row(row):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ========================= LOOP PRINCIPAL =========================

def run_live_loop(model):
    state = load_or_init_state(model, fetch_live_price() or model["last_known_price"])
    ensure_log_header()

    print(f"\nArrancando paper trading en vivo con colmena de {N_FLIES} moscas. "
          f"Chequeo cada {POLL_INTERVAL_MINUTES} minutos. Ctrl+C para cortar "
          f"(el progreso queda guardado en {STATE_FILE}).\n")

    consecutive_errors = 0

    while True:
        try:
            price = fetch_live_price()
            if price is None:
                print("  No se pudo obtener el precio esta vez, reintento en el proximo ciclo.")
                time.sleep(POLL_INTERVAL_MINUTES * 60)
                continue

            raw_return = (price - state["last_price"]) / state["last_price"]
            norm_return = (raw_return - model["ret_mean"]) / model["ret_std"]

            # Cada mosca actualiza su propio estado interno y vota.
            votes = []
            new_x_list = []
            for fly, x in zip(model["flies"], state["x_list"]):
                x_new = step_reservoir(model["W"], fly["W_in"], x, norm_return)
                new_x_list.append(x_new)
                pred = fly["readout"].predict(x_new.reshape(1, -1))[0]
                votes.append(1 if pred > 0.5 else 0)
            state["x_list"] = new_x_list

            votes_up = sum(votes)
            votes_total = len(votes)
            agreement_up = votes_up / votes_total
            agreement_down = 1 - agreement_up

            if agreement_up >= MIN_AGREEMENT:
                hive_signal = "sube"
            elif agreement_down >= MIN_AGREEMENT:
                hive_signal = "baja"
            else:
                hive_signal = "dividido"  # no hay mayoria clara -> no se actua

            action = "hold"
            if hive_signal == "sube" and state["position_sol"] == 0.0 and state["cash"] > 0:
                state["position_sol"] = state["cash"] / price
                state["cash"] = 0.0
                action = "buy"
            elif hive_signal == "baja" and state["position_sol"] > 0:
                state["cash"] = state["position_sol"] * price
                state["position_sol"] = 0.0
                action = "sell"

            state["last_price"] = price
            portfolio_value = state["cash"] + state["position_sol"] * price
            buy_hold_value = state["buy_hold_sol"] * price

            ts = datetime.now(timezone.utc).isoformat()
            print(f"[{ts}] precio=${price:.4f}  votos={votes_up}/{votes_total} sube  "
                  f"colmena={hive_signal}  accion={action}  "
                  f"portfolio=${portfolio_value:.2f}  buy&hold=${buy_hold_value:.2f}")

            log_row([ts, price, votes_up, votes_total, hive_signal, action,
                      state["cash"], state["position_sol"], portfolio_value,
                      buy_hold_value])
            save_state(state)
            consecutive_errors = 0

        except KeyboardInterrupt:
            raise

        except Exception as e:
            consecutive_errors += 1
            ts = datetime.now(timezone.utc).isoformat()
            print(f"[{ts}] [ERROR] Fallo inesperado en el ciclo (error consecutivo "
                  f"#{consecutive_errors}): {type(e).__name__}: {e}")
            if consecutive_errors >= 5:
                print("  5 errores seguidos - probablemente algo cambio de forma "
                      "estructural (ej. la API). Corto el script para que lo revises "
                      "a mano. El estado sigue guardado, podes correrlo de nuevo "
                      "despues de arreglarlo.")
                break

        time.sleep(POLL_INTERVAL_MINUTES * 60)


# ========================= MAIN =========================

if __name__ == "__main__":
    model = train_hive()
    try:
        run_live_loop(model)
    except KeyboardInterrupt:
        print("\nCortado por el usuario. El estado y el log quedaron guardados.")
        print(f"  - Historial de operaciones: {LOG_FILE}")
        print(f"  - Estado para reanudar: {STATE_FILE}")
