"""
paper_trading_mosca.py
---------------------------------------------------------------------------
Simulacion de trading EN VIVO (sin dinero real, sin wallet, sin claves)
usando el reservorio de la mosca entrenado para predecir DIRECCION del
precio de SOL (mismo modelo que entrenar_con_mercado.py).

Que hace:
  1. Entrena el lector (Ridge) con 90 dias de historia real de SOL.
  2. "Calienta" el estado del reservorio con los datos historicos mas
     recientes, para no arrancar con el reservorio en cero.
  3. Cada POLL_INTERVAL_MINUTES minutos, consulta el precio actual real
     (CoinGecko), calcula el retorno, lo mete en el reservorio, y le
     pregunta al lector si predice que sube o baja.
  4. Segun la senal, simula comprar/vender con un portfolio FICTICIO
     que arranca en STARTING_BALANCE_USD. No ejecuta ninguna operacion
     real en ningun exchange ni wallet.
  5. Anota cada paso en paper_trading_log.csv y guarda el estado en
     paper_trading_state.json para poder cortar (Ctrl+C) y reanudar
     despues sin perder el progreso ni el balance acumulado.
  6. Tambien trackea "comprar y mantener" (buy & hold) desde el mismo
     punto de partida, para tener un punto de comparacion honesto.

Como correrlo:
  python paper_trading_mosca.py

Para dejarlo corriendo varios dias en background (recomendado, porque
15 min por paso significa que necesitas MUCHOS pasos para que los
resultados digan algo):
  nohup python paper_trading_mosca.py > paper_trading.log 2>&1 &

Requisitos: los mismos que los otros scripts del proyecto (numpy,
pandas, requests, scikit-learn) + el archivo fly_adjacency.npy en la
misma carpeta.

IMPORTANTE: esto es una simulacion educativa. El modelo de direccion,
en el ultimo backtest, dio 52.2% vs 52.2% de linea base - o sea, no le
ganaba a adivinar. Es esperable que el paper trading tampoco le gane
al principio. El objetivo de este script es justamente medir eso con
datos en vivo, honestamente, ANTES de pensar en plata real.
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
RANDOM_SEED = 42

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 20


# ========================= DATOS =========================

def fetch_historical_prices(days=HISTORICAL_DAYS):
    """Mismos datos y misma llamada que entrenar_con_mercado.py."""
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
    """Precio actual real. Reintenta ante errores transitorios (rate
    limit, timeout de red) en vez de tirar abajo el loop entero."""
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


# ========================= RESERVORIO =========================

def load_reservoir():
    print("Cargando el reservorio de la mosca...")
    W = np.load(ADJACENCY_FILE)
    n_neurons = W.shape[0]
    spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (SPECTRAL_RADIUS_TARGET / spectral_radius)

    rng = np.random.RandomState(RANDOM_SEED)
    W_in = rng.uniform(-1, 1, size=n_neurons) * 0.5
    return W, W_in, n_neurons


def step_reservoir(W, W_in, x, norm_return):
    """Un solo paso del reservorio (misma formula que los scripts de
    entrenamiento), devuelve el nuevo estado."""
    return np.tanh(W @ x + W_in * norm_return)


def run_reservoir_sequence(W, W_in, n_neurons, signal, x0=None):
    states = np.zeros((len(signal), n_neurons))
    x = np.zeros(n_neurons) if x0 is None else x0.copy()
    for i in range(len(signal)):
        x = step_reservoir(W, W_in, x, signal[i])
        states[i] = x
    return states, x


# ========================= ENTRENAMIENTO =========================

def train_model():
    prices = fetch_historical_prices()
    returns = np.diff(prices) / prices[:-1]
    ret_mean, ret_std = returns.mean(), returns.std()
    returns_norm = (returns - ret_mean) / ret_std

    W, W_in, n_neurons = load_reservoir()

    split = int(len(returns_norm) * 0.8)
    train_signal = returns_norm[:split]
    test_signal = returns_norm[split:]

    print("Entrenando el lector con datos historicos...")
    train_states, _ = run_reservoir_sequence(W, W_in, n_neurons, train_signal)
    X_train = train_states[WASHOUT:-1]
    y_train = (train_signal[WASHOUT + 1:] > 0).astype(int)

    readout = Ridge(alpha=RIDGE_ALPHA)
    readout.fit(X_train, y_train)

    # Backtest rapido, solo para mostrar en pantalla al arrancar (mismo
    # calculo que entrenar_con_mercado.py)
    test_states, _ = run_reservoir_sequence(W, W_in, n_neurons, test_signal)
    X_test = test_states[WASHOUT:-1]
    y_test = (test_signal[WASHOUT + 1:] > 0).astype(int)
    test_pred = (readout.predict(X_test) > 0.5).astype(int)
    test_acc = (test_pred == y_test).mean()
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    print(f"Backtest de referencia -> modelo: {test_acc:.1%} | linea base: {baseline_acc:.1%}")
    print("(Este backtest es solo informativo. El paper trading de aca en mas mide")
    print(" el desempeno REAL en vivo, que es lo que realmente importa.)\n")

    # "Calentamos" el estado del reservorio con TODA la serie historica
    # (no solo train), para arrancar el loop en vivo con un estado
    # realista en vez de en cero.
    full_states, warmed_x = run_reservoir_sequence(W, W_in, n_neurons, returns_norm)

    return {
        "W": W,
        "W_in": W_in,
        "n_neurons": n_neurons,
        "readout": readout,
        "ret_mean": ret_mean,
        "ret_std": ret_std,
        "warmed_x": warmed_x,
        "last_known_price": float(prices[-1]),
    }


# ========================= ESTADO / PORTFOLIO =========================

def load_or_init_state(model, live_price_now):
    """Si ya habia un paper_trading_state.json de una corrida anterior,
    lo retoma (balance, posicion, estado del reservorio). Si no, arranca
    de cero con STARTING_BALANCE_USD."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            saved = json.load(f)
        print(f"Retomando estado guardado de una corrida anterior "
              f"(valor de portfolio: ${saved['cash'] + saved['position_sol'] * live_price_now:.2f})")
        return {
            "cash": saved["cash"],
            "position_sol": saved["position_sol"],
            "last_price": saved["last_price"],
            "x": np.array(saved["x"]),
            "buy_hold_sol": saved["buy_hold_sol"],
        }

    print(f"No hay estado previo. Arrancando con ${STARTING_BALANCE_USD:.2f} ficticios.")
    return {
        "cash": STARTING_BALANCE_USD,
        "position_sol": 0.0,
        "last_price": model["last_known_price"],
        "x": model["warmed_x"],
        # Punto de comparacion: si hubieras comprado todo al arrancar y
        # nunca mas tocado nada.
        "buy_hold_sol": STARTING_BALANCE_USD / model["last_known_price"],
    }


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump({
            "cash": state["cash"],
            "position_sol": state["position_sol"],
            "last_price": state["last_price"],
            "x": state["x"].tolist(),
            "buy_hold_sol": state["buy_hold_sol"],
        }, f)


def ensure_log_header():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp_utc", "price_usd", "predicted_signal", "action",
                "cash_usd", "position_sol", "portfolio_value_usd",
                "buy_hold_value_usd",
            ])


def log_row(row):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ========================= LOOP PRINCIPAL =========================

def run_live_loop(model):
    state = load_or_init_state(model, fetch_live_price() or model["last_known_price"])
    ensure_log_header()

    print(f"\nArrancando paper trading en vivo. Chequeo cada "
          f"{POLL_INTERVAL_MINUTES} minutos. Ctrl+C para cortar (el "
          f"progreso queda guardado en {STATE_FILE}).\n")

    while True:
        price = fetch_live_price()
        if price is None:
            print("  No se pudo obtener el precio esta vez, reintento en el proximo ciclo.")
            time.sleep(POLL_INTERVAL_MINUTES * 60)
            continue

        raw_return = (price - state["last_price"]) / state["last_price"]
        norm_return = (raw_return - model["ret_mean"]) / model["ret_std"]

        state["x"] = step_reservoir(model["W"], model["W_in"], state["x"], norm_return)
        pred = model["readout"].predict(state["x"].reshape(1, -1))[0]
        predicted_up = pred > 0.5

        action = "hold"
        if predicted_up and state["position_sol"] == 0.0 and state["cash"] > 0:
            state["position_sol"] = state["cash"] / price
            state["cash"] = 0.0
            action = "buy"
        elif (not predicted_up) and state["position_sol"] > 0:
            state["cash"] = state["position_sol"] * price
            state["position_sol"] = 0.0
            action = "sell"

        state["last_price"] = price
        portfolio_value = state["cash"] + state["position_sol"] * price
        buy_hold_value = state["buy_hold_sol"] * price

        ts = datetime.now(timezone.utc).isoformat()
        print(f"[{ts}] precio=${price:.4f}  senal={'sube' if predicted_up else 'baja'}  "
              f"accion={action}  portfolio=${portfolio_value:.2f}  "
              f"buy&hold=${buy_hold_value:.2f}")

        log_row([ts, price, int(predicted_up), action,
                  state["cash"], state["position_sol"], portfolio_value,
                  buy_hold_value])
        save_state(state)

        time.sleep(POLL_INTERVAL_MINUTES * 60)


# ========================= MAIN =========================

if __name__ == "__main__":
    model = train_model()
    try:
        run_live_loop(model)
    except KeyboardInterrupt:
        print("\nCortado por el usuario. El estado y el log quedaron guardados.")
        print(f"  - Historial de operaciones: {LOG_FILE}")
        print(f"  - Estado para reanudar: {STATE_FILE}")
