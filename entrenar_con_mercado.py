"""
Paso 4: Entrenar el reservorio con datos de mercado reales (Fases 3-5)
---------------------------------------------------------------------------
Descarga precios historicos reales de SOL (via CoinGecko, gratis),
los usa como entrada del reservorio de la mosca, y entrena un lector
simple para predecir si el precio va a subir o bajar en el proximo
paso. Despues hace backtesting: prueba contra datos que NUNCA vio
durante el entrenamiento.

Esto NO usa plata real todavia - es pura validacion con datos pasados.
"""

import numpy as np
import requests
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

# ========== DESCARGAR DATOS REALES ==========

print("Descargando precios historicos de SOL (ultimos 90 dias, CoinGecko)...")
resp = requests.get(
    "https://api.coingecko.com/api/v3/coins/solana/market_chart",
    params={"vs_currency": "usd", "days": "90"},
    timeout=30,
)
resp.raise_for_status()
data = resp.json()
prices = np.array([p[1] for p in data["prices"]])
print(f"Puntos de precio descargados: {len(prices)}")

# --- Normalizar (cambios porcentuales, no precios crudos) ---
returns = np.diff(prices) / prices[:-1]
returns = (returns - returns.mean()) / returns.std()  # normalizar a media 0, desvio 1

# --- Dividir en entrenamiento (80%) y prueba (20%, "el futuro" que no vio) ---
split = int(len(returns) * 0.8)
train_signal = returns[:split]
test_signal = returns[split:]

# ========== CARGAR EL RESERVORIO DE LA MOSCA ==========

print("Cargando el reservorio de la mosca...")
W = np.load("fly_adjacency.npy")
n_neurons = W.shape[0]

spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
W = W * (0.9 / spectral_radius)

np.random.seed(42)
W_in = np.random.uniform(-1, 1, size=n_neurons) * 0.5


def run_reservoir(signal):
    states = np.zeros((len(signal), n_neurons))
    x = np.zeros(n_neurons)
    for i in range(len(signal)):
        x = np.tanh(W @ x + W_in * signal[i])
        states[i] = x
    return states


# ========== ENTRENAR EL LECTOR ==========

print("Simulando el reservorio con datos de entrenamiento...")
train_states = run_reservoir(train_signal)

washout = 50
X_train = train_states[washout:-1]
# Tarea: predecir si el PROXIMO cambio va a ser positivo (1) o negativo (0)
y_train = (train_signal[washout + 1:] > 0).astype(int)

readout = Ridge(alpha=1.0)
readout.fit(X_train, y_train)

train_pred = (readout.predict(X_train) > 0.5).astype(int)
train_acc = accuracy_score(y_train, train_pred)
print(f"Precision en entrenamiento: {train_acc:.1%}")

# ========== BACKTESTING: PROBAR CONTRA DATOS QUE NUNCA VIO ==========

print("\nProbando contra datos de PRUEBA (el 'futuro' que nunca vio)...")
test_states = run_reservoir(test_signal)
X_test = test_states[washout:-1]
y_test = (test_signal[washout + 1:] > 0).astype(int)

test_pred = (readout.predict(X_test) > 0.5).astype(int)
test_acc = accuracy_score(y_test, test_pred)

print(f"\n{'=' * 50}")
print(f"PRECISION EN DATOS NUNCA VISTOS: {test_acc:.1%}")
print(f"{'=' * 50}")

# --- Comparar contra una linea base (adivinar siempre "sube") ---
baseline_acc = max(y_test.mean(), 1 - y_test.mean())
print(f"\nLinea base (adivinar siempre la clase mas comun): {baseline_acc:.1%}")

if test_acc > baseline_acc + 0.03:
    print("✅ El modelo le gana a la linea base por un margen real.")
elif test_acc > baseline_acc:
    print("⚠️ Le gana apenas a la linea base - podria ser casualidad, no confiar mucho todavia.")
else:
    print("❌ NO le gana a simplemente adivinar la clase mas comun. No usar esto para operar.")

print("\n⚠️ Recordatorio: esto predice DIRECCION en datos historicos de SOL, no memecoins")
print("especificas, y una buena precision en backtesting NO garantiza resultados futuros.")
