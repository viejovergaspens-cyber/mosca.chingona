"""
Paso 5: Predecir VOLATILIDAD (no direccion) usando precio + volumen
---------------------------------------------------------------------------
Cambios respecto al intento anterior:
1. La entrada ahora tiene 2 variables por paso: cambio de precio Y
   cambio de volumen (antes solo era el precio).
2. La tarea ahora es predecir si el PROXIMO movimiento va a ser de
   alta o baja volatilidad (cuanto se mueve, no hacia donde) - una
   pregunta genuinamente mas facil de responder bien que la direccion.
"""

import numpy as np
import requests
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

# ========== DESCARGAR DATOS REALES (precio Y volumen) ==========

print("Descargando precios y volumen historicos de SOL (90 dias, CoinGecko)...")
resp = requests.get(
    "https://api.coingecko.com/api/v3/coins/solana/market_chart",
    params={"vs_currency": "usd", "days": "90"},
    timeout=30,
)
resp.raise_for_status()
data = resp.json()
prices = np.array([p[1] for p in data["prices"]])
volumes = np.array([v[1] for v in data["total_volumes"]])
print(f"Puntos descargados: {len(prices)}")

# --- Normalizar ambas variables ---
returns = np.diff(prices) / prices[:-1]
returns_norm = (returns - returns.mean()) / returns.std()

volume_changes = np.diff(volumes) / volumes[:-1]
volume_changes_norm = (volume_changes - volume_changes.mean()) / volume_changes.std()

# Entrada combinada: 2 variables por paso de tiempo
combined_input = np.stack([returns_norm, volume_changes_norm], axis=1)

# --- Objetivo: volatilidad del PROXIMO paso (valor absoluto del retorno) ---
abs_returns = np.abs(returns_norm)

# --- Dividir en entrenamiento (80%) y prueba (20%) ---
split = int(len(combined_input) * 0.8)
train_input = combined_input[:split]
test_input = combined_input[split:]
train_target_abs = abs_returns[:split]
test_target_abs = abs_returns[split:]

# El umbral de "alta volatilidad" se define SOLO con datos de entrenamiento
# (nunca mirar el conjunto de prueba para tomar esta decision)
volatility_threshold = np.median(train_target_abs)
print(f"Umbral de volatilidad (mediana de entrenamiento): {volatility_threshold:.4f}")

# ========== CARGAR EL RESERVORIO DE LA MOSCA ==========

print("Cargando el reservorio de la mosca...")
W = np.load("fly_adjacency.npy")
n_neurons = W.shape[0]

spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
W = W * (0.9 / spectral_radius)

np.random.seed(42)
W_in = np.random.uniform(-1, 1, size=(n_neurons, 2)) * 0.5  # ahora 2 entradas


def run_reservoir(input_sequence):
    states = np.zeros((len(input_sequence), n_neurons))
    x = np.zeros(n_neurons)
    for i in range(len(input_sequence)):
        x = np.tanh(W @ x + W_in @ input_sequence[i])
        states[i] = x
    return states


# ========== ENTRENAR EL LECTOR ==========

print("Simulando el reservorio con datos de entrenamiento...")
train_states = run_reservoir(train_input)

washout = 50
X_train = train_states[washout:-1]
# Prediciendo si el proximo paso sera de ALTA volatilidad (1) o baja (0)
y_train = (train_target_abs[washout + 1:] > volatility_threshold).astype(int)

readout = Ridge(alpha=1.0)
readout.fit(X_train, y_train)

train_pred = (readout.predict(X_train) > 0.5).astype(int)
print(f"Precision en entrenamiento: {accuracy_score(y_train, train_pred):.1%}")

# ========== BACKTESTING ==========

print("\nProbando contra datos de PRUEBA (nunca vistos)...")
test_states = run_reservoir(test_input)
X_test = test_states[washout:-1]
y_test = (test_target_abs[washout + 1:] > volatility_threshold).astype(int)

test_pred = (readout.predict(X_test) > 0.5).astype(int)
test_acc = accuracy_score(y_test, test_pred)

print(f"\n{'=' * 50}")
print(f"PRECISION PREDICIENDO VOLATILIDAD ALTA/BAJA: {test_acc:.1%}")
print(f"{'=' * 50}")

baseline_acc = max(y_test.mean(), 1 - y_test.mean())
print(f"\nLinea base (adivinar siempre la clase mas comun): {baseline_acc:.1%}")

if test_acc > baseline_acc + 0.03:
    print("✅ El modelo le gana a la linea base por un margen real.")
elif test_acc > baseline_acc:
    print("⚠️ Le gana apenas - podria ser casualidad, no confiar mucho todavia.")
else:
    print("❌ NO le gana a simplemente adivinar la clase mas comun.")

print("\n⚠️ Aunque esto funcione, predecir volatilidad NO es lo mismo que predecir")
print("ganancias - saber que 'viene movimiento' no te dice para que lado.")
