"""
Paso 3: Probar el reservorio con una tarea simple (Fase 2 del plan)
------------------------------------------------------------------------
Antes de meterle datos de mercado, confirmamos que la red de la mosca
es ESTABLE prediciendo algo simple y conocido: una onda seno. Si esto
funciona sin "explotar" (el problema de panico de antes), pasamos a
datos financieros con confianza.

Como funciona (Echo State Network / Reservoir Computing):
1. La matriz de conexiones de la mosca (fija) transforma una señal de
   entrada en un patron de actividad interno complejo.
2. Solo entrenamos una capa de lectura simple (regresion) que aprende
   a interpretar ese patron para la tarea que le pidamos.
"""

import numpy as np
from sklearn.linear_model import Ridge

np.random.seed(42)

print("Cargando la matriz de conexiones de la mosca...")
W = np.load("fly_adjacency.npy")
n_neurons = W.shape[0]
print(f"Reservorio: {n_neurons} neuronas")

# --- Normalizar el radio espectral (evita el "panico"/inestabilidad) ---
eigenvalues = np.linalg.eigvals(W)
spectral_radius = np.max(np.abs(eigenvalues))
print(f"Radio espectral original: {spectral_radius:.4f}")

TARGET_SPECTRAL_RADIUS = 0.9  # zona estable pero con dinamica rica
if spectral_radius > 0:
    W = W * (TARGET_SPECTRAL_RADIUS / spectral_radius)
print(f"Matriz reescalada a radio espectral {TARGET_SPECTRAL_RADIUS}")

# --- Generar la tarea simple: predecir el siguiente valor de un seno ---
t = np.linspace(0, 50, 2000)
signal = np.sin(t)

# Pesos de entrada aleatorios: conectan la señal de 1 valor a las 2000 neuronas
W_in = np.random.uniform(-1, 1, size=(n_neurons, 1)) * 0.5

# --- Simular el reservorio a lo largo del tiempo ---
states = np.zeros((len(signal), n_neurons))
x = np.zeros(n_neurons)

for i in range(len(signal) - 1):
    u = signal[i]
    x = np.tanh(W @ x + (W_in.flatten() * u))
    states[i] = x

# --- Entrenar la capa de lectura (regresion ridge) ---
# Descartamos los primeros 100 pasos (transitorio, antes de que la red "arranque")
washout = 100
X_train = states[washout:-1]
y_train = signal[washout + 1:]

readout = Ridge(alpha=1.0)
readout.fit(X_train, y_train)

predictions = readout.predict(X_train)
mse = np.mean((predictions - y_train) ** 2)

print(f"\nError cuadratico medio prediciendo el seno: {mse:.6f}")
if mse < 0.01:
    print("✅ La red esta ESTABLE y aprendiendo bien la tarea simple.")
elif mse < 0.5:
    print("⚠️ Funciona pero no muy bien. Puede necesitar ajustar el radio espectral.")
else:
    print("❌ Algo anda mal (posible inestabilidad). Revisemos juntos.")
