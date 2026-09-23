"""
Paso 1: Explorar los archivos descargados de FlyWire/Codex
--------------------------------------------------------------
Antes de construir el reservorio, miramos que columnas tiene cada
archivo para saber como cruzarlos (por ID de neurona).
"""

import pandas as pd

print("=" * 60)
print("ARCHIVO: neurons.csv.gz (neurotransmisores)")
print("=" * 60)
neurons = pd.read_csv("neurons.csv.gz")
print(f"Filas: {len(neurons)}")
print(f"Columnas: {list(neurons.columns)}")
print(neurons.head())

print()
print("=" * 60)
print("ARCHIVO: connections...csv.gz (conexiones)")
print("=" * 60)
# Ajustamos el nombre exacto mas abajo segun lo que confirmes
import glob
conn_file = glob.glob("connections*.csv.gz")[0]
print(f"Usando archivo: {conn_file}")
connections = pd.read_csv(conn_file, nrows=1000)  # solo las primeras 1000 filas para explorar rapido
print(f"Columnas: {list(connections.columns)}")
print(connections.head())
