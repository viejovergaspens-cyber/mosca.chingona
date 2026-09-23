"""
Paso 2: Construir la matriz de conexiones (el "reservorio")
----------------------------------------------------------------
Toma un subconjunto manejable de neuronas del conectoma real de la
mosca (todas las dopaminergicas + una muestra del resto) y arma la
matriz de conexiones entre ellas, para usar como reservorio fijo.

Esto puede tardar unos minutos la primera vez (el archivo de
conexiones es grande).
"""

import pandas as pd
import numpy as np

N_NEURONS = 2000  # tamano manejable para empezar a probar

print("Cargando neuronas y sus neurotransmisores...")
neurons = pd.read_csv("neurons.csv.gz")

da_neurons = neurons[neurons["nt_type"] == "DA"]["root_id"].tolist()
n_others_needed = max(0, N_NEURONS - len(da_neurons))
other_neurons = (
    neurons[neurons["nt_type"] != "DA"]["root_id"]
    .sample(n=n_others_needed, random_state=42)
    .tolist()
)

selected_ids = set(da_neurons + other_neurons)
print(f"Neuronas seleccionadas: {len(selected_ids)} "
      f"(de las cuales {len(da_neurons)} son dopaminergicas)")

print("Cargando el archivo completo de conexiones (puede tardar un momento)...")
connections = pd.read_csv("connections_princeton.csv.gz")
print(f"Conexiones totales en el conectoma: {len(connections)}")

mask = connections["pre_root_id"].isin(selected_ids) & connections["post_root_id"].isin(selected_ids)
sub_connections = connections[mask]
print(f"Conexiones dentro de nuestro subconjunto: {len(sub_connections)}")

id_list = sorted(selected_ids)
id_to_index = {rid: i for i, rid in enumerate(id_list)}
n = len(id_list)

adjacency = np.zeros((n, n), dtype=np.float32)
pre_idx = sub_connections["pre_root_id"].map(id_to_index).values
post_idx = sub_connections["post_root_id"].map(id_to_index).values
weights = sub_connections["syn_count"].values.astype(np.float32)
np.add.at(adjacency, (pre_idx, post_idx), weights)

print(f"Matriz de adyacencia: {adjacency.shape}, "
      f"conexiones no-cero: {np.count_nonzero(adjacency)}")

np.save("fly_adjacency.npy", adjacency)

id_to_nt = dict(zip(neurons["root_id"], neurons["nt_type"]))
nt_types = [id_to_nt.get(rid, "UNKNOWN") for rid in id_list]
pd.DataFrame({"root_id": id_list, "nt_type": nt_types}).to_csv(
    "fly_neuron_metadata.csv", index=False
)

print("\nListo. Archivos guardados:")
print("  - fly_adjacency.npy (la matriz de conexiones)")
print("  - fly_neuron_metadata.csv (que neurotransmisor tiene cada neurona)")
