import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

center = (0.5, 0.5)
r = 0.3333 / 2.0
tol = 1e-3

nodes = [
    (0.0, 0.0),  
    (0.25, 0.0),
    (0.5, 0.0),
    (0.75, 0.0),
    (1.0, 0.0),

    (1.0, 0.25),
    (1.0, 0.5),
    (1.0, 0.75),
    (1.0, 1.0),

    (0.75, 1.0),
    (0.5, 1.0),
    (0.25, 1.0),
    (0.0, 1.0),

    (0.0, 0.75),
    (0.0, 0.5),
    (0.0, 0.25),

    (0.25, 0.25),
    (0.75, 0.25),
    (0.25, 0.75),
    (0.75, 0.75),

    (0.5 + r, 0.5),
    (0.5 - r, 0.5),
    (0.5, 0.5 + r),
    (0.5, 0.5 - r),

    (0.5, 0.5),


    (0.5, 0.25),
    (0.5, 0.75),
    (0.25, 0.5)
]

coords = np.array(nodes)
n_nodes = coords.shape[0]


types = []
distances = np.sqrt((coords[:,0]-center[0])**2 + (coords[:,1]-center[1])**2)

for (xk, yk), d in zip(coords, distances):

    if np.isclose(xk, 0.0) or np.isclose(xk, 1.0) or np.isclose(yk, 0.0) or np.isclose(yk, 1.0):
        types.append("Dirichlet")

    elif d < r - tol:
        types.append("Externo")

    elif abs(d - r) <= tol:
        types.append("Neumann")

    else:
        types.append("Interno")

df_nodes = pd.DataFrame({
    "N° nodo": np.arange(n_nodes),
    "x": coords[:,0],
    "y": coords[:,1],
    "dist_centro": distances,
    "Tipo": types
})

print("\n==============================")
print("   TABLA DE NODOS (Punto 1)")
print("==============================\n")
print(df_nodes)
print("\nConteo por tipo:\n", df_nodes["Tipo"].value_counts())


valid_mask = df_nodes["Tipo"] != "Externo"
coords_valid = coords[valid_mask]
indices_map = np.where(valid_mask)[0]  

tri = Delaunay(coords_valid)

elements = []
for t in tri.simplices:
    
    n0 = indices_map[t[0]]
    n1 = indices_map[t[1]]
    n2 = indices_map[t[2]]
    elements.append((n0, n1, n2))

df_elem = pd.DataFrame(elements, columns=["Nodo 1", "Nodo 2", "Nodo 3"])
df_elem["N° elemento"] = df_elem.index

print("\n==============================")
print("   TABLA DE ELEMENTOS (Punto 2)")
print("==============================\n")
print(df_elem)
