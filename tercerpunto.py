import numpy as np
import pandas as pd

n = 7
L = 1.0
x = np.linspace(0, L, n)
y = np.linspace(0, L, n)
h = x[1] - x[0]
X, Y = np.meshgrid(x, y)
coords = np.column_stack((X.flatten(), Y.flatten()))

# Centro y radio del hueco
center = (0.5, 0.5)
r = 0.3333 / 2.0

# Clasificación base
is_dirichlet = (np.isclose(coords[:,0], 0) | np.isclose(coords[:,0], L) |
                np.isclose(coords[:,1], 0) | np.isclose(coords[:,1], L))
dist_to_center = np.sqrt((coords[:,0] - center[0])**2 + (coords[:,1] - center[1])**2)
is_inside_hole = dist_to_center < r - 1e-12

# Crear matriz de tipo (inicialmente 'Interno')
node_type = np.full(len(coords), 'Interno', dtype=object)

# Asignar tipos base
node_type[is_dirichlet] = 'Dirichlet'
node_type[is_inside_hole] = 'Externo'

# Detectar nodos Neumann (adyacentes al hueco)
def ij_to_k(i, j): return i*n + j
for i in range(n):
    for j in range(n):
        k = ij_to_k(i, j)
        if node_type[k] != 'Interno':
            continue
        # Ver si alguno de sus 4 vecinos cae dentro del hueco
        for (ii, jj) in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
            if 0 <= ii < n and 0 <= jj < n:
                kk = ij_to_k(ii, jj)
                if is_inside_hole[kk]:
                    node_type[k] = 'Neumann'
                    break

# Crear mapa de incógnitas
idx_map = -np.ones(len(coords), dtype=int)
idx = 0
for k in range(len(coords)):
    if node_type[k] in ['Interno', 'Neumann']:
        idx_map[k] = idx
        idx += 1

# Crear DataFrame
data = {
    'N° nodo': np.arange(len(coords)),
    'x': np.round(coords[:,0], 3),
    'y': np.round(coords[:,1], 3),
    'Tipo': node_type,
    'Índice incógnita': idx_map
}
tabla_nodos = pd.DataFrame(data)

# Mostrar tabla ordenada por tipo (solo para claridad)
tabla_nodos = tabla_nodos.sort_values(by=['Tipo', 'N° nodo']).reset_index(drop=True)
print(tabla_nodos)

# Guardar si querés a Excel o CSV
# tabla_nodos.to_excel("tabla_nodos.xlsx", index=False)
