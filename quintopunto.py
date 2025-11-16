import numpy as np

# --- Parámetros del problema ---
n = 7
L = 1.0
x = np.linspace(0, L, n)
y = np.linspace(0, L, n)
h = x[1] - x[0]

# Mallado
X, Y = np.meshgrid(x, y)
coords = np.column_stack((X.flatten(), Y.flatten()))

# Hueco circular
center = (0.5, 0.5)
r = 0.3333/2

dist = np.sqrt((coords[:,0]-center[0])**2 + (coords[:,1]-center[1])**2)

# Clasificaciones
node_type = np.full(49, 'Interno', dtype=object)
node_type[(coords[:,0]==0) | (coords[:,0]==1) | (coords[:,1]==0) | (coords[:,1]==1)] = 'Dirichlet'
node_type[dist < r] = 'Externo'

# Detectar nodos Neumann
def ij_to_k(i,j): return i*n + j
for i in range(n):
    for j in range(n):
        k = ij_to_k(i,j)
        if node_type[k] != 'Interno': continue
        # Si un vecino cae dentro del círculo → es Neumann
        for ii,jj in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
            if 0 <= ii < n and 0 <= jj < n:
                kk = ij_to_k(ii,jj)
                if node_type[kk] == 'Externo':
                    node_type[k] = 'Neumann'
                    break

# Crear índice de incógnitas
idx_map = -np.ones(49, dtype=int)
counter = 0
for k in range(49):
    if node_type[k] != 'Externo':
        idx_map[k] = counter
        counter += 1

N = counter   # número de incógnitas
A = np.zeros((N,N))
b = np.zeros(N)

# --- Llenado del sistema ---
for i in range(n):
    for j in range(n):
        k = ij_to_k(i,j)
        row = idx_map[k]
        if row == -1: continue

        tipo = node_type[k]

        # -----------------------------
        # Dirichlet
        # -----------------------------
        if tipo == 'Dirichlet':
            A[row,row] = 1.0
            b[row] = 0.0
            continue

        # Coordenadas
        xk, yk = coords[k]

        # -----------------------------
        # Neumann
        # -----------------------------
        if tipo == 'Neumann':
            # normal geométrica
            nx = (xk - center[0])/dist[k]
            ny = (yk - center[1])/dist[k]

            # vecinos
            west  = ij_to_k(i, j-1)
            east  = ij_to_k(i, j+1)
            south = ij_to_k(i-1, j)
            north = ij_to_k(i+1, j)

            # Ecuación normal
            if idx_map[east]  != -1: A[row, idx_map[east]]  += nx/(2*h)
            if idx_map[west]  != -1: A[row, idx_map[west]]  -= nx/(2*h)
            if idx_map[north] != -1: A[row, idx_map[north]] += ny/(2*h)
            if idx_map[south] != -1: A[row, idx_map[south]] -= ny/(2*h)

            b[row] = 0.0
            continue

        # -----------------------------
        # Interno: Laplaciano 5 puntos
        # -----------------------------
        neighbors = [
            ij_to_k(i,j),        # centro
            ij_to_k(i,j-1),      # west
            ij_to_k(i,j+1),      # east
            ij_to_k(i-1,j),      # south
            ij_to_k(i+1,j)       # north
        ]

        A[row,row] = 4/h**2
        for nb in neighbors[1:]:
            if idx_map[nb] != -1:
                A[row, idx_map[nb]] = -1/h**2

        b[row] = 0.0
