import numpy as np

# ---------------------------------------------------
# 1. Definir la función f(x,y) = fa(x,y) del punto analítico
# ---------------------------------------------------
def fa(x, y):
    # ❗❗❗ COMPLETAR con tu función real
    # Ejemplo: return np.sin(np.pi * x) * np.sin(np.pi * y)
    return np.sin(np.pi * x) * np.sin(np.pi * y)


# ---------------------------------------------------
# 2. Parámetros y mallado (mismo que antes)
# ---------------------------------------------------
n = 7
L = 1.0
x = np.linspace(0, L, n)
y = np.linspace(0, L, n)
h = x[1] - x[0]

X, Y = np.meshgrid(x, y)
coords = np.column_stack((X.flatten(), Y.flatten()))

center = (0.5, 0.5)
r = 0.3333 / 2

dist = np.sqrt((coords[:,0]-center[0])**2 + (coords[:,1]-center[1])**2)
node_type = np.full(49, 'Interno', dtype=object)

# Dirichlet
node_type[(coords[:,0]==0) | (coords[:,0]==1) | (coords[:,1]==0) | (coords[:,1]==1)] = 'Dirichlet'

# Externo
node_type[dist < r] = 'Externo'

# Identificar Neumann
def ij_to_k(i,j): return i*n + j
for i in range(n):
    for j in range(n):
        k = ij_to_k(i,j)
        if node_type[k] != 'Interno': continue
        for ii,jj in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
            if 0 <= ii < n and 0 <= jj < n:
                kk = ij_to_k(ii,jj)
                if node_type[kk] == 'Externo':
                    node_type[k] = 'Neumann'
                    break

# Mapa de incógnitas
idx_map = -np.ones(49, dtype=int)
counter = 0
for k in range(49):
    if node_type[k] != 'Externo':
        idx_map[k] = counter
        counter += 1

N = counter
A = np.zeros((N,N))
b = np.zeros(N)


# ---------------------------------------------------
# 3. Armado de A y b con f(x,y) incluida
# ---------------------------------------------------
for i in range(n):
    for j in range(n):
        k = ij_to_k(i,j)
        row = idx_map[k]
        if row == -1:
            continue

        tipo = node_type[k]
        xk, yk = coords[k]

        # -------------------------
        # Dirichlet
        # -------------------------
        if tipo == 'Dirichlet':
            A[row,row] = 1.0
            b[row] = 0.0
            continue

        # -------------------------
        # Neumann
        # -------------------------
        if tipo == 'Neumann':
            nx = (xk - center[0]) / dist[k]
            ny = (yk - center[1]) / dist[k]

            west  = ij_to_k(i, j-1)
            east  = ij_to_k(i, j+1)
            south = ij_to_k(i-1, j)
            north = ij_to_k(i+1, j)

            if idx_map[east]  != -1: A[row, idx_map[east]]  += nx/(2*h)
            if idx_map[west]  != -1: A[row, idx_map[west]]  -= nx/(2*h)
            if idx_map[north] != -1: A[row, idx_map[north]] += ny/(2*h)
            if idx_map[south] != -1: A[row, idx_map[south]] -= ny/(2*h)

            b[row] = 0.0
            continue

        # -------------------------
        # Interno → Laplaciano
        # -------------------------
        A[row, row] = 4/h**2

        neighbors = [
            ij_to_k(i, j-1),   # West
            ij_to_k(i, j+1),   # East
            ij_to_k(i-1, j),   # South
            ij_to_k(i+1, j)    # North
        ]

        for nb in neighbors:
            if idx_map[nb] != -1:
                A[row, idx_map[nb]] = -1/h**2

        # Cargar f(x,y) en el vector b
        b[row] = fa(xk, yk)


# ---------------------------------------------------
# 4. Resolver sistema lineal
# ---------------------------------------------------
u_sol = np.linalg.solve(A, b)

# Reconstruir solución completa incluyendo nodos externos
u_full = np.zeros(49)
for k in range(49):
    if idx_map[k] != -1:
        u_full[k] = u_sol[idx_map[k]]
    else:
        u_full[k] = np.nan   # nodos fuera del dominio


# ---------------------------------------------------
# 5. Mostrar la solución en formato de matriz 7×7
# ---------------------------------------------------
U = u_full.reshape((n,n))
print("Solución U(x,y):")
print(U)
