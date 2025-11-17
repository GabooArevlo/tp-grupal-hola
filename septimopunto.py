import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 1) Definir malla 7x7 → 49 nodos
# ==========================================
N = 7
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)

xx, yy = np.meshgrid(x, y)
xx = xx.flatten()
yy = yy.flatten()

# Dominio: círculo de radio 1
inside = xx**2 + yy**2 <= 1.0

# Índices de nodos internos del círculo
idx_inside = np.where(inside)[0]
n_inside = len(idx_inside)

# Paso de malla
h = x[1] - x[0]

# ==========================================
# 2) Definir la solución analítica y f(x,y)
# ==========================================
def ua(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def f(x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# ==========================================
# 3) Armar sistema Au = b (solo nodos internos)
# ==========================================
A = np.zeros((n_inside, n_inside))
b = np.zeros(n_inside)

# Función auxiliar para mapear coordenadas → índice interno
pos = {idx_inside[i]: i for i in range(n_inside)}

# Para cada nodo interno, aplicar Laplaciano por diferencias finitas
for k, p in enumerate(idx_inside):
    xi, yi = xx[p], yy[p]

    # índice 2D en la grilla
    i = np.where(x == xi)[0][0]
    j = np.where(y == yi)[0][0]

    # vecinos (i±1, j) y (i, j±1)
    neighbors = [
        ((i+1, j), 1/h**2),
        ((i-1, j), 1/h**2),
        ((i, j+1), 1/h**2),
        ((i, j-1), 1/h**2),
    ]

    A[k, k] = -4/h**2  # parte central

    # Cargar vecinos
    for (ii, jj), val in neighbors:
        # Check límites
        if ii < 0 or ii >= N or jj < 0 or jj >= N:
            continue

        p2 = jj*N + ii
        if inside[p2]:  # vecino interior
            A[k, pos[p2]] += val
        else:
            # Dirichlet fuera: u = 0
            b[k] -= val * 0  

    # Cargar término fuente f(x,y)
    b[k] += f(xi, yi)

# ==========================================
# 4) Resolver sistema
# ==========================================
u_num = np.linalg.solve(A, b)

# Reconstruir vector de 49 nodos (incluye externos)
u_full = np.zeros(49)
for k, p in enumerate(idx_inside):
    u_full[p] = u_num[k]

# ==========================================
# 5) Solución analítica en los 49 nodos
# ==========================================
u_exact = ua(xx, yy)

# ==========================================
# 6) Error absoluto en los 49 nodos
# ==========================================
EA = np.abs(u_full - u_exact)

EA_promedio = np.mean(EA[inside])
EA_maximo  = np.max(EA[inside])

print("Error absoluto promedio =", EA_promedio)
print("Error absoluto máximo    =", EA_maximo)

# ==========================================
# 7) Gráfico 3D del error
# ==========================================
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(xx, yy, EA, linewidth=0.2, antialiased=True)

ax.set_title("Error absoluto |u_DF - u_a| en 49 nodos")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Error absoluto")

plt.show()
