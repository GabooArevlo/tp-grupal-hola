import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------------
# SOLUCIÓN ANALÍTICA UA y FUENTE f(x,y)
# -----------------------------------------------------------
def ua(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def f(x, y):
    return 2 * np.pi**2 * ua(x, y)

# -----------------------------------------------------------
# RUTINA PARA CREAR MALLA, ARMAR SISTEMA Y RESOLVERLO
# -----------------------------------------------------------
def solve_for_N(N):
    # Malla de NxN puntos
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)

    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()

    # Paso de malla
    h = x[1] - x[0]

    # Dominio circular
    inside = xx**2 + yy**2 <= 1.0
    idx_inside = np.where(inside)[0]
    n_inside = len(idx_inside)

    # Matriz A y vector b
    A = np.zeros((n_inside, n_inside))
    b = np.zeros(n_inside)

    # Mapeo índice global → índice interno en A
    pos = {idx_inside[i]: i for i in range(n_inside)}

    # Construir sistema
    for k, p in enumerate(idx_inside):
        xi, yi = xx[p], yy[p]

        # índice 2D
        i = np.where(x == xi)[0][0]
        j = np.where(y == yi)[0][0]

        # vecinos
        neighbors = [
            ((i+1, j), 1/h**2),
            ((i-1, j), 1/h**2),
            ((i, j+1), 1/h**2),
            ((i, j-1), 1/h**2),
        ]

        A[k, k] = -4/h**2

        for (ii, jj), val in neighbors:
            if ii < 0 or ii >= N or jj < 0 or jj >= N:
                continue

            p2 = jj*N + ii
            if inside[p2]:
                A[k, pos[p2]] += val
            else:
                # Dirichlet fuera: u = 0
                b[k] -= val * 0

        # Término fuente
        b[k] += f(xi, yi)

    # Resolver
    u_num = np.linalg.solve(A, b)

    # Vector completo incluyendo nodos externos
    u_full = np.zeros(N*N)
    for k, p in enumerate(idx_inside):
        u_full[p] = u_num[k]

    # Solución analítica en toda la malla
    u_exact = ua(xx, yy)

    # Error absoluto
    EA = np.abs(u_full - u_exact)

    return xx, yy, u_full, u_exact, EA, inside

# -----------------------------------------------------------
# GRÁFICOS 3D PARA UNA MALLA DADA
# -----------------------------------------------------------
def plot_3D(xx, yy, zz, titulo):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(xx, yy, zz, linewidth=0.2, antialiased=True)

    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()

# -----------------------------------------------------------
# LISTA DE MALLADOS PARA EL PUNTO 8
# -----------------------------------------------------------
mallados = [81, 121, 441, 961, 1681, 2601]  # nodos totales
Ns = [int(np.sqrt(m)) for m in mallados]    # tamaños NxN

# -----------------------------------------------------------
# EJECUCIÓN Y GRÁFICOS
# -----------------------------------------------------------
for N in Ns:
    print(f"\n=== Mallado {N}xN ({N*N} nodos) ===")

    xx, yy, u_num, u_exact, EA, inside = solve_for_N(N)

    print("Error promedio:", np.mean(EA[inside]))
    print("Error máximo:  ", np.max(EA[inside]))

    # Graficar solución numérica
    plot_3D(xx, yy, u_num, f"Solución numérica u_DF para N={N}")

    # Graficar solución analítica
    plot_3D(xx, yy, u_exact, f"Solución analítica u_a para N={N}")

    # Graficar error absoluto
    plot_3D(xx, yy, EA, f"Error absoluto |u_DF - u_a| para N={N}")
