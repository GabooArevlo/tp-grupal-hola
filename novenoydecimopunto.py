# Código autocontenido: define solve_for_N y construye la tabla del punto 9
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Parámetros del problema ----
center = (0.5, 0.5)
r = 0.3333 / 2.0   # radio del hueco
L = 1.0

# ---- Funciones analítica (reemplazá por las tuyas si las tenés) ----
# Por defecto pongo un ejemplo que funciona: ua = sin(pi x) sin(pi y)
# y fa = -Δ ua = 2*pi^2 sin(pi x) sin(pi y).
# REEMPLAZA estas funciones por ua(x,y) y fa(x,y) que obtuviste analíticamente.
def ua(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def fa(x, y):
    return 2 * (np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y)

# ---- Rutina general para armar y resolver el problema en una malla NxN ----
def solve_for_N(N):
    """
    N: número de nodos por eje (ej: 9, 11, 21, ...)
    Devuelve:
      x_flat, y_flat, u_full (vector tamaño N*N con np.nan para nodos 'Externo'),
      ua_full (valor analítico en cada nodo), EA (error absoluto por nodo),
      h (separación)
    """
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    h = x[1] - x[0]
    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    total = N * N

    # Clasificación: Dirichlet (borde exterior), Externo (dentro del hueco), Interno/Neumann
    is_dirichlet = (np.isclose(x_flat, 0.0) | np.isclose(x_flat, L) |
                    np.isclose(y_flat, 0.0) | np.isclose(y_flat, L))
    dist = np.sqrt((x_flat - center[0])**2 + (y_flat - center[1])**2)
    is_externo = dist < r - 1e-12

    # Inicialmente marcar todos como 'Interno'
    node_type = np.array(['Interno'] * total, dtype=object)
    node_type[is_dirichlet] = 'Dirichlet'
    node_type[is_externo] = 'Externo'

    # Detectar Neumann: internos que tienen al menos un vecino 'Externo'
    def ij_to_k(i, j): return i * N + j
    for ii in range(N):
        for jj in range(N):
            k = ij_to_k(ii, jj)
            if node_type[k] != 'Interno':
                continue
            # vecinos 4-conectividad
            for (ni, nj) in [(ii+1, jj),(ii-1, jj),(ii, jj+1),(ii, jj-1)]:
                if 0 <= ni < N and 0 <= nj < N:
                    kk = ij_to_k(ni, nj)
                    if node_type[kk] == 'Externo':
                        node_type[k] = 'Neumann'
                        break

    # Mapear incógnitas: solo nodos que no son Externo forman parte del sistema
    idx_map = -np.ones(total, dtype=int)
    counter = 0
    for k in range(total):
        if node_type[k] != 'Externo':
            idx_map[k] = counter
            counter += 1
    N_unknowns = counter

    # Construcción de A y b
    A = np.zeros((N_unknowns, N_unknowns), dtype=float)
    b = np.zeros(N_unknowns, dtype=float)

    # Armado
    for i in range(N):
        for j in range(N):
            k = ij_to_k(i, j)
            row = idx_map[k]
            if row == -1:
                continue
            typ = node_type[k]
            xk = x_flat[k]; yk = y_flat[k]

            if typ == 'Dirichlet':
                # Imponer u = 0
                A[row, row] = 1.0
                b[row] = 0.0
                continue

            if typ == 'Neumann':
                # Aproximación de la derivada normal: nx*(u_E - u_W)/(2h) + ny*(u_N - u_S)/(2h) = 0
                nx = (xk - center[0]) / dist[k]
                ny = (yk - center[1]) / dist[k]

                # Indices de vecinos (si están dentro de la malla)
                west  = ij_to_k(i, j-1) if j-1 >= 0 else None
                east  = ij_to_k(i, j+1) if j+1 < N else None
                south = ij_to_k(i-1, j) if i-1 >= 0 else None
                north = ij_to_k(i+1, j) if i+1 < N else None

                # Llenar fila con coeficientes correspondientes
                # Notar: si el vecino es Externo no existe idx_map -> será -1 y no sumamos
                if east is not None and idx_map[east] != -1:
                    A[row, idx_map[east]] +=  nx / (2*h)
                if west is not None and idx_map[west] != -1:
                    A[row, idx_map[west]] += -nx / (2*h)
                if north is not None and idx_map[north] != -1:
                    A[row, idx_map[north]] +=  ny / (2*h)
                if south is not None and idx_map[south] != -1:
                    A[row, idx_map[south]] += -ny / (2*h)

                b[row] = 0.0
                continue

            # Interno normal: Laplaciano 5 puntos (discretización central)
            A[row, row] = 4.0 / h**2
            # vecinos W,E,S,N
            neighbors = []
            if j-1 >= 0: neighbors.append(ij_to_k(i, j-1))
            if j+1 < N:  neighbors.append(ij_to_k(i, j+1))
            if i-1 >= 0: neighbors.append(ij_to_k(i-1, j))
            if i+1 < N:  neighbors.append(ij_to_k(i+1, j))

            for nb in neighbors:
                if idx_map[nb] != -1:
                    A[row, idx_map[nb]] = -1.0 / h**2
                else:
                    # vecino fuera del dominio (Externo) o borde Dirichlet (en ese caso idx_map puede ser -1 si era Externo)
                    # si fuera Dirichlet el valor es 0 por lo que no se añade a b
                    pass

            # término fuente f(x,y)
            b[row] = fa(xk, yk)

    # Resolver sistema lineal
    # Protección si A no es cuadrada o singular
    try:
        u_unknown = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"Error al resolver sistema para N={N}: {e}")

    # Reconstruir vector completo (N*N) con np.nan en 'Externo'
    u_full = np.full(total, np.nan)
    for k in range(total):
        idx = idx_map[k]
        if idx != -1:
            u_full[k] = u_unknown[idx]
        else:
            u_full[k] = np.nan

    # Solución analítica y error absoluto
    ua_full = ua(x_flat, y_flat)
    EA = np.abs(u_full - ua_full)
    # Para nodos Externo dejamos EA como nan para que no formen parte de estadísticas
    EA[is_externo] = np.nan

    return x_flat, y_flat, u_full, ua_full, EA, h, node_type

# ---- Lista de mallados tal como pediste (cantidad total de nodos) ----
mallados = [81, 121, 441, 961, 1681, 2601]
Ns = [int(np.sqrt(m)) for m in mallados]

# ---- Recorrer mallados y construir la tabla ----
resultados = []
for N in Ns:
    print(f"Resolviendo para malla {N}x{N} ...")
    x_flat, y_flat, u_full, ua_full, EA, h, node_type = solve_for_N(N)

    # Estadísticas ignorando nan (nodos Externo)
    EA_valid = EA[~np.isnan(EA)]
    ea_prom = np.mean(EA_valid)
    ea_max  = np.max(EA_valid)

    resultados.append({
        "Cantidad de nodos": N*N,
        "N (por eje)": N,
        "Separación h": h,
        "EA promedio": ea_prom,
        "EA máximo": ea_max
    })

# DataFrame final
df = pd.DataFrame(resultados)
# Ordenar por cantidad de nodos
df = df.sort_values(by="Cantidad de nodos").reset_index(drop=True)
print("\nTabla resumen (punto 9):")
print(df.to_string(index=False))

plt.figure(figsize=(8,5))
plt.plot(df["Cantidad de nodos"], df["EA máximo"], marker="o")
plt.xlabel("Cantidad de nodos")
plt.ylabel("Error Absoluto Máximo (EA_max)")
plt.title("EA máximo vs Cantidad de nodos")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----- Gráfico del EA promedio -----
plt.figure(figsize=(8,5))
plt.plot(df["Cantidad de nodos"], df["EA promedio"], marker="o")
plt.xlabel("Cantidad de nodos")
plt.ylabel("Error Absoluto Promedio (EA_prom)")
plt.title("EA promedio vs Cantidad de nodos")
plt.grid(True)
plt.tight_layout()
plt.show()
