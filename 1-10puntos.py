# Super-script corregido (todo en uno) - DIFERENCIAS FINITAS con ua definida
# Requiere: numpy, scipy, matplotlib, pandas, sympy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import pandas as pd
import sympy as sp
import os, time, math

# ---------------------------
# Parámetros globales
# ---------------------------
L = 1.0
center = (0.5, 0.5)
r = 1.0 / 6.0
outdir = "tp_output"
os.makedirs(outdir, exist_ok=True)

# ---------------------------
# Parte analítica: ud, un, ua y fa = Δua (simbólico)
# ---------------------------
x_s, y_s = sp.symbols('x y')
ud_s = x_s*(1 - x_s)*y_s*(1 - y_s)
un_s = ((x_s - sp.Rational(1,2))**2 + (y_s - sp.Rational(1,2))**2 - sp.Rational(1,6)**2)**2
ua_s = sp.simplify(ud_s * un_s)
lap_ua_s = sp.simplify(sp.diff(ua_s, x_s, 2) + sp.diff(ua_s, y_s, 2))  # Δ ua

ua_func = sp.lambdify((x_s, y_s), ua_s, "numpy")
fa_func = sp.lambdify((x_s, y_s), lap_ua_s, "numpy")  # fa = Δ ua

# Opcional: mostrar fórmulas (resumen)
print("ua(x,y) simbólica (resumen):")
sp.pprint(sp.simplify(ua_s))
print("\nfa(x,y) = Δ ua (resumen):")
sp.pprint(sp.simplify(lap_ua_s))

# ---------------------------
# Funciones utilitarias: malla y clasificación
# ---------------------------
def build_grid(N):
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    return X.flatten(), Y.flatten(), x[1]-x[0]

def classify_nodes(x_flat, y_flat, N):
    total = N*N
    is_dir = (np.isclose(x_flat,0.0) | np.isclose(x_flat, L) |
              np.isclose(y_flat,0.0) | np.isclose(y_flat, L))
    dist = np.sqrt((x_flat-center[0])**2 + (y_flat-center[1])**2)
    is_externo = dist < (r - 1e-12)
    node_type = np.array(['Interno']*total, dtype=object)
    node_type[is_dir] = 'Dirichlet'
    node_type[is_externo] = 'Externo'

    # detectar Neumann: interno con vecino Externo
    def ij_to_k(i,j): return i*N + j
    for i in range(N):
        for j in range(N):
            k = ij_to_k(i,j)
            if node_type[k] != 'Interno': continue
            for (ni,nj) in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                if 0 <= ni < N and 0 <= nj < N:
                    kk = ij_to_k(ni,nj)
                    if node_type[kk] == 'Externo':
                        node_type[k] = 'Neumann'
                        break
    return node_type, is_dir, is_externo, dist

# ---------------------------
# Construir sistema A u = b (consistente)
# Discretización: (uE+uW+uN+uS-4uC)/h^2 = f(x,y)
# ---------------------------
def build_system(N, verbose=False):
    x_flat, y_flat, h = build_grid(N)
    total = N*N
    node_type, is_dir, is_externo, dist = classify_nodes(x_flat, y_flat, N)

    # mapa incógnitas (excluye Externo)
    idx_map = -np.ones(total, dtype=int)
    counter = 0
    for k in range(total):
        if node_type[k] != 'Externo':
            idx_map[k] = counter
            counter += 1
    N_unknowns = counter
    if verbose:
        print(f"N={N}: total={total}, unknowns={N_unknowns}, Externo={np.sum(node_type=='Externo')}, Dirichlet={np.sum(node_type=='Dirichlet')}, Neumann={np.sum(node_type=='Neumann')}, Interno={np.sum(node_type=='Interno')}")

    A = lil_matrix((N_unknowns, N_unknowns), dtype=float)
    b = np.zeros(N_unknowns, dtype=float)

    def ij_to_k(i,j): return i*N + j

    for i in range(N):
        for j in range(N):
            k = ij_to_k(i,j)
            row = idx_map[k]
            if row == -1: continue
            typ = node_type[k]
            xk = x_flat[k]; yk = y_flat[k]

            # Dirichlet: u = 0
            if typ == 'Dirichlet':
                A[row, row] = 1.0
                b[row] = 0.0
                continue

            # Neumann
            if typ == 'Neumann':
                # normal (unitaria)
                distk = dist[k] if dist[k] != 0 else r
                nx = (xk - center[0]) / distk
                ny = (yk - center[1]) / distk

                # vecinos (global indices)
                west  = ij_to_k(i, j-1) if j-1 >= 0 else None
                east  = ij_to_k(i, j+1) if j+1 < N else None
                south = ij_to_k(i-1, j) if i-1 >= 0 else None
                north = ij_to_k(i+1, j) if i+1 < N else None

                # nx*(uE - uW)/(2h) + ny*(uN - uS)/(2h) = 0
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

            # Interno: Laplaciano central
            A[row, row] = -4.0 / (h**2)
            # W
            if j-1 >= 0:
                nb = ij_to_k(i, j-1)
                if idx_map[nb] != -1:
                    A[row, idx_map[nb]] = 1.0 / (h**2)
                else:
                    # vecino Externo -> this should not happen for 'Interno' nodes (because we mark adjacent ones as Neumann)
                    pass
            # E
            if j+1 < N:
                nb = ij_to_k(i, j+1)
                if idx_map[nb] != -1:
                    A[row, idx_map[nb]] = 1.0 / (h**2)
            # S
            if i-1 >= 0:
                nb = ij_to_k(i-1, j)
                if idx_map[nb] != -1:
                    A[row, idx_map[nb]] = 1.0 / (h**2)
            # N
            if i+1 < N:
                nb = ij_to_k(i+1, j)
                if idx_map[nb] != -1:
                    A[row, idx_map[nb]] = 1.0 / (h**2)

            # RHS: fa(x,y)
            b[row] = fa_func(xk, yk)

    return csr_matrix(A), b, idx_map, node_type, x_flat, y_flat, h

# ---------------------------
# Resolver y postprocesar
# ---------------------------
def solve_and_process(N, plot3d=True, save_csv=True, verbose=False):
    A, b, idx_map, node_type, x_flat, y_flat, h = build_system(N, verbose=verbose)
    t0 = time.time()
    u_unknown = spsolve(A, b)
    t1 = time.time()
    if verbose:
        print(f"Solved N={N} in {t1-t0:.3f}s (unknowns={len(u_unknown)})")

    total = N*N
    u_full = np.full(total, np.nan)
    for k in range(total):
        idx = idx_map[k]
        if idx != -1:
            u_full[k] = u_unknown[idx]

    ua_full = ua_func(x_flat, y_flat)
    EA = np.abs(u_full - ua_full)
    EA[node_type == 'Externo'] = np.nan

    EA_valid = EA[~np.isnan(EA)]
    ea_prom = float(np.mean(EA_valid)) if EA_valid.size>0 else np.nan
    ea_max  = float(np.max(EA_valid)) if EA_valid.size>0 else np.nan

    df_nodes = pd.DataFrame({
        "n": np.arange(total),
        "x": x_flat,
        "y": y_flat,
        "tipo": node_type,
        "idx_unknown": idx_map,
        "u_num": u_full,
        "u_a": ua_full,
        "EA": EA
    })

    if save_csv:
        df_nodes.to_csv(os.path.join(outdir, f"nodos_N{N}.csv"), index=False)

    if plot3d:
        def plot_trisurf(vals, title, fname):
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(x_flat, y_flat, vals, linewidth=0.1, antialiased=True)
            ax.set_title(title)
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, fname), dpi=200)
            plt.show()
        plot_trisurf(u_full, f"u_num N={N}", f"u_num_N{N}.png")
        plot_trisurf(ua_full, f"u_a N={N}", f"ua_N{N}.png")
        plot_trisurf(EA, f"EA N={N}", f"EA_N{N}.png")

    return {"N":N, "h":h, "EA_prom":ea_prom, "EA_max":ea_max, "df_nodes":df_nodes, "u_full":u_full, "ua_full":ua_full, "EA":EA}

# ---------------------------
# Ejecución Puntos 1..10
# ---------------------------

# Punto 1: 7x7 (49 puntos)
N0 = 7
x7, y7, h7 = build_grid(N0)
coords7 = np.column_stack((x7,y7))
print("\nPunto 1: 7x7 coordenadas (index: x,y)")
for k,(xx,yy) in enumerate(coords7):
    print(f"{k}: ({xx:.6f}, {yy:.6f})")
pd.DataFrame({"n":np.arange(N0*N0),"x":x7,"y":y7}).to_csv(os.path.join(outdir,"coords_7x7.csv"), index=False)

# Punto 2/3: clasificar nodos 7x7 y tabla
res7 = solve_and_process(7, plot3d=False, save_csv=True, verbose=True)
print("\nTabla nodal 7x7 (primeras filas):")
print(res7["df_nodes"].head(12))
res7["df_nodes"].to_csv(os.path.join(outdir,"tabla_nodos_7x7.csv"), index=False)

# Punto 4: teórico (no hace falta ejecutar)

# Punto 5: mostrar ejemplo de filas Dirichlet y Neumann (7x7)
A7, b7, idx_map7, node_type7, xf7, yf7, h7b = build_system(7, verbose=False)
A7 = A7.tocsr()
dfn = res7["df_nodes"]
try:
    row_dir = int(dfn[dfn["tipo"]=="Dirichlet"].iloc[0]["idx_unknown"])
    print(f"\nEjemplo fila Dirichlet index unknown {row_dir} -> nonzero cols: {A7.getrow(row_dir).nonzero()}")
except:
    print("No Dirichlet row found")
try:
    row_neu = int(dfn[dfn["tipo"]=="Neumann"].iloc[0]["idx_unknown"])
    rr = A7.getrow(row_neu).tocoo()
    print(f"Ejemplo fila Neumann index unknown {row_neu} -> cols/vals: {list(zip(rr.col, rr.data))}")
except:
    print("No Neumann row found")

# Punto 6/7: resuelto para 7x7 (resultado en res7)
print(f"\nPunto 6/7 (7x7): EA_prom={res7['EA_prom']:.6e}, EA_max={res7['EA_max']:.6e}")

# Punto 8: mallados grandes
mallados_totales = [81,121,441,961,1681,2601]
Ns = [int(math.sqrt(m)) for m in mallados_totales]
results = []
print("\nPunto 8: resolviendo mallados grandes...")
for N in Ns:
    print(f"\nResolviendo N={N} (total nodos = {N*N}) ...")
    rres = solve_and_process(N, plot3d=True, save_csv=True, verbose=True)
    results.append(rres)
    print(f"N={N}: h={rres['h']:.6e}, EA_prom={rres['EA_prom']:.6e}, EA_max={rres['EA_max']:.6e}")

# Punto 9: tabla resumen
tabla = []
for rres in results:
    tabla.append({
        "Cantidad de nodos": rres["N"]*rres["N"],
        "N (por eje)": rres["N"],
        "Separación h": rres["h"],
        "EA máximo": rres["EA_max"],
        "EA promedio": rres["EA_prom"]
    })
df_resumen = pd.DataFrame(tabla).sort_values("Cantidad de nodos").reset_index(drop=True)
df_resumen.to_csv(os.path.join(outdir,"tabla_resumen_p9.csv"), index=False)
print("\nPunto 9 - tabla resumen:")
print(df_resumen)

# Punto 10: gráficas 2D y log-log
plt.figure(figsize=(7,4))
plt.plot(df_resumen["Cantidad de nodos"], df_resumen["EA máximo"], 'o-')
plt.xlabel("Cantidad de nodos")
plt.ylabel("EA máximo")
plt.title("EA máximo vs Cantidad de nodos")
plt.grid(True)
plt.savefig(os.path.join(outdir,"EAmax_vs_nodos.png"), dpi=200)
plt.show()

plt.figure(figsize=(7,4))
plt.plot(df_resumen["Cantidad de nodos"], df_resumen["EA promedio"], 'o-')
plt.xlabel("Cantidad de nodos")
plt.ylabel("EA promedio")
plt.title("EA promedio vs Cantidad de nodos")
plt.grid(True)
plt.savefig(os.path.join(outdir,"EAprom_vs_nodos.png"), dpi=200)
plt.show()

plt.figure(figsize=(7,5))
plt.loglog(df_resumen['Separación h'], df_resumen['EA máximo'], 'o-', label='EA máximo')
plt.loglog(df_resumen['Separación h'], df_resumen['EA promedio'], 's-', label='EA promedio')
plt.xlabel('h')
plt.ylabel('Error absoluto')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.savefig(os.path.join(outdir,"EA_loglog.png"), dpi=200)
plt.show()

# Estimar orden de convergencia (EA_max ~ C * h^p)
logh = np.log(df_resumen['Separación h'].values)
logEA = np.log(df_resumen['EA máximo'].values)
coeffs = np.polyfit(logh, logEA, 1)
p_est = coeffs[0]; C_est = np.exp(coeffs[1])
print(f"\nEstimación orden convergencia (EA_max ≈ C h^p): p ≈ {p_est:.4f}, C ≈ {C_est:.4e}")

# Guardar resultados
df_resumen.to_csv(os.path.join(outdir,"tabla_resumen_final.csv"), index=False)
for rres in results:
    rres["df_nodes"].to_csv(os.path.join(outdir, f"detalle_nodos_N{rres['N']}.csv"), index=False)

print(f"\nScript completado. Salida guardada en carpeta '{outdir}'.")
