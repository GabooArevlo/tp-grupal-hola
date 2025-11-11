import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Parámetros de la malla
n = 7               # 7x7 = 49 puntos
L = 1.0
x = np.linspace(0, L, n)
y = np.linspace(0, L, n)
h = x[1] - x[0]

# Centro y radio del hueco (según enunciado)
center = (0.5, 0.5)
r = 0.3333 / 2.0    # radio aprox 0.16665

# Crear arrays de coordenadas
X, Y = np.meshgrid(x, y)    # formato (n,n)
coords = np.column_stack((X.flatten(), Y.flatten()))

# Clasificar nodos:
# - exterior (Dirichlet): si x==0, x==1, y==0, y==1
# - inside_hole: si distancia al centro < r  (estos caerán fuera del dominio)
# - internos efectivos: los que no son frontera exterior ni inside_hole
is_dirichlet = (np.isclose(coords[:,0], 0) | np.isclose(coords[:,0], L) |
                np.isclose(coords[:,1], 0) | np.isclose(coords[:,1], L))

dist_to_center = np.sqrt((coords[:,0] - center[0])**2 + (coords[:,1] - center[1])**2)
is_inside_hole = dist_to_center < r - 1e-12   # margen numérico

# Mapear índices: -1 => no incógnita (o sea: Dirichlet o dentro del hueco)
N_total = n*n
idx_map = -np.ones(N_total, dtype=int)
idx = 0
for k in range(N_total):
    if (not is_dirichlet[k]) and (not is_inside_hole[k]):
        idx_map[k] = idx
        idx += 1
N_unknowns = idx
print("Nodos totales:", N_total)
print("Nodos Dirichlet (exteriores):", is_dirichlet.sum())
print("Nodos dentro del hueco (excluidos):", is_inside_hole.sum())
print("Nodos incógnita (efectivos):", N_unknowns)   # debe dar 48

# Función fuente f(x,y) -> ejemplo: podemos usar f=1 o usar el f_a obtenido analíticamente
def f_xy(xv, yv):
    # Aquí pones la función fuente que quieras. Como ejemplo, usemos f=1:
    return 1.0

# Construir matriz A (lil para armar) y vector b
A = lil_matrix((N_unknowns, N_unknowns), dtype=float)
b = np.zeros(N_unknowns, dtype=float)

# Helper para convertir (i,j) a índice lineal k
def ij_to_k(i, j):
    return i*n + j

# Recorremos todos los nodos y armamos ecuaciones solo para nodos incógnita
for i in range(n):
    for j in range(n):
        k = ij_to_k(i, j)
        unk = idx_map[k]
        if unk == -1:
            continue  # no es incógnita

        # RHS = f(x,y) * h^2 (usamos convención multiplicando por h^2)
        px, py = coords[k]
        b_val = f_xy(px, py) * (h**2)

        # Estencil de 5 puntos: vecinos en (i+1,j),(i-1,j),(i,j+1),(i,j-1)
        # Empezamos con coeficiente central -4 (luego lo ajustamos si faltan vecinos por hueco).
        center_coeff = -4

        neighbors = [ (i+1, j), (i-1, j), (i, j+1), (i, j-1) ]
        for (ii, jj) in neighbors:
            # ¿vecino dentro de la malla?
            if ii < 0 or ii >= n or jj < 0 or jj >= n:
                # esto no ocurre porque las fronteras exteriores son nodos Dirichlet ya marcados
                continue
            kk = ij_to_k(ii, jj)
            if is_inside_hole[kk]:
                # Vecino cae dentro del hueco -> condición Neumann (∂u/∂n=0)
                # Implementación simple: sustituimos u_vecino = u_center,
                # lo que equivale a aumentar el coeficiente central en +1 (reduce su magnitud)
                center_coeff += 1
                # no añadimos término a b porque u_center se queda en la diagonal
            elif is_dirichlet[kk]:
                # Vecino es frontera exterior con u=0 (Dirichlet)
                # valor conocido g=0 -> mover g/h^2 al RHS (pero g=0 entonces no cambia)
                g = 0.0
                b_val -= g  # * 1/h^2 pero ya b_val incluye h^2, y g=0, permanece 0
                # sin coeficiente para la incógnita
            else:
                # Vecino es otra incógnita: agregamos entrada A[unk, neighbor_idx] = 1
                neighbor_idx = idx_map[kk]
                A[unk, neighbor_idx] = 1.0

        # fijar coeficiente diagonal
        A[unk, unk] = center_coeff

        # fijar RHS
        b[unk] = b_val

# Convertir A a CSR y resolver
A_csr = csr_matrix(A)
# Nota: A está en la forma del estencil multiplicado por 1 (luego dividiremos por h^2 en la solución),
# pero como b ya fue multiplicado por h^2, resolvemos A u = b.
u_vec = spsolve(A_csr, b)

# Reconstruir solución en la malla completa (poner NaN dentro del hueco)
U_grid = np.full(N_total, np.nan, dtype=float)
# poner Dirichlet = 0 en fronteras
U_grid[is_dirichlet] = 0.0
# poner incógnitas resueltas
for k in range(N_total):
    mk = idx_map[k]
    if mk != -1:
        U_grid[k] = u_vec[mk]
# U_grid para graficar en forma (n,n)
U_grid = U_grid.reshape((n, n))

# Graficar: mostrar valores, con el hueco en blanco
plt.figure(figsize=(6,6))
# Usamos pcolormesh o imshow; imshow requiere matriz con orientacion; usamos imshow y ajustamos extent
plt.imshow(U_grid, origin='lower', extent=(0,1,0,1))
plt.colorbar(label='u (aprox)')
plt.scatter(coords[~is_dirichlet & ~is_inside_hole][:,0],
            coords[~is_dirichlet & ~is_inside_hole][:,1],
            c='k', s=10, label='nodos incógnita')
# marcar el hueco y la frontera
circle = plt.Circle(center, r, color='white', fill=True, zorder=10)
plt.gca().add_patch(circle)
plt.title("Solución aproximada (nodos Dirichlet=0, hueco en blanco)")
plt.xlabel('x'); plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.show()
