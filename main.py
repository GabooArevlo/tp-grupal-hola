# ==========================================
# PLANTEO ANALÍTICO
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1️⃣ Definir variables simbólicas
x, y = sp.symbols('x y')

# 2️⃣ Definir funciones u_d y u_n según el enunciado
u_d = x*(1 - x)*y*(1 - y)
u_n = ((x - 0.5)**2 + (y - 0.5)**2 - (1/6)**2)**2

# 3️⃣ Definir la función analítica combinada
u_a = u_d - u_n
u_a_simpl = sp.simplify(u_a)
print("Función analítica ua(x,y) = ")
sp.pprint(u_a_simpl)

# ============ Punto 1 ============
# Verificar condiciones de borde
print("\n--- Verificación de condiciones de borde ---")

# Condición Dirichlet: u=0 en el contorno exterior
borde_exterior = [
    (0, 0.5),   # x=0
    (1, 0.5),   # x=1
    (0.5, 0),   # y=0
    (0.5, 1)    # y=1
]

for (xx, yy) in borde_exterior:
    val = u_a.subs({x: xx, y: yy})
    print(f"u_a({xx},{yy}) =", float(val))

# Condición Neumann: derivada normal = 0 en el círculo interior
# Círculo centrado en (0.5,0.5) con radio 0.3333/2 ≈ 0.16665
r = 0.16665
ux = sp.diff(u_a, x)
uy = sp.diff(u_a, y)

# Derivada normal (∂u/∂n) = ux*cosθ + uy*sinθ
θ = sp.symbols('θ')
x_circ = 0.5 + r*sp.cos(θ)
y_circ = 0.5 + r*sp.sin(θ)
un = ux.subs({x:x_circ, y:y_circ})*sp.cos(θ) + uy.subs({x:x_circ, y:y_circ})*sp.sin(θ)
un_simpl = sp.simplify(un)
print("\nDerivada normal en el círculo (∂u/∂n):")
sp.pprint(un_simpl)

# ============ Punto 2 ============
# Calcular f_a(x,y) = Δu_a = ∂²u/∂x² + ∂²u/∂y²
lap_u_a = sp.diff(u_a, x, 2) + sp.diff(u_a, y, 2)
lap_u_a_simpl = sp.simplify(lap_u_a)

print("\nFunción fuente f_a(x,y) = Δu_a =")
sp.pprint(lap_u_a_simpl)

# ============ Punto 3 ============
# Graficar u_a y f_a en 3D
u_a_func = sp.lambdify((x, y), u_a, "numpy")
f_a_func = sp.lambdify((x, y), lap_u_a_simpl, "numpy")

X = np.linspace(0, 1, 100)
Y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(X, Y)
Z_u = u_a_func(X, Y)
Z_f = f_a_func(X, Y)

fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 5))

axs[0].plot_surface(X, Y, Z_u, cmap='viridis')
axs[0].set_title("Solución analítica ua(x,y)")
axs[0].set_xlabel("x"); axs[0].set_ylabel("y")

axs[1].plot_surface(X, Y, Z_f, cmap='plasma')
axs[1].set_title("Fuente f_a(x,y) = Δua")
axs[1].set_xlabel("x"); axs[1].set_ylabel("y")

plt.tight_layout()
plt.show()
