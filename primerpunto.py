import numpy as np
import matplotlib.pyplot as plt

# Definir el número de puntos por eje
n = 7  

# Crear los vectores de puntos igualmente espaciados
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)

# Crear la malla (grid)
X, Y = np.meshgrid(x, y)

# Mostrar las coordenadas de los 49 puntos (opcional)
puntos = np.column_stack((X.flatten(), Y.flatten()))
print("Coordenadas de los 49 puntos:")
print(puntos)

# Graficar los puntos
plt.figure(figsize=(6,6))
plt.scatter(X, Y, color='blue')
plt.title("49 puntos igualmente espaciados en Ω = [0,1]x[0,1]")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.gca().set_aspect('equal')
plt.show()
