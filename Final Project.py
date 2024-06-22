import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
dt = 0.1
dx = 0.1
D = 0.01  # Example diffusion coefficient

L = 50  # Length of the plate
B = 50  # Width of the plate

# Heating device shaped like an X
Gr = np.eye(10) * 2000
for iGr in range(10):
    Gr[iGr, -iGr-1] = 2000

# Function to set M values corresponding to non-zero Gr values
def assert_heaters(M, Gr):
    M[20:30, 10:20] = np.where(Gr > 0, Gr, M[20:30, 10:20])
    M[20:30, 30:40] = np.where(Gr > 0, Gr, M[20:30, 30:40])

# Initialize concentration matrix
M = np.zeros([L, B])  # Matrix to hold the concentration values
assert_heaters(M, Gr)

# Build MM, a list of matrices, each element corresponding to M at a given step
T = np.arange(0, 10, dt)
MM = []
for itime in range(len(T)):
    M_new = M.copy()
    for j in range(1, L-1):
        for i in range(1, B-1):
            # Fick's second law for diffusion
            d2C_dx2 = (M[i+1, j] - 2*M[i, j] + M[i-1, j]) / dx**2
            d2C_dy2 = (M[i, j+1] - 2*M[i, j] + M[i, j-1]) / dx**2
            M_new[i, j] += D * dt * (d2C_dx2 + d2C_dy2)
    
    # Re-assert heaters
    assert_heaters(M_new, Gr)

    MM.append(M_new.copy())
    M = M_new

# Initialize the figure
fig = plt.figure()
pcm = plt.pcolormesh(MM[0], cmap='hot', shading='auto')
plt.colorbar(label='Concentration (φ) ')

# Function called to update the graphic
def step(i):
    if i >= len(MM): return
    pcm.set_array(MM[i].ravel())
    plt.title(f'Time: {i*dt:.2f} seconds')
    plt.draw()

# Create the animation
anim = FuncAnimation(fig, step, frames=len(MM), interval=50, repeat=False)
plt.show()