import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from copy import deepcopy
# Generate synthetic 2D velocity field data for von Kármán vortex street
np.random.seed(42)
nx, ny = 150, 150  # Number of spatial points in x and y directions
nt = 100  # Number of time snapshots

# Create a grid
x = np.linspace(0, 2 * np.pi, nx)
y = np.linspace(0, 2 * np.pi, ny)
X, Y = np.meshgrid(x, y)

# Generate von Kármán vortex street velocity field data with gaps
vortex_strength = 5
omega = 0.5

velocity_data = np.zeros((nt, nx, ny, 2))
for t in range(nt):
    velocity_data[t, :, :, 0] = np.cos(X - omega * t) * np.exp(-((Y - np.pi) / vortex_strength)**2)
    velocity_data[t, :, :, 1] = -np.sin(Y - np.pi) * np.exp(-((Y - np.pi) / vortex_strength)**2)

# Introduce gaps in the data
gap_start = 30
gap_end = 60
origin_veldata                            = deepcopy(velocity_data)

velocity_data[:, gap_start:gap_end, gap_start:gap_end, ::2] = np.nan

velocity_gapped_data= deepcopy(velocity_data)
velocity_gapped_data[:, gap_start:gap_end, gap_start:gap_end, :] = 0
# Reshape the data for POD
reshaped_data = velocity_data.reshape((nt, nx * ny * 2))

# Mask out NaN values
nan_mask = np.isnan(reshaped_data)
reshaped_data[nan_mask] = 0

# Perform SVD


r = 10
U, s, Vt = svd(reshaped_data, full_matrices=False)

U = U[:,:r]
s = s[:r]
Vt = Vt[:r]



# Reconstruct the gappy data using the truncated SVD components
reconstructed_data = (U @ np.diag(s) @ Vt).reshape((nt, nx, ny, 2))




# Plot the original nd reconstructed velocity fields at a specific time step
time_step_to_plot = 10

plt.subplot(2, 2, 1)
plt.quiver(X, Y, velocity_gapped_data[time_step_to_plot, :, :, 0], velocity_gapped_data[time_step_to_plot, :, :, 1],
           scale=120, 
           color='b')
plt.title('Gappy Velocity Field')
# plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(2, 2, 2)
plt.quiver(X, Y, reconstructed_data[time_step_to_plot, :, :, 0], reconstructed_data[time_step_to_plot, :, :, 1],
           scale=120, 
           color='r',
            #  width=0.005,
           )
plt.title('Reconstructed Velocity Field')
plt.xlabel('X')
plt.ylabel('Y')


plt.subplot(2, 2, 3)
plt.quiver(X, Y, origin_veldata[time_step_to_plot, :, :, 0], origin_veldata[time_step_to_plot, :, :, 1],
           scale=120, 
           color='g', 
        #    width=0.005,
           )
plt.title('Original Velocity Field')
plt.xlabel('X')
# plt.ylabel('Y')

plt.tight_layout()


plt.savefig('gappy')
