import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import loadmat
import cmocean
import cmocean.cm as cmo

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 1)
plt.rc('font', size = 16)
plt.rc('legend', fontsize = 16)              
plt.rc('xtick', labelsize = 16)             
plt.rc('ytick', labelsize = 16)

data = loadmat('../data/cylinder_nektar_wake.mat')
u = data['U_star'][:, 0]
v = data['U_star'][:, 1]
p = data['p_star']

x = data['X_star'][:, 0]
y = data['X_star'][:, 1]
t = data['t']

X = np.concatenate((u, v), axis = 0)
U, s, vh = np.linalg.svd(X, full_matrices=False)

nt = 71
a = vh[1, :nt]
t = t[:nt]

u = u.reshape((-1, 100, 200))
v = v.reshape((-1, 100, 200))
p = p.reshape((-1, 100, 200))

x = x.reshape((-1, 100))
y = y.reshape((-1, 100))

u = u[:, :, :nt]
v = v[:, :, :nt]
p = p[:, :, :nt]

np.random.seed(24)
c = 10.0 

u_noise = u + np.random.normal(0, c, np.shape(u)) * u / 100
v_noise = v + np.random.normal(0, c, np.shape(v)) * v / 100
u_noise = np.stack([u_noise, v_noise])

data_pinn = np.load(f'../results/res_cylinder_Gn{c}.npz')
u_pinn = data_pinn['up']

vh_pinn = np.linalg.lstsq(U @ np.diag(s), u_pinn[:2].reshape((-1, 71)), rcond=None)[0]
#%%
uy = np.gradient(u, y[:, 0], axis = 0, edge_order=2)
vx = np.gradient(v, x[0], axis = 1, edge_order=2)
w = vx - uy
# w = np.where(np.abs(w) < 0.12, 0, w)

uy_noise = np.gradient(u_noise[0], y[:, 0], axis = 0, edge_order=2)
vx_noise = np.gradient(u_noise[1],  x[0], axis = 1, edge_order=2)
w_noise = vx_noise - uy_noise
# w_noise = np.where(np.abs(w_noise) < 0.12, 0, w_noise)

uy_pinn = np.gradient(u_pinn[0], y[:, 0], axis = 0, edge_order=2)
vx_pinn = np.gradient(u_pinn[1],  x[0], axis = 1, edge_order=2)
w_pinn = vx_pinn - uy_pinn
# w_pinn = np.where(np.abs(w_pinn) < 0.12, 0, w_pinn)
#%%
fig, ax = plt.subplots(3, 3, figsize=(15.5, 6))
# plt.set_cmap('cmo.tarn')
cmap = "cmo.tarn"
# plt.rcParams['image.cmap'] = 'cmo.tarn'
l = 12

c0 = ax[0, 0].contourf(x, y, w[:, :, 18], cmap=cmap,levels = l, vmin = -2, vmax = 2)
c1 = ax[0, 1].contourf(x, y, w[:, :, 53], cmap=cmap,levels = l, vmin = -2, vmax = 2)
ax[1, 0].contourf(x, y, w_noise[:, :, 18],cmap=cmap, levels = l, vmin = -2, vmax = 2)
ax[1, 1].contourf(x, y, w_noise[:, :, 53],cmap=cmap, levels = l, vmin = -2, vmax = 2)
ax[2, 0].contourf(x, y, w_pinn[:, :, 18], cmap=cmap,levels = l, vmin = -2, vmax = 2)
ax[2, 1].contourf(x, y, w_pinn[:, :, 53], cmap=cmap,levels = l, vmin = -2, vmax = 2)

n = 3
ax[0, 2].plot(t, vh[n, :nt], c = 'tab:blue', label = 'Reference')
ax[0, 2].plot(t, vh_pinn[n, :nt], c = 'tab:orange', label = 'PINN', ls = '-.')
ax[0, 2].legend(frameon = False, loc = (0.0, 1.03), ncol = 3)

n = 5
ax[1, 2].plot(t, vh[n, :nt], c = 'tab:blue')
ax[1, 2].plot(t, vh_pinn[n, :nt], c = 'tab:orange', ls = '-.')

n = 7
ax[2, 2].plot(t, vh[n, :nt], c = 'tab:blue')
ax[2, 2].plot(t, vh_pinn[n, :nt], c = 'tab:orange', ls = '-.')


##################
## Evaluate the l2-norm error obtained from the temporal coefficients
##################
from numpy import linalg as LA

e_a3 = LA.norm(vh_pinn[3,:nt] - vh[3,:nt])/LA.norm(vh[3,:nt])
e_a5 = LA.norm(vh_pinn[5,:nt] - vh[5,:nt])/LA.norm(vh[5,:nt])
e_a7 = LA.norm(vh_pinn[7,:nt] - vh[7,:nt])/LA.norm(vh[7,:nt])
print(f"The relative Ecduliean norm error: \n{e_a3*100:.3f}\n{e_a5*100:.3f}\n{e_a7*100:.3f} ")

from scipy.stats import pearsonr
r_a3,_ = pearsonr(vh_pinn[3,:nt], vh[3,:nt])
r_a5,_ = pearsonr(vh_pinn[5,:nt], vh[5,:nt])
r_a7,_ = pearsonr(vh_pinn[7,:nt], vh[7,:nt])
print(f"The cross correlation: \n{r_a3*100:.3f}\n{r_a5*100:.3f}\n{r_a7*100:.3f} ")




for axx in ax[:, -1]:
    axx.set_xlim(0.0, 7.0)
    axx.set_xticks([0.0, 3.5, 7.0])
    
for axx in ax[:, :2].flatten():
    axx.set_aspect('equal')
    axx.set_xticks([1, 4.5, 8])
    
for axx in ax[:-1].flatten():
    axx.set_xticklabels([])
    
for axx in ax[:, 1].flatten():
    axx.set_yticklabels([])
    
ax[0, 0].set_ylabel('Reference \n $y$')
ax[1, 0].set_ylabel('Reference + noise \n $y$')
ax[2, 0].set_ylabel('PINN \n $y$')

for axx in ax[-1, :2]:
    axx.set_xlabel('$x$')
    
ax[0, 0].set_title('$t = 1.8$')
ax[0, 1].set_title('$t = 5.3$')
ax[-1, -1].set_xlabel('$t$')
ax[0, 2].set_ylabel('$a_{3}$')
ax[1, 2].set_ylabel('$a_{5}$')
ax[2, 2].set_ylabel('$a_{7}$')

ax[0, 2].grid(visible=True, axis='x', c = 'pink', ls = '--', lw = 2)
ax[1, 2].grid(visible=True, axis='x', c = 'pink', ls = '--', lw = 2)
ax[2, 2].grid(visible=True, axis='x', c = 'pink', ls = '--', lw = 2)
    
cb0 = fig.colorbar(c1, ax = ax[:, :2], format = '%.2f', location = 'left', shrink = 1, pad = 0.13, aspect = 40)
cb0.ax.locator_params(nbins = 6)

# plt.savefig('cylinder_res_pinn.png', bbox_inches = 'tight', dpi = 300)

#%%
# data_pinn = np.load('../results/res_cylinder_Gn0.0.npz')
u_pinn = data_pinn['up']
u_pinn[2] = u_pinn[2] - u_pinn[2].mean() + p.mean()

n = 18
e_pinn = np.abs(u_pinn[2, :, :, n] - p[:, :, n])

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
# plt.set_cmap('cmo.tarn_r')
cmap = 'cmo.tarn_r'

c0 = ax[1].contourf(x, y, p[:, :, n],    cmap = cmap, levels = l, vmin = -0.5, vmax = 0.05)
ax[0].contourf(x, y, u_pinn[2, :, :, n], cmap = cmap, levels = l, vmin = -0.5, vmax = 0.05)
c1 = ax[2].contourf(x, y, e_pinn,        cmap = cmap, levels = l)

for axx in ax.flatten():
    axx.set_aspect('equal')
    axx.set_xticks([1, 4.5, 8])
    
cb0 = fig.colorbar(c0, ax = ax[:2], format = '%.2f', shrink = 0.4)
cb0.ax.locator_params(nbins = 5)

cb1 = fig.colorbar(c1, ax = ax[2], format = '%.2f', shrink = 0.4, pad = 0.1)
cb1.ax.locator_params(nbins = 5)

for axx in ax:
    axx.set_xlabel('$x$')
    
for axx in ax[1:3]:
    axx.set_yticklabels([])

ax[0].set_ylabel('$y$')

ax[1].set_title('Reference')
ax[0].set_title('PINNs')
ax[2].set_title('$\\varepsilon = | p - \\tilde{p} |$', fontsize = 16)


# plt.savefig('cylinder_res_p.png', bbox_inches = 'tight', dpi = 300)
