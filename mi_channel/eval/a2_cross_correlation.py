"""
Print the cross correlation of the prediction 
"""
import numpy as np
from matplotlib import pyplot as plt 
import cmocean
import cmocean.cm as cmo
from scipy.signal import correlate2d
"""
Visualisation of the mini-channel flow output
"""
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 1)
plt.rc('font', size = 16)
plt.rc('legend', fontsize = 16)              
plt.rc('xtick', labelsize = 16)             
plt.rc('ytick', labelsize = 16)

ref = np.load('../data/min_channel_sr.npz')

x = ref['x'] 
y = ref['y']
z = ref['z']
u = ref['u'] # dimensions  = (nz, ny, nx, nt)
v = ref['v']
w = ref['w']
t = ref['t']

u = np.stack([u, v, w])

yy, zz, xx = np.meshgrid(y, z, x)
dt = 0.2
nv = 4
nz, ny, nx = xx.shape

pred1 = np.load('../results/res_pinn_t3_s16.npz')
u_pred1 = pred1['up'][:3]

pred2 = np.load('../results/res_pinn_t3_s8.npz')
u_pred2 = pred2['up'][:3]


uy = np.gradient(u[0], y, axis = 1, edge_order=2)
uz = np.gradient(u[0], z, axis = 0, edge_order=1)

vx = np.gradient(u[1], x, axis = 2, edge_order=2)
vz = np.gradient(u[1], z, axis = 0, edge_order=1)

wx = np.gradient(u[2], x, axis = 2, edge_order=2)
wy = np.gradient(u[2], y, axis = 1, edge_order=2)

omega_z = vx - uy

uy_pred1 = np.gradient(u_pred1[0], y, axis = 1, edge_order=2)
uz_pred1 = np.gradient(u_pred1[0], z, axis = 0, edge_order=1)

vx_pred1 = np.gradient(u_pred1[1], x, axis = 2, edge_order=2)
vz_pred1 = np.gradient(u_pred1[1], z, axis = 0, edge_order=1)

wx_pred1 = np.gradient(u_pred1[2], x, axis = 2, edge_order=2)
wy_pred1 = np.gradient(u_pred1[2], y, axis = 1, edge_order=2)

omega_z_pred1 = vx_pred1 - uy_pred1

uy_pred2 = np.gradient(u_pred2[0], y, axis = 1, edge_order=2)
uz_pred2 = np.gradient(u_pred2[0], z, axis = 0, edge_order=1)

vx_pred2 = np.gradient(u_pred2[1], x, axis = 2, edge_order=2)
vz_pred2 = np.gradient(u_pred2[1], z, axis = 0, edge_order=1)

wx_pred2 = np.gradient(u_pred2[2], x, axis = 2, edge_order=2)
wy_pred2 = np.gradient(u_pred2[2], y, axis = 1, edge_order=2)

omega_z_pred2 = vx_pred2 - uy_pred2
#%%
x = xx[0]
y = yy[0]

omega_z = omega_z[0]
omega_z_pred1 = omega_z_pred1[0]
omega_z_pred2 = omega_z_pred2[0]

u = u[:, 0]
u_pred1 = u_pred1[:, 0]
u_pred2 = u_pred2[:, 0]



n = 5 # t=1.0

upc1   = np.empty(shape=(3,ny,nx))
upc2   = np.empty(shape=(3,ny,nx))
omc1   = np.empty(shape=(ny,nx))
omc2   = np.empty(shape=(ny,nx))

for i in range(3):
    print(u[i,:,:,n].shape)
    upc1[i]   = correlate2d(u_pred1[i,:,:,n],u[i,:,:,n], mode='same', boundary='symm')
    upc2[i]   = correlate2d(u_pred2[i,:,:,n],u[i,:,:,n], mode='same', boundary='symm')

omc1   = correlate2d( omega_z_pred1[:,:,n], omega_z[:,:,n], mode='same', boundary='symm')
omc2   = correlate2d( omega_z_pred2[:,:,n], omega_z[:,:,n], mode='same', boundary='symm')


fig, ax = plt.subplots(2, 4, figsize=(9, 6), sharex = True, sharey = True)
# plt.set_cmap('cmo.tarn')


mi0 = min(upc1[0, :, :].min(), upc2[0, :, :].min())
mi1 = min(upc1[1, :, :].min(), upc2[1, :, :].min())
mi2 = min(upc1[2, :, :].min(), upc2[2, :, :].min())


mx0 = max(upc1[0, :, :].max(), upc2[0, :, :].max())
mx1 = max(upc1[1, :, :].max(), upc2[1, :, :].max())
mx2 = max(upc1[2, :, :].max(), upc2[2, :, :].max())



mi3 = min(omc1.min(),omc2.min())
mx3 = max(omc1.max(),omc2.max())
# mx3 = omega_z[:, :, n].max()

vmin2 = u[:, :, n].min()
vmax2 = u[:, :, n].max()

l = 12

print(x.shape,y.shape, upc1[0,:,:].shape)
c0 = ax[0, 0].contourf(x, y, upc1[0, :, :], levels = l, vmin = mi0, vmax = mx0)
c1 = ax[0, 1].contourf(x, y, upc1[1, :, :], levels = l, vmin = mi1, vmax = mx1)
c2 = ax[0, 2].contourf(x, y, upc1[2, :, :], levels = l, vmin = mi2, vmax = mx2)
c3 = ax[0, 3].contourf(x, y, omc1[:, :], levels = l, vmin = mi3, vmax = mx3)

ax[1, 0].contourf(x, y, upc2[0, :, :], levels = l, vmin = mi0, vmax = mx0)
ax[1, 1].contourf(x, y, upc2[1, :, :], levels = l, vmin = mi1, vmax = mx1)
ax[1, 2].contourf(x, y, upc2[2, :, :], levels = l, vmin = mi2, vmax = mx2)
ax[1, 3].contourf(x, y, omc2[:, :], levels = l, vmin = mi3, vmax = mx3)
    

for axx in ax.flatten():
    axx.set_aspect('equal')


ax[0, 0].set_ylabel('PINN \n t3--s16 \n $y$')
ax[1, 0].set_ylabel('PINN \n t3--s8 \n $y$')

for axx in ax[-1]:
    axx.set_xlabel('$x$')
    
tit = ['$u$', '$v$', '$w$', '$\\omega_z$']
i = 0
for axx in ax[0]:
    axx.set_title(tit[i])
    i += 1
    
cb0 = fig.colorbar(c0, ax = ax[:, 0], format = '%.2f', orientation = 'horizontal', shrink = 0.9, pad = 0.15, aspect = 20)
cb0.ax.locator_params(nbins = 3)

cb1 = fig.colorbar(c1, ax = ax[:, 1], format = '%.2f', orientation = 'horizontal', shrink = 0.9, pad = 0.15, aspect = 20)
cb1.ax.locator_params(nbins = 3)

cb2 = fig.colorbar(c2, ax = ax[:, 2], format = '%.2f', orientation = 'horizontal', shrink = 0.9, pad = 0.15, aspect = 20)
cb2.ax.locator_params(nbins = 3)

cb3 = fig.colorbar(c3, ax = ax[:, 3], format = '%.2f', orientation = 'horizontal', shrink = 0.9, pad = 0.15, aspect = 20)
cb3.ax.locator_params(nbins = 3)


plt.savefig('channel_res_pinn.png', bbox_inches = 'tight', dpi = 300)
