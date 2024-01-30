"""
Examine the velocity similarity obatained through 2 models
@yuningw
"""
import numpy as np
from matplotlib import pyplot as plt 
import cmocean
import cmocean.cm as cmo
from matplotlib import gridspec
from scipy.stats import pearsonr
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
nz, ny, nx = xx.shape

pred1 = np.load('../results/res_pinn_t3_s16.npz')
u_pred1 = pred1['up'][:3]

pred2 = np.load('../results/res_pinn_t3_s8.npz')
u_pred2 = pred2['up'][:3]
#%%
x = xx[0]
y = yy[0]

u = u[:, 0]
u_pred1 = u_pred1[:, 0]
u_pred2 = u_pred2[:, 0]

locx = int(nx/2)
locy = int(ny/1.3)

Name = ["U","V","W"]
print(f"At location: {locx}, {locy}")
for i in range(3):
    print(f"For {Name[i]}")
    r_p1,_ = pearsonr(u_pred1[i,locy,locx].flatten(), u[i,locy,locx].flatten())
    r_p2,_ = pearsonr(u_pred2[i,locy,locx].flatten(), u[i,locy,locx].flatten())
    print(f"PINN--t3--s16: {r_p1:.3f}")
    print(f"PINN--t3--s8: {r_p2:.3f}")


# ax1.plot(t, u[0, locy, locx], label = 'Reference')
# ax2.plot(t, u[1, locy, locx])
# ax3.plot(t, u[2, locy, locx])


# ax1.plot(t, u_pred2[0, locy, locx], ls = '--', label = 'PINN--t3--s8')
# ax2.plot(t, u_pred2[1, locy, locx], ls = '--')
# ax3.plot(t, u_pred2[2, locy, locx], ls = '--')

# ax1.plot(t, u_pred1[0, locy, locx], ls = '-.', label = 'PINN--t3--s16')
# ax2.plot(t, u_pred1[1, locy, locx], ls = '-.')
# ax3.plot(t, u_pred1[2, locy, locx], ls = '-.')
