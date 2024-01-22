"""
Post-processing of the Cylinder wake 
@yuningw
"""

import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import loadmat
from pyDOE import lhs
from time import time
from pathlib import Path

import cmocean
import cmocean.cm as cmo

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 1)
plt.rc('font', size = 16)
plt.rc('legend', fontsize = 16)              
plt.rc('xtick', labelsize = 16)             
plt.rc('ytick', labelsize = 16)


import argparse
argparser   = argparse.ArgumentParser()
argparser.add_argument("--c",default=5,type=float, help="Level of gaussian noise: 0, 2.5, 5, and 10%")
args        = argparser.parse_args()

data_path   = 'data/'
res_path    = 'res/'
model_path  = 'model/'
fig_path    = 'figs/'

fileName    = f"cylinder_PINN_{args.c:.1f}"

data = loadmat(data_path + 'cylinder_nektar_wake.mat')

def get_test_Data(noise_level):
    u = data['U_star'][:, 0]
    v = data['U_star'][:, 1]
    p = data['p_star']
    print(u.shape)
    x = data['X_star'][:, 0]
    y = data['X_star'][:, 1]
    t = data['t']
    u = u.reshape((-1, 100, 200))
    v = v.reshape((-1, 100, 200))
    p = p.reshape((-1, 100, 200))

    x = x.reshape((-1, 100))
    y = y.reshape((-1, 100))

    u = u[:, :, :70]
    v = v[:, :, :70]
    p = p[:, :, :70]

    xx = x[:, :]
    yy = y[:, :]

    x = xx[0, :]
    y = yy[:, 0]

    t = t[:70]

    u_ns = u + np.random.normal(0,noise_level,np.shape(u)) * u / 100  
    v_ns = v + np.random.normal(0,noise_level,np.shape(v)) * v / 100  

    ny, nx = xx.shape
    return x,y, xx,yy ,t, u, v, p, u_ns, v_ns 

x,y,xx,yy ,t, u, v, p, u_ns, v_ns = get_test_Data(args.c)

dp   = np.load(res_path + fileName + ".npz")
print(f"INFO: the predcition of {fileName} has been loaded!")
p    = dp['up']


print(f'Time: {t}')

t18 = 18
t53 = 53

print(t18, t53)

# Load the prediction, order: U, V, P 
up   = p[0]
vp   = p[1]
pp   = p[2]


uy = np.gradient(u, y, axis = 0, edge_order=2)
vx = np.gradient(v, x, axis = 1, edge_order=2)
omega_z = vx - uy

uy_p = np.gradient(up, y, axis = 0, edge_order=2)
vx_p = np.gradient(vp, x, axis = 1, edge_order=2)
omega_z_p = vx_p - uy_p

uy_ns = np.gradient(u_ns, y, axis = 0, edge_order=2)
vx_ns = np.gradient(v_ns, x, axis = 1, edge_order=2)
omega_z_ns = vx_ns - uy_ns

mi18,mx18  = omega_z[:,:,t18].min(),omega_z[:,:,t18].max()
mi53,mx53  = omega_z[:,:,t53].min(),omega_z[:,:,t53].max()

fig, axs = plt.subplots(3,2,figsize=(9,6),sharex=True, sharey=True)
cmap = 'cmo.tarn'
# plt.set_cmap('cmo.tarn')
levels = 10
plt.title("")


c0 = axs[0,0].contourf(x,y,omega_z[:,:,t18],    vmin=mi18, vmax = mx18,levels =levels, cmap=cmap)
c0 = axs[1,0].contourf(x,y,omega_z_ns[:,:,t18], vmin=mi18, vmax = mx18,levels =levels, cmap=cmap)
c0 = axs[2,0].contourf(x,y,omega_z_p[:,:,t18],  vmin=mi18, vmax = mx18,levels =levels, cmap=cmap)
c0 = axs[0,1].contourf(x,y,omega_z[:,:,t53],    vmin=mi53, vmax = mx53,levels =levels, cmap=cmap)
c0 = axs[1,1].contourf(x,y,omega_z_ns[:,:,t53], vmin=mi53, vmax = mx53,levels =levels, cmap=cmap)
c0 = axs[2,1].contourf(x,y,omega_z_p[:,:,t53],  vmin=mi53, vmax = mx53,levels =levels, cmap=cmap)

for i in range(3):
    axs[i,0].set_ylabel("y")
for i in range(2):
    axs[-1,i].set_xlabel('x')
axs[1,0].set_title(f'Noise level = {args.c}%')
axs[0,0].set_title(f't = 1.8')
axs[0,1].set_title(f't = 5.3')
axs[2,0,].set_title(f'PINNS')
plt.savefig(fig_path + f"Post_Omegaz_{args.c}.pdf",bbox_inches='tight')
