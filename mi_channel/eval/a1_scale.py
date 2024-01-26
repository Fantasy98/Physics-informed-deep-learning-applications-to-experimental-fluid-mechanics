"""
Check the data and exmaine the characteristic length scale
@yuningw

"""

import numpy as np
from matplotlib import pyplot as plt 
import cmocean
import cmocean.cm as cmo
from  math import pi
import argparse
argparser   = argparse.ArgumentParser()
argparser.add_argument("--t",default=5,type=int, help="The number of points sampled from Time")
argparser.add_argument("--s",default=16,type=int, help="The number of points sampled from Space")
argparser.add_argument("--c",default=0,type=float, help="Level of gaussian noise")
args        = argparser.parse_args()

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 1)
plt.rc('font', size = 16)
plt.rc('legend', fontsize = 16)              
plt.rc('xtick', labelsize = 16)             
plt.rc('ytick', labelsize = 16)


#########################
## Read Data
#########################
ref = np.load('../data/min_channel_sr.npz')

x = ref['x'] 
y = ref['y']
z = ref['z']
u = ref['u'] # dimensions  = (nz, ny, nx, nt)
v = ref['v']
w = ref['w']
t = ref['t']

u = np.stack([u, v, w])

nz,ny,nx,nt = v.shape
yy, zz, xx = np.meshgrid(y, z, x)
dt = 0.2
nv = 4
nz, ny, nx = xx.shape


print(f"Summary of data:")
print(f"Data loaded, Nz = {nz}\nNy = {ny}\nNx = {nx}\nNt = {nt}")
print(f'\nTime duration: t = {t.min():.2f} to {t.max():.2f}, nt = {len(t)}')
print(f"\nDomain size\tX = {x.min()/pi:.1f}pi to {x.max()/pi:.1f}pi")
print(f"Domain size\tY = {y.min():.1f} to {y.max():.1f}")
print(f"Domain size\tZ = {z.min()/pi:.5f}pi to {z.max()/pi:.4f}pi")


#-----------------------------------
# Some given information in the reference 
Recl    = 5000 
Retau   = 202 
h       = 1  # Non-dimensional half-height of channel 

print(f"\nCharacteristic values from reference")
print(f"Half height of channel h:\t{h}")
print(f"Central line Reynolds number Re_cl:\t{Recl}")
print(f"Friction Reynolds number Re_tau:\t{Retau}")

#------------------------------------
# Kolmogorov Length scale estimation 
# L/eta  = Re^{3/4}
eta  = h /  Recl**(3/4)
print(f"\nKolmgorov length scale approximation: {eta} ")

#------------------------------------
# Kolmogorov Time scale estimation 
# tL/teta  = Re^{1/2}
teta  =  1 /  Recl**(1/2)
print(f"\nKolmgorov time scale approximation: {teta} ")

#------------------------------------
#Estimate the spatial and temporal interval via Kolmogorov length and time scale 

sx = 1
sy = 1
sz = 1
st = 1
dx = x[::sx][1] - x[::sx][0]
dy = y[::sy][1:] - y[::sy][:-1]
dz = z[::1][1] -  z[::1][0]
dt = t[::st,0][1] - t[::st,0][0]

dx_s = np.abs(dx/eta)
dy_s_max = np.abs(dy/eta).max()
dy_s_min = np.abs(dy/eta).min()
dz_s = np.abs(dz/eta)
dt_s = np.abs(dt/teta)

print(f"\nExamine the resolution by scale:")
print(f"dx = {dx}, dymin = {dy.min()}, dymax = {dy.max()}, dz = {dz}, dt = {dt}")
print(f"\nScaled:\ndx_s = {dx_s:.2f}eta\ndy_s_min = {dy_s_min:.2f}eta dy_s_max = {dy_s_max:.2f}eta\ndz_s = {dz_s:.2f}eta\ndt_s = {dt_s:.2f}teta")


n_space = args.s
n_time  = args.t
print(f"\nAs case of s = {n_space}, t = {n_time}")
sx = int(nx/n_space)
sy = int(ny/n_space)
sz = int(1)
st = int(nt/(n_time-1))

dx = x[::sx][1] - x[::sx][0]
dy = y[::sy][1:] - y[::sy][:-1]
dz = z[::1][1] -  z[::1][0]
dt = t[::st,0][1] - t[::st,0][0]

dx_s = np.abs(dx/eta)
dy_s_max = np.abs(dy/eta).max()
dy_s_min = np.abs(dy/eta).min()
dz_s = np.abs(dz/eta)
dt_s = np.abs(dt/teta)


print(f"\nScaled:\ndx_s = {dx_s:.2f}eta\ndy_s_min = {dy_s_min:.2f}eta dy_s_max = {dy_s_max:.2f}eta\ndz_s = {dz_s:.2f}eta\ndt_s = {dt_s:.2f}teta")

quit()
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
