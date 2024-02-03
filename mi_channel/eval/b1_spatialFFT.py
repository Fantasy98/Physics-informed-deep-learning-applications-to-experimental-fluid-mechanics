"""
Apply the FFT on a snapshots to check the scale 
@yuningw
"""

import numpy as np
from matplotlib import pyplot as plt 
import cmocean
import cmocean.cm as cmo
from matplotlib import gridspec
from scipy.stats import pearsonr
from numpy import fft


def PSD(data,Nx,Ny,Nz, nt,Lx,Ly,Lz, dir=0):
    import numpy as np
    
    # Fluctuation
    for i in range(nt):
        data[:,:,:,:,i] = data[:,:,:,:,i] - np.average(data,-1)
    data = np.average(data,-1)
    
    print(data.shape)
    
    if dir == 0: 
        Lx = Ly; Lz = Lz
        Nx = Ny; Nz = Nz
        Theta_fluc_targ= np.average(data,axis=-1)
        print(Theta_fluc_targ.shape)
        
    elif dir ==1:
        Lx = Lx; Lz = Lz
        Nx = Nx; Nz = Nz
        Theta_fluc_targ= np.average(data,axis=-2)
        print(Theta_fluc_targ.shape)
        
    else:
        Lx = Lx; Lz = Ly
        Nx = Nx; Nz = Ny
        Theta_fluc_targ= np.average(data,axis=-3)
        print(Theta_fluc_targ.shape)
        

    # Wavenumber spacing
    dkx = 2*np.pi/Lx
    dkz = 2*np.pi/Lz

    x_range=np.linspace(1e-3,Lx,Nx)
    z_range=np.linspace(1e-3,Lz,Nz)
    
    kx = dkx * np.append(x_range[:Nx//2], -x_range[Nx//2:0:-1])
    kz = dkz * np.append(z_range[:Nz//2], -z_range[Nz//2:0:-1])
    
    [kkx,kkz]=np.meshgrid(kx,kz)
    kkx_norm= np.sqrt(kkx**2)
    kkz_norm = np.sqrt(kkz**2)
    
    Re_Tau = 202 #Direct from simulation
    Re = 5000 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu
    Lambda_x = (2*np.pi/kkx_norm)*u_tau/nu
    Lambda_z = (2*np.pi/kkz_norm)*u_tau/nu
    # Lambda_x = (2*np.pi/kkx_norm)
    # Lambda_z = (2*np.pi/kkz_norm)
    # # Theta_fluc_targ=data-np.mean(data)
    print(Theta_fluc_targ.shape)
    # We compute the 2 dimensional discrete Fourier Transform
    fourier_image_targ = np.fft.fftn(Theta_fluc_targ)
    # We also compute the pre-multiplication with the wavenumber vectors
    # print(fourier_image_targ.shape, kkx.shape, kkz.shape)
    # fourier_amplitudes_targ = np.mean(np.absolute(fourier_image_targ)**2,axis=0)*kkx*kkz
    fourier_amplitudes_targ = np.sum(np.abs(fourier_image_targ)**2,axis=0)*kkx*kkz
    
    return fourier_amplitudes_targ, Lambda_x, Lambda_z

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


Lx     =   np.abs(x.max()- x.min())
Ly     =   np.abs(y.max()- y.min())
Lz     =   np.abs(z.max()- z.min())


u = np.stack([u, v, w])

yy, zz, xx = np.meshgrid(y, z, x)

nz, ny, nx = xx.shape
nt         = len(t)
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

x = xx[0]
y = yy[0]

t      = 5 
yp     = 0
utype  = 2

direction = 0


sp_u,xx,zz = PSD(u,
                Nx = nx,
                Ny = ny,
                Nz = nz,
                nt = nt,
                Lx = Lx,
                Ly = Ly,
                Lz = Lz,
                dir= direction)

sp_up,_,_ = PSD(
                u_pred1,
                Nx = nx,
                Ny = ny,
                Nz = nz,
                nt = nt,
                Lx = Lx,
                Ly = Ly,
                Lz = Lz,
                dir= direction
                )

sp_up2,_,_ = PSD(
                u_pred2,
                Nx = nx,
                Ny = ny,
                Nz = nz,
                nt = nt,
                Lx = Lx,
                Ly = Ly,
                Lz = Lz,
                dir= direction
                )
fig, axs = plt.subplots(1,1,sharex=True, sharey=True)

pct100 = sp_u.max()
pct10 = pct100 * 0.1
pct50 = pct100 * 0.5
pct90 = pct100 * 0.9

levelist = [pct10,pct50,pct90,pct100]
# levelist = np.linspace(0.1,1,4) * pct100

# xx = 2 * np.pi / xx
# zz = 2 * np.pi / zz

axs.contourf(xx,zz, sp_u,   levelist, cmap='cmo.gray')
axs.contour( xx,zz, sp_up,  levelist, colors='r')
axs.contour( xx,zz, sp_up2, levelist, colors='y')

# axs.set_ylim(0,100)
# axs.set_xlim(390,400)
# axs.set_ylim(390,450)
# for ax in axs:
# axs.set_xticks(np.linspace(xx.min(),xx.max(),4))
# axs.set_yticks(np.linspace(zz.min(),zz.max(),4))
# axs[1].contourf(xx,zz,sp_up2, levels = 100)
fig.savefig(f"Spatial_scale_Dir{direction}.jpg",bbox_inches='tight',dpi=300)