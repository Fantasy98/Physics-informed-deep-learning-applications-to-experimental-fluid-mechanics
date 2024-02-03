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


def PSD(data,Nx,Nz,Lx,Lz):
    import numpy as np
    
    # Computational box and dimensions of DNS daa
    # Nx = 256
    # Nz  = 256
    # Lx  = 12
    # Lz  = 6

    # Wavenumber spacing
    dkx = 2*np.pi/Lx
    dkz = 2*np.pi/Lz

    x_range=np.linspace(1,Lx,Nx)
    z_range=np.linspace(1,Lz,Nz)
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

    # Theta_fluc_targ=data-np.mean(data)
    Theta_fluc_targ=data-np.mean(data)

    # We compute the 2 dimensional discrete Fourier Transform
    fourier_image_targ = np.fft.fftn(Theta_fluc_targ)
    # We also compute the pre-multiplication with the wavenumber vectors
    # print(fourier_image_targ.shape, kkx.shape, kkz.shape)
    fourier_amplitudes_targ = np.mean(np.abs(fourier_image_targ)**2,axis=0).T*kkx*kkz
    
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

print(Lx,Ly,Lz)

u = np.stack([u, v, w])

yy, zz, xx = np.meshgrid(y, z, x)

nz, ny, nx = xx.shape
nt         = len(t)
pred1 = np.load('../results/res_pinn_t3_s16.npz')
u_pred1 = pred1['up'][:3]

pred2 = np.load('../results/res_pinn_t3_s8.npz')
u_pred2 = pred2['up'][:3]

x = xx[0]
y = yy[0]

t      = 5 
yp     = 0
utype  = 0
u_wall   =       u[utype,  yp , :  , : ,:].reshape(nt,nx,ny)
u_wallp  = u_pred1[utype,  yp , :  , : ,:].reshape(nt,nx,ny)
u_wallp2 = u_pred2[utype,  yp , :  , : ,:].reshape(nt,nx,ny)
print(u_wall.shape)

sp_u,xx,zz = PSD(u_wall,
                Nx = nx,
                Nz = ny,
                Lx = Lx,
                Lz = Ly )

sp_up,_,_ = PSD(u_wallp,
                Nx = nx,
                Nz = ny,
                Lx = Lx,
                Lz = Ly )

# sp_up2,_,_ = PSD(u_wallp2,
#                 Nx = nx,
#                 Nz = nz,
#                 Lx = Lx,
#                 Lz = Lz )


fig, axs = plt.subplots(1,1,sharex=True, sharey=True)

pct100 = sp_u.max()
pct10 = pct100 * 0.1
pct50 = pct100 * 0.5
pct90 = pct100 * 0.9

axs.contourf(xx,zz,sp_u, [pct10, pct50, pct90, pct100])
axs.contour( xx,zz,sp_up, [pct10, pct50, pct90, pct100],colors='r')
# axs.set_ylim(0,100)
# axs.set_xlim(360,370)
# for ax in axs:
# axs.set_xticks(np.linspace(xx.min(),xx.max(),4))
# axs.set_yticks(np.linspace(zz.min(),zz.max(),4))
# axs[1].contourf(xx,zz,sp_up2, levels = 100)
fig.savefig("Spatial_scale.jpg")