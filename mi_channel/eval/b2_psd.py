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
from utils.plot import colorplate as cc
from utils import plt_rc_setup

font_dict = {"fontsize":25,'weight':'bold'}

def PSD_1D(data,Nx,Ny,Nz, nt,Lx,Ly,Lz, ):
    import numpy as np
    
    yp = 40
    utype = 0
    # Streamwise velocity at wall
    data = data[utype]
    for t in range(nt):
        data[:,:,:,t] = data[:,:,:,t] - data.mean(-1)

    data    = data[:,yp,:,:]
    # Fluctuation
    # Wavenumber spacing
    dkx = 2*np.pi/Lx
    dky = 2*np.pi/Ly
    dkz = 2*np.pi/Lz

    # Wave number 
    x_range =  dkx *  np.linspace(-Nx/2 , Nx/2 -1, Nx)
    y_range =  dky *  np.linspace(-Ny/2 , Ny/2 -1, Ny)
    z_range =  dkz *  np.linspace(-Nz/2 , Nz/2 -1, Nz)

    kx = dkx * np.append(x_range[:Nx//2], -x_range[Nx//2:0:-1])
    ky = dky * np.append(y_range[:Ny//2], -y_range[Ny//2:0:-1])
    kz = dkz * np.append(z_range[:Nz//2], -z_range[Nz//2:0:-1])

    kx = np.sqrt(kx**2)
    ky = np.sqrt(ky**2)
    kz = np.sqrt(kz**2)

    Re_Tau = 202 #Direct from simulation
    Re = 5000 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu
    eta = 1/Re**-0.75
    Lambda_x =kx * eta 
    Lambda_y =ky
    Lambda_z =kz
    spectra = np.empty(shape=(Nx,nt))
    for t in range(nt):
        # At each timestep, fft on x-z plane
        u_hat = np.fft.fft(data[:,:,t])
        u_hat = np.fft.fftshift(u_hat)
        eng   = np.absolute(u_hat.mean(0))**2
        spectra[:,t] = eng

    # spectra = spectra/spectra.max()
    spectra = np.mean(spectra,-1)
    # spectra = spectra[:,0]

    return spectra, Lambda_x



    

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


sp_u, wvnumber = PSD_1D(u,
                Nx = nx,
                Ny = ny,
                Nz = nz,
                nt = nt,
                Lx = Lx,
                Ly = Ly,
                Lz = Lz,
                )

sp_t3s8, wvnumber = PSD_1D(u_pred2,
                Nx = nx,
                Ny = ny,
                Nz = nz,
                nt = nt,
                Lx = Lx,
                Ly = Ly,
                Lz = Lz,
                )



sp_t3s16, wvnumber = PSD_1D(u_pred1,
                Nx = nx,
                Ny = ny,
                Nz = nz,
                nt = nt,
                Lx = Lx,
                Ly = Ly,
                Lz = Lz,
                )



fig, axs = plt.subplots(1,1,sharex=True, sharey=True, figsize=(6,4))

axs.loglog( 
            wvnumber,
            sp_u,
            '-.',
            c= cc.black,
            lw = 2.5,
            label = 'Reference'
        )

axs.loglog( 
            wvnumber,
            sp_t3s8,
            c = cc.blue,
            lw = 2,
            label = r'PINN--t3--s8'
        )


axs.loglog( 
            wvnumber,
            sp_t3s16,
            c = cc.red,
            lw = 2,
            label = r'PINN--t3--s16'
        )

axs.set_ylabel(r"$E_{u}(k)$",font_dict )
axs.set_xlabel(r"$k\eta$", font_dict)   
axs.set_ylim(10e-7, 10e0)
axs.legend(frameon=False, ncol = 3, loc = (0.0, 1.05), fontsize=12)

# axs.set_xticks([wvnumber.min(), 0.5 * wvnumber.max(), wvnumber.max()])

fig.savefig(f"PSD1d.jpg",bbox_inches='tight',dpi=300)