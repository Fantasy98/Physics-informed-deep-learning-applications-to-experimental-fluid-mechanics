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


def PSD_1D(data,Nx,Ny,Nz, nt,Lx,Ly,Lz,utype=0):
    import numpy as np
    Re_Tau = 202 #Direct from simulation
    Re = 5000 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu
    
    eta = nu / u_tau
    
    # eta = 1/Re**-0.75
    # eta = 1
    # nu =1
    # u_tau =1 
    yp =  30
    y_loc = Ly * (yp/Ny) /eta

    print(f"At y = {y_loc}")
    # print(yp)

    xp = int(Nx/2)
    # utype = 2
    # Streamwise velocity at wall
    data = data[utype] 
    # for t in range(nt):
    #     data[:,:,:,t] = data[:,:,:,t] - data.mean(-1)

    data    = data[:,yp,:,:]
    data    = data[:,:,xp,:]
    # Fluctuation
    # Wavenumber spacing
    dkx = 2*np.pi/(Lx) 
    dky = 2*np.pi/(Ly)
    dkz = 2*np.pi/(Lz)

    # Wave number 
    x_range =  dkx *  np.linspace(-Nx/2 , Nx/2 -1, Nx)
    y_range =  dky *  np.linspace(-Ny/2 , Ny/2 -1, Ny)
    z_range =  dkz *  np.linspace(-Nz/2 , Nz/2 -1, Nz)

    kx = np.append(x_range[:Nx//2], -x_range[Nx//2:0:-1])
    ky = np.append(y_range[:Ny//2], -y_range[Ny//2:0:-1])
    kz = np.append(z_range[:Nz//2], -z_range[Nz//2:0:-1])

    # kx = np.sqrt((kx)**2)
    # ky = np.sqrt((ky)**2)
    # kz = np.sqrt((kz)**2)

    Lambda_x = kx 
    Lambda_y = ky 
    Lambda_z = kz 
    spectra = np.empty(shape=(Nx,nt))
    # spectra = np.empty(shape=(Ny,nt))
    for t in range(nt):
        # At each timestep, fft on x-z plane
        u_hat = np.fft.fftn(data[:,:,t])
        u_hat = np.fft.fftshift(u_hat)
        eng   = np.absolute(u_hat.mean(0))**2
        spectra[:,t] = eng
        # spectra[:,t] = eng 

    # spectra = spectra/spectra.max()
    spectra = np.mean(spectra,-1)
    wvnumber = Lambda_x * eta
    return spectra, wvnumber



    

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 1.5)
plt.rc('font', size = 16)
plt.rc('legend', fontsize = 14)              
plt.rc('xtick', labelsize = 18)             
plt.rc('ytick', labelsize = 18)
font_dict = {"fontsize":22,'weight':'bold'}


ref = np.load('../data/min_channel_sr.npz')

x = ref['x'] 
y = ref['y']
z = ref['z']
u = ref['u'] # dimensions  = (nz, ny, nx, nt)
v = ref['v']
w = ref['w']
t = ref['t']


# Lx     =   np.abs(x.max()- x.min())
# Ly     =   np.abs(y.max()- y.min())
# Lz     =   np.abs(z.max()- z.min())

Lx     =  0.6 * np.pi 
Ly     =  1 
Lz     =  0.01125*np.pi


u = np.stack([u, v, w])

yy, zz, xx = np.meshgrid(y, z, x)

nz, ny, nx = xx.shape
nt         = len(t)
pred1 = np.load('../results/res_pinn_t3_s16.npz')
u_pred1 = pred1['up'][:3]

pred2 = np.load('../results/res_pinn_t3_s8.npz')
u_pred2 = pred2['up'][:3]

t      = 5 
yp     = 0
utype  = 2


Ylabels = [  
            r"$E_{u}/(\nu \cdot u_{\tau})$",
            r"$E_{v}/(\nu \cdot u_{\tau})$",
            r"$E_{w}/(\nu \cdot u_{\tau})$",
            ]

Ylabels = [  
            r"$E_{u}$",
            r"$E_{v}$",
            r"$E_{w}$",
            ]

Fname  =  ["u","v",'w']

for utype in range(3):

    sp_u, wvnumber = PSD_1D(u,
                    Nx = nx,
                    Ny = ny,
                    Nz = nz,
                    nt = nt,
                    Lx = Lx,
                    Ly = Ly,
                    Lz = Lz,
                    utype= utype,
                    )

    sp_t3s8, wvnumber = PSD_1D(u_pred2,
                    Nx = nx,
                    Ny = ny,
                    Nz = nz,
                    nt = nt,
                    Lx = Lx,
                    Ly = Ly,
                    Lz = Lz,
                    utype= utype,
                    )



    sp_t3s16, wvnumber = PSD_1D(u_pred1,
                    Nx = nx,
                    Ny = ny,
                    Nz = nz,
                    nt = nt,
                    Lx = Lx,
                    Ly = Ly,
                    Lz = Lz,
                    utype= utype,
                    )



    fig, axs = plt.subplots(1,1,sharex=True, sharey=True, figsize=(6,4))

    axs.loglog( 
                wvnumber,
                sp_u,
                '-.',
                c= cc.black,
                lw = 3,
                label = 'Reference'
            )

    axs.loglog( 
                wvnumber,
                sp_t3s8,
                "-o",
                c = cc.blue,
                lw = 2,
                markersize = 7.5,
                label = r'PINN--t3--s8'
            )


    axs.loglog( 
                wvnumber,
                sp_t3s16,
                "-^",
                c = cc.red,
                lw = 2,
                markersize = 7.5,
                label = r'PINN--t3--s16'
            )

    axs.set_ylabel(Ylabels[utype],font_dict )
    axs.set_xlabel(r"$k_x l^*$", font_dict)   
    # axs.set_xlabel(r"$k_{y^+}$", font_dict)   
    # axs.set_xlabel(r"$k_{y^+}$", font_dict)   
    
    axs.set_ylim(5 * 10e-7, 10e0)
    
    axs.legend(frameon=False, ncol = 3, loc = (0.0, 1.05), fontsize=13)

    # axs.set_xticks([wvnumber.min(), 0.5 * wvnumber.max(), wvnumber.max()])

    fig.savefig( f"Figs/PSD1d_{Fname[utype]}.pdf",bbox_inches='tight',dpi=1000)