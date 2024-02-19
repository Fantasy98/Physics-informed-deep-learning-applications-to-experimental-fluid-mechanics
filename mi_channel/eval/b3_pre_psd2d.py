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
    yp = 10
    y_loc = Ly * (yp/Ny) /eta
    print(f"At y+ = {y_loc}")
    y_loc = Ly * (yp/Ny)
    print(f"At y = {y_loc}")
    xp = int(Nx/2)
    
    
    # utype = 2
    # Streamwise velocity at wall
    data_ = data[utype] 
    data  = data - np.mean(data,-1,keepdims=True)
    
    Yplus = np.linspace(0,Ly,Ny)/eta
    
    Spectra = np.empty(shape=(Nx,Ny))
    for yp in range(Ny):
        data    = data_[:,yp,:,:]
        
        dkx = 2*np.pi/(Lx) 
        dky = 2*np.pi/(Ly)
        dkz = 2*np.pi/(Lz)

        # Wave number 
        x_range = np.linspace(1, Nx, Nx)
        y_range = np.linspace(1, Ny, Ny)
        z_range = np.linspace(1, Nz, Nz)

        kx =dkx *  np.append(x_range[:Nx//2], -x_range[Nx//2:0:-1])
        ky =dky *  np.append(y_range[:Ny//2], -y_range[Ny//2:0:-1])
        kz =dkz *  np.append(z_range[:Nz//2], -z_range[Nz//2:0:-1])

        kkx, kkz = np.meshgrid(kx,kz)
        
        kkx_norm = np.sqrt(kkx[0,:]**2)
        kkz_norm = np.sqrt(kkz[:,0]**2)


        Lambda_x = (2*np.pi/kkx_norm)/eta
        Lambda_z = (2*np.pi/kkz_norm)/eta
        u_hat = np.fft.fftn(data)
        # u_hat = np.fft.fftshift(u_hat)
        spectra = np.empty(shape=(Nx,nt))
        
        for t in range(nt):

            spectra[:,t] = np.mean(np.absolute(u_hat[:,:,t]),axis=0) * kx /u_tau**2
        
        spectra = np.mean(spectra,-1) 
        Spectra[:,yp] = spectra

    return Spectra,Lambda_x, Yplus

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

Ylabels = [  
            r"$E_{u}/(\nu \cdot u_{\tau})$",
            r"$E_{v}/(\nu \cdot u_{\tau})$",
            r"$E_{w}/(\nu \cdot u_{\tau})$",
            ]

Ylabels = [  
            r"$k_x E_{u}$",
            r"$k_x E_{v}$",
            r"$k_x E_{w}$",
            ]

Fname  =  ["u","v",'w']

for utype in range(3):

    sp_u,wvnumber,yP = PSD_1D(u,
                    Nx = nx,
                    Ny = ny,
                    Nz = nz,
                    nt = nt,
                    Lx = Lx,
                    Ly = Ly,
                    Lz = Lz,
                    utype= utype,
                    )

    sp_t3s8,wvnumber,yP = PSD_1D(u_pred2,
                    Nx = nx,
                    Ny = ny,
                    Nz = nz,
                    nt = nt,
                    Lx = Lx,
                    Ly = Ly,
                    Lz = Lz,
                    utype= utype,
                    )



    sp_t3s16,wvnumber,yP = PSD_1D(u_pred1,
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
    
    axs.contourf( 
                yP,
                wvnumber,
                sp_u,
                levels = np.array([0.1,0.5,0.9])*sp_u.max(),
                cmap = 'cmo.gray_r',
                # '-.',
                # c= cc.black,
                # lw = 3,
                # label = 'Reference'
            )

    axs.contour(yP,wvnumber,sp_t3s16,
                levels = np.array([0.1,0.5,0.9])*sp_u.max(),
                colors=cc.red)
    axs.contour(yP,wvnumber,sp_t3s8,
                levels = np.array([0.1,0.5,0.9])*sp_u.max(),
                colors=cc.blue)
    # axs.set_xscale("log")
    axs.set_yscale("log")
    # axs.set_aspect('equal')
    # axs.set_xlim(0,yP.max()``)
    # axs.set_ylim(0,wvnumber.max())
    
    # axs.loglog( 
    #             wvnumber,
    #             sp_t3s8,
    #             "-o",
    #             c = cc.blue,
    #             lw = 2,
    #             markersize = 7.5,
    #             label = r'PINN--t3--s8'
    #         )


    # axs.loglog( 
    #             wvnumber,
    #             sp_t3s16,
    #             "-^",
    #             c = cc.red,
    #             lw = 2,
    #             markersize = 7.5,
    #             label = r'PINN--t3--s16'
    #         )


    axs.set_title(Ylabels[utype],font_dict )
    # axs.set_ylabel(r"$k_{x^+}$", font_dict)   
    axs.set_xlabel(r'$y^+$')
    # axs.set_ylabel(Ylabels[utype],font_dict )
    axs.set_ylabel(r"$\lambda^+_{x}$", font_dict)   
    
    # axs.set_ylim(5 * 10e-7, 10e0)
    fig.savefig( f"Figs/PSD2d_{Fname[utype]}.jpg",bbox_inches='tight',dpi=200)