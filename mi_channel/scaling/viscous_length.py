"""
To compute the Kolmogorov scale by levearging the public dataset and compare with the current results 
@yuningw
"""

import numpy as np
from matplotlib import pyplot as plt 
import cmocean
import cmocean.cm as cmo
from  math import pi
import argparse
import os
class cc:
    red = "#D23918" # luoshenzhu
    blue = "#2E59A7" # qunqing
    yellow = "#E5A84B" # huanghe liuli
    cyan = "#5DA39D" # er lv
    black = "#151D29" # lanjian
    green = "#2A6E3F" # guan lv
    brown = "#9F6027" # huang liu 
    purple = "#A76283" # zi jing pin feng 
    orange = "#EA5514" # huang dan

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

font_dict = {'size':25,"weight":'bold'}


class mini_channel:
    Recl    = 5000 
    Retau   = 202 
    h       = 1  # Non-dimensional half-height of channel 
    mu      = 1/Recl # Viscousity 
    utau    = Retau/Recl

def compute_scale(mu, utau):
    """
    Compute the Kolmogorov scale by viscousity and friction velocity 
    
    We follow the equation for viscous time:  
            
            teta = mu/utau**2 

            eta = teta * utau = mu/utau
    """

    teta = mu / utau**2
    eta  = teta * utau # Length scale

    return eta, teta



def viscous_scale(args):
    """
    Function for showcasting the current scale

    Args:   
        args    :   class passed from argparser
    """
    print("\n"+"#"*30)
    print(f"The actual case in hand:")
    #########################
    ## Read Data
    #########################
    ref = np.load('../data/min_channel_sr.npz')

    x = ref['x'] 
    y = ref['y']
    z = ref['z']
    u = ref['u'] # dimensions  = (nz, ny, nx, nt)
    t = ref['t']

    nz,ny,nx,nt = u.shape
    yy, zz, xx = np.meshgrid(y, z, x)
    dt = 0.2
    nv = 4
    nz, ny, nx = xx.shape


    print(f"\nSummary of data:")
    print(f"Data loaded, Nz = {nz}\nNy = {ny}\nNx = {nx}\nNt = {nt}")
    print(f'\nTime duration: t = {t.min():.2f} to {t.max():.2f}, nt = {len(t)}')
    print(f"\nDomain size\tX = {x.min()/pi:.1f}pi to {x.max()/pi:.1f}pi")
    print(f"Domain size\tY = {y.min():.1f} to {y.max():.1f}")
    print(f"Domain size\tZ = {z.min()/pi:.5f}pi to {z.max()/pi:.4f}pi")


    #-----------------------------------
    # Some given information in the reference 
    Recl    = mini_channel.Recl
    Retau   = mini_channel.Retau
    h       = mini_channel.h  # Non-dimensional half-height of channel 

    print(f"\nCharacteristic values from reference:")
    print(f"Half height of channel h:\t{h}")
    print(f"Central line Reynolds number Re_cl:\t{Recl}")
    print(f"Friction Reynolds number Re_tau:\t{Retau}")

    #------------------------------------
    # Kolmogorov Length scale estimation 
    # L/eta  = Re^{3/4}
    eta,teta  = compute_scale(mu=mini_channel.mu,utau=mini_channel.utau)
    print(f"\nViscous length scale approximation: {eta:.3f} ")
    #------------------------------------
    # Kolmogorov Time scale estimation 
    print(f"\nViscous time scale approximation: {teta:.3f} ")


    # We deomstrate the scaled wall-normal positions

    y_scale = y/eta
    fig,axs = plt.subplots(1,1,figsize=(5,5))
    axs.plot(y_scale,y,'-o',c=cc.black,lw=2.5)
    axs.set_xlabel(r'$\frac{y u_\tau }{\mu}$',font_dict)
    axs.set_ylabel(r'$y/h$',font_dict)
    axs.set_title(r"$t_{\rm visc }$" + f" = {teta:.3e}\n"+\
                r"$l_{\rm visc}$" + f"= {eta:.3e}"  )
    fig.savefig("Figs/visous_normal.pdf",bbox_inches='tight',dpi=500)
    




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
    print(f"Scaled:\ndx_s = {dx_s:.2f}eta\ndy_s_min = {dy_s_min:.2f}eta dy_s_max = {dy_s_max:.2f}eta\ndz_s = {dz_s:.2f}eta\ndt_s = {dt_s:.2f}teta")


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



if __name__ == '__main__':
    

    viscous_scale(args)