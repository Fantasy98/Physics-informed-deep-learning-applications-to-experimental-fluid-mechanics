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




class mini_channel:
    Recl    = 5000 
    Retau   = 202 
    h       = 1  # Non-dimensional half-height of channel 
    mu      = 1/Recl # Viscousity 
    utau    = Retau/Recl

def kolmogorov_scale( mu, dissip):
    """
    Compute the Kolmogorov scale by dissipation term
    
    We follow the equation eta = (mu**3/epsilon)**(1/3)
    """
    eta= ( mu**3. / dissip )
    eta= eta **(0.25)
    return eta


def ref_scale(Re_tau):
    """
    Load the data of Re_tau from the DNS channel case 
    and scaled back to our case by u_tau

    We follow the equation eta = (mu**3/epsilon)**(1/3)
    """
    print("#"*30)
    print(f"Reference scale from public DNS Dataset")
    # File which stores the sum of dissapsion 
    file_path = f"data/Re{Re_tau}/balances/Re{Re_tau}.bal.kbal"
    print(f"Loading the file from:\n{file_path}")


# Initialise the diction for data
#--------------------
    key_list = ["y/h",
                "y+",
                "dissip",
                "produc",
                "p-strain",
                "p-diff",
                "t-diff",
                "v-diff",
                "bal",
                "tp-kbal"]
    
    data_dict = {}
    for k in key_list:
        data_dict[k] = []

# Load the file 
#--------------------

    with open(file_path,'r') as f:
        
        i = 0 
        head = f.readline().split()
        header = head[0]
        header_next= header
        
        while( (header_next[0] == header)):
            line_next = f.readline().split()
            header_next = line_next[0]
            i+=1
        print(f"At {i} We stop the header of the data!")
        
        for i, val in enumerate(line_next):
            data_dict[key_list[i]].append(float(val))

        if_load = True
        while if_load:
            line_next = f.readline().split()
            if len(line_next) != 0:
                for i, val in enumerate(line_next):
                    data_dict[key_list[i]].append(float(val))
                if_load = True
            else:
                if_load = False
    f.close()
    print(f"DATA LOADING FINISH:")

# Acuqire data 
#--------------------
    dissip = np.array(data_dict['dissip'])
    yh     = np.array(data_dict['y/h'])
    print(f"Dissipation Term Loaded: There are {len(dissip)} elements along the wall")

# Compute Kolmogorov Scale as a function of wall
#--------------------
    ## Basic info for our 
    
    dissip *= mini_channel.utau**4 * mini_channel.Recl
    eta = kolmogorov_scale(mini_channel.mu, dissip=np.abs(dissip))
    print(f"Komlgorov Scaled computed, max: {eta.max():.3f}, min: {eta.min():.3f}")
    fig,axs = plt.subplots(1,1,figsize=(5,5))
    axs.plot(eta,yh,'-o',c=cc.black,lw=2.5)
    axs.set_xlabel(r'$\eta$')
    axs.set_ylabel(r'$y/h$')
    axs.set_title(r"$Re_{\tau}$" + f" = {mini_channel.Retau}")
    fig.savefig("Figs/kolmogorov_normal.pdf",bbox_inches='tight',dpi=500)
    
    return

def actual_scale(args):
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
    Recl    = 5000 
    Retau   = 202 
    h       = 1  # Non-dimensional half-height of channel 

    print(f"\nCharacteristic values from reference:")
    print(f"Half height of channel h:\t{h}")
    print(f"Central line Reynolds number Re_cl:\t{Recl}")
    print(f"Friction Reynolds number Re_tau:\t{Retau}")

    #------------------------------------
    # Kolmogorov Length scale estimation 
    # L/eta  = Re^{3/4}
    eta  = h /  Recl**(3/4)
    print(f"\nKolmgorov length scale approximation: {eta:.3f} ")

    #------------------------------------
    # Kolmogorov Time scale estimation 
    # tL/teta  = Re^{1/2}
    teta  =  1 /  Recl**(1/2)
    print(f"\nKolmgorov time scale approximation: {teta:.3f} ")

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
    
    Re_tau = 180
    
    ref_scale(Re_tau)

    actual_scale(args)