"""
Visualisation of the output results regarding the solution
"""
# Environ 
import numpy as np
from scipy.io import loadmat
import argparse
from matplotlib import pyplot as plt 
from pyDOE import lhs
from scipy.interpolate import interp1d
from lib import gen_data, name_PINNs
from utils.plot import colorplate as cc

plt.rc("font",family = "serif")
plt.rc('text',usetex=True)
plt.rc("font",size = 20)
plt.rc("axes",labelsize = 16, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 16)
plt.rc("ytick",labelsize = 16)
font_dict = {"fontsize":25,'weight':'bold'}



def load_pred(case_name):
    """
    Load the prediction from training and analytical results
    """
    
    data_path   = '02_pred/'
    d_res           = loadmat(data_path + case_name + ".mat")
    d_ana           = loadmat(data_path + 'Analy_' + case_name + '.mat')

    return d_res, d_ana 


def vis_profiles(
                u,up,
                y,
                ):
    """
    Visualisation of the 
    """

    print(f"The ground Truth: {up.shape}, {u.shape}")

    markers = ["^","^"]
    colors  = ["g",'r']

    fig, axs = plt.subplots(2,3,figsize=(16,8),sharey=True)
    
    names = [[r"$U$"+" "+"[m/s]",
        r"$\overline{u^2}$"+" "+"[m"+ r"$^2$" +"/s"+r"$^2$" +"]",
        r"$\overline{v^2}$"+" "+"[m"+ r"$^2$" +"/s"+r"$^2$" +"]",
        r"$\overline{uv}$"+" "+"[m"+ r"$^2$" +"/s"+r"$^2$" +"]" ],
        [r"$V [m/s]$",r"$P [pa]$"]
        ]

    axf = axs.flatten()

    for j in range(6):
        if j <=3:
            axf[j].plot(u[:,j],y,"-ko",lw =3)
            axf[j].plot(up[:,j],y,'o',c='r',markersize = 10)
            axf[j].set_xlabel(names[0][j],font_dict)

        else:
            axf[j].plot(up[:,j],y,'o-',c='b',markersize = 10)
            axf[j].set_xlabel(names[1][j-4],font_dict)

    axs[0,0].set_ylabel(r"$y$" + " [m]",font_dict)
    axs[1,0].set_ylabel(r"$y$" + " [m]",font_dict)
    fig.subplots_adjust(hspace=0.3)
    fig.savefig('figs/out.jpg',bbox_inches='tight',dpi=300)
    
    return 

def vis_Diff(da,y):
    """
    Visualisation of the derivatives as Profile of y 
    """

    ## For contiunity euqation 

    names_cont = [
                r"$\frac{\partial U}{\partial x}$",
                r"$\frac{\partial V}{\partial y}$",
                ]
    
    fig, axs = plt.subplots(1,2,figsize=(6,2),sharey=True)
    axs[0].plot(da['dUdx'].flatten(),y,"-o",c=cc.green,lw=2)
    axs[1].plot(da['dVdy'].flatten(),y,"-o",c=cc.orange,lw=2)
    
    for i,n in enumerate(names_cont):
        axs[i].set_xlabel(n,font_dict)
    axs[0].set_ylabel(r"$y$" + " [m]",font_dict)
    fig.savefig('figs/Contiunity.jpg',bbox_inches='tight',dpi=300)


    ## For X-Momentum:
    names_cont = [
                r"$U\frac{\partial U}{\partial x}$",
                r"$V\frac{\partial U}{\partial y}$",
                r"$\frac{\partial P}{\partial x}$",
                r"$\frac{\partial \overline{u^2}}{\partial x}$",
                r"$\frac{\partial \overline{uv}}{\partial y}$",
                ]
    fig, axs = plt.subplots(2,3,figsize=(12,5),sharey=True)
    
    axs[0,0].plot((da["U"]*da['dUdx']).flatten(),y,"-o",c=cc.green,lw=2)
    axs[0,1].plot((da["V"]*da['dUdy']).flatten(),y,"-o",c=cc.orange,lw=2)
    axs[0,2].plot((da['dPdx']).flatten(),y,"-o",c=cc.green,lw=2)
    axs[1,0].plot((da['duudx']).flatten(),y,"-o",c=cc.green,lw=2)
    axs[1,1].plot((da['duvdy']).flatten(),y,"-o",c=cc.orange,lw=2)
    axs[1,2].axis('off')
    
    axf = axs.flatten()
    for i,n in enumerate(names_cont):
        axf[i].set_xlabel(n,font_dict)
    
    axs[0,0].set_ylabel(r"$y$" + " [m]",font_dict)
    axs[1,0].set_ylabel(r"$y$" + " [m]",font_dict)
    
    fig.subplots_adjust(hspace=0.55)

    fig.savefig('figs/X-Momentum.jpg',bbox_inches='tight',dpi=300)


    ## For Y-Momentum:
    names_cont = [
                r"$U\frac{\partial V}{\partial x}$",
                r"$V\frac{\partial V}{\partial y}$",
                r"$\frac{\partial P}{\partial y}$",
                r"$\frac{\partial \overline{uv}}{\partial x}$",
                r"$\frac{\partial \overline{v^2}}{\partial y}$",
                ]
    fig, axs = plt.subplots(2,3,figsize=(12,5),sharey=True)
    
    axs[0,0].plot((da["U"]*da['dVdx']).flatten(),y,"-o",c=cc.green,lw=2)
    axs[0,1].plot((da["V"]*da['dVdy']).flatten(),y,"-o",c=cc.orange,lw=2)
    axs[0,2].plot((da['dPdy']).flatten(),y,"-o",c=cc.orange,lw=2)
    axs[1,0].plot((da['duvdx']).flatten(),y,"-o",c=cc.green,lw=2)
    axs[1,1].plot((da['dvvdy']).flatten(),y,"-o",c=cc.orange,lw=2)
    axs[1,2].axis('off')
    
    axf = axs.flatten()
    for i,n in enumerate(names_cont):
        axf[i].set_xlabel(n,font_dict)
    
    axs[0,0].set_ylabel(r"$y$" + " [m]",font_dict)
    axs[1,0].set_ylabel(r"$y$" + " [m]",font_dict)
    
    fig.subplots_adjust(hspace=0.55)

    fig.savefig('figs/Y-Momentum.jpg',bbox_inches='tight',dpi=300)

def vis_residual(r):
    """
    Visualisation of Residual evolution 
    """


    fig, axs = plt.subplots(1,1,figsize=(8,4))

    axs.semilogy(np.arange(len(r[:,0])),r[:,0],c=cc.blue,label='Continuity')
    axs.semilogy(np.arange(len(r[:,0])),r[:,1],c=cc.red,label='X-Momentum')
    axs.semilogy(np.arange(len(r[:,0])),r[:,2],c=cc.green,label='Y-Momentum')

    axs.legend(loc='upper right')
    axs.set_xlabel('Epoch',font_dict)
    axs.set_ylabel("Residual",font_dict)

    rect_width =  0.9*(1000/len(r[:,0]))  * (axs.get_xlim()[1] - axs.get_xlim()[0])
    rect_height = axs.get_ylim()[1] - axs.get_ylim()[0]
    rect_left = axs.get_xlim()[0]
    axs.add_patch(plt.Rectangle((0, axs.get_ylim()[0]), rect_width, 
                                rect_height, 
                                color='#e6f7ff', alpha=1))
    axs.set_xlim(0,len(r[:,0]))

    fig.savefig("figs/Residual.jpg",bbox_inches='tight',dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINN training')
    parser.add_argument('--cp', default= 50, type=int, help='Number of grid point')
    parser.add_argument('--nl', default= 4 , type=int, help='Number of layer')
    parser.add_argument('--nn', default= 40 , type=int, help='Number of neuron')
    parser.add_argument('--epoch', default=1000, type=int, help='Training Epoch')
    parser.add_argument('--sw', default=10, type=int, help='Weight of supervise learning loss')
    parser.add_argument('--uw', default=1, type=int, help='Weight of unsupervise learning loss')
    parser.add_argument('--f', default=3, type=int, help='Sample Frequency of the reference data')
    args = parser.parse_args() 
    case_name = name_PINNs(args)
    d,da = load_pred(case_name)
    ic,cp,gt,cp_test,cp_spine = gen_data(args)
    y = cp_test[:,-1]
    vis_profiles(gt,d['up'],y)
    vis_Diff(da,y)
    r = d['residual'].squeeze().mean(1)
    vis_residual(r)