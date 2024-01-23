"""
A script for answering the reviewer's question and evaluation
@yuningw
"""
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt 
from scipy.io import loadmat
from pyDOE import lhs
from time import time
from pathlib import Path
from scipy.stats import pearsonr

import cmocean
import cmocean.cm as cmo

import argparse
argparser   = argparse.ArgumentParser()
argparser.add_argument("--c",default=2.5,type=float, help="Level of gaussian noise: 0, 2.5, 5, and 10%")
args        = argparser.parse_args()

data_path   = 'data/'
res_path    = 'res/'
model_path  = 'model/'
fig_path    = 'figs/'

fileName    = f"cylinder_PINN_{args.c:.1f}"

data = loadmat(data_path + 'cylinder_nektar_wake.mat')

def l2_norm_error(p,g): 
    """
    Compute the l2 norm error 
    """
    error = (LA.norm(p-g,axis=(0,1)))/LA.norm(g,axis =(0,1))
    # error = (LA.norm(p-g))/LA.norm(g)
    # if len(p.shape) == 3:
    #     error = np.empty(shape=(p.shape[-1]))
    #     for il in range(p.shape[-1]):
    #         error[il] = (LA.norm(p[:,:,il]-g[:,:,il]))/LA.norm(g[:,:,il])
    # else:
    #     error = (LA.norm(p-g))/LA.norm(g)
    
    # print(error.shape)
    return error.mean() 

def get_Data(noise_level):
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

    ncp = 2000
    lb = np.array([x.min(), y.min(), t.min()])
    ub = np.array([x.max(), y.max(), t.max()])

    cp = lb + (ub-lb) * lhs(3, ncp)

    ns = len(xx.flatten())

    ic = np.array([xx.flatten(), yy.flatten(), np.zeros((ns,)) + t[0],
                    u_ns[:, :, 0].flatten(), v_ns[:, :, 0].flatten()]).T

    pr = 0.8
    ind_ic = np.random.choice([False, True], len(ic), p=[1 - pr, pr])
    ic = ic[ind_ic]

    ind_bc = np.zeros(xx.shape, dtype = bool)
    ind_bc[[0, -1], :] = True; ind_bc[:, [0, -1]] = True

    X, Y, T = np.meshgrid(x, y, t)

    x_bc = X[ind_bc].flatten()
    y_bc = Y[ind_bc].flatten()
    t_bc = T[ind_bc].flatten()

    u_bc = u_ns[ind_bc].flatten()
    v_bc = v_ns[ind_bc].flatten()
    # p_bc = p_ns[ind_bc].flatten()

    bc = np.array([x_bc, y_bc, t_bc, u_bc, v_bc]).T

    pr = 0.2
    indx_bc = np.random.choice([False, True], len(bc), p=[1 - pr, pr])
    bc = bc[indx_bc]

    return X,Y,T,xx,yy,u,v,p,ic,bc,cp 


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
    p_ns = p + np.random.normal(0,noise_level,np.shape(v)) * p / 100  

    ny, nx = xx.shape
    return x,y, xx,yy ,t, u, v, p, u_ns, v_ns,p_ns 

x,y,xx,yy ,t, u, v, p, u_ns, v_ns, p_ns = get_test_Data(args.c)

# Get training data
# X,Y,T, xx,yy,u,v,p,ic, bc, cp = get_Data(noise_level=args.c)


dp   = np.load(res_path + fileName + ".npz")
print(f"INFO: the predcition of {fileName} has been loaded!")
dp    = dp['up']
t18, t53 = 18, 53

# Load the prediction, order: U, V, P 
up   = dp[0]
vp   = dp[1]
pp   = dp[2]


# Compute the vorticity of the flow 
uy = np.gradient(u, y, axis = 0, edge_order=2)
vx = np.gradient(v, x, axis = 1, edge_order=2)
omega_z = vx - uy

uy_p = np.gradient(up, y, axis = 0, edge_order=2)
vx_p = np.gradient(vp, x, axis = 1, edge_order=2)
omega_z_p = vx_p - uy_p

uy_ns = np.gradient(u_ns, y, axis = 0, edge_order=2)
vx_ns = np.gradient(v_ns, x, axis = 1, edge_order=2)
omega_z_ns = vx_ns - uy_ns



#-------------------------------------------------------
##################
## Question c: Why shift the pressure??
##################
print("#"*30)
print(f"Question 1")
deltap  = p.mean()- pp.mean()
pp      += deltap
print(f"After shiftting, mean pressure of ref: {p.mean():.3f}, of pred: {pp.mean():.3f} ")


eu   = l2_norm_error(up,u)
ev   = l2_norm_error(vp,v)
ep   = l2_norm_error(pp,p)

print(f"Error of u:\t{eu * 100}")
print(f"Error of v:\t{ev * 100}")
print(f"Error of p:\t{ep * 100}")
# Answer: What we compute is pressure gradient, so we need some global information to bring it back to pressure

#-------------------------------------------------------
##################
## Question b: Why We can tell the results are GOOD? 
##################

# Idea here: Compute the error for vorticity

print("#"*30)
print("Question 2")


eu_noise    = l2_norm_error(u_ns,u)
ev_noise    = l2_norm_error(v_ns,v)
ep_noise    = l2_norm_error(p_ns,p)
print(f"Adding noise {args.c:.2f}%, gives error on \neu:\t{eu_noise * 100:.2f}\nev:\t{ev_noise * 100:.2f}\nep:\t{ep_noise * 100:.2f}")
print(f"For PINNS: ")
print(f"Error of u:\t{eu * 100:.2f}")
print(f"Error of v:\t{ev * 100:.2f}")
print(f"Error of p:\t{ep * 100:.2f}")

e_omega_p       = l2_norm_error(omega_z_p[:,:,t18],omega_z[:,:,t18])
e_omega_ns      = l2_norm_error(omega_z_ns[:,:,t18],omega_z[:,:,t18])
print(f"At Time = {t18}")
print(f"For PINNS, omega error:{e_omega_p*100:.2f}")
print(f"For NOISE, omega error:{e_omega_ns*100:.2f}")


e_omega_p       = l2_norm_error(omega_z_p[:,:,t53],omega_z[:,:,t53])
e_omega_ns      = l2_norm_error(omega_z_ns[:,:,t53],omega_z[:,:,t53])
print(f"At Time = {t53}")
print(f"For PINNS, omega error:{e_omega_p*100:.2f}")
print(f"For NOISE, omega error:{e_omega_ns*100:.2f}")


e_omega_p       = l2_norm_error(omega_z_p,omega_z)
e_omega_ns      = l2_norm_error(omega_z_ns,omega_z)
print(f"For time averged")
print(f"For PINNS, omega error:{e_omega_p*100:.2f}")
print(f"For NOISE, omega error:{e_omega_ns*100:.2f}")


r_omega_p, _       = pearsonr(omega_z.flatten(),omega_z_p.flatten())
r_omega_ns,_      = pearsonr(omega_z.flatten(),omega_z_ns.flatten())
print(f"For PINNS, omega Pearson Correlation:{r_omega_p:.2f}")
print(f"For NOISE, omega Pearson Correlation:{r_omega_ns:.2f}")


#-------------------------------------------------------
##################
## Question d: What about peak of the vorticity? 
##################

print("#"*30)
print("Question d")

# Find the peak of vorticity and compare the value 
Nt = omega_z.shape[-1]
fig,axs = plt.subplots(1,1,figsize = (8,4))

Pp  =  []
P   =  []
for ti in range(Nt):
    max_id = np.argmax(omega_z[:,:,ti].flatten())

    # print(max_id)
    peak_omega      = p[:,:,ti].flatten()[max_id]
    peak_omega_p    = pp[:,:,ti].flatten()[max_id]
    
    # print(f"At time = {ti}, Ref: Pressure at peak of voritcity: {peak_omega:.2f}")
    
    # print(f"At time = {ti}, Pred: Pressure at peak of voritcity: {peak_omega_p:.2f}")
    Pp.append(peak_omega_p)
    P.append(peak_omega)
print("Pressure at the peak of voritcity")
print(f'At T = {t18}, Reference: {P[t18]}, Reference: {Pp[t18]}, ')
print(f'At T = {t53}, Reference: {P[t53]}, Reference: {Pp[t53]}, ')

axs.plot(t.flatten(), P,"-ok",label='Ground truth')
axs.plot(t.flatten(), Pp,"-sr",label ="Prediction")
axs.set_xlabel("Time")
axs.set_ylabel("P")
axs.set_title(f'Pressure at peak voriticity (Noise Level = {args.c}%)')
axs.legend()
plt.savefig(fig_path + f'{args.c}_Peak_pressure.jpg',bbox_inches='tight')

#-------------------------------------------------------
##################
## Question : Why not show the relative error plot for Pressure? 
##################

# Of course! 

cmap = 'cmo.tarn'
cmapr = 'cmo.tarn_r'
if args.c ==0:
    
    p53 = p[:,:, t53]
    vmax,vmin = p53.max(), p53.min()
    pp53 = pp[:,:,t53]

    fig, axs = plt.subplots(2,2,sharex=True, sharey=True, figsize=(6,4))

    c0 = axs[0,1].contourf(xx,yy,p53 ,vmin = vmin, vmax = vmax,cmap=cmap)
    c1 = axs[0,0].contourf(xx,yy,pp53,vmin = vmin, vmax = vmax,cmap=cmap)
    axs[0,0].set_aspect('equal')
    axs[0,1].set_aspect('equal')
    
    axs[0,0].set_title("PINNs")
    axs[0,1].set_title("Reference")
    
    plt.colorbar(c0,ax=axs[0,0])
    plt.colorbar(c1,ax=axs[0,1])
    
    pabs = np.abs(p53-pp53)
    prel = np.abs((p53-pp53)/p53)

    c2 = axs[1,0].contourf(xx,yy,pabs,cmap=cmapr,levels=50)
    c3 = axs[1,1].contourf(xx,yy,prel,cmap=cmapr,levels=50)

    plt.colorbar(c2,ax=axs[1,0])
    plt.colorbar(c3,ax=axs[1,1])

    axs[1,0].set_aspect('equal')
    axs[1,1].set_aspect('equal')

    axs[1,0].set_title(r"$\epsilon = |p - \hat{p}|$")
    axs[1,1].set_title(r"$\epsilon = |\frac{p - \hat{p}}{p}|$")
    

    axs[0,0].set_ylabel("y")
    axs[1,0].set_ylabel("y")
    axs[1,1].set_xlabel("x")
    axs[1,0].set_xlabel("x")
    plt.savefig(fig_path + "clean_pressure.pdf",bbox_inches='tight',dpi=500)
    plt.savefig(fig_path + "clean_pressure.jpg",bbox_inches='tight',dpi=500)
