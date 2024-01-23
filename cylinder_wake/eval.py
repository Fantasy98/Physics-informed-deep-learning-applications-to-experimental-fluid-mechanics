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
## Question 2: Why shift the pressure??
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
## Question 1: Why We can tell the results are GOOD? 
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



quit()


mi18,mx18  = omega_z[:,:,t18].min(),omega_z[:,:,t18].max()
mi53,mx53  = omega_z[:,:,t53].min(),omega_z[:,:,t53].max()


