"""
Try to implement Gappy POD on the data
@yuningw
"""

import numpy as np
from pyDOE import lhs
from time import time

import argparse

argparser   = argparse.ArgumentParser()
argparser.add_argument("--t",default=5,type=int, help="The number of points sampled from Time")
argparser.add_argument("--s",default=16,type=int, help="The number of points sampled from Space")
argparser.add_argument("--c",default=0,type=float, help="Level of gaussian noise")
args        = argparser.parse_args()


n_time  = args.t #resolution in time
n_space = args.s #resolution in space
c       = args.c # std of gaussian noise

np.random.seed(24)
# data
data = np.load('../data/min_channel_sr.npz')
x = data['x'] 
y = data['y']
z = data['z']
u = data['u'] # dimensions  = (nz, ny, nx, nt)
v = data['v']
w = data['w']
t = data['t']
nz, ny, nx, nt = u.shape
grid = (x, y, z, t)
    # low resolution and noisy data for supervised learning
sy = int(ny / n_space)
sx = int(nx / n_space)
st = int(nt / (n_time - 1))
    
u_lr = u[:, ::sy, ::sx, ::st]
v_lr = v[:, ::sy, ::sx, ::st]
w_lr = w[:, ::sy, ::sx, ::st]
    
x_lr = x[::sx]
y_lr = y[::sy]
z_lr = z.copy()
t_lr = t[::st]
        
u_lr = u_lr + np.random.normal(0.0, c, np.shape(u_lr)) * u_lr / 100
v_lr = v_lr + np.random.normal(0.0, c, np.shape(v_lr)) * v_lr / 100
w_lr = w_lr + np.random.normal(0.0, c, np.shape(w_lr)) * w_lr / 100
    
Y, Z, X, T = np.meshgrid(y_lr, z_lr, x_lr, t_lr)

print(f"INFO: Get dataset, Origin data shape: {u.shape}")
print(f"INFO: Down-Sampled data has shape: {u_lr.shape}")
reduce_percentage = 1 - (len(u_lr.flatten())/len(u.flatten()))
print(f"INFO: The reduced percentage: {reduce_percentage * 100 }")



##################################
# Gappy POD 
################################
# Apply a naive Gappy POD method on u 

#-------------------------------------------------
# generate a mask to identify the missing data
mask                    = np.zeros_like(u,dtype=np.int32) 
mask[:,::sy,::sx,::st]  = 1

V                       = u.reshape(nt,-1)
M                       = mask.reshape(V.shape)
tV                      = M * V

print(tV.shape)
V[np.where(M==0)]                = 0                  
print(tV == V)
# V_guess                 = np.ones_like(u) * u.mean()


#-------------------------------------------------
# Flatten the 

