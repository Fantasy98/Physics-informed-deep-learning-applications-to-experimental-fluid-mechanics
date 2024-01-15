#%%
import numpy as np 
from pyDOE import lhs
from Solver import Solver
from time import time
"""
Training PINNs 

2D RANS for inflow data profile 

Without scaling 

Consider X direction derivatives    
"""
#%%
import argparse
parser = argparse.ArgumentParser(description='PINN training')
parser.add_argument('--cp', default= 50, type=int, help='Number of grid point')
parser.add_argument('--nl', default= 4 , type=int, help='Number of layer')
parser.add_argument('--nn', default= 60 , type=int, help='Number of neuron')
parser.add_argument('--epoch', default=1000, type=int, help='Training Epoch')
parser.add_argument('--noise', default=1, type=int, help='Noise level in %')
parser.add_argument('--sw', default=1, type=int, help='Weight of supervise learning loss')
parser.add_argument('--uw', default=1, type=int, help='Weight of unsupervise learning loss')
args = parser.parse_args()
# %%
file_name ='01_data/inflow.dat'
ds = np.genfromtxt(file_name,skip_header=1)

y = ds[:,0]
x = np.ones(shape=y.shape) * 3

u = ds[:,1] ;uv = -ds[0:,2]; uu = ds[0:,3]
vv = ds[0:,4] ;ww = ds[0:,5]; 
print(f"The min U = {u.min()}, the max U = {u.max()}")
print(f"The min vv = {vv.min()}, the max vv = {vv.max()}")
print(f"The min uv = {uv.min()}, the max uv = {uv.max()}")
gt = np.array([u,uu,vv,uv]).T
#%%
np.random.seed(24)
noise_level = 0.01 * args.noise
noise = np.random.normal(size = y.shape) * noise_level
print(f"The noise level is {args.noise}")
print(f"The noise is at range {noise.min()} ~ {noise.max()}")
x = x
y = y
u = u + noise
uv = uv + noise
uu = uu + noise
vv = vv + noise
gn = np.array([u,uu,vv,uv]).T
#%%
name = [
        'u','v','w',
        'uu','vv','uv']
for i,n in enumerate(name):
    print(f"The MIN and MAX value for {n} is {ds[:,i].min()}, {ds[:,i].max()}\n")
#%%
lb = np.array([x.min(),y.min()])
ub = np.array([x.max(),y.max()])
ncp = args.cp
cp = lb + (ub-lb) * lhs(2, ncp)
print(cp.shape)

#%%
ic = np.array([
                x.flatten(),
                y.flatten(),
                u.flatten(),  
                uu.flatten(),
                vv.flatten(),uv.flatten(),
                ]).T
print(ic.shape)
print(f"Collection point = {ic.flatten().shape} ")
#%%
nl = args.nl
nn = args.nn
epoch = args.epoch
s_w = args.sw
u_w = args.uw
solv = Solver(nn=nn,nl=nl,epoch=epoch,
              s_w=s_w,u_w=u_w)

case_name = f"DN_noise{args.noise}_cp{ncp}_nl{nl}_nn{nn}_epoch{epoch}_S{s_w}_U{u_w}"
print(f"INFO: Solver has been setup, case name is {case_name}")

hist, comp_time = solv.fit(ic=ic,cp=cp)
print(f"INFO: Training end, time cost: {np.round(comp_time,2)}s")
y = ds[:,0]
x = np.ones(shape=y.shape) * 3
cp = np.array([ x.flatten(),y.flatten()]).T
up,error = solv.pred(cp=cp,gt=gt)
print(f"The prediction error are {np.round(error,3)}%")
hist = np.array(hist)
#%%
hist = np.array(hist)
np.savez_compressed("02_pred/"+ case_name + ".npz", up = up,
                                                    gn = gn, # Noise reference
                                                    hist = hist,
                                                    comp_time = comp_time)
solv.model.save("03_model/"+case_name +".h5")
# %%
