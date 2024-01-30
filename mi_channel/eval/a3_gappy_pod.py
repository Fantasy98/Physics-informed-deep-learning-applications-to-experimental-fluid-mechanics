"""
Try to implement Gappy POD on the data
@yuningw
"""
from copy import deepcopy
import numpy as np
from pyDOE import lhs
from time import time
import matplotlib.pyplot as plt 
import numpy.linalg as LA 
import argparse
import cmocean as cmo
from tqdm import tqdm
def l2_norm_error(p,g): 
    """
    Compute the l2 norm error 
    """
    import numpy.linalg as LA
    error = (LA.norm(p-g,axis=(0,1)))/LA.norm(g,axis =(0,1))
    return error.mean() * 100
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

yy, zz, xx = np.meshgrid(y, z, x)
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

# An empty tensor which full of nan
u_s                     = np.empty_like(u)
u_s[:,:,:,:]            = np.nan
# Give the same initial condition that PINNs has 
u_s[:,::sx,::sy,::st]   = u[:,::sx,::sy,::st]
mask                    = np.isnan(u_s)


nP                      = len(mask.flatten()==True)
# Use the mean value as initial guess, based on the 
gmean                   = u_s[~mask].mean()
u_s[mask]               = gmean
V                       = u_s.reshape(nt,-1)
# Implement a svd to find adequate rank
U, S_, Vh   = LA.svd(u.reshape(nt,-1), full_matrices=False)
rank        = 5
energy      = np.sum(S_[:rank])/np.sum(S_)
print(f'Use rank = {rank}, energy = {energy:.2f}')
ev                      = l2_norm_error(u_s, u)
print(f"After initial guessing, error = {ev:.2f}")
Niter = 100

# Start for loop for fitting data 
for i in tqdm(range(Niter)):
    # We implement SVD on data, create a basis to use 
    U, S_, Vh   = LA.svd(V,full_matrices=False)
    U           = U[:,:rank] 
    S           = S_[:rank]
    Vh          = Vh[:rank]

    coeff       = LA.lstsq(U,V)[0]
    V           = U @ coeff
    V           = V.reshape(nz,ny,nx,nt)
    
    # Use the reconstructed repaired data to replace our initial guess
    
    u_s[mask]   = V[mask]
    u_s[~mask]  = u[~mask]
    # Go for next iteration
    V           = u_s.reshape(nt,-1)

ur          = V.reshape(nz,ny,nx,nt)

ev = l2_norm_error(ur, u)
print(f"using rank of {rank}, error = {ev:.2f}")


np.savez_compressed(f'gappy_u_r{rank}_iter{Niter}.npz',
                    U=U,
                    S=S_,
                    Vh = Vh,
                    r = rank)

# #################
# ## Visualisation
# ################
fig, axs = plt.subplots(3,1)
axs[0].contourf(x,y,u[0,:,:,0])
axs[1].contourf(x,y,ur[0,:,:,0])
axs[2].plot(1 - S_/S_.max())
axs[2].plot(1 - S_[:rank]/S_.max(),'or')
plt.show()


#-------------------------------------------------

"""
Conclusion: 

The gappy POD still requires the spatial modes of the original data 
and use lstsq for correcting the temporal coefficient obtained by POD at each step 

Therefore, it is not comparable with the PINNs method as PINNs does not require prior reference during training. 
We train on the noisy and uncomplete data 
"""
