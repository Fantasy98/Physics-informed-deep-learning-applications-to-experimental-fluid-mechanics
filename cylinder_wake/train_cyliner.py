import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import loadmat
from pyDOE import lhs

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, activations
from ScipyOP import optimizer as SciOP

from PINN_cylinder import PINNs
from time import time
from pathlib import Path

import argparse
argparser   = argparse.ArgumentParser()
argparser.add_argument("--c",default=2.5,type=float, help="Level of gaussian noise: 0, 2.5, 5, and 10%")
args        = argparser.parse_args()

np.random.seed(24)


data_path   = 'data/'
res_path    = 'res/'
model_path  = 'model/'
fig_path    = 'figs/'

Path(data_path).mkdir(exist_ok=True)
Path(res_path).mkdir(exist_ok=True)
Path(model_path).mkdir(exist_ok=True)
Path(fig_path).mkdir(exist_ok=True)

data = loadmat(data_path + 'cylinder_nektar_wake.mat')

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


print(f"INFO: The Noise level = {args.c}%")
print(f"INFO: Data Generated!")
X,Y,T, xx,yy,u,v,p,ic, bc, cp = get_Data(noise_level=args.c)
###########################
## Compile the model and training
###########################
act = activations.tanh
inp = layers.Input(shape = (3,))
hl = inp
for i in range(10):
    hl = layers.Dense(100, activation = act)(hl)
out = layers.Dense(3)(hl)

model = models.Model(inp, out)
print(model.summary())

lr = 1e-3
opt = optimizers.Adam(lr)
sopt = SciOP(model)

st_time = time()

pinn = PINNs(model, opt, sopt, 1000)
hist = pinn.fit(ic, bc, cp)

en_time = time()
comp_time = en_time - st_time

print(f"INFO: Training Finish, training time = {comp_time:.2f}s")

###########################
## Inference 
###########################
ny, nx = xx.shape
cp = np.array([X.flatten(), Y.flatten(), T.flatten()]).T
up = pinn.predict(cp).T
up = up.reshape((3, ny, nx, -1), order = 'C')

###########################
## Save File 
###########################
np.savez_compressed( res_path +  f'cylinder_PINN_{args.c}', 
                    up = up, comp_time = comp_time)
model.save( model_path + f'cylinder_PINN_{args.c}.h5')

print(f"INFO: Results saved!")

###########################
## Visualisation 
###########################

fig, ax = plt.subplots(2, 2)
n = -1
ax[0, 0].contourf(xx, yy, up[0, :, :, n])
ax[0, 1].contourf(xx, yy, u[:, :, n])

ax[1, 0].contourf(xx, yy, up[2, :, :, n])
ax[1, 1].contourf(xx, yy, p[:, :, n])
plt.savefig(fig_path + f'prediction.jpg',bbox_inches='tight')

print("INFO: Figures saved!")
