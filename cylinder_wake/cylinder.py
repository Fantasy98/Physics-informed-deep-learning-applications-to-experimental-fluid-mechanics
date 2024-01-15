import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import loadmat
from pyDOE import lhs


import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, activations
from ScipyOP import optimizer as SciOP

from PINN_cylinder import PINNs

from time import time

data = loadmat('./cylinder_nektar_wake.mat')
#%%
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

ny, nx = xx.shape
#%%
np.random.seed(24)
ncp = 2000
lb = np.array([x.min(), y.min(), t.min()])
ub = np.array([x.max(), y.max(), t.max()])

cp = lb + (ub-lb) * lhs(3, ncp)

#%%
ns = len(xx.flatten())

ic = np.array([xx.flatten(), yy.flatten(), np.zeros((ns,)) + t[0],
                u[:, :, 0].flatten(), v[:, :, 0].flatten()]).T

pr = 0.8
ind_ic = np.random.choice([False, True], len(ic), p=[1 - pr, pr])
ic = ic[ind_ic]
#%%
ind_bc = np.zeros(xx.shape, dtype = bool)
ind_bc[[0, -1], :] = True; ind_bc[:, [0, -1]] = True

X, Y, T = np.meshgrid(x, y, t)

x_bc = X[ind_bc].flatten()
y_bc = Y[ind_bc].flatten()
t_bc = T[ind_bc].flatten()

u_bc = u[ind_bc].flatten()
v_bc = v[ind_bc].flatten()
p_bc = p[ind_bc].flatten()

bc = np.array([x_bc, y_bc, t_bc, u_bc, v_bc]).T

pr = 0.2
indx_bc = np.random.choice([False, True], len(bc), p=[1 - pr, pr])
bc = bc[indx_bc]
#%%
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
#%%
cp = np.array([X.flatten(), Y.flatten(), T.flatten()]).T
up = pinn.predict(cp).T
up = up.reshape((3, ny, nx, -1), order = 'C')
#%%
fig, ax = plt.subplots(2, 2)
n = -1
ax[0, 0].contourf(xx, yy, up[0, :, :, n])
ax[0, 1].contourf(xx, yy, u[:, :, n])

ax[1, 0].contourf(xx, yy, up[2, :, :, n])
ax[1, 1].contourf(xx, yy, p[:, :, n])
plt.savefig('prediction.jpg',bbox_inches='tight')


#%%
np.savez_compressed('cylinder_PINN', up = up, comp_time = comp_time)
model.save('cylinder_PINN.h5')
