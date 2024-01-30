import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import loadmat

c = 10.0 
filename_pinn = f'../results/res_cylinder_Gn{c}.npz'
#%%
data = loadmat('../data/cylinder_nektar_wake.mat')
u = data['U_star'][:, 0]
v = data['U_star'][:, 1]
p = data['p_star']

u = u.reshape((-1, 100, 200))
v = v.reshape((-1, 100, 200))
p = p.reshape((-1, 100, 200))

nt = 71
u = u[:, :, :nt]
v = v[:, :, :nt]
p = p[:, :, :nt]

u = np.stack((u, v, p), axis = 0)
#%%
def error(u, up):
    return np.linalg.norm((u - up), axis = (1, 2))/np.linalg.norm(u, axis = (1, 2)) * 100

up_pinn = np.load('../results/' + filename_pinn)['up']

up_pinn[2] = up_pinn[2] - up_pinn[2].mean() + p.mean()
e_pinn = error(u, up_pinn).mean(1)

print(e_pinn)