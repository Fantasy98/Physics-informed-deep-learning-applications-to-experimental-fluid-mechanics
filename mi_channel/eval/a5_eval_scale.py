"""
Investigate which scale we can reconstruct and which we can not? 
"""
import numpy as np
from matplotlib import pyplot as plt 
import cmocean
import cmocean.cm as cmo
from scipy.signal import correlate2d
from scipy.stats import pearsonr
import argparse
argparser   = argparse.ArgumentParser()
argparser.add_argument("--t",default=3,type=int, help="The number of points sampled from Time")
argparser.add_argument("--s",default=16,type=int, help="The number of points sampled from Space")
argparser.add_argument("--c",default=0,type=float, help="Level of gaussian noise")
args        = argparser.parse_args()


def l2_norm_error(p,g): 
    """
    Compute the l2 norm error 
    """
    import numpy.linalg as LA
    error = (LA.norm(p-g,axis=(0,1)))/LA.norm(g,axis =(0,1))
    return error.mean() * 100
"""
Visualisation of the mini-channel flow output
"""
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 1)
plt.rc('font', size = 16)
plt.rc('legend', fontsize = 16)              
plt.rc('xtick', labelsize = 16)             
plt.rc('ytick', labelsize = 16)

ref = np.load('../data/min_channel_sr.npz')

x = ref['x'] 
y = ref['y']
z = ref['z']
u = ref['u'] # dimensions  = (nz, ny, nx, nt)
v = ref['v']
w = ref['w']
t = ref['t']

nz,ny,nx,nt = u.shape
u = np.stack([u, v, w])

yy, zz, xx = np.meshgrid(y, z, x)
dt = 0.2
nv = 4
nz, ny, nx = xx.shape

if args.c == 0.0:
    pred1 = np.load(f'../results/res_pinn_t{args.t}_s{args.s}.npz')
else:
    pred1 = np.load(f'../results/res_pinn_t{args.t}_s{args.s}_c{args.c:.1f}.npz')

u_pred1 = pred1['up'][:3]

pred2 = np.load('../results/res_pinn_t3_s8.npz')
u_pred2 = pred2['up'][:3]


uy = np.gradient(u[0], y, axis = 1, edge_order=2)
uz = np.gradient(u[0], z, axis = 0, edge_order=1)

vx = np.gradient(u[1], x, axis = 2, edge_order=2)
vz = np.gradient(u[1], z, axis = 0, edge_order=1)

wx = np.gradient(u[2], x, axis = 2, edge_order=2)
wy = np.gradient(u[2], y, axis = 1, edge_order=2)

omega_z = vx - uy

uy_pred1 = np.gradient(u_pred1[0], y, axis = 1, edge_order=2)
uz_pred1 = np.gradient(u_pred1[0], z, axis = 0, edge_order=1)

vx_pred1 = np.gradient(u_pred1[1], x, axis = 2, edge_order=2)
vz_pred1 = np.gradient(u_pred1[1], z, axis = 0, edge_order=1)

wx_pred1 = np.gradient(u_pred1[2], x, axis = 2, edge_order=2)
wy_pred1 = np.gradient(u_pred1[2], y, axis = 1, edge_order=2)

omega_z_pred1 = vx_pred1 - uy_pred1

uy_pred2 = np.gradient(u_pred2[0], y, axis = 1, edge_order=2)
uz_pred2 = np.gradient(u_pred2[0], z, axis = 0, edge_order=1)

vx_pred2 = np.gradient(u_pred2[1], x, axis = 2, edge_order=2)
vz_pred2 = np.gradient(u_pred2[1], z, axis = 0, edge_order=1)

wx_pred2 = np.gradient(u_pred2[2], x, axis = 2, edge_order=2)
wy_pred2 = np.gradient(u_pred2[2], y, axis = 1, edge_order=2)

omega_z_pred2 = vx_pred2 - uy_pred2

x = xx[0]
y = yy[0]


print(u.shape, np.expand_dims(omega_z,0).shape)

u       = np.concatenate([u,np.expand_dims(omega_z,0)])
u_pred1 = np.concatenate([u_pred1, np.expand_dims(omega_z_pred1,0)])
u_pred2 = np.concatenate([u_pred2, np.expand_dims(omega_z_pred2,0)])

##############
# Reconstrcution scale 
##############
"""
Reconstruction scale: We compute pixel-wise error and we compare the distance as we know the distance between pixel scaled by Komlogorov scale 
"""


# t = 5 #  t= 5 == 1s; t = 15 == 5s
t = 15 #  t= 5 == 1s; t = 15 == 5s
threshold1= 0
threshold2= [3, 50, 50 ,50]

fig, axs = plt.subplots(2,4,sharex=True,sharey=True,figsize=(9,5))
for i in range(4):
    Var                   = u[i]
    Var_pred1             = u_pred1[i]
    Var_pred2             = u_pred2[i]
    

    var_min,var_max = Var.min(), Var.max()
    
    
    # Var       =  1 -  2*(Var - var_min)/(var_max - var_min)
    # Var_pred1 =  1 -  2*(Var_pred1 - var_min)/(var_max - var_min)
    # Var_pred2 =  1 -  2*(Var_pred2 - var_min)/(var_max - var_min)
    # var_min,var_max = Var.min(), Var.max()

    abs_err1 = np.abs((Var_pred1 - Var)/Var) * 100
    mask1    =(abs_err1 <= threshold2[i]) & (abs_err1 > threshold1)
    
    zero1     = np.ones_like(Var_pred1)
    zero1[mask1] = 0
    
    Var_pred1[~mask1] = np.nan
    vmin1, vmax1 = Var_pred1[mask1].min(),Var_pred1[mask1].max()
    

    abs_err2 = np.abs((Var_pred2 - Var)/Var) *100
    mask2    =(abs_err2 <= threshold2[i]) & (abs_err2 > threshold1)
    
    zero2     = np.ones_like(Var_pred2)
    zero2[mask2] = 0
    Var_pred2[~mask2] = np.nan
    vmin2, vmax2 = Var_pred2[mask2].min(),Var_pred2[mask2].max()
    
    c1 = axs[0,i].contourf(x,y,Var_pred1[0,:,:,t],levels= 30, vmin = var_min, vmax = var_max, cmap ='cmo.tarn')
    axs[0,i].contour(x,y,zero1[0,:,:,t],levels=[0,1],colors='r',linewidths=1)
    
    c2 = axs[1,i].contourf(x,y,Var_pred2[0,:,:,t],levels= 30, vmin = var_min, vmax = var_max, cmap ='cmo.tarn')
    axs[1,i].contour(x,y,zero2[0,:,:,t],levels=[0,1],colors='r',linewidths=1)
    
    
    clb1 = plt.colorbar(c1,ax = axs[:,i],format='%.2f',orientation='horizontal')
    
    clb1.ax.locator_params(nbins=3)
    axs[0,i].set_aspect('equal')
    axs[1,i].set_aspect('equal')
    axs[1,i].set_xlabel('x')
axs[0, 0].set_ylabel('PINN \n t3--s16 \n $y$')
axs[1, 0].set_ylabel('PINN \n t3--s8 \n $y$')

tit = [ r'$|\frac{u - \hat{u}}{u}| \leq $'                      + f"{threshold2[0]}"+r"\%", 
        r'$|\frac{v - \hat{v}}{v}| \leq $'                      + f"{threshold2[1]}"+r"\%", 
        r'$|\frac{w - \hat{w}}{w}| \leq $'                      + f"{threshold2[2]}"+r"\%",
        r'$|\frac{\omega_z - \hat{\omega_z}}{\omega_z}| \leq $' + f"{threshold2[-1]}"+r"\%"
        ]
i = 0
for axx in axs[0]:
    axx.set_title(tit[i],
                pad = 8.8)
    i += 1

plt.savefig(f"channel_t{t}_reconstruct_scale.png",bbox_inches='tight',dpi=500)