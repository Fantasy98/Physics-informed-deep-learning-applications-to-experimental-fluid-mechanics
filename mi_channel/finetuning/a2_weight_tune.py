import numpy as np
from pyDOE import lhs
from tensorflow.keras import models, layers, optimizers, activations
from PINN_channel import PINN
from time import time
import argparse
import os 
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

argparser   = argparse.ArgumentParser()
argparser.add_argument("--t",default=5,type=int, help="The number of points sampled from Time")
argparser.add_argument("--s",default=8,type=int, help="The number of points sampled from Space")
argparser.add_argument("--c",default=0,type=float, help="Level of gaussian noise")
args        = argparser.parse_args()

data_path  = 'tune_data/'
model_path = 'tune_model/'
from pathlib import Path

Path(data_path).mkdir(exist_ok= True)
Path(model_path).mkdir(exist_ok= True)




def get_data(n_time, n_space, c, n_cp=2000):
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
    
    # collocation points for unsupervised learning 
    lb = np.array([x.min(), y.min(), z.min(), t.min()])
    ub = np.array([x.max(), y.max(), z.max(), t.max()])
    cp = lb + (ub-lb) * lhs(4, n_cp)
    
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
    
    x_sp = X.flatten()
    y_sp = Y.flatten()
    z_sp = Z.flatten()
    t_sp = T.flatten()
    
    u_sp = u_lr.flatten()
    v_sp = v_lr.flatten()
    w_sp = w_lr.flatten()
    
    sp = np.array([x_sp, y_sp, z_sp, t_sp, u_sp, v_sp, w_sp]).T
    return sp, cp, grid

n_time  = args.t #resolution in time
n_space = args.s #resolution in space
c       = args.c # std of gaussian noise

sp, cp, grid = get_data(n_time, n_space, c)
#%% model training 
# sw  = 1 
# uw  = 1
# NL  = [4,  6,  10]
# NN  = [20, 60, 100]
nv = 4 #(u, v, w, p)
    

nl = 10 
nn = 100 

SW = [1, 5, 10 ]
UW = [1, 5, 10 ]

for sw in SW: 
    for uw in UW:

        case_name = f"channel_nl{nl}_nn{nn}_sw{sw}_uw{uw}_t{args.t}_s{args.s}_Gn{c}"
        
        if os.path.exists(data_path + "res_" + case_name + '.npz'):
            print(f"EXISTS: {case_name}")
            continue
        else:
            print(f"\nSTART: {case_name}\n")
            act = activations.tanh
            inp = layers.Input(shape = (4,))
            hl = inp
            for i in range(nl):
                hl = layers.Dense(nn, activation = act)(hl)
            out = layers.Dense(nv)(hl)

            model = models.Model(inp, out)
            print(model.summary())

            lr = 1e-3
            opt = optimizers.Adam(lr)
            n_training_with_adam = 1000

            st_time = time()

            pinn = PINN(model, opt, n_training_with_adam)
            hist = pinn.fit(sp, cp)

            en_time = time()
            comp_time = en_time - st_time
            #%% prediction
            x, y, z, t = grid
            yy, zz, xx = np.meshgrid(y, z, x)
            nz, ny, nx = xx.shape

            Y, Z, X, T = np.meshgrid(y, z, x, t)
            cp = np.array([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()]).T
            up = pinn.predict(cp).T
            up = up.reshape((nv, nz, ny, nx, -1), order = 'C')

            #%% save model and predictions
            np.savez_compressed(data_path + "res_"+ case_name + '.npz', up = up, hist = hist, comp_time = comp_time)
            model.save( model_path + case_name + '.h5')

            quit()