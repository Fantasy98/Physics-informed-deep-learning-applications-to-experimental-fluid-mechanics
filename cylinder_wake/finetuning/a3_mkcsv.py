"""
Create tables for the finetuning results
"""
import numpy as np 
import pandas as pd 
from scipy.io import loadmat
data_path  = 'tune_data/'
model_path = 'tune_model/'
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m",default="arch",type=str)
args = parser.parse_args()

Path(data_path).mkdir(exist_ok= True)
Path(model_path).mkdir(exist_ok= True)

def error(u, up):
    return np.linalg.norm((u - up), axis = (1, 2))/np.linalg.norm(u, axis = (1, 2)) * 100

data = loadmat('../data/cylinder_nektar_wake.mat')
u = data['U_star'][:, 0]
v = data['U_star'][:, 1]
p = data['p_star']

u = u.reshape((-1, 100, 200))
v = v.reshape((-1, 100, 200))
p = p.reshape((-1, 100, 200))
x = data['X_star'][:, 0]
y = data['X_star'][:, 1]
t = data['t']
x = x.reshape((-1, 100))
y = y.reshape((-1, 100))
nt = 71
u = u[:, :, :nt]
v = v[:, :, :nt]
p = p[:, :, :nt]
u = np.stack((u, v, p), axis = 0)
c   = 5.0



error_dict = {}

items = ['nn','nl','sw','uw']
for i in items:
    error_dict[i] = []


Names = ["E_U", "E_V", "E_P",'time',"Avg"]
for n in Names:
    error_dict[n] = []


if args.m =='arch':
    sw  = 1
    uw  = 1
    NL  = [4,  6,  10]
    NN  = [20, 60, 100]
    for nl in NL:
        for nn in NN:
            case_name = f"cyliner_nl{nl}_nn{nn}_sw{sw}_uw{uw}_Gn{c}"
            print(f"INFO: Testing\t{case_name}")
            dp   = np.load(data_path + "res_" + case_name + ".npz")
            u_pinn = dp['up']
            ctime  = dp['comp_time']
            u_pinn[2] = u_pinn[2] - u_pinn[2].mean() + p.mean()
            e_pinn = error(u, u_pinn).mean(1)
            print(e_pinn)
            error_dict['nn'].append(nn)
            error_dict['nl'].append(nl)
            error_dict['sw'].append(sw)
            error_dict['uw'].append(uw)

            for i in range(len(Names)-2):
                error_dict[Names[i]].append(np.round(e_pinn[i],2))
            error_dict['Avg'].append(np.round(e_pinn.mean(),2))
            error_dict['time'].append(np.round(ctime,2))
            
    for n in Names:
        error_dict[n] = np.array(error_dict[n])
    for i in items:
        error_dict[i] = np.array(error_dict[i])
    
    df = pd.DataFrame(error_dict)
    df.to_csv(f"cylinder_tune_arch_sw{sw}_uw{uw}.csv")
else:
    nl  = 4
    nn  = 20 
    SW  = [1,  5,  10]
    UW  = [1,  5,  10]

    for sw in SW:
        for uw in UW :
            case_name = f"cyliner_nl{nl}_nn{nn}_sw{sw}_uw{uw}_Gn{c}"
            print(f"INFO: Testing\t{case_name}")
            dp   = np.load(data_path + "res_" + case_name + ".npz")
            u_pinn = dp['up']
            ctime  = dp['comp_time']
            u_pinn[2] = u_pinn[2] - u_pinn[2].mean() + p.mean()
            e_pinn = error(u, u_pinn).mean(1)

            error_dict['nn'].append(nn)
            error_dict['nl'].append(nl)
            error_dict['sw'].append(sw)
            error_dict['uw'].append(uw)

            for i in range(len(Names)-2):
                error_dict[Names[i]].append(np.round(e_pinn[i],2))
            error_dict['Avg'].append(np.round(e_pinn.mean(),2))
            error_dict['time'].append(np.round(ctime,2))
            
    for n in Names:
        error_dict[n] = np.array(error_dict[n])
    for i in items:
        error_dict[i] = np.array(error_dict[i])
    
    
    df = pd.DataFrame(error_dict)
    df.to_csv(f"cylinder_tune_para_nl{nl}_nn{nn}.csv")