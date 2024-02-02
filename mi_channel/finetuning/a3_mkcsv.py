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
    error= np.linalg.norm((u - up), axis = (0, 1))/np.linalg.norm(u, axis = (0, 1)) * 100
    return error.mean()

ref = np.load('../data/min_channel_sr.npz')

x = ref['x'] 
y = ref['y']
z = ref['z']
u = ref['u'] # dimensions  = (nz, ny, nx, nt)
v = ref['v']
w = ref['w']
t = ref['t']

u = np.stack([u, v, w])



error_dict = {}

items = ['nn','nl','sw','uw']
for i in items:
    error_dict[i] = []


Names = ["E_U", "E_V", "E_W",'time','Avg']
for n in Names:
    error_dict[n] = []

c = 0 
t = 5 
s = 8 


if args.m =='arch':
    sw  = 1
    uw  = 1
    NL  = [4,  6,  10]
    NN  = [20, 60, 100]
    for nl in NL:
        for nn in NN:
            case_name = f"channel_nl{nl}_nn{nn}_sw{sw}_uw{uw}_t{t}_s{s}_Gn{c}"

            print(f"INFO: Testing\t{case_name}")
            dp   = np.load(data_path + "res_" + case_name + ".npz")
            u_pinn = dp['up'][:3]
            ctime  = dp['comp_time']
            
            error_dict['nn'].append(nn)
            error_dict['nl'].append(nl)
            error_dict['sw'].append(sw)
            error_dict['uw'].append(uw)

            es = np.empty(shape=(3,))
            for i in range(len(Names)-2):
                e_pinn = error(u[i],u_pinn[i])
                error_dict[Names[i]].append(np.round(e_pinn,2))
                es[i] = e_pinn

            error_dict['Avg'].append(np.round(es.mean(),2))
            
            error_dict['time'].append(np.round(ctime,2))
            
    for n in Names:
        error_dict[n] = np.array(error_dict[n])
    for i in items:
        error_dict[i] = np.array(error_dict[i])
    
    df = pd.DataFrame(error_dict)
    df.to_csv("channel_tune_arch.csv")
else:
    nl  = 10
    nn  = 100 
    SW  = [1,  5,  10]
    UW  = [1,  5,  10]

    for sw in SW:
        for uw in UW :
            case_name = f"channel_nl{nl}_nn{nn}_sw{sw}_uw{uw}_t{t}_s{s}_Gn{c}"
            # print(f"INFO: Testing\t{case_name}")
            print(f"INFO: Testing\t{case_name}")
            dp   = np.load(data_path + "res_" + case_name + ".npz")
            u_pinn = dp['up'][:3]
            ctime  = dp['comp_time']
            
            error_dict['nn'].append(nn)
            error_dict['nl'].append(nl)
            error_dict['sw'].append(sw)
            error_dict['uw'].append(uw)

            es = np.empty(shape=(3,))
            for i in range(len(Names)-2):
                e_pinn = error(u[i],u_pinn[i])
                error_dict[Names[i]].append(np.round(e_pinn,2))
                es[i] = e_pinn

            error_dict['Avg'].append(np.round(es.mean(),2))
            
            error_dict['time'].append(np.round(ctime,2))
            
    for n in Names:
        error_dict[n] = np.array(error_dict[n])
    for i in items:
        error_dict[i] = np.array(error_dict[i])
    
    df = pd.DataFrame(error_dict)
    df.to_csv("channel_tune_para.csv")
