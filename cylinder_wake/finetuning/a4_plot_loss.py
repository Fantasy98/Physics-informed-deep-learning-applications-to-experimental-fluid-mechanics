"""
Visualisation of the loss evolution 
"""
import numpy as np 
import pandas as pd 
from scipy.io import loadmat
import matplotlib.pyplot as plt 


data_path  = 'tune_data/'
model_path = 'tune_model/'
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m",default="arch",type=str)
args = parser.parse_args()

Path(data_path).mkdir(exist_ok= True)
Path(model_path).mkdir(exist_ok= True)


error_dict = {}

items = ['nn','nl','sw','uw']
for i in items:
    error_dict[i] = []


Names = [ "l_tot","l_s", "l_e"]
for n in Names:
    error_dict[n] = []

c = 5.0

if args.m =='arch':
    sw  = 1
    uw  = 10 
    NL  = [4,  6,  10]
    NN  = [20, 60, 100]

    fig, ax = plt.subplots(1,1, figsize=(8,6) )

    il = 0 
    jl = 0
    maxlen =0
    colors = ['b','r','g']
    for il, nl in enumerate(NL):
        for jl, nn in enumerate(NN):
            print(il, jl )
            case_name = f"cyliner_nl{nl}_nn{nn}_sw{sw}_uw{uw}_Gn{c}"
            print(f"INFO: Testing\t{case_name}")
            dp   = np.load(data_path + "res_" + case_name + ".npz")
            
            hist = dp['hist']
            error_dict['nn'].append(nn)
            error_dict['nl'].append(nl)
            error_dict['sw'].append(sw)
            error_dict['uw'].append(uw)

            for i in range(len(Names)):
                error_dict[Names[i]].append(hist[-1,i])
            
            if il == 1 :
                # ax.semilogy(hist[:,0],'-',c=colors[jl], label = case_name)
                # ax.semilogy(hist[:,1],'--',c=colors[jl], label = case_name)
                ax.semilogy(hist[:,2],'-.', c=colors[jl], label = case_name)
                if len(hist) >= maxlen:
                    maxlen = len(hist)

            
    plt.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_xlim([0-1,maxlen*1.05])
    rect_width = (1000/maxlen) * (ax.get_xlim()[1] - ax.get_xlim()[0])
    rect_height = ax.get_ylim()[1] - ax.get_ylim()[0]
    # rect_left = ax.get_xlim()[0] + 0.4 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    rect_left = ax.get_xlim()[0]
    # Create a rectangle with a different color
    ax.add_patch(plt.Rectangle((0, ax.get_ylim()[0]), rect_width, rect_height, color='#e6f7ff', alpha=0.5))
            
    plt.savefig('loss_trend.jpg',bbox_inches='tight',dpi=300)

            
    for n in Names:
        error_dict[n] = np.array(error_dict[n])
    for i in items:
        error_dict[i] = np.array(error_dict[i])
    
    df = pd.DataFrame(error_dict)
    df.to_csv("tune_arch_loss_convergence.csv")


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

            hist = dp['hist']
            error_dict['nn'].append(nn)
            error_dict['nl'].append(nl)
            error_dict['sw'].append(sw)
            error_dict['uw'].append(uw)

            for i in range(len(Names)):
                error_dict[Names[i]].append(hist[-1,i])

            
    for n in Names:
        error_dict[n] = np.array(error_dict[n])
    for i in items:
        error_dict[i] = np.array(error_dict[i])
    
    df = pd.DataFrame(error_dict)
    df.to_csv("tune_arch_loss_convergence.csv")

    
    
    df = pd.DataFrame(error_dict)
    df.to_csv("cylinder_tune_para.csv")