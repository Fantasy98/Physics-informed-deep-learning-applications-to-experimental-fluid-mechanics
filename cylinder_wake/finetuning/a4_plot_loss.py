"""
Visualisation of the loss evolution 
"""
import numpy as np 
import pandas as pd 
from scipy.io import loadmat
import matplotlib.pyplot as plt 
from utils.plot import colorplate as cc 
from utils import plt_rc_setup


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

    
    il = 0 
    jl = 0
    maxlen1 =0
    maxlen2 =0
    colors = [cc.red, cc.blue, cc.yellow]
    colors2 = [cc.blue, cc.yellow, cc.red]
    fig1, ax = plt.subplots(1, 1, figsize=(6,4))
    fig2, ax2 = plt.subplots(1, 1, figsize=(6,4))

    for il, nl in enumerate(NL):
        for jl, nn in enumerate(NN):
            print(il, jl )

            case_name = f"cyliner_nl{nl}_nn{nn}_sw{sw}_uw{uw}_Gn{c}"
            print(f"INFO: Testing\t{case_name}")
            dp   = np.load(data_path + "res_" + case_name + ".npz")
            hist = dp['hist']

            if il == 0:
                
                label_name = f"l = {nl}, n = {nn}, " + r"$\alpha$" + f" = {sw}, " + r"$\beta$" + f" = {uw}"  

                if len(hist[:,2]) > maxlen1:
                    maxlen1 = len(hist)
                ax.semilogy(np.arange(len(hist)),
                            hist[:,2],
                            '-', 
                            c=colors[jl],
                            label = label_name,
                            lw = 2)

            if jl == 0:
                
                label_name = f"l = {nl}, n = {nn}, " + r"$\alpha$" + f" = {sw}, " + r"$\beta$" + f" = {uw}"  
                if len(hist[:,2]) > maxlen2:
                    maxlen2 = len(hist)
                ax2.semilogy(np.arange(len(hist)),
                            hist[:,2],
                            '-', 
                            c=colors2[il],
                            label = label_name,
                            lw = 2)


    ax.set_xlabel("Epochs")
    ax.set_ylabel(r"L")
    ax.set_xlim([0-1,maxlen1*1.05])
    ax.legend(loc = 'upper right')
    rect_width = (1000/maxlen1) * (ax.get_xlim()[1] - ax.get_xlim()[0])
    rect_height = ax.get_ylim()[1] - ax.get_ylim()[0]
    rect_left = ax.get_xlim()[0]
    ax.add_patch(plt.Rectangle((0, ax.get_ylim()[0]), rect_width, rect_height, color='#e6f7ff', alpha=1))
    
    fig1.savefig('cylinder_loss_trend_nn.jpg',bbox_inches='tight',dpi=300)
    

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel(r"L")
    ax2.set_xlim([0-1,maxlen2*1.05])
    ax2.legend(loc = 'upper right')
    rect_width = (1000/maxlen2) * (ax2.get_xlim()[1] - ax2.get_xlim()[0])
    rect_height = ax2.get_ylim()[1] - ax2.get_ylim()[0]
    rect_left = ax2.get_xlim()[0]
    ax2.add_patch(plt.Rectangle((0, ax2.get_ylim()[0]), rect_width, rect_height, color='#e6f7ff', alpha=1))
    
    fig2.savefig('cylinder_loss_trend_nl.jpg',bbox_inches='tight',dpi=300)
    


else:
    nl  = 4 
    nn  = 20 
    SW  = [1,  5,  10]
    UW  = [1,  5,  10]
    maxlen1 =0
    maxlen2 =0
    colors = [cc.blue, cc.yellow, cc.red]
    colors2 = [cc.red, cc.yellow, cc.blue]
    fig1, ax = plt.subplots(1, 1, figsize=(6,4))
    fig2, ax2 = plt.subplots(1, 1, figsize=(6,4))
    for il, sw in enumerate(SW):
        for jl, uw in enumerate(UW):
            print(il, jl )

            case_name = f"cyliner_nl{nl}_nn{nn}_sw{sw}_uw{uw}_Gn{c}"
            print(f"INFO: Testing\t{case_name}")
            dp   = np.load(data_path + "res_" + case_name + ".npz")
            hist = dp['hist']

            if il == 0:
                
                label_name = f"l = {nl}, n = {nn}, " + r"$\alpha$" + f" = {sw}, " + r"$\beta$" + f" = {uw}"  

                if len(hist[:,2]) > maxlen1:
                    maxlen1 = len(hist)
                ax.semilogy(np.arange(len(hist)),
                            hist[:,2],
                            '-', 
                            c=colors[jl],
                            label = label_name,
                            lw = 2)

            if jl == 2:
                
                label_name = f"l = {nl}, n = {nn}, " + r"$\alpha$" + f" = {sw}, " + r"$\beta$" + f" = {uw}"  
                if len(hist[:,2]) > maxlen2:
                    maxlen2 = len(hist)
                ax2.semilogy(np.arange(len(hist)),
                            hist[:,2],
                            '-', 
                            c=colors2[il],
                            label = label_name,
                            lw = 2)


    ax.set_xlabel("Epochs")
    ax.set_ylabel(r"L")
    ax.set_xlim([0-1,maxlen1*1.05])
    ax.legend(loc = 'upper right')
    rect_width = (1000/maxlen1) * (ax.get_xlim()[1] - ax.get_xlim()[0])
    rect_height = ax.get_ylim()[1] - ax.get_ylim()[0]
    rect_left = ax.get_xlim()[0]
    ax.add_patch(plt.Rectangle((0, ax.get_ylim()[0]), rect_width, rect_height, color='#e6f7ff', alpha=1))
    
    fig1.savefig('cylinder_loss_trend_beta.jpg',bbox_inches='tight',dpi=300)
    

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel(r"L")
    ax2.set_xlim([0-1,maxlen2*1.05])
    ax2.legend(loc = 'upper right')
    rect_width = (1000/maxlen2) * (ax2.get_xlim()[1] - ax2.get_xlim()[0])
    rect_height = ax2.get_ylim()[1] - ax2.get_ylim()[0]
    rect_left = ax2.get_xlim()[0]
    ax2.add_patch(plt.Rectangle((0, ax2.get_ylim()[0]), rect_width, rect_height, color='#e6f7ff', alpha=1))
    
    fig2.savefig('cylinder_loss_trend_alpha.jpg',bbox_inches='tight',dpi=300)
    

