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
fig_path   = 'Figs/'
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m",default="arch",type=str)
args = parser.parse_args()

Path(data_path).mkdir(exist_ok= True)
Path(model_path).mkdir(exist_ok= True)
Path(fig_path).mkdir(exist_ok= True)


error_dict = {}

items = ['nn','nl','sw','uw']
for i in items:
    error_dict[i] = []


Names = [ "l_tot","l_s", "l_e"]
for n in Names:
    error_dict[n] = []

t = 5 
s = 8 
c = 0

if args.m =='arch':
    sw  = 1
    uw  = 1
    maxlen1 =0
    maxlen2 =0
    
    colors = [cc.black, cc.red, cc.yellow]
    linestyle = ['--','-','-']
    fig1, ax = plt.subplots(1, 1, figsize=(6,4))
    nn = 100 
    NL = [4,6,10]

    for il, nl in enumerate(NL):
        case_name = f"channel_nl{nl}_nn{nn}_sw{sw}_uw{uw}_t{t}_s{s}_Gn{c}"
        print(f"INFO: Testing\t{case_name}")
        dp   = np.load(data_path + "res_" + case_name + ".npz")
        hist = dp['hist']
        label_name = f"l = {nl}, n = {nn}, " + r"$\alpha$" + f" = {sw}, " + r"$\beta$" + f" = {uw}"  
        if len(hist[:,2]) > maxlen1:
            maxlen1 = len(hist)
        ax.semilogy(np.arange(len(hist)),
                            hist[:,2],
                            linestyle[il], 
                            c=colors[il],
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
    fig1.tight_layout()
    fig1.savefig('Figs/channel_loss_trend_nn.pdf',bbox_inches='tight',dpi=1000)
    


    nl = 10  
    NN = [20,60,100]   
    colors = [ cc.blue, cc.cyan, cc.black]
    linestyle = ['-','-','--']
    fig2, ax2 = plt.subplots(1, 1, figsize=(6,4))
    for jl, nn in enumerate(NN):

        label_name = f"l = {nl}, n = {nn}, " + r"$\alpha$" + f" = {sw}, " + r"$\beta$" + f" = {uw}"  
        case_name = f"channel_nl{nl}_nn{nn}_sw{sw}_uw{uw}_t{t}_s{s}_Gn{c}"
        print(f"INFO: Testing\t{case_name}")
        dp   = np.load(data_path + "res_" + case_name + ".npz")
        hist = dp['hist']
        
        if len(hist[:,2]) > maxlen2:
            maxlen2 = len(hist)
        ax2.semilogy(np.arange(len(hist)),
                        hist[:,2],
                        linestyle[jl], 
                        c=colors[jl],
                        label = label_name,
                        lw = 2)

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel(r"L")
    ax2.set_xlim([0-1,maxlen2*1.05])
    ax2.legend(loc = 'upper right')
    rect_width = (1000/maxlen2) * (ax2.get_xlim()[1] - ax2.get_xlim()[0])
    rect_height = ax2.get_ylim()[1] - ax2.get_ylim()[0]
    rect_left = ax2.get_xlim()[0]
    ax2.add_patch(plt.Rectangle((0, ax2.get_ylim()[0]), rect_width, rect_height, color='#e6f7ff', alpha=1))    
    fig1.tight_layout()
    fig2.savefig('Figs/channel_loss_trend_nl.pdf',bbox_inches='tight',dpi=1000)
    


else:
    nn  = 100
    nl  = 10
    maxlen1 =0
    maxlen2 =0
    
    colors = [cc.black, cc.brown, cc.purple]
    linestyle = ['--','-','-']
    fig1, ax = plt.subplots(1, 1, figsize=(6,4))
    

    sw = 1 
    UW = [1,5,10]
    
    for il, uw in enumerate(UW):
        case_name = f"channel_nl{nl}_nn{nn}_sw{sw}_uw{uw}_t{t}_s{s}_Gn{c}"
        print(f"INFO: Testing\t{case_name}")
        dp   = np.load(data_path + "res_" + case_name + ".npz")
        hist = dp['hist']
        label_name = f"l = {nl}, n = {nn}, " + r"$\alpha$" + f" = {sw}, " + r"$\beta$" + f" = {uw}"  
        if len(hist[:,2]) > maxlen1:
            maxlen1 = len(hist)
        ax.semilogy(np.arange(len(hist)),
                            hist[:,2],
                            linestyle[il], 
                            c=colors[il],
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
    fig1.savefig('Figs/channel_loss_trend_uw.pdf',bbox_inches='tight',dpi=300)
    


    uw = 1
    SW = [1,5,10]   
    

    linestyle = ['--','-','-']
    colors = [cc.black, cc.green, cc.orange]

    # linestyle = ['--','-','-']
    fig2, ax2 = plt.subplots(1, 1, figsize=(6,4))
    for jl, sw in enumerate(SW):

        label_name = f"l = {nl}, n = {nn}, " + r"$\alpha$" + f" = {sw}, " + r"$\beta$" + f" = {uw}"  
        case_name = f"channel_nl{nl}_nn{nn}_sw{sw}_uw{uw}_t{t}_s{s}_Gn{c}"
        print(f"INFO: Testing\t{case_name}")
        dp   = np.load(data_path + "res_" + case_name + ".npz")
        hist = dp['hist']
        
        if len(hist[:,2]) > maxlen2:
            maxlen2 = len(hist)
        ax2.semilogy(np.arange(len(hist)),
                        hist[:,2],
                        linestyle[jl], 
                        c=colors[jl],
                        label = label_name,
                        lw = 2)

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel(r"L")
    ax2.set_xlim([0-1,maxlen2*1.05])
    ax2.legend(loc = 'upper right')
    rect_width = (1000/maxlen2) * (ax2.get_xlim()[1] - ax2.get_xlim()[0])
    rect_height = ax2.get_ylim()[1] - ax2.get_ylim()[0]
    rect_left = ax2.get_xlim()[0]
    ax2.add_patch(plt.Rectangle((0, ax2.get_ylim()[0]), rect_width, rect_height, color='#e6f7ff', alpha=1))    
    fig2.savefig('Figs/channel_loss_trend_sw.pdf',bbox_inches='tight',dpi=300)
    

