"""
Explore the PINNs reagrding GradTape
@yuningw 
Mar 21
"""
# Env SetUp
import argparse
import numpy    as np
import tensorflow as tf 
from matplotlib import pyplot as plt 
from scipy.io   import loadmat, savemat
from pyDOE      import lhs
from Solver     import Solver
from time       import time
from utils.plot import colorplate as cc 
from utils      import plt_rc_setup

# Path define 
model_path = '03_model/'
pred_path  = '02_pred/'


def name_PINNs(args):
    """
    Name the file by using the input arguments
    """
    ncp         = args.cp
    nl          = args.nl
    nn          = args.nn
    epoch       = args.epoch
    s_w         = args.sw
    u_w         = args.uw
    SampleFreq  = args.f
    
    case_name = f"SR_cp{ncp}_nl{nl}_nn{nn}_epoch{epoch}_{s_w}S_{u_w}U_{SampleFreq}Sample"
    print(f"INFO: Solver has been setup, case name is {case_name}")
    return case_name 

def gen_data(args): 
    """
    Using the argument for generating the data for training 
    Args: 
        args    : The argumentation for this training 

    Returns:
        ic  :   (NumpyArrary)
    """
    file_name ='01_data/inflow.dat'
    ds = np.genfromtxt(file_name,skip_header=1)
    y   = ds[:,0]
    x   = np.ones(shape=y.shape) * 3
    u   = ds[:,1] ;uv = -ds[:,2]; uu = ds[:,3]
    vv  = ds[:,4] ;ww = ds[:,5]; 
    gt  = np.array([u,uu,vv,uv]).T

    # Down Sampling for Super resolution
    SampleFreq  = args.f
    yall        = u.shape[0]
    indx        = np.arange(0,yall,SampleFreq)
    x           = x[indx]; y = y[indx]
    u           = u[indx]; uv = uv[indx]; uu = uu[indx]; vv = vv[indx]
    name = [
            'u','v','w',
            'uu','vv','uv']
    for i,n in enumerate(name):
        print(f"The MIN and MAX value for {n} is {ds[:,i].min()}, {ds[:,i].max()}\n")

    np.random.seed(24)
    lb  = np.array([x.min(),y.min()])
    ub  = np.array([x.max(),y.max()])
    ncp = args.cp
    cp  = lb + (ub-lb) * lhs(2, ncp)

    ic = np.array([
                x.flatten(),y.flatten(),
                u.flatten(),uu.flatten(),vv.flatten(),uv.flatten(),
                ]).T

    print(f"Supervised Learning = {ic.flatten().shape} ")
    y           = ds[:,0]
    x           = np.ones(shape=y.shape) * 3
    cp_test          = np.array([ x.flatten(),y.flatten()]).T
    # Generate the donw-sampled data for test 
    cp_          = ds[:,0]
    y_spine     = np.zeros(shape=cp_.shape[0]*SampleFreq)
    y_spine[::SampleFreq] = cp_
    y_spine[-1] = cp_[-1]
    x_spine     = np.ones(shape=y_spine.shape)*3
    cp_spine    = np.array([x_spine.flatten(), y_spine.flatten()]).T

    return ic, cp, gt, cp_test, cp_spine

def run_PINNs(case_name,solv,ic,cp,cp_test,cp_spine,gt):
    """
    Train and infer the PINN solver 
    Args: 

        solv    : The solver 
        
        ic      : The supervised data for training 
        
        cp      : The random-sampled coordinates 

    Returns: 
        up          : Prediction without DownSample 
        hist        : History of loss 
        residual    : Residual of Gov Eq
        up_sp       : Prediction For DownSample
        e           : The Error Obtained 
        comp_time   : The cost time 
    """

    print(f"Start Training: {case_name}")

    hist, residual, comp_time = solv.fit(ic=ic,cp=cp)
    up,error    = solv.pred(cp=cp_test,gt=gt)
    
    print(f"The prediction error are {np.round(error,3)}%")
    hist        = np.array(hist)
    residual    = np.array(residual)
    up_sp       = solv.pred(cp_spine,gt=gt,return_error=False)

    d          = {}
    d['up']    = up 
    d["up_sp"] =up_sp, # interpolated prediction
    d["hist"]  = hist,
    d["residual"] = residual,
    d["comp_time"] = comp_time

    savemat(pred_path + case_name + ".mat", d)
    print(f"Data Saved!")

    solv.model.save("03_model/"+case_name +".h5")
    return 


def load_pretrained_PINNs(args):
    """
    Load the pretrained object from the h5 file 
    """

    ffmat = '.h5'
    case_name = name_PINNs(args)

    solv = Solver(nn=args.nn,
                nl=args.nl, 
                epoch=args.epoch,
                s_w=args.sw,
                u_w=args.uw
                )
    
    wb_file = model_path + case_name + ffmat
    
    preModel = tf.keras.models.load_model(wb_file)
    solv.model = preModel
    
    print(f"Model Loaded, Summary: {solv.model.summary()}")

    return solv



def main(args):

    case_name = name_PINNs(args)
    ic,cp,gt,cp_test,cp_spine = gen_data(args)

    
    solv = Solver(nn=args.nn,
                nl=args.nl, 
                epoch=args.epoch,
                s_w=args.sw,
                u_w=args.uw
                )
    
    # Run the Model and Save the data through training
    run_PINNs(case_name,solv,ic,cp,cp_test,cp_spine,gt)
    # Load the model and analysis the results
    solv = load_pretrained_PINNs(args)
    d = solv.auto_diff(cp_test)
    savemat(pred_path + "Analy_" + case_name + ".mat", d)
    
    print(f"Analytial Data Saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINN training')
    parser.add_argument('--cp', default= 50, type=int, help='Number of grid point')
    parser.add_argument('--nl', default= 4 , type=int, help='Number of layer')
    parser.add_argument('--nn', default= 40 , type=int, help='Number of neuron')
    parser.add_argument('--epoch', default=1000, type=int, help='Training Epoch')
    parser.add_argument('--sw', default=10, type=int, help='Weight of supervise learning loss')
    parser.add_argument('--uw', default=1, type=int, help='Weight of unsupervise learning loss')
    parser.add_argument('--f', default=3, type=int, help='Sample Frequency of the reference data')

    args = parser.parse_args() 

    main(args)