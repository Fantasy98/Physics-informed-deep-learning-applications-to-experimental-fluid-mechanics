import numpy as np 
from pyDOE      import lhs


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