"""
PINNs for Investigating the derviatives and residual for replying the conservation of mass and momentum 
@yuningw Mar21 
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import models

class PINNs(models.Model):
    def __init__(self, model, optimizer, sopt, epochs, s_w, u_w, **kwargs):
        super(PINNs, self).__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.sopt = sopt
        self.epochs = epochs
        
        self.s_w = s_w
        self.u_w = u_w
        
        self.hist = []
        self.epoch = 0


        #YW: Adding new features for the investigation 
        # List for residual history 
        self.residual = []

    @tf.function
    def net_f(self, cp):
        """
        Apply the Governing Equations by using output of model and Auto Differentiation

        """
        
        x = cp[:, 0]
        y = cp[:, 1]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            X = tf.stack([x,y],axis=-1)
            # X = self.scalex(X)
            pred = self.model(X)
            # pred = self.scale_r(pred)
            u = pred[:, 0]
            uu = pred[:, 1]
            vv = pred[:, 2]
            uv = pred[:, 3]
            # Predicting as if there is no reference
            v = pred[:, 4]
            p = pred[:,5]


            u_x = tape.gradient(u, x)
            v_x = tape.gradient(v, x)
            p_x = tape.gradient(p, x)

            u_y = tape.gradient(u, y)
            v_y = tape.gradient(v, y)
            p_y = tape.gradient(p, y)


        uu_x = tape.gradient(uu,y)    
        uv_x = tape.gradient(uv,x)    
    
        uv_y = tape.gradient(uv,y)    
        vv_y = tape.gradient(vv,y)    
    
        # continuity 
        f0 = u_x + v_y
        # Momentum X
        f1 = u * u_x +  v * u_y + p_x + uu_x  + uv_y 
        # Momentum Y
        f2 = u * v_x +  v * v_y + p_y + uv_x  + vv_y 
        
        f = tf.stack([f0, f1, f2], axis = -1)
        return f
    
    
#---------------------------------------------------------
    @tf.function
    def train_step(self, ic, cp):
        
        with tf.GradientTape() as tape:
            u_p_ic = self.model(ic[:, :2])
            
            f = self.net_f(cp)
            
            loss_ic = tf.reduce_mean(tf.square(ic[:, 2:] - u_p_ic[:, :-2]))
            
            loss_f = tf.reduce_mean(tf.square(f))
            
            loss_u =loss_ic
            loss = self.s_w*loss_u + self.u_w * loss_f
            
        
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        
        l1 = tf.reduce_mean(loss)
        l2 = tf.reduce_mean(loss_u)
        l3 = tf.reduce_mean(loss_f)
        
        tf.print('loss:', l1, 'loss_u:', l2, 'loss_f:', l3)

        ll = tf.stack([l1,l2,l3],axis=-1)

        # YW: Note I add the residual history right after the outputs 
        return loss, grads, ll, f
    
#---------------------------------------------------------
    def fit(self, ic,cp):
        ic = tf.convert_to_tensor(ic, tf.float32)
        cp = tf.convert_to_tensor(cp, tf.float32)
    
        def func(params_1d):
            self.sopt.assign_params(params_1d)
            tf.print('epoch:', self.epoch)
            loss, grads,ll,f = self.train_step(ic, cp)
            grads = tf.dynamic_stitch(self.sopt.idx, grads)
            self.epoch += 1
            self.hist.append(ll.numpy())
            self.residual.append(f.numpy())
            return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
        
        for epoch in range(self.epochs):
            tf.print('epoch:', self.epoch)
            loss, grads, ll,f = self.train_step(ic, cp)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(ll.numpy())
            self.residual.append(f.numpy())
            
            
        self.sopt.minimize(func)
            
        return self.hist, self.residual

#---------------------------------------------------------
    def predict(self, cp):

        cp = tf.convert_to_tensor(cp, tf.float32)

        u_p = self.model(cp)

        return u_p.numpy()


# New function for computing the derivatives 
#----------------------------------------------------------
    def auto_diff(self,cp):
        """
        Using the coordinates information to compute the derivatives 
        
        Args: 
            cp: A stack of the coordinates 

        Returns:

            d : (dict) A dictionary to store all the variables and derivatives in the Gov Eqs
        """ 

        # Initialize a dictionary
        cp = tf.convert_to_tensor(cp,tf.float32)

        d = {}

        x = cp[:, 0]
        y = cp[:, 1]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            X = tf.stack([x,y],axis=-1)
            # X = self.scalex(X)
            pred = self.model(X)
            # pred = self.scale_r(pred)
            u = pred[:, 0]
            uu = pred[:, 1]
            vv = pred[:, 2]
            uv = pred[:, 3]
            # Predicting as if there is no reference
            v = pred[:, 4]
            p = pred[:,5]


            u_x = tape.gradient(u, x)
            v_x = tape.gradient(v, x)
            p_x = tape.gradient(p, x)

            u_y = tape.gradient(u, y)
            v_y = tape.gradient(v, y)
            p_y = tape.gradient(p, y)


        uu_x = tape.gradient(uu,y)    
        uv_x = tape.gradient(uv,x)    
    
        uv_y = tape.gradient(uv,y)    
        vv_y = tape.gradient(vv,y)    
    
        # continuity 
        f0 = u_x + v_y
        # Momentum X
        f1 = u * u_x +  v * u_y + p_x + uu_x  + uv_y 
        # Momentum Y
        f2 = u * v_x +  v * v_y + p_y + uv_x  + vv_y 
        
        f = tf.stack([f0, f1, f2], axis = -1)

        # Writting all the variables
        
        ## No diff:
        ### U, V, P , uu, uv, vv
        d['U']  = u.numpy()
        d['V']  = v.numpy()
        d['P']  = p.numpy()
        d['uu'] = uu.numpy()
        d['uv'] = uv.numpy()
        d['vv'] = vv.numpy()
        
        ## 1-order diff 
        ### X-dir
        d['dUdx']   = u_x.numpy()
        d['dVdx']   = v_x.numpy()
        d['dPdx']   = p_x.numpy()
        d['duudx']  = uu_x.numpy()
        d['duvdx']  = uv_x.numpy()
        
        ### Y-dir 
        d['dUdy']   = u_y.numpy()
        d['dVdy']   = v_y.numpy()
        d['dPdy']   = p_y.numpy()
        d['duvdy']  = uv_y.numpy()
        d['dvvdy']  = uv_y.numpy()
        

        return d 