# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:04:27 2021

@author: Wang
"""

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from scipy.io import savemat

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, Z_theta, Z_RE, Z_T, Z_heat, theta, T, layers_theta, layers_k, layers_psi, layers_T, lb, ub):# Z_all for choose

        self.lb = lb
        self.ub = ub

        self.z = Z_theta[:, 0:1]
        self.t = Z_theta[:, 1:2]
        self.z_re = Z_RE[:, 0:1]
        self.t_re = Z_RE[:, 1:2]
        self.z_T = Z_T[:, 0:1]
        self.t_T = Z_T[:, 1:2]
        self.z_heat = Z_heat[:, 0:1]
        self.t_heat = Z_heat[:, 1:2]
        
        
        self.adaptive_constant_val = np.array(1.0)
        self.beta = 0.9
        self.theta = theta
        self.T = T

        self.layers_theta = layers_theta
        self.layers_k = layers_k
        self.layers_psi = layers_psi
        self.layers_T = layers_T

        # Initialize NNs
        self.weights_psi, self.biases_psi = self.initialize_NN('psi_', layers_psi)
        self.weights_T, self.biases_T = self.initialize_NN('Tem_', layers_T)
        self.weights_k, self.biases_k = self.initialize_NN('k_', layers_k)
        self.weights_kc, self.biases_kc = self.initialize_NN('kc_', layers_k)
        self.weights_theta, self.biases_theta = self.initialize_MNN('theta_', layers_theta)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.z_tf = tf.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.theta_tf = tf.placeholder(tf.float32, shape=[None, self.theta.shape[1]])
        self.T_tf = tf.placeholder(tf.float32, shape=[None, self.T.shape[1]])
        self.z_heat_tf = tf.placeholder(tf.float32, shape=[None, self.z_heat.shape[1]])
        self.t_heat_tf = tf.placeholder(tf.float32, shape=[None, self.t_heat.shape[1]])
        self.z_T_tf = tf.placeholder(tf.float32, shape=[None, self.z_T.shape[1]])
        self.t_T_tf = tf.placeholder(tf.float32, shape=[None, self.t_T.shape[1]])
        self.z_re_tf = tf.placeholder(tf.float32, shape=[None, self.z_re.shape[1]])
        self.t_re_tf = tf.placeholder(tf.float32, shape=[None, self.t_re.shape[1]])
        self.psi_tf = tf.placeholder(tf.float32, shape = [None, self.theta.shape[1]])
        
        self.adaptive_constant_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_val.shape)
        all_vars = tf.trainable_variables()
        psi_vars = [var for var in all_vars if 'psi_' in var.name]
        k_vars = [var for var in all_vars if 'k_' in var.name]
        theta_vars = [var for var in all_vars if 'theta_' in var.name]
        T_vars = [var for var in all_vars if 'Tem_' in var.name]
        kc_vars = [var for var in all_vars if 'kc_' in var.name]
        RRE_vars = psi_vars + k_vars + theta_vars
        couple_vars = T_vars + kc_vars

        self.T_pred = self.net_T(self.z_T_tf, self.t_T_tf, self.weights_T, self.biases_T)
        self.theta_pred, self.k_pred, self.psi_pred = self.net_RE(self.z_tf, self.t_tf)
        self.kc_pred = self.net_k(-tf.math.log(-self.psi_pred), self.weights_kc, self.biases_kc)
        self.theta_t, self.psi_z, self.psi_zz, self.K_z, self.f_RE, self.f_RE_c = self.net_f_RE(self.z_re_tf, self.t_re_tf)
        self.f_heat, self.lam, self.f1, self.f2, self.f3, self.D = self.net_f_heat_RE(self.z_heat_tf, self.t_heat_tf)  
              
        tf.log_h = tf.math.log(-self.psi_tf)
        self.WRC_theta = self.net_theta(-tf.log_h, self.weights_theta, self.biases_theta)
        self.HCF_K = self.net_k(-tf.log_h, self.weights_k, self.biases_k)
        self.HCF_Kc = self.net_k(-tf.log_h, self.weights_kc, self.biases_kc)
        
        
        self.loss_l2 = (self.cal(T_vars))/ tf.Variable([4500.0], dtype=tf.float32)
        self.loss_T1 = tf.reduce_mean(tf.square(self.T_tf - self.T_pred)) + self.loss_l2*tf.Variable([0.001], dtype=tf.float32) 
        self.loss_T = tf.reduce_mean(tf.square(self.T_tf - self.T_pred))
        self.loss_theta = tf.reduce_mean(tf.square(self.theta_tf - self.theta_pred))
        self.loss_heat = tf.reduce_mean(tf.square(self.f_heat))
        self.loss_RE = tf.reduce_mean(tf.square(self.f_RE_c))*self.adaptive_constant_tf
        self.loss_sm = self.loss_theta* tf.Variable([10.0], dtype=tf.float32) + tf.reduce_mean(tf.square(self.f_RE))

        
        self.heat_gr = tf.gradients(self.loss_heat,self.weights_kc)[0]
        self.re_gr = tf.gradients(self.loss_RE,self.weights_kc)[0]
        self.max_heat_gr= tf.reduce_max(tf.abs(self.heat_gr))
        self.mean_re_gr= tf.reduce_mean(tf.abs(self.re_gr)) 
        self.adaptive_constant = self.max_heat_gr / self.mean_re_gr
        self.loss_c = self.loss_heat  + self.loss_T* tf.Variable([100000.0], dtype=tf.float32) + self.loss_RE

        #learning rate  
        self.learning_rate = tf.train.exponential_decay(learning_rate=0.0005, global_step=tf.Variable(0, trainable=False), decay_steps=10000, decay_rate=0.99, staircase=True)
        self.learning_rate1 = tf.train.exponential_decay(learning_rate=0.0001, global_step=tf.Variable(0, trainable=False), decay_steps=10000, decay_rate=0.99, staircase=True)
        self.learning_rate2 = tf.train.exponential_decay(learning_rate=0.001, global_step=tf.Variable(0, trainable=False), decay_steps=10000, decay_rate=0.99, staircase=True)

        

        # L-BFGS-B method----------------------------------------------------------------------------------------------------------------------------------
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss_sm,
                                                        var_list = RRE_vars,
                                                        method = 'L-BFGS-B',
                                                        options = {'maxiter': 50000,
                                                                   'maxfun': 50000,
                                                                   'maxcor': 50,
                                                                   'maxls': 50,
                                                                   'ftol' : 1.0 * np.finfo(float).eps})

        # Adam method
        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate2)
        self.train_op_Adam = self.optimizer_Adam.minimize(loss=self.loss_sm, var_list=RRE_vars)
        



        # optimizer for T--------------------------------------------------------------------------------------------------------------------------------------------------
        self.optimizer_T = tf.contrib.opt.ScipyOptimizerInterface(self.loss_T1,
                                                        var_list = T_vars,
                                                        method = 'L-BFGS-B',
                                                        options = {'maxiter': 50000,
                                                                   'maxfun': 50000,
                                                                   'maxcor': 50,
                                                                   'maxls': 50,
                                                                   'ftol' : 1.0 * np.finfo(float).eps})
        # Adam method for T
        self.optimizer_Adam_T = tf.train.AdamOptimizer()
        self.train_op_Adam_T = self.optimizer_Adam_T.minimize(loss=self.loss_T1, var_list=T_vars)
       
        # optimizer for heat re----------------------------------------------------------------------------------------------------------------------------------
        self.optimizer_heat_re = tf.contrib.opt.ScipyOptimizerInterface(self.loss_c,
                                                        var_list = couple_vars,
                                                        method = 'L-BFGS-B',
                                                        options = {'maxiter': 50000,
                                                                    'maxfun': 50000,
                                                                    'maxcor': 50,
                                                                    'maxls': 50,
                                                                    'ftol' : 1.0 * np.finfo(float).eps})
        # Adam method for heat
        self.optimizer_Adam_heat_re = tf.train.AdamOptimizer(learning_rate=self.learning_rate2)
        self.train_op_Adam_heat_re = self.optimizer_Adam_heat_re.minimize(loss=self.loss_c, var_list=couple_vars)        

        
        #-----------------------------------------------------------------------------------------------------------------------------------------------------
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        # tf.saver
        self.saver = tf.train.Saver()
        print(tf.trainable_variables)
        
    def cal(self, W):
        for i, each in enumerate(W):
            if i == 0:
                f = tf.reduce_sum(tf.reduce_sum(tf.square(each)))
            else:
                f = f + tf.reduce_sum(tf.reduce_sum(tf.square(each)))
        return f
    def initialize_NN(self, name, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(name, size=[layers[l], layers[l + 1]])
            with tf.variable_scope(name):
                b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
    
    def initialize_MNN(self, name, layers): # monotonic neural network
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers-1):
            W = self.xavier_init(name, size = [layers[l],layers[l+1]])
            W2 = W**2
            with tf.variable_scope(name):
                b = tf.Variable(tf.zeros([1, layers[l+1]], dtype = tf.float32), dtype = tf.float32)
            weights.append(W2)
            biases.append(b)
        return weights, biases

    def xavier_init(self, name, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        with tf.variable_scope(name):
            w = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
        return w
    
    def net_k(self, X, weights, biases): 
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        K = tf.exp(Y)  
        return K
       
    
    def net_lam(self, X, weights, biases): 
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        lam = tf.exp(Y)  
        return lam
        
    def net_psi(self, z, t, weights, biases): 
        X = tf.concat([z, t],1)
        num_layers = len(weights) + 1
        # H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0 #normalization
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        psi = -tf.exp(Y)  
        return psi
    
    def net_theta(self, X, weights, biases): 
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        theta = tf.sigmoid(Y)
        return theta
    
    def net_T(self, z, t, weights, biases): 
        X = tf.concat([z, t],1)
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        T = tf.exp(Y)
        return T


    def net_RE(self, z, t):
        psi = self.net_psi(z, t, self.weights_psi, self.biases_psi)
        log_h = tf.math.log(-psi)
        theta = self.net_theta(-log_h, self.weights_theta, self.biases_theta)
        k = self.net_k(-log_h, self.weights_k, self.biases_k)
        
        return theta, k, psi
    
    def net_f_RE(self, z, t):  
        theta, k, psi = self.net_RE(z, t)
        kc = self.net_k(-tf.math.log(-psi), self.weights_kc, self.biases_kc)
        
        theta_t = tf.gradients(theta, t)[0]
        psi_z = tf.gradients(psi, z)[0]
        psi_zz = tf.gradients(psi_z, z)[0]
        K_z = tf.gradients(k, z)[0]
        Kc_z = tf.gradients(kc, z)[0]
        f = theta_t - K_z*psi_z - k*psi_zz - K_z
        f_pc = theta_t - Kc_z*psi_z - kc*psi_zz - Kc_z
        
        return theta_t, psi_z, psi_zz, K_z, f, f_pc


    def net_f_heat_RE(self, z, t): #loam
        # b1 = tf.constant(1.56728, dtype=tf.float32) #loam
        # b2 = tf.constant(2.53474, dtype=tf.float32)
        # b3 = tf.constant(9.89388, dtype=tf.float32)

        b1 = tf.constant(1.47054, dtype=tf.float32) #silt loam
        b2 = tf.constant(1.1287, dtype=tf.float32)
        b3 = tf.constant(7.29464, dtype=tf.float32)

        # b1 = tf.constant(1.65758, dtype=tf.float32) #sandy loam
        # b2 = tf.constant(-0.163178, dtype=tf.float32)
        # b3 = tf.constant(16.3307, dtype=tf.float32)


        beta = tf.constant(5, dtype=tf.float32)
        Cn_N = tf.constant(0.7882985, dtype=tf.float32) # TODO
        
        Cw = tf.constant(3.12035, dtype=tf.float32)      #10**14
        mult = tf.constant(100, dtype=tf.float32)
        eps = tf.constant(0.0000001, dtype=tf.float32)
        
        theta, k, psi = self.net_RE(z, t)
        k =  self.net_k(-tf.math.log(-psi), self.weights_kc, self.biases_kc)
        T = self.net_T(z, t, self.weights_T, self.biases_T)
        T_z = tf.gradients(T, z)[0]
        T_zz = tf.gradients(T_z, z)[0]
        theta_t = tf.gradients(theta, t)[0]
        
        psi_z = tf.gradients(psi, z)[0]
        flux = -k*(psi_z + 1.0)
        lam1 = (b1 + b2*theta + b3*tf.sqrt(theta+eps)) * mult
        lam2 = beta*Cw*tf.abs(flux) #10**16  10**14
        Cp = Cn_N + Cw*theta  #10**14

        Cpt = Cp*T

        f1 = tf.gradients(Cpt, t)[0]
        f21 = tf.gradients(lam1*T_z, z)[0]
        f22 = beta*Cw*(-flux*T_zz + T_z * theta_t)
        f2 = f21 + f22
        f3 = Cw * (flux*T_z - theta_t*T)
        f = (f1 - f2 + f3)
        D = beta*T_z+T
        return f, lam1+lam2, f1, f21+f22, f3, D
    

        
    def choose_plots_for_heat_train_3parts(self, Z_choose, N_heat, N_heat2,a, b): # a soil moisture, b psi_z
        tf_dict = {self.z_tf: Z_choose[:, 0:1], self.t_tf: Z_choose[:, 1:2], 
                   self.z_re_tf: Z_choose[:, 0:1], self.t_re_tf: Z_choose[:, 1:2]}
        theta_star = self.sess.run(self.theta_pred, tf_dict)
        psi_z_star = self.sess.run(self.psi_z, tf_dict)
        
        theta_max = np.max(theta_star)
        threshold1 = theta_max*a
        threshold2 = theta_max
        threshold3 = theta_max*0.9
        
        z_filt = Z_choose[:, 0:1] < 0 #TODO
        zr = np.array(z_filt ,dtype=np.int)
        
        theta_filt = theta_star > threshold3
        theta_filtt =threshold1 > theta_star
        ar1 = np.array(theta_filt ,dtype=np.int)
        ar11 = np.array(theta_filtt ,dtype=np.int)
        theta_filt2 = theta_star > threshold1
        theta_filt22 = threshold2 > theta_star
        ar2 = np.array(theta_filt2 ,dtype=np.int)
        ar22 = np.array(theta_filt22 ,dtype=np.int)
        theta_filt3 = theta_star > 0
        theta_filt33 = threshold1 > theta_star
        ar3 = np.array(theta_filt3 ,dtype=np.int)
        ar33 = np.array(theta_filt33 ,dtype=np.int)
        
        theta_t_filt = np.abs(psi_z_star) < b
        arpz = np.array(theta_t_filt ,dtype=np.int)
        arheat1 = ar1 + arpz + ar11 + zr
        heat_last1 = arheat1 > 3
        arheat2 = ar2 + arpz + ar22 +zr
        heat_last2 = arheat2 > 3
        arheat3 = ar3  + ar33
        heat_last3 = arheat3 > 1
        
        z_for_heat1 = Z_choose[:, 0:1][heat_last1].reshape((Z_choose[:, 0:1][heat_last1].shape[0], 1))
        t_for_heat1 = Z_choose[:, 1:2][heat_last1].reshape((Z_choose[:, 1:2][heat_last1].shape[0], 1))
        z_for_heat2 = Z_choose[:, 0:1][heat_last2].reshape((Z_choose[:, 0:1][heat_last2].shape[0], 1))
        t_for_heat2 = Z_choose[:, 1:2][heat_last2].reshape((Z_choose[:, 1:2][heat_last2].shape[0], 1))
        z_for_heat3 = Z_choose[:, 0:1][heat_last3].reshape((Z_choose[:, 0:1][heat_last3].shape[0], 1))
        t_for_heat3 = Z_choose[:, 1:2][heat_last3].reshape((Z_choose[:, 1:2][heat_last3].shape[0], 1))        
        
        N_heat3 = 500
        if Z_choose[:, 0:1][heat_last1].shape[0] < N_heat:
            N_heat = int(Z_choose[:, 0:1][heat_last1].shape[0]*0.8)
        if Z_choose[:, 0:1][heat_last2].shape[0] < N_heat2:
            N_heat2 = int(Z_choose[:, 0:1][heat_last2].shape[0]*0.8)
        if Z_choose[:, 0:1][heat_last3].shape[0] < N_heat3:
            N_heat3 = int(Z_choose[:, 0:1][heat_last3].shape[0]*0.8)
            
        heat_random1 = np.random.choice(z_for_heat1.shape[0], N_heat, False)
        heat_random2 = np.random.choice(z_for_heat2.shape[0], N_heat2, False)
        heat_random3 = np.random.choice(z_for_heat3.shape[0], N_heat3, False)

        return z_for_heat1[heat_random1], t_for_heat1[heat_random1], z_for_heat2[heat_random2], t_for_heat2[heat_random2], z_for_heat3[heat_random3], t_for_heat3[heat_random3]

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, N_iter):
        
        start_time = time.time()
        tf_dict = {self.z_tf: self.z, self.t_tf: self.t, self.theta_tf: self.theta,
                   self.z_re_tf: np.vstack((self.z_re, self.z)), self.t_re_tf: np.vstack((self.t_re, self.t))}
        # Adam
        for it in range(N_iter):
            self.sess.run(self.train_op_Adam, tf_dict)

            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss_sm, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %(it, loss_value, elapsed))
                start_time = time.time()
                
        # L-BFGS-B
        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss_sm],
                                loss_callback=self.callback)
        
        loss_value = self.sess.run(self.loss_sm, tf_dict)

        
    def train_for_T_batch(self, max_epoch, batch_size):      # ANN  batch   
    
        def get_batch(z, t, T, batch_size, it_one_epoch):
            idx = np.random.choice(T.shape[0], T.shape[0], replace = False)
            z = z[idx]
            t = t[idx]
            T = T[idx]
            z_batch = []
            t_batch = []
            T_batch = []
            for it in range(it_one_epoch):
                if it == it_one_epoch-1:
                    z_batch.append(z[-batch_size:,:])
                    t_batch.append(t[-batch_size:,:])
                    T_batch.append(T[-batch_size:,:])
                else:
                    z_batch.append(z[it*batch_size:(it+1)*batch_size,:])
                    t_batch.append(t[it*batch_size:(it+1)*batch_size,:])
                    T_batch.append(T[it*batch_size:(it+1)*batch_size,:])
            return z_batch, t_batch, T_batch
        it_one_epoch = int(np.ceil(self.T.shape[0]/batch_size))
            
        start_time = time.time()
        
        
        # Adam
        for it in range(max_epoch):
            z_batch, t_batch, T_batch = get_batch(self.z_T, self.t_T, self.T, batch_size, it_one_epoch)
            for i in range(it_one_epoch):
                train_T = T_batch[i]
                train_z = z_batch[i]
                train_t = t_batch[i]
                
                tf_dict = {self.z_T_tf: train_z, self.t_T_tf: train_t, self.T_tf: train_T}
                self.sess.run(self.train_op_Adam_T, tf_dict)
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss_T, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %(it, loss_value, elapsed))
                start_time = time.time()     
        # L-BFGS-B
        tf_dict = {self.z_T_tf: self.z_T, self.t_T_tf: self.t_T, self.T_tf: self.T}
        self.optimizer_T.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss_T],
                                loss_callback=self.callback)
        loss_value = self.sess.run(self.loss_T, tf_dict)




    def train_heat_annealing(self,z_heat, t_heat,z_re, t_re, N_iter):      
        start_time = time.time()
        
        # Adam 更新w
        for it in range(N_iter):
            tf_dict = {self.z_heat_tf: z_heat, self.t_heat_tf: t_heat, 
                   self.z_T_tf: self.z_T, self.t_T_tf: self.t_T, self.T_tf: self.T, 
                   self.z_re_tf: z_re, self.t_re_tf: t_re, self.adaptive_constant_tf: self.adaptive_constant_val}
            self.sess.run(self.train_op_Adam_heat_re, tf_dict)
            if it % 10 == 0:
                elapsed = time.time() - start_time
                adaptive_constant_value = self.sess.run(self.adaptive_constant, tf_dict)
                self.adaptive_constant_val = adaptive_constant_value * (1.0 - self.beta) + self.beta * self.adaptive_constant_val      
                
                loss_value = self.sess.run(self.loss_c, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f, ad: %.3e' %(it, loss_value, elapsed, self.adaptive_constant_val))
                start_time = time.time()    
        # L-BFGS-B
        self.optimizer_heat_re.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss_c],
                                loss_callback=self.callback)
        loss_value = self.sess.run(self.loss_c, tf_dict)

        
    def predict(self, Z_star):

        tf_dict = {self.z_tf: Z_star[:, 0:1], self.t_tf: Z_star[:, 1:2], 
                   self.z_T_tf: Z_star[:, 0:1], self.t_T_tf: Z_star[:, 1:2],
                   self.z_heat_tf: Z_star[:, 0:1], self.t_heat_tf: Z_star[:, 1:2],
                   self.z_re_tf: Z_star[:, 0:1], self.t_re_tf: Z_star[:, 1:2]}      
        psi_star = self.sess.run(self.psi_pred, tf_dict)
        K_star = self.sess.run(self.k_pred, tf_dict)
        Kc_star = self.sess.run(self.kc_pred, tf_dict)
        f_star = self.sess.run(self.f_RE, tf_dict)
        theta_t_star = self.sess.run(self.theta_t, tf_dict)
        psi_z_star = self.sess.run(self.psi_z, tf_dict)
        psi_zz_star = self.sess.run(self.psi_zz, tf_dict)
        K_z_star = self.sess.run(self.K_z, tf_dict)
        flux_star = -K_star*(psi_z_star + 1.0)
        fluxc_star = -Kc_star*(psi_z_star + 1.0)
        h_star = self.sess.run(self.f_heat, tf_dict)
        T_star = self.sess.run(self.T_pred, tf_dict)

        return theta_star, psi_star, K_star, f_star, theta_t_star, psi_z_star, psi_zz_star, K_z_star, flux_star, h_star, T_star, fluxc_star
    
    def predict_T(self, Z_star):

        T_star = self.sess.run(self.T_pred, {self.z_T_tf: Z_star[:, 0:1], self.t_T_tf: Z_star[:, 1:2]})

        return T_star
    
    def save_model(self, name, path):
        name = name + '.ckpt'
        save_path = self.saver.save(self.sess, path + '/' + name)   
        print("Model saved in path: %s" % save_path)
        
    def load_model(self, name, path):
        name = name + '.ckpt'
        self.saver.restore(self.sess, path + '/' + name)
        
    def WRC_HCF(self, psi_star):
        tf_dict = {self.psi_tf: psi_star}

        theta = self.sess.run(self.WRC_theta, tf_dict)
        K = self.sess.run(self.HCF_K, tf_dict)
        Kc = self.sess.run(self.HCF_Kc, tf_dict)
        
        return theta, K, Kc
    




if __name__ == "__main__":
    
    
    noise = 0.0
    noise_theta = 0

    N_u = 100
    N_f = 1500
    N_re = 2000
    layers_psi = [2, 40, 40, 40, 40, 40, 40, 1]
    layers_k = [1, 10, 1]
    layers_theta = [1, 10, 1]
    layers_T = [2, 40, 40, 40, 40, 1] 

    # data
    data = pd.read_csv("D:/case331_siltloam.csv")
    savepath = 'D:/'
    t_star = data['time'].values[:,None]
    z_star = data['depth'].values[:,None]
    psi_star = data['head'].values[:,None]
    K_star = data['K'].values[:,None]
    C_star = data['C'].values[:,None]
    theta_star = data['theta'].values[:,None]
    flux_star = data['flux'].values[:,None]
    T_star = data['T'].values[:,None]
    Z_star = np.hstack((z_star, t_star))
    for i in range(251):
        if i == 0:
            Z_choose = Z_star[i*1001: i*1001 + 201]
        else:
            Z_choose = np.vstack((Z_choose, Z_star[i*1001: i*1001 + 201]))
    
    depth_increment = 4 # 4
    fixed_position_full = [-0.05, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65, -0.75, -0.85, -0.95]  # dimensionless depth 
    fixed_position = fixed_position_full[::depth_increment] # change the number of virtual sensors 
    fixed_list_p = [int(i * (-200)) for i in fixed_position]
    
    for i in range(251):
        if i == 0:
            fixed_list = fixed_list_p
        else:
            fixed_list_plus = [each + 1001*i for each in fixed_list_p]
            fixed_list = np.append(fixed_list, fixed_list_plus)
            
    depth_increment_T = 2 # 5
    fixed_position_T = fixed_position_full[::depth_increment_T] # change the number of virtual sensors 
    fixed_list_p_T = [int(i * (-200)) for i in fixed_position_T]
    
    for i in range(251):
        if i == 0:
            fixed_list_T = fixed_list_p_T
        else:
            fixed_list_plus_T = [each + 1001*i for each in fixed_list_p_T]
            fixed_list_T = np.append(fixed_list_T, fixed_list_plus_T)
    
    # noise
    noise_theta = noise_theta*np.random.randn(theta_star.shape[0], theta_star.shape[1]) 
    theta_noise = theta_star + noise_theta
    noise_T = noise*np.random.randn(T_star.shape[0], T_star.shape[1])*np.std(T_star)
    T_noise = T_star + noise_T

    # fixed points 
    Z_theta_train = Z_star[fixed_list,:]
    theta_train = theta_star[fixed_list, :]
    Z_T_train = Z_star[fixed_list_T,:]
    T_train = T_noise[fixed_list_T, :]
   
    # Doman bounds, collocation points
    lb = Z_star.min(0)
    ub = Z_star.max(0)
    lb[0] = -20
    lhs_f = lhs(2, N_f)
    Z_heat_train = lb + (ub - lb) * lhs_f
    lhs_re = lhs(2, N_re)
    Z_RE_train = lb + (ub - lb) * lhs_re
    
    tf.reset_default_graph()    
    model = PhysicsInformedNN(Z_theta_train, Z_RE_train, Z_T_train, Z_heat_train, theta_train, T_train, layers_theta, layers_k, 
                              layers_psi, layers_T, lb, ub)
    start_time = time.time()
    
    # train T------------------------------------------------------------------------------------------------------------------------------------------------------
    model.train_for_T_batch(2000, 16) 
    
    # record T
    T_pred = model.predict_T(Z_star)
    dataset = pd.DataFrame({'z': z_star.flatten(), 't': t_star.flatten(),
                        'T_actual': T_star.flatten(), 'T_pred': T_pred.flatten(),})
    dataset.to_csv(f"{savepath}data_T.csv")
    
    # Train RRE------------------------------------------------------------------------------------------------------------------------------------------------------------
    model.train(50000)
    theta_pred, psi_pred, K_pred, f_pred, theta_t_pred, psi_z_pred, psi_zz_pred, K_z_pred, flux_pred, h_pred, T_pred, fluxc = model.predict(Z_star)
    dataset = pd.DataFrame({'z': z_star.flatten(), 't': t_star.flatten(),
                    'theta_actual': theta_star.flatten(), 'theta_pred': theta_pred.flatten(),
                    'T_actual': T_star.flatten(), 'T_pred': T_pred.flatten(),
                    'theta_noise': theta_noise.flatten(),
                    'psi_actual': psi_star.flatten(), 'psi_pred': psi_pred.flatten(),
                    'K_actual': K_star.flatten(), 'K_pred': K_pred.flatten(),
                    'flux_actual': flux_star.flatten(), 'flux_pred': flux_pred.flatten(),
                    'f_pred': f_pred.flatten(), 'theta_t_pred': theta_t_pred.flatten(),
                    'psi_z_pred': psi_z_pred.flatten(), 'psi_zz_pred': psi_zz_pred.flatten(),
                    'K_z_pred': K_z_pred.flatten(), 'h_pred': h_pred.flatten(),
                    'kflux_pred': fluxc.flatten(),})
    dataset.to_csv(f"{savepath}data_rre.csv")

    plots_heat_z1, plots_heat_t1, plots_heat_z2, plots_heat_t2, plots_heat_z3, plots_heat_t3 = model.choose_plots_for_heat_train_3parts(Z_choose, 500, 500, 0, 0.05) # a soil moisture, b psi_z
    savemat(f'{savepath}collocation_points.mat', {'z_heat_plot1': plots_heat_z1, 't_heat_plot1': plots_heat_t1,
                                                                          'z_heat_plot2': plots_heat_z2, 't_heat_plot2': plots_heat_t2,
                                                                          'z_heat_plot3': plots_heat_z3, 't_heat_plot3': plots_heat_t3,})      
    ## lookup table
    log_h_look = np.arange(-5, 3.4, 0.021)
    h_look = 10**log_h_look
    psi_look = -h_look.reshape(400,1)
    theta_look, K_look, kK = model.WRC_HCF(psi_look)

    lookup = pd.DataFrame({'theta': theta_look.flatten(), 'psi': psi_look.flatten(),
                            'K': K_look.flatten(), 'kK': kK.flatten()})

    lookup.to_csv(f"{savepath}lookup_rre.csv", index = False)   
    
    
       
    # train heat and rre(coupled)-------------------------------------------------------------------------------------------------------------------------------------------------------
    model.train_heat_annealing(np.vstack((plots_heat_z2)), np.vstack((plots_heat_t2)), Z_RE_train[:, 0:1], Z_RE_train[:, 1:2],50000)


    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    theta_pred, psi_pred, K_pred, f_pred, theta_t_pred, psi_z_pred, psi_zz_pred, K_z_pred, flux_pred, h_pred, T_pre, fluxc  = model.predict(Z_star)
    dataset = pd.DataFrame({'z': z_star.flatten(), 't': t_star.flatten(),
                    'theta_actual': theta_star.flatten(), 'theta_pred': theta_pred.flatten(),
                    'T_actual': T_star.flatten(), 'T_pred': T_pred.flatten(),
                    'theta_noise': theta_noise.flatten(),
                    'psi_actual': psi_star.flatten(), 'psi_pred': psi_pred.flatten(),
                    'K_actual': K_star.flatten(), 'K_pred': K_pred.flatten(),
                    'flux_actual': flux_star.flatten(), 'flux_pred': flux_pred.flatten(),
                    'f_pred': f_pred.flatten(), 'theta_t_pred': theta_t_pred.flatten(),
                    'psi_z_pred': psi_z_pred.flatten(), 'psi_zz_pred': psi_zz_pred.flatten(),
                    'kflux_pred': fluxc.flatten()})
    dataset.to_csv(f"{savepath}data_coupled.csv")
    
    ## lookup table
    theta_look, K_look, kK  = model.WRC_HCF(psi_look)

    lookup = pd.DataFrame({'theta': theta_look.flatten(), 'psi': psi_look.flatten(),
                            'K': K_look.flatten(), 'kK': kK.flatten()})

    lookup.to_csv(f"{savepath}lookup_coupled.csv", index = False) 
    
    # model.save_model()